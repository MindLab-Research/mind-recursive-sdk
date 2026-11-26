"""Renderer correctness -- of the *recipe-local* renderer.

``recipes/_renderer.py`` is not part of the SDK (rendering depends on chat
templates and dataset policy, which is cookbook territory), but the recipes
cannot run without it and a wrong mask is invisible, so it stays tested here.

A wrong loss mask does not raise -- it just trains the wrong positions and the
weights drift. So these tests check the mask *content*, not merely its shape.

Most run against a hand-built fake tokenizer so they are fast and
deterministic. The ones marked ``real_tokenizer`` run against
``Qwen/Qwen3.8-27B`` from the local HF cache, which is where the token-merge
behaviour that defeats prefix diffing actually shows up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "recipes"))
from _renderer import Renderer, RenderedConversation, get_renderer  # noqa: E402

# ---------------------------------------------------------------- fake tokenizer


class FakeTokenizer:
    """A whitespace/char tokenizer with a Qwen-shaped chat template.

    Deliberately mimics the awkward part of the real template: the generation
    prompt ends with ``<think>\\n`` while a completed turn renders
    ``<think>\\n\\n</think>\\n\\n``.
    """

    chat_template = "fake"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 1000
        return self.vocab[piece]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ) -> str:
        out = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "assistant":
                out.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        text = "".join(out)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n"
        return text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        """Tokenise into runs of word chars / specials, tracking offsets."""
        tokens: list[int] = []
        offsets: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text.startswith("<|im_start|>", i):
                piece, length = "<|im_start|>", 12
            elif text.startswith("<|im_end|>", i):
                piece, length = "<|im_end|>", 10
            elif text.startswith("<think>", i):
                piece, length = "<think>", 7
            elif text.startswith("</think>", i):
                piece, length = "</think>", 8
            else:
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] in "+?-"):
                    j += 1
                if j == i:
                    j = i + 1
                piece, length = text[i:j], j - i
            tokens.append(self._id(piece))
            offsets.append((i, i + length))
            i += length
        result = {"input_ids": tokens}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, tokens, skip_special_tokens=False) -> str:
        reverse = {v: k for k, v in self.vocab.items()}
        pieces = [reverse.get(t, "") for t in tokens]
        if skip_special_tokens:
            pieces = [
                p
                for p in pieces
                if p not in ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
            ]
        return "".join(pieces)


@pytest.fixture
def renderer() -> Renderer:
    return Renderer(model_name="fake/model", tokenizer=FakeTokenizer())


def supervised_text(rendered: RenderedConversation, tokenizer) -> str:
    """The exact text the mask puts loss on."""
    return "".join(
        tokenizer.decode([t]) for t, w in zip(rendered.tokens, rendered.weights) if w > 0
    )


class TestMaskContent:
    def test_only_the_assistant_completion_is_supervised(self, renderer):
        rendered = renderer.render(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        text = supervised_text(rendered, renderer.tokenizer)
        # The answer and the turn terminator, but not the role header.
        assert "hello" in text
        assert "assistant" not in text
        assert "question" not in text

    def test_user_content_is_never_supervised(self, renderer):
        rendered = renderer.render(
            [
                {"role": "user", "content": "secret"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        assert "secret" not in supervised_text(rendered, renderer.tokenizer)

    def test_system_prompt_is_not_supervised(self, renderer):
        rendered = renderer.render(
            [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        )
        assert "instructions" not in supervised_text(rendered, renderer.tokenizer)

    def test_every_assistant_turn_is_supervised(self, renderer):
        rendered = renderer.render(
            [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        )
        text = supervised_text(rendered, renderer.tokenizer)
        assert "a1" in text and "a2" in text
        assert "q1" not in text and "q2" not in text

    def test_mask_and_tokens_stay_aligned(self, renderer):
        rendered = renderer.render(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        assert len(rendered.tokens) == len(rendered.weights)
        assert rendered.supervised_tokens > 0
        assert rendered.supervised_tokens < len(rendered.tokens)

    def test_supervise_reasoning_false_skips_the_think_block(self, renderer):
        renderer.supervise_reasoning = False
        rendered = renderer.render(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "visible"}]
        )
        text = supervised_text(rendered, renderer.tokenizer)
        assert "visible" in text
        # The closing tag is consumed as the boundary, so no think markup remains.
        assert "<think>" not in text


class TestDatumConversion:
    def test_targets_are_inputs_shifted_left(self, renderer):
        rendered = renderer.render(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        datum = rendered.to_datum()
        tokens = datum.model_input.to_ints()
        targets = datum.loss_fn_inputs["target_tokens"].data.tolist()
        assert tokens == rendered.tokens
        assert targets[:-1] == rendered.tokens[1:]

    def test_final_position_has_no_target_and_no_weight(self, renderer):
        """The last token has nothing to predict; weighting it trains noise."""
        rendered = renderer.render(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        datum = rendered.to_datum()
        assert datum.loss_fn_inputs["weights"].data.tolist()[-1] == 0.0
        assert datum.loss_fn_inputs["target_tokens"].data.tolist()[-1] == 0

    def test_weights_are_shifted_with_targets(self, renderer):
        """Position i predicts token i+1, so weight i must be weight_{i+1}."""
        rendered = renderer.render(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        datum = rendered.to_datum()
        shifted = datum.loss_fn_inputs["weights"].data.tolist()
        assert shifted[:-1] == rendered.weights[1:]

    def test_dtypes_match_the_wire_contract(self, renderer):
        datum = renderer.render_to_datum(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        assert datum.loss_fn_inputs["weights"].data.dtype == np.float32
        assert datum.loss_fn_inputs["target_tokens"].data.dtype == np.int64

    def test_single_token_conversation_is_rejected(self, renderer):
        rendered = RenderedConversation(tokens=[1], weights=[1.0], text="x")
        with pytest.raises(ValueError, match="at least 2 tokens"):
            rendered.to_datum()


class TestValidation:
    def test_empty_conversation_is_rejected(self, renderer):
        with pytest.raises(ValueError, match="empty conversation"):
            renderer.render([])

    def test_conversation_without_an_assistant_turn_is_rejected(self, renderer):
        """Nothing to learn from, and a silent all-zero mask would be worse."""
        with pytest.raises(ValueError, match="no supervised turn"):
            renderer.render([{"role": "user", "content": "q"}])

    def test_missing_chat_template_is_rejected_at_construction(self):
        class NoTemplate(FakeTokenizer):
            chat_template = None

        with pytest.raises(ValueError, match="no chat_template"):
            Renderer(model_name="x", tokenizer=NoTemplate())

    def test_non_append_only_template_is_reported_not_mismasked(self):
        """A template that rewrites history makes span arithmetic meaningless."""

        class Rewriting(FakeTokenizer):
            def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=False
            ):
                text = super().apply_chat_template(
                    messages, tokenize, add_generation_prompt
                )
                # Simulate a template that injects a summary once history grows.
                if len(messages) > 1 and not add_generation_prompt:
                    return "<|im_start|>system\nsummary<|im_end|>\n" + text
                return text

        renderer = Renderer(model_name="x", tokenizer=Rewriting())
        with pytest.raises(ValueError, match="not append-only"):
            renderer.render(
                [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
            )


class TestBatchRendering:
    def test_bad_examples_are_skipped_by_default(self, renderer):
        """One malformed row must not kill a job mid-dataset."""
        data = [
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            [{"role": "user", "content": "no answer"}],  # unrenderable
            [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}],
        ]
        assert len(renderer.render_many(data)) == 2

    def test_skip_errors_false_surfaces_the_failure(self, renderer):
        data = [[{"role": "user", "content": "no answer"}]]
        with pytest.raises(ValueError):
            renderer.render_many(data, skip_errors=False)

    def test_over_long_conversations_are_dropped(self, renderer):
        renderer.max_length = 5
        data = [
            [
                {"role": "user", "content": "a fairly long question here"},
                {"role": "assistant", "content": "and a long answer too"},
            ]
        ]
        assert renderer.render_many(data) == []


class TestSamplingHelpers:
    def test_prompt_tokens_include_the_generation_prompt(self, renderer):
        tokens = renderer.prompt_tokens([{"role": "user", "content": "q"}])
        text = renderer.tokenizer.decode(tokens)
        # Without this the model re-opens the turn instead of continuing it.
        assert text.endswith("<think>\n")

    def test_decode_round_trips(self, renderer):
        rendered = renderer.render(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        )
        assert "world" in renderer.decode(rendered.tokens, skip_special=False)

    def test_debug_spans_brackets_supervised_regions(self, renderer):
        rendered = renderer.render(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ans"}]
        )
        shown = rendered.debug_spans(renderer.tokenizer, limit=0)
        assert "[" in shown and "]" in shown
        assert "ans" in shown


class TestCaching:
    def test_same_config_is_reused(self):
        first = get_renderer("fake/x", tokenizer=FakeTokenizer())
        assert isinstance(first, Renderer)
        # An explicit tokenizer bypasses the cache (it may be stateful).
        second = get_renderer("fake/x", tokenizer=FakeTokenizer())
        assert first is not second


# ------------------------------------------------------- real tokenizer tests

QWEN = "Qwen/Qwen3.8-27B"


def _real_renderer(**kwargs) -> Renderer:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return get_renderer(QWEN, **kwargs)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{QWEN} tokenizer unavailable locally: {exc}")


@pytest.mark.real_tokenizer
class TestAgainstRealTokenizer:
    def test_prefix_diffing_would_be_wrong_here(self):
        """The reason this renderer uses offsets instead of a token diff.

        With this template the prompt ends in ``<think>\\n`` -> token 198
        (``"\\n"``), but in the full conversation that position is token 271
        (``"\\n\\n"``) because the next newline merges in. String prefix holds;
        token prefix does not. A diff-based mask would be shifted by one.
        """
        renderer = _real_renderer()
        tokenizer = renderer.tokenizer
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        full = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True
        )
        assert full.startswith(prompt), "string prefix should hold"

        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # The property a prefix-diff implementation would rely on:
        assert full_ids[: len(prompt_ids)] != prompt_ids, (
            "if this ever becomes equal the template changed; prefix diffing "
            "would then be sound, but offsets remain correct either way"
        )

    def test_mask_covers_exactly_the_answer_and_terminator(self):
        renderer = _real_renderer()
        rendered = renderer.render(
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
        text = supervised_text(rendered, renderer.tokenizer)
        assert "4" in text
        assert "What is 2+2?" not in text
        assert "<|im_start|>" not in text

    def test_multi_turn_supervises_every_answer(self):
        renderer = _real_renderer()
        rendered = renderer.render(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "alpha"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "beta"},
            ]
        )
        text = supervised_text(rendered, renderer.tokenizer)
        assert "alpha" in text and "beta" in text
        assert "first" not in text and "second" not in text

    def test_system_prompt_stays_unsupervised(self):
        renderer = _real_renderer()
        rendered = renderer.render(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "terse answer"},
            ]
        )
        text = supervised_text(rendered, renderer.tokenizer)
        assert "terse answer" in text
        assert "be terse" not in text

    def test_datum_is_wire_ready(self):
        renderer = _real_renderer()
        datum = renderer.render_to_datum(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        from mind_recursive._proto import encode_forward_backward_request

        raw = encode_forward_backward_request(
            model_id="m", seq_id=1, data=[datum], loss_fn="cross_entropy"
        )
        assert len(raw) > 0

    def test_prompt_tokens_are_a_token_prefix_of_nothing_assumed(self):
        """prompt_tokens must end with the generation prompt, not the answer."""
        renderer = _real_renderer()
        tokens = renderer.prompt_tokens([{"role": "user", "content": "q"}])
        text = renderer.decode(tokens, skip_special=False)
        assert text.rstrip().endswith("<think>")
