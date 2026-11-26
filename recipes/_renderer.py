"""Messages -> tokens + loss mask. **Recipe-local, not part of the SDK.**

Rendering is not the SDK's job: it depends on the model's chat template and on
how a given dataset wants its reasoning blocks supervised, which is dataset and
experiment policy, not transport. ``mind_recursive`` deals in ``Datum`` -- tokens plus
tensors -- and stops there. This file lives with the recipes so they can run,
and is the kind of thing that belongs in the cookbook.

It is pure local computation: no network call. That is exactly why it has to be
right. The renderer must agree with the server's tokenizer token-for-token,
because a mask that is off by one trains on the wrong positions and the
resulting weights drift *without any error being raised*.

## How the mask is built, and why not the obvious way

The obvious approach is to render the prompt, render the full conversation, and
diff the token lists. **That is wrong**, and measurably so. With
``Qwen/Qwen3.8-27B`` the generation prompt ends in ``<think>\\n``, whose final
newline tokenises as ``198`` (``"\\n"``) on its own, but in the full
conversation that position is ``271`` (``"\\n\\n"``) because the following
newline merges into the same token. So::

    encode(prompt_text)  is NOT  a prefix of  encode(full_text)

even though ``full_text.startswith(prompt_text)`` holds. A prefix-diff would
silently shift every mask by one token.

What we do instead: tokenise the full conversation **once**, with
``return_offsets_mapping=True``, and ask the tokenizer where each token sits in
the original string. Supervised regions are character spans, computed from the
chat template, and a token is supervised when it falls inside one. Token
boundaries never need to line up with span boundaries, so the merge above is
harmless.

## Relationship to the training image's renderer

Platform-side renderers exist (for example
``compute/_mint_glm52.py``, ~1200 lines) which subclass ``tinker_cookbook``
classes and handle a richer message contract -- structured thinking parts,
serving-leftover ``</think>`` fragments, vision inputs. Those remain the
reference for parity with the training image.

This file is deliberately *not* that. It is built on the model's own HF chat
template, so it has no cookbook dependency and works for any model whose
template is published. Where the two can disagree is reasoning-block handling;
``supervise_reasoning`` below is the control for that. It has **not** been
compared token-for-token against the training image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from mind_recursive import Datum, ModelInput, TensorData

__all__ = ["Renderer", "RenderedConversation", "get_renderer", "load_tokenizer"]

log = logging.getLogger("recipes.renderer")

Message = dict[str, Any]


@dataclass
class RenderedConversation:
    """One conversation as tokens plus a per-token loss weight."""

    tokens: list[int]
    weights: list[float]
    #: The exact string the tokenizer saw. Keep it: when a mask looks wrong this
    #: is the only way to see what the template actually produced.
    text: str

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.weights):
            raise ValueError(
                f"tokens ({len(self.tokens)}) and weights ({len(self.weights)}) "
                "must be the same length"
            )

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def supervised_tokens(self) -> int:
        return int(sum(1 for w in self.weights if w > 0))

    def to_datum(self) -> Datum:
        """Convert to a ``cross_entropy`` datum.

        Next-token prediction, so position *i* is trained to predict token
        *i+1*: targets are the input shifted left, and the final position has
        nothing to predict and gets weight 0.
        """
        if len(self.tokens) < 2:
            raise ValueError(
                "a conversation needs at least 2 tokens to form a training "
                f"example, got {len(self.tokens)}"
            )
        targets = [*self.tokens[1:], 0]
        weights = [*self.weights[1:], 0.0]
        return Datum(
            model_input=ModelInput.from_ints(self.tokens),
            loss_fn_inputs={
                "weights": TensorData(np.asarray(weights, dtype=np.float32), "float32"),
                "target_tokens": TensorData(np.asarray(targets, dtype=np.int64), "int64"),
            }
        )

    def debug_spans(self, tokenizer: Any, limit: int = 40) -> str:
        """Render the mask as readable text, for eyeballing it.

        ``[supervised]`` vs plain text. Reach for this whenever loss looks
        wrong before suspecting the optimiser.
        """
        out: list[str] = []
        run: list[int] = []
        state = False
        for token, weight in zip(self.tokens, self.weights):
            supervised = weight > 0
            if supervised != state and run:
                piece = tokenizer.decode(run)
                out.append(f"[{piece}]" if state else piece)
                run = []
            state = supervised
            run.append(token)
        if run:
            piece = tokenizer.decode(run)
            out.append(f"[{piece}]" if state else piece)
        text = "".join(out)
        if limit and len(text) > limit * 40:
            return text[: limit * 40] + " ...(truncated)"
        return text


def load_tokenizer(model_name: str, **kwargs: Any) -> Any:
    """Load a HF tokenizer, with a clear error when it is not installed.

    ``transformers`` is a recipe dependency, not an SDK one: the data plane
    never tokenises anything.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "rendering needs a tokenizer: pip install transformers jinja2"
        ) from exc
    kwargs.setdefault("trust_remote_code", False)
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


@dataclass
class Renderer:
    """Turns chat messages into tokens and a loss mask for one model.

    Args:
        model_name: HF model id, e.g. ``Qwen/Qwen3.8-27B``. Must be the model
            you are training: a mask built with a different tokenizer is
            wrong in a way nothing will report.
        tokenizer: a preloaded tokenizer; loaded from ``model_name`` if absent.
        supervise_reasoning: whether tokens inside a ``<think>`` block count as
            supervised. Default ``True``, matching the training image. Set
            ``False`` to train only the visible answer.
        max_length: drop conversations longer than this many tokens. ``None``
            keeps everything, and the server will reject anything over its
            context limit.

    Example::

        r = get_renderer("Qwen/Qwen3.8-27B")
        datum = r.render([
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ]).to_datum()
    """

    model_name: str
    tokenizer: Any = None
    supervise_reasoning: bool = True
    max_length: int | None = None
    #: Roles whose content is supervised. Only assistant turns by default.
    supervised_roles: tuple[str, ...] = ("assistant",)
    _think_close: str = field(default="</think>", repr=False)

    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.model_name)
        template = getattr(self.tokenizer, "chat_template", None)
        if not template:
            raise ValueError(
                f"{self.model_name} has no chat_template, so conversations "
                "cannot be rendered; supply a tokenizer that has one"
            )

    # -- template helpers -----------------------------------------------------
    def _apply(self, messages: Sequence[Message], *, add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    def _supervised_spans(self, messages: Sequence[Message], full: str) -> list[tuple[int, int]]:
        """Character spans of the text to put loss on.

        For each supervised turn, the span runs from the end of the generation
        prompt that precedes it to the end of that turn -- i.e. exactly what the
        model would have had to produce. The role header and template
        scaffolding stay outside, because the model does not generate them.
        """
        spans: list[tuple[int, int]] = []
        for index, message in enumerate(messages):
            if message.get("role") not in self.supervised_roles:
                continue
            prompt = self._apply(messages[:index], add_generation_prompt=True)
            through = self._apply(messages[: index + 1], add_generation_prompt=False)

            # Templates are expected to be append-only across turns. If this
            # fails the template rewrites history (some tool-calling templates
            # do), and span arithmetic would be meaningless -- so say so rather
            # than emit a quietly wrong mask.
            #
            # Both renders must be checked. For the final message ``through``
            # equals ``full`` even under a rewriting template, so testing only
            # that would miss single-turn rewrites; ``prompt`` is what actually
            # anchors the span start.
            if not full.startswith(prompt) or not full.startswith(through):
                raise ValueError(
                    f"{self.model_name}'s chat template is not append-only at "
                    f"message {index}: rendering messages[:{index + 1}] is not a "
                    "prefix of the full conversation, so loss spans cannot be "
                    "located. Render turns separately, or use a renderer written "
                    "for this template."
                )

            start, end = len(prompt), len(through)
            if self.supervise_reasoning is False:
                start = self._skip_reasoning(full, start, end)
            if end > start:
                spans.append((start, end))
        return spans

    def _skip_reasoning(self, text: str, start: int, end: int) -> int:
        """Move ``start`` past a ``</think>`` inside this turn, if present."""
        close = text.find(self._think_close, start, end)
        if close == -1:
            return start
        return close + len(self._think_close)

    # -- the public call ------------------------------------------------------
    def render(self, messages: Sequence[Message]) -> RenderedConversation:
        """Render one conversation.

        Raises:
            ValueError: no supervised turn, or the template is not append-only.
        """
        if not messages:
            raise ValueError("cannot render an empty conversation")
        full = self._apply(messages, add_generation_prompt=False)
        spans = self._supervised_spans(messages, full)
        if not spans:
            roles = [m.get("role") for m in messages]
            raise ValueError(
                f"no supervised turn in this conversation (roles: {roles}); "
                f"at least one {'/'.join(self.supervised_roles)} message is required"
            )

        encoded = self.tokenizer(
            full, add_special_tokens=False, return_offsets_mapping=True
        )
        tokens = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])

        # Offsets, not a token-level prefix diff: see the module docstring for
        # the token-merge that makes prefix diffing unsound.
        weights = [0.0] * len(tokens)
        for position, (begin, stop) in enumerate(offsets):
            if stop <= begin:  # zero-width: special tokens carry no source span
                continue
            for span_start, span_end in spans:
                if begin >= span_start and stop <= span_end:
                    weights[position] = 1.0
                    break

        return RenderedConversation(tokens=tokens, weights=weights, text=full)

    def render_to_datum(self, messages: Sequence[Message]) -> Datum:
        """``render`` then ``to_datum``, the common path."""
        return self.render(messages).to_datum()

    def render_many(
        self, conversations: Iterable[Sequence[Message]], *, skip_errors: bool = True
    ) -> list[Datum]:
        """Render a batch, optionally skipping the examples that fail.

        Args:
            conversations: message lists.
            skip_errors: when true (default) an unrenderable or over-long
                conversation is logged and dropped rather than killing a
                training job partway through a dataset. Set false while
                developing a data pipeline, when you want to see the failure.
        """
        out: list[Datum] = []
        skipped = 0
        for index, messages in enumerate(conversations):
            try:
                rendered = self.render(messages)
                if self.max_length is not None and len(rendered) > self.max_length:
                    raise ValueError(
                        f"{len(rendered)} tokens exceeds max_length={self.max_length}"
                    )
                if rendered.supervised_tokens == 0:
                    raise ValueError("no supervised tokens after masking")
                out.append(rendered.to_datum())
            except (ValueError, KeyError) as exc:
                if not skip_errors:
                    raise
                skipped += 1
                log.debug("skipping conversation %d: %s", index, exc)
        if skipped:
            log.warning("skipped %d of %d conversations", skipped, skipped + len(out))
        return out

    def prompt_tokens(self, messages: Sequence[Message]) -> list[int]:
        """Tokens for a prompt awaiting a completion -- the sampling entry point.

        Includes the generation prompt, so the model continues the assistant
        turn rather than re-opening it.
        """
        text = self._apply(messages, add_generation_prompt=True)
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(self, tokens: Sequence[int], *, skip_special: bool = True) -> str:
        """Tokens back to text, for reading what the model sampled."""
        return self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special)

    def __repr__(self) -> str:
        return (
            f"<Renderer {self.model_name} "
            f"supervise_reasoning={self.supervise_reasoning}>"
        )


#: Renderers are expensive to build (the tokenizer load dominates) and are
#: stateless once built, so they are cached per configuration.
_CACHE: dict[tuple[str, bool, int | None], Renderer] = {}


def get_renderer(
    model_name: str,
    *,
    supervise_reasoning: bool = True,
    max_length: int | None = None,
    tokenizer: Any = None,
) -> Renderer:
    """Get a renderer for a model, reusing one when possible.

    Example::

        r = get_renderer("Qwen/Qwen3.8-27B", max_length=4096)
    """
    key = (model_name, supervise_reasoning, max_length)
    if tokenizer is None and key in _CACHE:
        return _CACHE[key]
    renderer = Renderer(
        model_name=model_name,
        tokenizer=tokenizer,
        supervise_reasoning=supervise_reasoning,
        max_length=max_length,
    )
    if tokenizer is None:
        _CACHE[key] = renderer
    return renderer
