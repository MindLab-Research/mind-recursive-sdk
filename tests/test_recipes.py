"""The recipes must actually run.

A recipe that does not execute is documentation, and documentation drifts. So
each one is driven end to end here against a real-socket fake relay with the
real Qwen tokenizer, and the pure functions (advantages, LR schedule, datum
construction) are tested directly because they carry the算法 correctness.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import numpy as np
import pytest

from tests.fake_relay import FakeRelay

RECIPES = Path(__file__).resolve().parent.parent / "recipes"
QWEN = "Qwen/Qwen3.8-27B"


def _load_recipe(name: str):
    """Import a recipe module by filename (they start with digits).

    Registered in ``sys.modules`` before execution: ``@dataclass`` resolves
    its own module by name at class-creation time, so a module that is not
    there yet fails with an opaque ``NoneType`` error.
    """
    if str(RECIPES) not in sys.path:
        sys.path.insert(0, str(RECIPES))
    module_name = name.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, RECIPES / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture(scope="module")
def tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if str(RECIPES) not in sys.path:
        sys.path.insert(0, str(RECIPES))
    try:
        from _renderer import get_renderer

        return get_renderer(QWEN).tokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{QWEN} tokenizer unavailable: {exc}")


@pytest.fixture
def relay(tokenizer) -> FakeRelay:
    r = FakeRelay()
    r.capabilities = {
        "supported_models": [
            {
                "model_name": QWEN,
                "max_context_length": 131072,
                "trainable": True,
                "sampleable": True,
            },
            {
                "model_name": "mock-model",
                "max_context_length": 4096,
                "trainable": True,
                "sampleable": True,
            },
        ]
    }
    r.sample_tokenizer = tokenizer
    return r


@pytest.fixture
def live(relay: FakeRelay, monkeypatch):
    """Serve the fake relay over real TCP and point the SDK at it."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def _dispatch(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            response = relay.handler(
                httpx.Request(
                    self.command,
                    f"http://127.0.0.1{self.path}",
                    content=body,
                    headers={k: v for k, v in self.headers.items()},
                )
            )
            payload = response.content
            self.send_response(response.status_code)
            self.send_header(
                "Content-Type", response.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _dispatch
        do_POST = _dispatch

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("MIND_RECURSIVE_API_KEY", "sk-recipe-test")
    monkeypatch.setenv("MIND_RECURSIVE_BASE_URL", f"http://127.0.0.1:{port}")
    try:
        yield relay
    finally:
        server.shutdown()


def _run(name: str, overrides: dict | None = None) -> int:
    """Execute a recipe's ``__main__`` with ``CONFIG`` fields overridden.

    The recipes take no command-line arguments -- you edit the ``CONFIG``
    dataclass at the top of the file and run it. So a test configures a run the
    same way a person does, by setting fields on that object, rather than by
    injecting ``sys.argv`` into a parser that no longer exists.

    Patched inside the module's own globals after import but before ``main()``,
    which is why the recipe is loaded with ``runpy`` under a non-``__main__``
    name first and its ``main()`` called explicitly.
    """
    module = _load_recipe(name)
    config = module.CONFIG
    for key, value in (overrides or {}).items():
        if not hasattr(config, key):
            raise AttributeError(f"{name} CONFIG has no field {key!r}")
        setattr(config, key, value)
    try:
        return int(module.main() or 0)
    except SystemExit as exc:
        return int(exc.code or 0)


# ------------------------------------------------------------------- execution
class TestRecipesRun:
    def test_smoke_recipe(self, live, capsys):
        assert _run("00_smoke.py", {"model": "mock-model"}) == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "mint://" in out

    def test_smoke_rejects_an_unavailable_model(self, live, capsys):
        code = _run("00_smoke.py", {"model": "not-served"})
        assert code == 1
        assert "not offered here" in capsys.readouterr().out

    def test_sft_recipe(self, live, capsys):
        assert _run("01_sft.py", {"epochs": 2, "batch_size": 3}) == 0
        out = capsys.readouterr().out
        assert "loss mask check" in out
        assert "sampler weights: mint://" in out
        # Two epochs over 8 demo conversations at batch 3 -> 3 steps each.
        assert "step 6/6" in out

    def test_sft_reports_metrics_and_summary(self, live):
        """Reporting is an HTTP write, so the recipe's numbers reach the relay."""
        _run("01_sft.py", {"epochs": 1, "batch_size": 4})

        metrics = live.requests_to("/api/mindrsi/v1/service/report/sess-0001/log")
        assert metrics
        # The recipes group metric keys as "<group>/<name>" so the console can
        # bucket them, so assert on the key the recipe actually sends -- a
        # substring check here would keep passing if the key were dropped.
        assert any(
            "loss/train" in json.loads(item.content)["metrics"] for item in metrics
        )

        summaries = live.requests_to("/api/mindrsi/v1/service/report/sess-0001/summary")
        assert any(
            "final_loss" in json.loads(item.content)["summary"]
            for item in summaries
        )

    def test_sft_writes_logs_and_samples_the_saved_weights(self, live):
        """The console's log tab needs type="log"; the saved path must be usable."""
        _run("01_sft.py", {"epochs": 1, "batch_size": 4})

        events = [
            event
            for item in live.requests_to("/api/mindrsi/v1/service/report/sess-0001/events")
            for event in json.loads(item.content)["events"]
        ]
        # 正常路径只走 log；event 留给跨侧交互（比如 run.failed），不该拿来讲进度。
        assert {event["type"] for event in events} == {"log"}
        log_messages = [event.get("message", "") for event in events]
        assert any("SFT started" in message for message in log_messages)
        assert any("step 2/2" in message for message in log_messages)

        # 存完权重才采样：先 publish，再 asample。
        paths = [r.path for r in live.requests]
        assert paths.index("/api/mindrsi/v1/train/save_weights_for_sampler") < paths.index(
            "/api/mindrsi/v1/sample/asample"
        )

    def test_reporting_posts_to_the_report_routes(self, live):
        """The write path is the report routes, not the platform read plane."""
        _run("01_sft.py", {"epochs": 1, "batch_size": 4})
        for suffix in ("log", "artifact", "summary"):
            assert live.requests_to(f"/api/mindrsi/v1/service/report/sess-0001/{suffix}")
        for suffix in ("log", "summary", "artifact", "events"):
            assert not live.requests_to(
                f"/api/mint/cli/v1/sessions/sess-0001/{suffix}"
            )

    def test_sft_sends_protobuf_and_records_the_artifact(self, live):
        _run("01_sft.py", {"epochs": 1, "batch_size": 8})
        fwdbwd = live.requests_to("/api/mindrsi/v1/train/forward_backward")
        assert fwdbwd
        assert fwdbwd[0].headers["content-type"] == "application/x-protobuf"

        artifacts = live.requests_to("/api/mindrsi/v1/service/report/sess-0001/artifact")
        assert artifacts
        body = json.loads(artifacts[0].content)
        assert body["type"] == "sampler_weights"
        assert body["uri"].startswith("mint://")

    def test_sft_full_recipe_with_eval_and_checkpoints(self, live, capsys):
        code = _run(
            "02_sft_full.py",
            {
                "epochs": 2,
                "batch_size": 2,
                "eval_every": 2,
                "save_every": 3,
                "patience": 0,
            },
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "val_loss" in out
        assert "checkpoint mint://" in out
        assert "best val loss" in out

    def test_sft_full_early_stops(self, live, capsys):
        """Flat validation loss must trigger the patience path, not run forever."""
        code = _run(
            "02_sft_full.py",
            {"epochs": 3, "batch_size": 2, "eval_every": 1, "patience": 2},
        )
        assert code == 0
        assert "no improvement" in capsys.readouterr().out

    def test_sft_full_resume_loads_weights(self, live):
        _run(
            "02_sft_full.py",
            {
                "epochs": 1,
                "batch_size": 4,
                "eval_every": 0,
                "resume_from": "mint://run-x/checkpoints/step-10-state",
            },
        )
        loads = live.requests_to("/api/mindrsi/v1/train/load_weights")
        assert loads and loads[0].json()["path"] == "mint://run-x/checkpoints/step-10-state"

    def test_grpo_recipe_with_discriminating_rewards(self, live, capsys):
        # Two correct, two wrong -> non-zero variance, so advantages exist.
        live.sample_texts = ["4", "7", "4", "nonsense"]
        code = _run("03_grpo.py", {"iterations": 2, "group_size": 4})
        assert code == 0
        out = capsys.readouterr().out
        assert "mean_reward" in out
        assert "sampler weights: mint://" in out

    def test_grpo_uses_importance_sampling(self, live):
        live.sample_texts = ["4", "7", "4", "nonsense"]
        _run("03_grpo.py", {"iterations": 1, "group_size": 4})
        # The loss function name lives in the protobuf body.
        from mind_recursive._proto import schema as S
        from mind_recursive._proto.codec import iter_fields

        body = live.requests_to("/api/mindrsi/v1/train/forward_backward")[0].content
        loss_fn = next(
            value
            for num, _wire, value in iter_fields(body)
            if num == S.ForwardBackwardRequest.LOSS_FN
        )
        assert loss_fn == b"importance_sampling"

    def test_grpo_publishes_weights_before_sampling(self, live):
        """A stale policy makes the importance ratio wrong, so order matters."""
        live.sample_texts = ["4", "7"]
        _run("03_grpo.py", {"iterations": 1, "group_size": 2})
        paths = [r.path for r in live.requests]
        assert paths.index("/api/mindrsi/v1/train/save_weights_for_sampler") < paths.index(
            "/api/mindrsi/v1/sample/asample"
        )

    def test_grpo_reports_degenerate_groups_instead_of_updating(self, live, capsys):
        """All-equal rewards carry no signal; a no-op update would be noise."""
        live.sample_texts = ["same", "same", "same", "same"]
        code = _run("03_grpo.py", {"iterations": 1, "group_size": 4})
        assert code == 0
        assert "degenerate" in capsys.readouterr().out
        assert live.count("/api/mindrsi/v1/train/optim_step") == 0

    def test_grpo_rejects_group_size_one(self, live, capsys):
        assert _run("03_grpo.py", {"group_size": 1}) == 2
        assert "at least 2" in capsys.readouterr().err

    def test_grpo_rejects_zero_temperature(self, live, capsys):
        assert _run("03_grpo.py", {"temperature": 0}) == 2
        assert "temperature" in capsys.readouterr().err


# -------------------------------------------------------------- pure functions
class TestGroupAdvantages:
    def test_normalises_to_zero_mean(self):
        grpo = _load_recipe("03_grpo.py")
        advantages = grpo.group_advantages([1.0, 2.0, 3.0, 4.0])
        assert advantages is not None
        assert sum(advantages) == pytest.approx(0.0, abs=1e-9)

    def test_better_than_average_is_positive(self):
        grpo = _load_recipe("03_grpo.py")
        advantages = grpo.group_advantages([0.0, 0.0, 1.0])
        assert advantages[2] > 0 and advantages[0] < 0

    def test_identical_rewards_return_none(self):
        """Zero variance: dividing would explode or silently yield zeros."""
        grpo = _load_recipe("03_grpo.py")
        assert grpo.group_advantages([1.0, 1.0, 1.0]) is None

    def test_single_sample_returns_none(self):
        grpo = _load_recipe("03_grpo.py")
        assert grpo.group_advantages([1.0]) is None


class TestRewardFunction:
    def test_exact_match_scores_above_one(self):
        grpo = _load_recipe("03_grpo.py")
        assert grpo.reward_fn("The answer is 4", {"answer": "4"}) > 1.0

    def test_wrong_answer_scores_zero(self):
        grpo = _load_recipe("03_grpo.py")
        assert grpo.reward_fn("The answer is 5", {"answer": "4"}) == 0.0

    def test_no_number_scores_zero(self):
        grpo = _load_recipe("03_grpo.py")
        assert grpo.reward_fn("I don't know", {"answer": "4"}) == 0.0

    def test_terser_correct_answers_score_higher(self):
        grpo = _load_recipe("03_grpo.py")
        terse = grpo.reward_fn("4", {"answer": "4"})
        verbose = grpo.reward_fn("Let me think. " * 20 + "4", {"answer": "4"})
        assert terse > verbose


class TestGrpoDatum:
    def test_prompt_positions_carry_no_gradient(self):
        """The prompt was not generated by the policy."""
        grpo = _load_recipe("03_grpo.py")
        datum = grpo.build_datum([1, 2, 3], [4, 5], [-0.5, -0.6], advantage=2.0)
        weights = datum.loss_fn_inputs["weights"].data
        # Position 2 predicts the first completion token, so weight starts there.
        assert weights[0] == 0.0 and weights[1] == 0.0
        assert weights[2] == pytest.approx(2.0)

    def test_advantage_lands_on_completion_positions(self):
        grpo = _load_recipe("03_grpo.py")
        datum = grpo.build_datum([1, 2], [3, 4, 5], [-0.1, -0.2, -0.3], advantage=-1.5)
        weights = datum.loss_fn_inputs["weights"].data
        assert set(np.unique(weights)) == {0.0, -1.5}

    def test_behaviour_logprobs_are_aligned_with_weights(self):
        """importance_sampling needs the sampler's logprob at the same position."""
        grpo = _load_recipe("03_grpo.py")
        datum = grpo.build_datum([1, 2], [3, 4], [-0.7, -0.8], advantage=1.0)
        weights = datum.loss_fn_inputs["weights"].data
        logprobs = datum.loss_fn_inputs["logprobs"].data
        for position, weight in enumerate(weights):
            if weight != 0:
                assert logprobs[position] != 0.0

    def test_targets_are_shifted_left(self):
        grpo = _load_recipe("03_grpo.py")
        datum = grpo.build_datum([1, 2], [3, 4], [-0.1, -0.2], advantage=1.0)
        tokens = datum.model_input.to_ints()
        targets = datum.loss_fn_inputs["target_tokens"].data.tolist()
        assert tokens == [1, 2, 3, 4]
        assert targets == [2, 3, 4, 0]

    def test_all_three_tensors_are_the_same_length(self):
        grpo = _load_recipe("03_grpo.py")
        datum = grpo.build_datum([1, 2, 3], [4, 5], [-0.1, -0.2], advantage=1.0)
        lengths = {
            name: len(tensor.data) for name, tensor in datum.loss_fn_inputs.items()
        }
        assert len(set(lengths.values())) == 1, lengths


class TestLearningRateSchedule:
    def test_warmup_starts_low_and_reaches_peak(self):
        sft = _load_recipe("02_sft_full.py")
        total, peak = 100, 1e-4
        assert sft.lr_at(0, total, peak) < peak
        warmup_end = max(1, int(total * 0.03))
        assert sft.lr_at(warmup_end - 1, total, peak) == pytest.approx(peak)

    def test_decays_toward_a_floor_not_zero(self):
        """A zero LR at the end wastes the final steps."""
        sft = _load_recipe("02_sft_full.py")
        final = sft.lr_at(99, 100, 1e-4)
        assert 0 < final < 1e-4
        assert final == pytest.approx(1e-5, rel=0.2)

    def test_monotonically_decreases_after_warmup(self):
        sft = _load_recipe("02_sft_full.py")
        values = [sft.lr_at(step, 100, 1e-4) for step in range(5, 100)]
        assert all(a >= b for a, b in zip(values, values[1:]))


class TestCommonHelpers:
    def test_batched_covers_every_item(self):
        common = _load_recipe("_common.py")
        batches = list(common.batched(list(range(7)), 3))
        assert [len(b) for b in batches] == [3, 3, 1]
        assert [x for b in batches for x in b] == list(range(7))

    def test_batched_rejects_zero(self):
        common = _load_recipe("_common.py")
        with pytest.raises(ValueError):
            list(common.batched([1, 2], 0))

    def test_load_jsonl_reports_the_offending_line(self, tmp_path):
        common = _load_recipe("_common.py")
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"messages": []}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2"):
            common.load_jsonl(bad)

    def test_load_jsonl_requires_the_field(self, tmp_path):
        common = _load_recipe("_common.py")
        path = tmp_path / "x.jsonl"
        path.write_text('{"other": 1}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="messages"):
            common.load_jsonl(path)

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        common = _load_recipe("_common.py")
        path = tmp_path / "x.jsonl"
        path.write_text('{"messages": [1]}\n\n{"messages": [2]}\n', encoding="utf-8")
        assert common.load_jsonl(path) == [[1], [2]]

    def test_empty_file_is_an_error_not_an_empty_run(self, tmp_path):
        common = _load_recipe("_common.py")
        path = tmp_path / "empty.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no usable records"):
            common.load_jsonl(path)

    def test_fmt_handles_absent_metrics(self):
        common = _load_recipe("_common.py")
        assert common.fmt(None) == "n/a"
        assert common.fmt(0.5) == "0.5000"
