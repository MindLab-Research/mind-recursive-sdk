"""``client/config`` parsing, the heartbeat thread, and the public types."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

import mind_recursive
from mind_recursive._session import ClientConfig, Heartbeat
from mind_recursive.types import MINT_ADAM_CONSTRAINT


class TestClientConfig:
    def test_observed_alpha_payload_parses(self):
        """The exact payload captured from alpha, so drift is visible here."""
        config = ClientConfig.from_wire(
            {
                "pjwt_auth_enabled": False,
                "credential_default_source": "api_key",
                "parallel_fwdbwd_chunks": True,
                "proto_write_fwdbwd": False,
                "proto_compress_fwdbwd": False,
                "fwd_via_fwdbwd": False,
                "fwdbwd_max_chunk_len": 1024,
                "fwdbwd_max_chunk_bytes_count": 50000000,
                "sample_max_concurrent_requests": 2000,
                "sample_join_sampling_session": True,
                "use_pyqwest_transport": False,
            }
        )
        assert config.chunk_budget() == (1024, 50_000_000)
        # pjwt_auth_enabled=false is why credential-command auth cannot work.
        assert config.pjwt_auth_enabled is False
        assert config.credential_default_source == "api_key"
        # An unmodelled flag is kept rather than dropped.
        assert config.extra["use_pyqwest_transport"] is False

    def test_missing_keys_keep_defaults(self):
        config = ClientConfig.from_wire({"fwdbwd_max_chunk_len": 64})
        assert config.fwdbwd_max_chunk_len == 64
        assert config.fwdbwd_max_chunk_bytes_count == 50_000_000

    def test_empty_or_none_payload_is_all_defaults(self):
        assert ClientConfig.from_wire(None).chunk_budget() == (1024, 50_000_000)
        assert ClientConfig.from_wire({}).chunk_budget() == (1024, 50_000_000)

    def test_malformed_value_is_ignored_not_fatal(self):
        """A surprising config value must not stop a run from starting."""
        config = ClientConfig.from_wire({"fwdbwd_max_chunk_len": "not a number"})
        assert config.fwdbwd_max_chunk_len == 1024

    def test_numeric_booleans_are_coerced(self):
        assert ClientConfig.from_wire({"proto_compress_fwdbwd": 1}).proto_compress_fwdbwd is True


class TestHeartbeat:
    def test_beats_repeatedly_until_stopped(self):
        beats = []
        heartbeat = Heartbeat(lambda: beats.append(time.monotonic()), interval=0.01)
        heartbeat.start()
        time.sleep(0.08)
        heartbeat.stop()
        assert len(beats) >= 2

    def test_does_not_beat_immediately(self):
        """The session was just created; it is already fresh."""
        beats = []
        heartbeat = Heartbeat(lambda: beats.append(1), interval=5.0)
        heartbeat.start()
        time.sleep(0.05)
        heartbeat.stop()
        assert beats == []

    def test_failures_are_survived_and_counted(self):
        """One missed beat must not kill the thread or the run."""
        calls = {"n": 0}

        def beat():
            calls["n"] += 1
            raise RuntimeError("network hiccup")

        heartbeat = Heartbeat(beat, interval=0.01)
        heartbeat.start()
        time.sleep(0.08)
        still_running = heartbeat.running
        heartbeat.stop()
        assert calls["n"] >= 2
        assert still_running
        assert heartbeat.consecutive_failures >= 2

    def test_a_terminal_failure_stops_the_loop(self):
        """410 Gone means the session is reclaimed; retrying the beat forever
        is noise, so the contract says stop."""
        from mind_recursive._errors import GoneError

        def beat():
            raise GoneError("session reclaimed")

        heartbeat = Heartbeat(beat, interval=0.01, stop_on=(GoneError,))
        heartbeat.start()
        time.sleep(0.08)
        assert not heartbeat.running
        assert isinstance(heartbeat.terminal_error, GoneError)
        heartbeat.stop()

    def test_failure_count_resets_after_success(self):
        state = {"n": 0}

        def beat():
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("one bad beat")

        heartbeat = Heartbeat(beat, interval=0.01)
        heartbeat.start()
        time.sleep(0.08)
        heartbeat.stop()
        assert heartbeat.consecutive_failures == 0

    def test_thread_is_daemon(self):
        """A forgotten heartbeat must never keep a process alive."""
        heartbeat = Heartbeat(lambda: None, interval=10)
        heartbeat.start()
        assert heartbeat._thread.daemon is True
        heartbeat.stop()

    def test_start_is_idempotent(self):
        heartbeat = Heartbeat(lambda: None, interval=10)
        heartbeat.start()
        first = heartbeat._thread
        heartbeat.start()
        assert heartbeat._thread is first
        heartbeat.stop()

    def test_stop_joins_the_thread(self):
        heartbeat = Heartbeat(lambda: None, interval=0.01)
        heartbeat.start()
        thread = heartbeat._thread
        heartbeat.stop()
        assert not thread.is_alive()
        assert heartbeat.running is False

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
    def test_nonsense_intervals_are_rejected(self, bad):
        """A zero or NaN interval spins or never fires; fail at construction."""
        with pytest.raises(ValueError):
            Heartbeat(lambda: None, interval=bad)

    def test_context_manager_stops_on_exit(self):
        with Heartbeat(lambda: None, interval=0.01) as heartbeat:
            assert heartbeat.running
        assert heartbeat.running is False


class TestAdamParams:
    def test_defaults_match_the_backend_constraint(self):
        """These differ from Tinker's defaults on purpose."""
        params = mind_recursive.AdamParams(learning_rate=1e-4)
        assert params.violations() == {}
        for name, required in MINT_ADAM_CONSTRAINT.items():
            assert getattr(params, name) == required

    def test_violations_report_given_and_required(self):
        params = mind_recursive.AdamParams(beta2=0.95, eps=1e-12)
        violations = params.violations()
        assert violations["beta2"] == (0.95, 0.999)
        assert violations["eps"] == (1e-12, 1e-8)

    def test_learning_rate_is_free(self):
        assert mind_recursive.AdamParams(learning_rate=3e-5).violations() == {}


class TestLoraConfig:
    def test_train_unembed_is_off_by_default(self):
        """Training the LM head yields weights that fail to deploy."""
        assert mind_recursive.LoraConfig().train_unembed is False

    def test_seed_is_omitted_when_unset(self):
        assert "seed" not in mind_recursive.LoraConfig(rank=8).to_wire()
        assert mind_recursive.LoraConfig(rank=8, seed=42).to_wire()["seed"] == 42


class TestModelInput:
    def test_from_ints_makes_one_text_chunk(self):
        model_input = mind_recursive.ModelInput.from_ints([1, 2, 3])
        assert len(model_input.chunks) == 1
        assert model_input.length == 3

    def test_append_extends_the_trailing_text_chunk(self):
        model_input = mind_recursive.ModelInput.from_ints([1])
        model_input.append_int(2)
        assert len(model_input.chunks) == 1
        assert model_input.to_ints() == [1, 2]

    def test_append_after_an_image_starts_a_new_chunk(self):
        model_input = mind_recursive.ModelInput(
            chunks=[mind_recursive.ImageChunk(data=b"x", format="png")]
        )
        model_input.append_int(5)
        assert len(model_input.chunks) == 2

    def test_to_ints_refuses_to_silently_drop_an_image(self):
        model_input = mind_recursive.ModelInput(
            chunks=[
                mind_recursive.EncodedTextChunk([1]),
                mind_recursive.ImageChunk(data=b"x", format="png"),
            ]
        )
        with pytest.raises(ValueError, match="cannot flatten"):
            model_input.to_ints()


class TestDatum:
    def test_from_tokens_builds_a_cross_entropy_datum(self):
        datum = mind_recursive.Datum.from_tokens([1, 2, 3])
        assert set(datum.loss_fn_inputs) == {"weights", "target_tokens"}
        # Default targets are the inputs shifted left by one.
        assert datum.loss_fn_inputs["target_tokens"].data.tolist() == [2, 3, 0]
        assert datum.loss_fn_inputs["weights"].data.tolist() == [1.0, 1.0, 1.0]

    def test_explicit_weights_are_kept(self):
        datum = mind_recursive.Datum.from_tokens([1, 2, 3], weights=[0.0, 0.0, 1.0])
        assert datum.loss_fn_inputs["weights"].data.tolist() == [0.0, 0.0, 1.0]

    def test_plain_lists_are_coerced_to_tensors(self):
        """Tinker coerces in __post_init__; a bare list used to survive
        construction and only fail later inside the protobuf encoder."""
        datum = mind_recursive.Datum(
            model_input=mind_recursive.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={"weights": [1.0, 1.0, 0.0], "target_tokens": [2, 3, 0]},
        )

        for value in datum.loss_fn_inputs.values():
            assert isinstance(value, mind_recursive.TensorData)

    def test_a_coerced_datum_encodes(self):
        from mind_recursive._proto.encode import encode_datum

        datum = mind_recursive.Datum(
            model_input=mind_recursive.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={"weights": [1.0, 1.0, 0.0], "target_tokens": [2, 3, 0]},
        )

        assert encode_datum(datum)

    def test_dtype_comes_from_the_key_not_the_values(self):
        """Integer-looking weights are still float32 on the wire."""
        datum = mind_recursive.Datum(
            model_input=mind_recursive.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={"weights": [1, 0, 1], "target_tokens": [2, 3, 0]},
        )

        assert datum.loss_fn_inputs["weights"].dtype == "float32"
        assert datum.loss_fn_inputs["target_tokens"].dtype == "int64"

    def test_numpy_arrays_are_accepted(self):
        datum = mind_recursive.Datum(
            model_input=mind_recursive.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={"advantages": np.array([0.5, 0.5, 0.5])},
        )

        assert datum.loss_fn_inputs["advantages"].dtype == "float32"

    def test_an_unknown_key_falls_back_to_the_arrays_kind(self):
        datum = mind_recursive.Datum(
            model_input=mind_recursive.ModelInput.from_ints([1]),
            loss_fn_inputs={"mystery": [1, 2, 3]},
        )

        assert datum.loss_fn_inputs["mystery"].dtype == "int64"

    def test_an_existing_tensor_is_not_rewrapped(self):
        tensor = mind_recursive.TensorData(np.array([1.0]), "float32")

        datum = mind_recursive.Datum(
            model_input=mind_recursive.ModelInput.from_ints([1]),
            loss_fn_inputs={"weights": tensor},
        )

        assert datum.loss_fn_inputs["weights"] is tensor

    def test_non_numeric_values_are_refused(self):
        with pytest.raises(TypeError, match="numbers"):
            mind_recursive.Datum(
                model_input=mind_recursive.ModelInput.from_ints([1]),
                loss_fn_inputs={"weights": ["a", "b"]},
            )

    def test_a_datum_is_frozen(self):
        """A training example is a value: rebinding one after it has been
        batched would silently change what a step trained on."""
        datum = mind_recursive.Datum.from_tokens([1, 2, 3])

        with pytest.raises(dataclasses.FrozenInstanceError):
            datum.loss_fn_inputs = {}


class TestForwardBackwardOutput:
    def test_named_accessors_read_the_metric_dict(self):
        out = mind_recursive.ForwardBackwardOutput(
            metrics={"loss:mean": 0.25, "num_tokens:sum": 512.0}
        )
        assert out.loss == 0.25
        assert out.num_tokens == 512
        assert isinstance(out.num_tokens, int)

    def test_absent_metrics_are_none_not_zero(self):
        out = mind_recursive.ForwardBackwardOutput()
        assert out.loss is None
        assert out.num_tokens is None


class TestImportHygiene:
    def test_importing_does_not_touch_os_environ(self, monkeypatch):
        """The deprecated toolkit rewrote env vars at import time."""
        monkeypatch.setenv("MIND_RECURSIVE_API_KEY", "sk-probe")
        monkeypatch.delenv("TINKER_API_KEY", raising=False)
        monkeypatch.delenv("TINKER_BASE_URL", raising=False)

        import importlib

        import mind_recursive as module

        importlib.reload(module)
        import os

        assert "TINKER_API_KEY" not in os.environ
        assert "TINKER_BASE_URL" not in os.environ
        assert os.environ["MIND_RECURSIVE_API_KEY"] == "sk-probe"

    def test_tinker_shim_reexports_our_own_types(self):
        from mind_recursive import tinker

        assert tinker.Datum is mind_recursive.Datum
        assert issubclass(tinker.ServiceClient, mind_recursive.ServiceClient)

    def test_shim_omits_the_over_privileged_clients(self):
        """RestClient/AdminClient reach global resources through the relay."""
        from mind_recursive import tinker

        assert not hasattr(tinker, "RestClient")
        assert not hasattr(tinker, "AdminClient")


class TestSessionName:
    def test_name_is_sent_in_create_session_and_exposed(self):
        from tests.fake_relay import FakeRelay, build_service

        relay = FakeRelay()
        service = build_service(relay, name="GRPO on GSM8K", client_id="exp-1")
        try:
            assert service.name == "GRPO on GSM8K"
            body = relay.requests_to("/api/mindrsi/v1/service/session/create")[0].json()
            assert body["name"] == "GRPO on GSM8K"
            assert body["client_id"] == "exp-1"
        finally:
            service.close()

    def test_name_is_omitted_when_not_given(self):
        from tests.fake_relay import FakeRelay, build_service

        relay = FakeRelay()
        service = build_service(relay)
        try:
            assert service.name is None
            body = relay.requests_to("/api/mindrsi/v1/service/session/create")[0].json()
            assert "name" not in body
        finally:
            service.close()
