"""End-to-end training against the fake relay.

These are the tests that matter most: they drive the real client code through
the real transport, future-polling, protobuf and chunking paths, and assert on
the bytes and headers the SDK put on the wire.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import mind_recursive
from mind_recursive._proto import schema as S
from mind_recursive._proto.codec import iter_fields
from mind_recursive._errors import RequestFailedError
from tests.fake_relay import FakeRelay, build_service, patch_user

SERVICE = "/api/mindrsi/v1/service"
SESSION = "/api/mindrsi/v1/service/session"
TRAIN = "/api/mindrsi/v1/train"
SERVICE = "/api/mindrsi/v1/service"


@pytest.fixture
def relay() -> FakeRelay:
    return FakeRelay()


@pytest.fixture
def service(relay: FakeRelay):
    client = build_service(relay, client_id="client-xyz")
    yield client
    client.close()


@pytest.fixture
def user(relay: FakeRelay):
    client = mind_recursive.UserClient(
        api_key="sk-test-key", base_url="https://relay.test"
    )
    patch_user(client, relay)
    yield client
    client.close()


def _batch(n: int = 2, length: int = 4) -> list[mind_recursive.Datum]:
    return [
        mind_recursive.Datum.from_tokens(list(range(i * length, i * length + length)))
        for i in range(n)
    ]


class TestUserClient:
    """The session-less half of the surface.

    The split exists so that asking what the backend serves cannot leave a
    billable session record behind -- the gap that made the old
    ``connect_only=True`` flag necessary.
    """

    def test_construction_sends_nothing(self, relay):
        client = mind_recursive.UserClient(
            api_key="sk-x", base_url="https://relay.test"
        )
        patch_user(client, relay)
        assert relay.requests == []
        client.close()

    def test_capabilities_creates_no_session(self, user, relay):
        caps = user.get_server_capabilities()
        assert caps.names() == ["Qwen/Qwen3.8-27B"]
        assert caps.find("Qwen/Qwen3.8-27B").max_context_length == 131072
        assert relay.count(f"{SESSION}/create") == 0

    def test_health_check_reports_rather_than_raising(self, user, relay):
        assert user.health_check() is True
        relay.fail_paths[f"{SERVICE}/healthz"] = (503, {"error": "down"})
        # The point of a probe is to be usable in an `if`.
        assert user.health_check() is False

    def test_queries_send_the_key_and_no_custom_headers(self, user, relay):
        user.get_server_capabilities()
        headers = relay.last(f"{SERVICE}/get_server_capabilities").headers
        assert headers["x-api-key"] == "sk-test-key"
        assert "authorization" not in headers
        assert not [k for k in headers if k.startswith("x-mind_recursive")]

    def test_a_closed_client_refuses_to_query(self, user):
        user.close()
        with pytest.raises(mind_recursive.ConfigurationError):
            user.get_server_capabilities()


class TestSessionLifecycle:
    def test_construction_opens_exactly_one_session(self, service, relay):
        """Constructing the client *is* creating the session: no lazy mode."""
        assert service.session_id == "sess-0001"
        assert relay.count(f"{SESSION}/create") == 1
        # Addressing the session again does not open a second one.
        assert service.require_session() == service.session_id
        assert relay.count(f"{SESSION}/create") == 1

    def test_config_is_negotiated_before_the_session_is_created(self, service, relay):
        paths = [r.path for r in relay.requests]
        # The config response governs how later requests must behave, so it
        # has to be fetched before the session is used.
        assert paths.index(f"{SERVICE}/client/config") < paths.index(
            f"{SESSION}/create"
        )

    def test_session_creation_prints_the_web_console_link(
        self, relay, monkeypatch, caplog
    ):
        """Opening a session hands the caller the page for this run.

        The link is terminal-only (it is not a report), and the env var exists
        so a non-production console can be pointed at without changing code.
        """
        monkeypatch.setenv(
            "MIND_RECURSIVE_WEB_URL", "https://web.example.com/sessions"
        )
        with caplog.at_level(logging.INFO, logger="mind_recursive"):
            client = build_service(relay)
        try:
            assert "Session created: sess-0001" in caplog.text
            assert (
                "Web console: https://web.example.com/sessions/sess-0001"
                in caplog.text
            )
        finally:
            client.close()

    def test_a_closed_client_refuses_to_hand_out_its_session(self, service):
        service.close()
        with pytest.raises(mind_recursive.SessionClosedError):
            service.require_session()

    def test_client_config_values_are_adopted(self, relay):
        relay.client_config = {
            "fwdbwd_max_chunk_len": 7,
            "fwdbwd_max_chunk_bytes_count": 999,
            "proto_compress_fwdbwd": True,
            "some_future_flag": "keep me",
        }
        client = build_service(relay, api_key="sk-x")
        config = client.client_config
        assert config.chunk_budget() == (7, 999)
        assert config.proto_compress_fwdbwd is True
        # Unknown flags are kept, not dropped, so a newer server loses nothing.
        assert config.extra["some_future_flag"] == "keep me"
        client.close()

    def test_config_failure_falls_back_to_defaults(self, relay):
        relay.fail_paths[f"{SERVICE}/client/config"] = (500, {"error": "down"})
        client = build_service(relay, api_key="sk-x", max_retries=0)
        # A wobbly config endpoint must not stop training from starting.
        assert client.client_config.fwdbwd_max_chunk_len == 1024
        client.close()

    def test_api_key_is_sent_raw_in_x_api_key(self, service, relay):
        headers = relay.requests[0].headers
        # No tml- disguise: owning the transport is what removes that patch.
        assert headers["x-api-key"] == "sk-test-key"
        assert "authorization" not in headers

    def test_client_id_is_a_create_session_field_not_a_header(self, service, relay):
        """It labels the session, so it belongs on the session.

        It is neither a credential nor a routing key, and neither Tinker nor
        Fireworks declares an attribution header; sending one would make every
        request carry a field only ``create_session`` reads.
        """
        created = relay.last(f"{SESSION}/create")
        assert created.json()["client_id"] == "client-xyz"
        for request in relay.requests:
            assert "x-mind_recursive-client-id" not in request.headers

    def test_no_client_id_means_the_field_is_absent(self, relay):
        client = build_service(relay, api_key="sk-x")
        assert "client_id" not in relay.last(f"{SESSION}/create").json()
        client.close()

    def test_load_weights_declares_whether_optimizer_state_comes_back(
        self, service, relay
    ):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.load_weights("mint://run-abc/checkpoints/step-1").result()
        tc.load_weights("mint://run-abc/checkpoints/step-1", optimizer=False).result()

        bodies = [
            request.json()
            for request in relay.requests_to(f"{TRAIN}/load_weights")
        ]
        assert [body["optimizer"] for body in bodies] == [True, False]

    def test_the_upstream_load_names_keep_their_meaning(self, service, relay):
        """``load_state`` is weights-only upstream; ours must not drift.

        These two cannot be aliases of ``load_weights`` -- its default is
        ``optimizer=True`` -- so the wire is the only place the promise is
        visible. Getting it wrong restores optimizer state a caller asked not
        to, which shows up as a resume that does not pick up cleanly rather
        than as an error.
        """
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.load_state("mint://run-abc/checkpoints/step-1").result()
        tc.load_state_with_optimizer("mint://run-abc/checkpoints/step-1").result()

        bodies = [
            request.json()
            for request in relay.requests_to(f"{TRAIN}/load_weights")
        ]
        assert [body["optimizer"] for body in bodies] == [False, True]


class TestTrainingStep:
    def test_full_step_round_trip(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=16)
        assert tc.model_id == "sess-0001_0"

        out = tc.forward_backward(_batch()).result()
        assert out.loss == pytest.approx(0.5)
        assert out.num_tokens == 128

        step = tc.optim_step().result()
        assert step["grad_norm"] == pytest.approx(1.25)

        path = tc.save_weights_only("sampler-001").result()
        assert path == "mint://run-abc/sampler/sampler-001"

    def test_sampler_save_sends_the_name_as_path(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.save_weights_only("sampler-001").result()

        sent = relay.requests_to(f"{TRAIN}/save_weights_for_sampler")[0]
        assert sent.json()["path"] == "sampler-001"

    def test_retrieve_future_408_is_pending_not_an_error(self, service, relay):
        """MinT signals pending with 408; the future must keep polling.

        Treating it as a transport error was a real regression: sampling failed
        on the first poll even though the request was still running.
        """
        relay.pending_status = 408
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        relay.pending_polls = 1

        path = tc.save_weights_only("sampler-408").result()

        assert path.endswith("sampler-408")
        polls = [
            item for item in relay.requests_to(f"{TRAIN}/retrieve_future")
            if b"req-save" in item.content
        ]
        assert len(polls) == 2

    def test_a_successful_save_reports_an_artifact(self, service, relay, monkeypatch):
        """The save knows the path and kind, so the caller should not repeat them."""
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        future = tc.save_weights_only(
            "sampler-001", user_metadata={"val_loss": "0.125"}
        )
        path = future.result()
        future.result()  # cached: the artifact must not be reported twice

        # Lifecycle events share this channel, so count artifacts only: the
        # claim under test is that one save reports one artifact, twice-called
        # future included.
        artifacts = [item for item in records if item[0] == "artifact"]
        assert len(artifacts) == 1
        kind, body = artifacts[0]
        assert body["uri"] == path
        assert body["type"] == "sampler_weights"
        assert body["name"] == "sampler-001"
        assert body["metadata"] == {"val_loss": "0.125"}

    def test_a_training_state_save_reports_a_checkpoint(self, service, relay, monkeypatch):
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        path = tc.save_state("state-001").result()

        assert [
            (kind, body["type"], body["uri"])
            for kind, body in records
            if kind == "artifact"
        ] == [("artifact", "checkpoint", path)]

    def test_creating_a_trainer_registers_it(self, service, relay, monkeypatch):
        """The SDK holds the ids, so the console should not depend on the caller."""
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=16)

        events = [body for kind, body in records if kind == "event"]
        assert [event["type"] for event in events] == ["trainer.created"]
        assert events[0]["data"] == {
            "model_id": tc._model_id,
            "base_model": "Qwen/Qwen3.8-27B",
            "rank": 16,
        }

    def test_resuming_registers_the_checkpoint_it_came_from(
        self, service, relay, monkeypatch
    ):
        """A resume is worth telling apart from a fresh create, with its source."""
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        state = "mint://run-abc/checkpoints/step-3"
        service.create_lora_trainer_from_state(
            state, base_model="Qwen/Qwen3.8-27B", load_optimizer=True
        )

        events = [body for kind, body in records if kind == "event"]
        assert [event["type"] for event in events] == ["trainer.resumed"]
        assert events[0]["data"]["state_path"] == state
        assert events[0]["data"]["load_optimizer"] is True

    def test_a_failed_create_registers_nothing(self, service, relay, monkeypatch):
        """Same rule as a failed save: no resolved id, no claim that it exists."""
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        monkeypatch.setattr(
            service,
            "_submit_on",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("create refused")),
        )
        with pytest.raises(RuntimeError):
            service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        assert [body["type"] for kind, body in records if kind == "event"] == []

    def test_closing_records_the_status_the_wire_drops(
        self, service, relay, monkeypatch
    ):
        """``close(status=...)`` is not transmitted, so the console is its only home."""
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        service.close("success", detail="all 3 steps done")

        events = [body for kind, body in records if kind == "event"]
        assert [event["type"] for event in events] == ["session.closing"]
        assert events[0]["data"]["status"] == "success"
        assert events[0]["data"]["detail"] == "all 3 steps done"

    def test_closing_twice_records_once(self, service, relay, monkeypatch):
        """Idempotent close must not double-report."""
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        service.close()
        service.close()

        assert [body["type"] for kind, body in records if kind == "event"] == [
            "session.closing"
        ]

    @pytest.mark.parametrize(
        ("method", "endpoint"),
        [
            ("save_weights_only", "save_weights_for_sampler"),
            ("save_weights_for_sampler", "save_weights_for_sampler"),
            ("save_state", "save_weights"),
        ],
    )
    def test_save_requires_a_name(self, service, relay, method, endpoint):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        save = getattr(tc, method)

        with pytest.raises(TypeError):
            save()

        for bad in ("", ".", "..", "has space", "a/b", "中文"):
            with pytest.raises(mind_recursive.ConfigurationError):
                save(bad)

        # None of those names reached the wire.
        assert not relay.requests_to(f"{TRAIN}/{endpoint}")

    def test_train_state_save_hits_the_checkpoint_endpoint(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        path = tc.save_state("state-001", overwrite=True).result()
        assert path == "mint://run-abc/state/state-001"

        sent = relay.requests_to(f"{TRAIN}/save_weights")[0]
        assert sent.json()["type"] == "save_weights"
        assert sent.json()["overwrite"] is True
        assert sent.json()["path"] == "state-001"

    def test_weights_only_forwards_overwrite(self, service, relay):
        """A named weights-only save conflicts on reuse unless overwrite is set."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        tc.save_weights_only("adapter-001").result()
        tc.save_weights_only("adapter-001", overwrite=True).result()

        sent = [r.json() for r in relay.requests_to(f"{TRAIN}/save_weights_for_sampler")]
        assert [b["overwrite"] for b in sent] == [False, True]

    def test_save_state_sends_the_72h_default(self, service, relay):
        """A resume path is good for three days, and the number travels."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        tc.save_state("step-3").result()

        body = relay.last(f"{TRAIN}/save_weights").json()
        assert body["ttl_seconds"] == 259200

    def test_save_state_takes_an_explicit_ttl(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        tc.save_state("step-3", ttl_seconds=600).result()

        assert relay.last(f"{TRAIN}/save_weights").json()["ttl_seconds"] == 600

    def test_save_state_none_follows_the_server(self, service, relay):
        """``None`` still means "send no TTL" -- the sentinel keeps it usable."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        tc.save_state("step-3", ttl_seconds=None).result()

        assert "ttl_seconds" not in relay.last(f"{TRAIN}/save_weights").json()

    def test_weights_only_still_omits_the_ttl_by_default(self, service, relay):
        """The 72h default is the checkpoint's, not every save's."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        tc.save_weights_only("step-3").result()

        body = relay.last(f"{TRAIN}/save_weights_for_sampler").json()
        assert "ttl_seconds" not in body

    def test_the_sampling_save_is_always_a_throwaway(self, service, relay, monkeypatch):
        """Keeping weights is save_weights_only's job, not the sampler path's.

        The export carries the fixed ephemeral TTL and registers no artifact,
        so a caller cannot accidentally keep weights through this route.
        """
        records: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            service._reporter, "_record", lambda kind, body: records.append((kind, body))
        )
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        tc.save_weights_and_get_sampler()

        body = relay.last(f"{TRAIN}/save_weights_for_sampler").json()
        assert body["path"].startswith("ephemeral-")
        assert body["ttl_seconds"] == 86400
        assert "overwrite" not in body
        assert [kind for kind, _ in records if kind == "artifact"] == []

    def test_the_sampling_save_takes_no_name_or_ttl(self, service):
        """The signature is the contract: no name, no ttl, no overwrite."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        for call in (
            lambda: tc.save_weights_and_get_sampler("step-1"),
            lambda: tc.save_weights_and_get_sampler(ttl_seconds=60),
            lambda: tc.save_weights_and_get_sampler(overwrite=True),
            lambda: tc.save_weights_and_get_sampler(user_metadata={"a": "b"}),
        ):
            with pytest.raises(TypeError):
                call()

    def test_weights_only_rejects_the_reserved_ephemeral_name(self, service, relay):
        """The server keeps each model's ephemeral export under ``ephemeral``."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        # save_weights_and_get_sampler is not in this list: it takes no name
        # at all now, so the reserved prefix is unreachable through it.
        for save in (
            lambda: tc.save_weights_only("ephemeral"),
            lambda: tc.save_weights_for_sampler("ephemeral"),
        ):
            with pytest.raises(mind_recursive.ConfigurationError, match="reserved"):
                save()

        assert not relay.requests_to(f"{TRAIN}/save_weights_for_sampler")
        # Train-state checkpoints live in another namespace; the name is fine there.
        tc.save_state("ephemeral").result()

    def test_weights_only_rejects_save_state_positional_shape(self, service, relay):
        """save_state's third positional is overwrite; here it is user_metadata."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        with pytest.raises(mind_recursive.ConfigurationError, match="user_metadata"):
            tc.save_weights_only("adapter-001", 3600, True)

        assert not relay.requests_to(f"{TRAIN}/save_weights_for_sampler")

    def test_forward_backward_is_sent_as_protobuf(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.forward_backward(_batch()).result()

        sent = relay.requests_to(f"{TRAIN}/forward_backward")[0]
        assert sent.headers["content-type"] == "application/x-protobuf"
        # And it must be parseable as the schema says, not merely non-empty.
        fields = {num for num, _wire, _val in iter_fields(sent.content)}
        assert S.ForwardBackwardRequest.MODEL_ID in fields
        assert S.ForwardBackwardRequest.DATA in fields

    def test_optim_step_is_json_not_protobuf(self, service, relay):
        """Only forward_backward is binary; the rest of the step is JSON."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.optim_step().result()
        sent = relay.requests_to(f"{TRAIN}/optim_step")[0]
        assert sent.headers["content-type"] == "application/json"

    def test_seq_id_starts_at_one_and_increments(self, service, relay):
        """A seq_id of 0 trips the server's range check."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.forward_backward(_batch()).result()
        tc.optim_step().result()

        sent = relay.requests_to(f"{TRAIN}/forward_backward")[0]
        seq = next(
            value
            for num, _wire, value in iter_fields(sent.content)
            if num == S.ForwardBackwardRequest.SEQ_ID
        )
        assert seq == 1
        assert relay.requests_to(f"{TRAIN}/optim_step")[0].json()["seq_id"] == 2

    def test_adam_defaults_satisfy_the_backend_constraint(self, service, relay):
        """Defaults must be MinT's required values, not Tinker's."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.optim_step().result()
        params = relay.requests_to(f"{TRAIN}/optim_step")[0].json()["adam_params"]
        assert params["beta1"] == 0.9
        assert params["beta2"] == 0.999
        assert params["eps"] == 1e-8

    def test_violating_adam_params_warn_but_are_sent_verbatim(self, service, relay, caplog):
        """We do not silently rewrite a caller's hyperparameters."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        with caplog.at_level("WARNING"):
            tc.optim_step(mind_recursive.AdamParams(learning_rate=1e-4, beta2=0.95)).result()
        assert "rejected by the backend" in caplog.text
        assert relay.requests_to(f"{TRAIN}/optim_step")[0].json()["adam_params"]["beta2"] == 0.95

    def test_train_unembed_defaults_off(self, service, relay):
        """Training the LM head produces weights that fail to deploy."""
        service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=8)
        body = relay.requests_to(f"{SERVICE}/train/create_model")[0].json()
        assert body["lora_config"]["train_unembed"] is False
        assert body["lora_config"]["rank"] == 8

    def test_per_datum_logprobs_survive_the_round_trip(self, relay):
        relay.per_datum_logprobs = [[-0.1, -0.2, -0.3], [-0.4, -0.5]]
        client = build_service(relay, api_key="sk-x")
        tc = client.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        out = tc.forward_backward(_batch()).result()

        assert len(out.loss_fn_outputs) == 2
        np.testing.assert_allclose(
            out.loss_fn_outputs[0]["logprobs"].data, [-0.1, -0.2, -0.3], rtol=1e-6
        )
        np.testing.assert_allclose(
            out.loss_fn_outputs[1]["logprobs"].data, [-0.4, -0.5], rtol=1e-6
        )
        client.close()

    def test_empty_batch_is_rejected_locally(self, service):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        with pytest.raises(ValueError):
            tc.forward_backward([])


class TestFutureProtocol:
    def test_two_phase_protocol_is_used(self, service, relay):
        """Submissions return only a request_id; results need retrieve_future."""
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        # Trainer creation goes through the service plane (the recursive
        # server owns trainer lifecycle); its future is polled there too.
        assert relay.requests_to(f"{SERVICE}/train/create_model")[0]
        assert relay.count(f"{TRAIN}/create_model") == 0
        assert relay.count(f"{SERVICE}/retrieve_future") >= 1
        assert tc.model_id

    def test_try_again_is_polled_until_resolved(self, relay):
        relay.pending_polls = 3
        client = build_service(relay, api_key="sk-x")

        tc = client.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        out = tc.forward_backward(_batch()).result()

        assert out.loss == pytest.approx(0.5)
        # 4 polls per future (3 try_again + 1 resolve): create_model polls the
        # service plane, forward_backward the train plane.
        assert relay.count(f"{SERVICE}/retrieve_future") == 4
        assert relay.count(f"{TRAIN}/retrieve_future") == 4
        client.close()

    def test_result_is_cached(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        future = tc.forward_backward(_batch())
        first = future.result()
        before = relay.count(f"{TRAIN}/retrieve_future")
        second = future.result()
        assert first is second
        assert relay.count(f"{TRAIN}/retrieve_future") == before

    def test_timeout_leaves_the_future_collectable(self, relay):
        """A timeout is not terminal: the work is still running upstream."""
        client = build_service(relay, api_key="sk-x")
        # Build the client while futures still resolve promptly; only the step
        # under test should be left pending.
        tc = client.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        relay.pending_polls = 10_000
        future = tc.forward_backward(_batch())
        with pytest.raises(mind_recursive.FutureTimeoutError) as excinfo:
            future.result(timeout=0.2)
        assert "did not resolve" in str(excinfo.value)
        assert future.done() is False

        relay.pending_polls = 0
        assert future.result(timeout=5).loss == pytest.approx(0.5)
        client.close()

    def test_request_failed_surfaces_server_message_and_category(self, relay):
        """This is how the FSDP2 Adam rejection reaches the caller."""
        constraint = (
            "UNSUPPORTED: FSDP2 requires beta1=0.9, beta2=0.999, eps=1e-8 "
            "and lora_plus_ratio=1"
        )
        relay.fail_paths[f"{SERVICE}/retrieve_future"] = (
            200,
            {"type": "request_failed", "error": constraint, "category": "user"},
        )
        client = build_service(relay, api_key="sk-x")

        with pytest.raises(RequestFailedError) as excinfo:
            client.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        assert "FSDP2 requires" in str(excinfo.value)
        assert excinfo.value.category == "user"
        client.close()


class TestChunking:
    def test_batch_over_the_token_budget_is_split(self, relay):
        """Exceeding the negotiated budget earns a 413, so we split first."""
        relay.client_config = {"fwdbwd_max_chunk_len": 8, "fwdbwd_max_chunk_bytes_count": 0}
        client = build_service(relay, api_key="sk-x")
        tc = client.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        # 4 datums x 4 tokens = 16 tokens against a budget of 8 -> 2 chunks.
        tc.forward_backward(_batch(n=4, length=4)).result()
        assert relay.count(f"{TRAIN}/forward_backward") == 2
        client.close()

    def test_chunked_metrics_are_recombined_token_weighted(self, relay):
        """A naive mean would bias toward whichever chunk was smaller."""
        relay.client_config = {"fwdbwd_max_chunk_len": 4, "fwdbwd_max_chunk_bytes_count": 0}
        relay.metrics = {"loss:mean": 0.5, "num_tokens:sum": 100.0}
        client = build_service(relay, api_key="sk-x")
        tc = client.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        out = tc.forward_backward(_batch(n=3, length=4)).result()
        assert relay.count(f"{TRAIN}/forward_backward") == 3
        # Sums add up; equal-weight means stay put.
        assert out.num_tokens == 300
        assert out.loss == pytest.approx(0.5)
        client.close()

    def test_batch_within_budget_is_one_request(self, service, relay):
        tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        tc.forward_backward(_batch(n=4, length=4)).result()
        assert relay.count(f"{TRAIN}/forward_backward") == 1


class TestSampling:
    def test_sample_returns_tokens_and_logprobs(self, service, relay):
        relay.sample_token_groups = [[5, 6, 7]]
        sc = service.create_sampler(base_model="Qwen/Qwen3.8-27B")
        response = sc.sample([1, 2, 3], num_samples=1).result()

        assert response.sequences[0].tokens == [5, 6, 7]
        # Token-aligned logprobs are the reason the sampler is token-level.
        np.testing.assert_allclose(
            response.sequences[0].logprobs, [-0.1, -0.1, -0.1], rtol=1e-6
        )
        assert response.prompt_cache_hit_tokens == 4

    def test_sampling_needs_a_target(self, service):
        with pytest.raises(ValueError):
            service.create_sampler()


def _record_events(service, monkeypatch) -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        service._reporter, "_record", lambda kind, body: records.append((kind, body))
    )
    return records


def test_a_successful_save_reports_completion(service, relay, monkeypatch):
    """A save future reports ``save.completed`` once it actually resolves."""
    records = _record_events(service, monkeypatch)
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    records.clear()

    path = trainer.save_state("state-1").result()

    events = [body for kind, body in records if kind == "event"]
    assert [event["type"] for event in events] == ["save.completed"]
    assert events[0]["data"] == {
        "path": path,
        "name": "state-1",
        "type": "checkpoint",
        "ttl_seconds": 259200,
    }


def test_a_weights_only_save_completion_omits_the_ttl(service, relay, monkeypatch):
    records = _record_events(service, monkeypatch)
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    records.clear()

    path = trainer.save_weights_only("weights-1").result()

    events = [body for kind, body in records if kind == "event"]
    assert [event["type"] for event in events] == ["save.completed"]
    assert events[0]["data"] == {
        "path": path,
        "name": "weights-1",
        "type": "sampler_weights",
    }


def test_a_failed_save_reports_no_completion(service, relay, monkeypatch):
    records = _record_events(service, monkeypatch)
    relay.fail_paths[f"{TRAIN}/save_weights"] = (400, {"error": "rejected"})
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    records.clear()

    with pytest.raises(Exception):
        trainer.save_state("state-1").result()

    assert [body for kind, body in records if kind == "event"] == []


def test_loading_reports_completion(service, relay, monkeypatch):
    records = _record_events(service, monkeypatch)
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    path = trainer.save_weights_only("weights-1").result()
    records.clear()

    trainer.load_weights(path, optimizer=False).result()

    events = [body for kind, body in records if kind == "event"]
    assert [event["type"] for event in events] == ["weights.loaded"]
    assert events[0]["data"] == {
        "path": path,
        "optimizer": False,
        "model_id": trainer.model_id,
    }
