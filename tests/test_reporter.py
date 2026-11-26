"""Reporting behaviour: five report types, one HTTP route each, no lifecycle.

Reporting is a surface on the ``ServiceClient`` itself, not a handle it
creates: one session has exactly one writer. It must not own
start/finish/heartbeat, and a reporting failure must never break training.

Every report is an HTTP POST through the client's own transport, so the
payload assertions here are wire-level. The network is the fake relay; the
routes are asserted explicitly because they are the contract.
"""

from __future__ import annotations

import json
import logging

import pytest

from mind_recursive.reporter import SYS_PREFIX
from tests.fake_relay import FakeRelay, build_service


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    """Retry backoff is correctness, not latency -- do not pay for it here."""
    monkeypatch.setattr("mind_recursive._transport.time.sleep", lambda _s: None)


@pytest.fixture
def relay() -> FakeRelay:
    return FakeRelay()


@pytest.fixture
def service(relay: FakeRelay):
    client = build_service(relay, api_key="sk-test", client_id="client-1")
    yield client
    client.close()


@pytest.fixture
def recorded(service, monkeypatch):
    """Capture what the reporter records, as objects rather than log text."""
    entries: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        service._reporter, "_record", lambda kind, body: entries.append((kind, body))
    )
    return entries


def _only(entries: list[tuple[str, dict]], kind: str) -> dict:
    """The single recorded body of one kind."""
    bodies = [body for name, body in entries if name == kind]
    assert len(bodies) == 1, f"expected one {kind}, got {len(bodies)}"
    return bodies[0]


class TestReporterSurface:
    def test_the_reporting_surface_lives_on_the_client(self, service):
        for name in ("log", "event", "metric", "artifact", "summary"):
            assert callable(getattr(service, name))
        for name in ("start", "finish", "phase", "record_timing"):
            assert not hasattr(service, name)

    def test_there_is_no_separate_reporter_to_create(self, service):
        """One writer per session: a handle would imply a choice that is not real."""
        assert not hasattr(service, "create_reporter")


class TestHttpSink:
    """Reporting is an HTTP write through the client's own transport."""

    def test_each_report_posts_to_its_route(self, service, relay):
        service.log("hi")
        service.metric({"loss": 0.1}, step=1)
        service.artifact("mint://x", type="sampler_weights")
        service.summary(final_loss=0.1)

        assert relay.count("/api/mindrsi/v1/service/report/sess-0001/events") == 1
        assert relay.count("/api/mindrsi/v1/service/report/sess-0001/log") == 1
        assert relay.count("/api/mindrsi/v1/service/report/sess-0001/artifact") == 1
        assert relay.count("/api/mindrsi/v1/service/report/sess-0001/summary") == 1

    def test_log_and_event_share_the_events_route(self, service, relay):
        service.log("one")
        service.event("custom", message="two")

        sent = relay.requests_to("/api/mindrsi/v1/service/report/sess-0001/events")
        assert len(sent) == 2
        bodies = [json.loads(item.content)["events"][0] for item in sent]
        assert [body["type"] for body in bodies] == ["log", "custom"]

    def test_a_report_is_written_into_the_session_that_exists(self, service, relay):
        """A report belongs to a session, so there is nowhere else to put it."""
        service.log("hi")
        assert service.session_id == "sess-0001"
        assert relay.count("/api/mindrsi/v1/service/session/create") == 1
        assert relay.count("/api/mindrsi/v1/service/report/sess-0001/events") == 1

    def test_reporting_after_close_is_swallowed(self, service, relay, caplog):
        """Closing finishes the session, so there is no longer a target.

        Reporting must not break training, and that holds for the shutdown
        race too -- a late report is a warning, never an exception.
        """
        service.close()
        with caplog.at_level(logging.WARNING, logger="mind_recursive.reporter"):
            service.log("too late")  # must not raise
        # ``log`` is an event upstream, which is the kind the sink reports.
        assert "report event failed" in caplog.text
        assert "has been closed" in caplog.text

    def test_a_failed_report_is_logged_not_raised(self, service, relay, caplog):
        relay.fail_paths["/api/mindrsi/v1/service/report/sess-0001/summary"] = (
            500,
            {"error": "down"},
        )
        with caplog.at_level(logging.WARNING, logger="mind_recursive.reporter"):
            service.summary(final_loss=0.4)  # must not raise
        assert "report summary failed" in caplog.text


class TestLog:
    def test_log_records_a_terminal_event(self, service, recorded):
        service.log("epoch 1 step 2 loss 0.42", level="warning", step=2)
        event = _only(recorded, "event")
        assert event["type"] == "log"
        assert event["level"] == "warning"
        assert event["message"] == "epoch 1 step 2 loss 0.42"
        assert event["step"] == 2
        assert event["seq"] == 1
        assert event["id"]

    def test_log_seq_increments(self, service, recorded):
        service.log("one")
        service.log("two")
        assert [body["seq"] for _, body in recorded] == [1, 2]

    def test_invalid_level_falls_back_to_info(self, service, recorded):
        service.log("hi", level="catastrophic")
        assert _only(recorded, "event")["level"] == "info"

    def test_log_is_best_effort(self, service, monkeypatch, caplog):
        monkeypatch.setattr(
            service._reporter, "_record", lambda *_a: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with caplog.at_level("WARNING"):
            service.log("hi")
        assert "ignored" in caplog.text


class TestEvent:
    def test_event_envelope_carries_the_structured_fields(self, service, recorded):
        service.event(
            "upstream.retrying",
            level="warning",
            code="429",
            message="backing off",
            data={"retry": 1},
            step=3,
        )
        event = _only(recorded, "event")
        assert event["type"] == "upstream.retrying"
        assert event["level"] == "warning"
        assert event["code"] == "429"
        assert event["message"] == "backing off"
        assert event["data"] == {"retry": 1}
        assert event["step"] == 3
        assert event["seq"] == 1
        assert event["id"]

    def test_log_and_event_share_the_sequence(self, service, recorded):
        service.log("one")
        service.event("custom", message="two")
        bodies = [body for _, body in recorded]
        assert [e["type"] for e in bodies] == ["log", "custom"]
        assert [e["seq"] for e in bodies] == [1, 2]

    def test_invalid_level_falls_back_to_info(self, service, recorded):
        service.event("x", level="catastrophic", message="hi")
        assert _only(recorded, "event")["level"] == "info"

    def test_event_id_fits_the_contract_cap(self, service, recorded):
        """Upstream rejects an id over 64 characters for the whole batch."""
        service.event("x", message="hi")
        assert len(_only(recorded, "event")["id"]) <= 64

    def test_long_message_is_truncated_to_the_cap(self, service, recorded):
        service.event("x", message="m" * 5000)
        assert len(_only(recorded, "event")["message"]) == 2048

    def test_non_object_data_is_rejected(self, service, caplog):
        """A list in ``data`` would make upstream reject the whole batch."""
        with caplog.at_level("WARNING"):
            service.event("x", data=[1, 2, 3])
        assert "ignored" in caplog.text

    def test_event_is_best_effort(self, service, monkeypatch, caplog):
        monkeypatch.setattr(
            service._reporter, "_record", lambda *_a: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with caplog.at_level("WARNING"):
            service.event("x", message="hi")
        assert "ignored" in caplog.text


class TestMetric:
    def test_metric_records_step_stage_and_total(self, service, recorded):
        service.metric({"loss": 0.42}, step=7, stage="training", total=800)
        body = _only(recorded, "metric")
        assert body["step"] == 7
        assert body["stage"] == "training"
        assert body["total"] == 800
        assert body["metrics"]["loss"] == 0.42

    def test_none_values_are_dropped_not_sent(self, service, recorded):
        """A null would occupy a key without locking its type."""
        service.metric({"loss": 0.1, "grad_norm": None}, step=1)
        assert "grad_norm" not in _only(recorded, "metric")["metrics"]

    def test_composite_metric_objects_pass_through_untouched(self, service, recorded):
        """The server owns the metric type system; we do not re-validate it."""
        histogram = {"_type": "histogram", "counts": [1, 2, 3], "edges": [0, 1, 2, 3]}
        service.metric({"grad": histogram}, step=1)
        assert _only(recorded, "metric")["metrics"]["grad"] == histogram

    def test_timings_become_the_system_distribution(self, service, recorded):
        """sys.* keeps built-ins from colliding with a user key's type lock."""
        service.metric(
            {"loss": 0.3},
            step=1,
            timings={"forward_backward": 1.5, "optim_step": 0.5},
        )
        timing = _only(recorded, "metric")["metrics"][f"{SYS_PREFIX}step_seconds"]
        assert timing["_type"] == "distribution"
        assert timing["labels"] == ["forward_backward", "optim_step"]
        assert timing["values"] == [1.5, 0.5]
        assert timing["unit"] == "s"

    def test_counters_are_namespaced(self, service, recorded):
        service.metric({}, step=1, counters={"retries": 2})
        assert _only(recorded, "metric")["metrics"][f"{SYS_PREFIX}retries"] == 2

    def test_metric_is_best_effort(self, service, monkeypatch, caplog):
        monkeypatch.setattr(
            service._reporter, "_record", lambda *_a: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with caplog.at_level("WARNING"):
            service.metric({"loss": 0.5}, step=1)
        assert "ignored" in caplog.text


class TestArtifact:
    def test_artifact_records_a_pointer(self, service, recorded):
        service.artifact(
            "mint://run-abc/sampler/step-100",
            type="sampler_weights",
            name="step-100",
            metadata={"step": 100},
        )
        body = _only(recorded, "artifact")
        assert body["uri"] == "mint://run-abc/sampler/step-100"
        assert body["type"] == "sampler_weights"
        assert body["metadata"] == {"step": 100}

    def test_uri_without_a_scheme_is_rejected(self, service, caplog):
        """Upstream 400s a bare string; catching it here names the real cause."""
        with caplog.at_level("WARNING"):
            service.artifact("run-abc/step-100", type="sampler_weights")
        assert "ignored" in caplog.text

    def test_artifact_is_best_effort(self, service, monkeypatch, caplog):
        monkeypatch.setattr(
            service._reporter, "_record", lambda *_a: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with caplog.at_level("WARNING"):
            service.artifact("mint://x", type="sampler_weights")
        assert "ignored" in caplog.text


class TestSummary:
    def test_summary_merges_mapping_and_kwargs(self, service, recorded):
        service.summary({"a": 1}, b=2)
        assert _only(recorded, "summary")["summary"] == {"a": 1, "b": 2}

    def test_summary_is_best_effort(self, service, monkeypatch, caplog):
        monkeypatch.setattr(
            service._reporter, "_record", lambda *_a: (_ for _ in ()).throw(RuntimeError("x"))
        )
        with caplog.at_level("WARNING"):
            service.summary(final_loss=0.4)
        assert "ignored" in caplog.text


class TestTerminalEcho:
    """The same information the web console gets is echoed to the terminal.

    Reporting to a web console only is invisible to the person running the
    script; the echo exists so a training script does not need its own
    ``print`` next to every report.
    """

    def test_log_echoes_with_step(self, service, caplog):
        with caplog.at_level(logging.INFO, logger="mind_recursive"):
            service.log("epoch 1 finished", step=2)
        assert "log: step=2 epoch 1 finished" in caplog.text

    def test_log_level_maps_to_logging_level(self, service, caplog):
        with caplog.at_level(logging.DEBUG, logger="mind_recursive"):
            service.log("careful", level="warning")
        echoed = [r for r in caplog.records if r.getMessage() == "log: careful"]
        assert echoed and echoed[0].levelno == logging.WARNING

    def test_metric_echoes_the_payload(self, service, caplog):
        with caplog.at_level(logging.INFO, logger="mind_recursive"):
            service.metric({"loss": 0.5, "dropped": None}, step=1, stage="training")
        assert "metric [training] step=1: {'loss': 0.5}" in caplog.text

    def test_artifact_echoes_uri_and_name(self, service, caplog):
        with caplog.at_level(logging.INFO, logger="mind_recursive"):
            service.artifact(
                "mint://run/adapters/step-1", type="sampler_weights", name="step-1"
            )
        assert (
            "artifact sampler_weights: mint://run/adapters/step-1 (name=step-1)"
            in caplog.text
        )

    def test_invalid_artifact_is_not_echoed_as_a_report(self, service, caplog):
        with caplog.at_level(logging.INFO, logger="mind_recursive"):
            service.artifact("no-scheme", type="sampler_weights")
        assert "artifact sampler_weights: no-scheme" not in caplog.text
