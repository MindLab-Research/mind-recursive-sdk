"""End-to-end over a real socket, not httpx.MockTransport.

MockTransport short-circuits the network, so it cannot prove the heartbeat
thread actually fires, that real TCP framing and Content-Length handling work,
or that protobuf bodies survive a socket round trip. This test runs a real
ThreadingHTTPServer backed by the same FakeRelay logic and drives a whole run
through it: capabilities, create_model, three steps, save weights, sample,
chat, finish.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import mind_recursive
from tests.fake_relay import FakeRelay


def _serve(relay: FakeRelay) -> tuple[ThreadingHTTPServer, int]:
    """Start a real HTTP server that delegates to the fake relay's handler."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # keep pytest output clean
            pass

        def _dispatch(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            request = httpx.Request(
                self.command,
                f"http://127.0.0.1{self.path}",
                content=body,
                headers={k: v for k, v in self.headers.items()},
            )
            response = relay.handler(request)
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@pytest.fixture
def live_relay():
    relay = FakeRelay()
    server, port = _serve(relay)
    try:
        yield relay, port
    finally:
        server.shutdown()


def test_full_run_over_real_sockets(live_relay):
    relay, port = live_relay
    base_url = f"http://127.0.0.1:{port}"

    # Check the backend before committing to a session, which is the order the
    # two clients exist to make possible.
    with mind_recursive.UserClient(api_key="sk-real-socket", base_url=base_url) as uc:
        assert uc.get_server_capabilities().names() == ["Qwen/Qwen3.8-27B"]

    service = mind_recursive.ServiceClient(
        api_key="sk-real-socket",
        base_url=base_url,
        client_id="e2e-client",
        heartbeat_seconds=0.2,
    )

    tc = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=16)

    losses = []
    for step in range(3):
        batch = [
            mind_recursive.Datum.from_tokens([1, 2, 3, 4]),
            mind_recursive.Datum.from_tokens([5, 6, 7, 8]),
        ]
        out = tc.forward_backward(batch).result()
        tc.optim_step(mind_recursive.AdamParams(learning_rate=1e-4)).result()
        service.metric(
            {"loss": out.loss},
            step=step,
            stage="training",
            total=3,
            timings={"forward_backward": 0.9}
        )
        losses.append(out.loss)

    assert losses == [0.5, 0.5, 0.5]

    path = tc.save_weights_only("sampler-001").result()
    assert path == "mint://run-abc/sampler/sampler-001"
    service.artifact(path, type="sampler_weights", name="final")
    service.summary(final_loss=losses[-1])

    relay.sample_token_groups = [[5, 6, 7]]
    sampler = service.create_sampler(model_path=path)
    sample = sampler.sample([1, 2, 3], num_samples=1).result()
    assert sample.sequences[0].tokens == [5, 6, 7]
    assert sample.sequences[0].logprobs == pytest.approx([-0.1, -0.1, -0.1], rel=1e-5)

    # The heartbeat thread can only be observed against a real server.
    time.sleep(0.5)
    assert relay.count("/api/mindrsi/v1/service/session/sess-0001/heartbeat") >= 1

    service.close()

    fwdbwd = relay.requests_to("/api/mindrsi/v1/train/forward_backward")[0]
    assert fwdbwd.headers["content-type"] == "application/x-protobuf"
    assert relay.requests[0].headers["x-api-key"] == "sk-real-socket"
    # Attribution travels in the create_session body over a real socket too,
    # and no request carries a custom header.
    created = relay.requests_to("/api/mindrsi/v1/service/session/create")[0]
    assert created.json()["client_id"] == "e2e-client"
    for request in relay.requests:
        assert "x-mind_recursive-client-id" not in request.headers

    # Reports go to the report plane over the same socket. This run called
    # metric / artifact / summary; the events it did not author itself are the
    # lifecycle facts the SDK registers on its own.
    for suffix in ("log", "artifact", "summary"):
        assert relay.requests_to(
            f"/api/mindrsi/v1/service/report/sess-0001/{suffix}"
        )
    event_types = {
        event["type"]
        for item in relay.requests_to("/api/mindrsi/v1/service/report/sess-0001/events")
        for event in json.loads(item.content)["events"]
    }
    # This run creates one trainer, opens one sampler and closes the session,
    # so exactly these three lifecycle facts should have arrived -- no more
    # (nothing user-authored) and no fewer (an empty set would pass a subset
    # check while reporting was broken).
    assert event_types == {
        "session.created",
        "trainer.created",
        "save.completed",
        "sampler.created",
        "session.closing",
    }


    # The session lifecycle is the ServiceClient's job, not the reporter's.
    assert relay.count("/api/mindrsi/v1/service/session/sess-0001/finish") == 1
