"""Transport: headers, retry policy, error mapping, pool isolation."""

from __future__ import annotations

import httpx
import pytest

from mind_recursive._config import build_config
from mind_recursive._errors import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
)
from mind_recursive._transport import Transport


def make_transport(handler, *, max_retries: int = 3, monkeypatch=None) -> Transport:
    """A transport whose pools all use the given mock handler."""
    config = build_config(
        api_key="sk-test", base_url="https://relay.test", max_retries=max_retries, env={}
    )
    transport = Transport(config)

    def build(pool: str, base_url: str) -> httpx.Client:
        key = f"{pool}:{base_url}"
        client = transport._clients.get(key)
        if client is None:
            client = httpx.Client(
                base_url=base_url, transport=httpx.MockTransport(handler)
            )
            transport._clients[key] = client
        return client

    transport._client = build  # type: ignore[method-assign]
    return transport


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retry backoff is correctness, not latency -- do not pay for it here."""
    monkeypatch.setattr("mind_recursive._transport.time.sleep", lambda _s: None)


class TestHeaders:
    def test_key_goes_in_x_api_key_verbatim(self):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update({k.lower(): v for k, v in request.headers.items()})
            return httpx.Response(200, json={})

        transport = make_transport(handler)
        transport.request("GET", "/x", base_url="https://relay.test")
        assert seen["x-api-key"] == "sk-test"
        transport.close()

    def test_json_body_is_compact(self):
        """Compact separators keep bodies stable for upstream fingerprinting."""
        seen: dict[str, bytes] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content
            return httpx.Response(200, json={})

        transport = make_transport(handler)
        transport.request(
            "POST", "/x", base_url="https://relay.test", json_body={"a": 1, "b": "two"}
        )
        assert seen["body"] == b'{"a":1,"b":"two"}'
        transport.close()

    def test_raw_bytes_are_sent_unmodified(self):
        """The protobuf path must not be re-encoded."""
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content
            seen["type"] = request.headers.get("content-type")
            return httpx.Response(200, json={})

        payload = b"\x08\x96\x01\xff\x00binary"
        transport = make_transport(handler)
        transport.request("POST", "/x", base_url="https://relay.test", content=payload)
        assert seen["body"] == payload
        assert seen["type"] == "application/x-protobuf"
        transport.close()

    def test_body_and_json_together_is_a_programming_error(self):
        transport = make_transport(lambda r: httpx.Response(200, json={}))
        with pytest.raises(ValueError):
            transport.request(
                "POST",
                "/x",
                base_url="https://relay.test",
                json_body={"a": 1},
                content=b"x",
            )
        transport.close()


class TestErrorMapping:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (400, BadRequestError),
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (409, ConflictError),
            (413, PayloadTooLargeError),
        ],
    )
    def test_statuses_map_to_their_exceptions(self, status, expected):
        transport = make_transport(
            lambda r: httpx.Response(status, json={"error": "no", "category": "user"})
        )
        with pytest.raises(expected):
            transport.request("GET", "/x", base_url="https://relay.test")
        transport.close()

    def test_category_is_preserved_from_the_envelope(self):
        transport = make_transport(
            lambda r: httpx.Response(400, json={"error": "bad", "category": "user"})
        )
        with pytest.raises(BadRequestError) as excinfo:
            transport.request("GET", "/x", base_url="https://relay.test")
        assert excinfo.value.category == "user"
        assert "bad" in str(excinfo.value)
        transport.close()

    def test_relay_message_envelope_is_also_read(self):
        """Two envelope shapes exist on these endpoints; read both."""
        transport = make_transport(
            lambda r: httpx.Response(400, json={"message": "relay says no"})
        )
        with pytest.raises(BadRequestError) as excinfo:
            transport.request("GET", "/x", base_url="https://relay.test")
        assert "relay says no" in str(excinfo.value)
        transport.close()

    def test_404_mentions_the_fixed_root_trap(self):
        """The usual cause is a plane suffix on the base, not an absent resource."""
        transport = make_transport(lambda r: httpx.Response(404, json={"error": "nope"}))
        with pytest.raises(NotFoundError) as excinfo:
            transport.request("GET", "/x", base_url="https://relay.test")
        assert "/api/mindrsi/v1" in str(excinfo.value)
        transport.close()

    def test_html_body_is_not_swallowed(self):
        """A relay misroute returns frontend HTML; show a snippet, not nothing."""
        transport = make_transport(
            lambda r: httpx.Response(404, content=b"<!DOCTYPE html><html>oops")
        )
        with pytest.raises(NotFoundError) as excinfo:
            transport.request("GET", "/x", base_url="https://relay.test")
        assert "DOCTYPE" in str(excinfo.value)
        transport.close()


class TestRetries:
    def test_5xx_is_retried_then_raises(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, json={"error": "down"})

        transport = make_transport(handler, max_retries=2)
        with pytest.raises(ServerError):
            transport.request("GET", "/x", base_url="https://relay.test")
        assert calls["n"] == 3  # initial + 2 retries
        transport.close()

    def test_retry_succeeds_on_a_later_attempt(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503, json={"error": "down"})
            return httpx.Response(200, json={"ok": True})

        transport = make_transport(handler, max_retries=3)
        response = transport.request("GET", "/x", base_url="https://relay.test")
        assert response.json() == {"ok": True}
        assert calls["n"] == 3
        transport.close()

    def test_4xx_is_not_retried(self):
        """Replaying a rejected request only burns quota."""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad"})

        transport = make_transport(handler, max_retries=3)
        with pytest.raises(BadRequestError):
            transport.request("GET", "/x", base_url="https://relay.test")
        assert calls["n"] == 1
        transport.close()

    def test_429_is_retried_and_carries_retry_after(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(
                429, json={"error": "slow down"}, headers={"Retry-After": "0"}
            )

        transport = make_transport(handler, max_retries=1)
        with pytest.raises(RateLimitError) as excinfo:
            transport.request("GET", "/x", base_url="https://relay.test")
        assert calls["n"] == 2
        assert excinfo.value.retry_after == 0
        assert excinfo.value.retryable is True
        transport.close()

    def test_connection_errors_are_retried_then_wrapped(self):
        calls = {"n": 0}

        def handler(request: httpx.Request):
            calls["n"] += 1
            raise httpx.ConnectError("no route to host")

        transport = make_transport(handler, max_retries=2)
        with pytest.raises(APIConnectionError):
            transport.request("GET", "/x", base_url="https://relay.test")
        assert calls["n"] == 3
        transport.close()

    def test_per_call_retry_override(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, json={"error": "x"})

        transport = make_transport(handler, max_retries=5)
        with pytest.raises(ServerError):
            transport.request(
                "GET", "/x", base_url="https://relay.test", max_retries=0
            )
        assert calls["n"] == 1
        transport.close()


class TestPools:
    def test_train_and_control_pools_are_separate(self):
        """A stalled forward_backward must not starve a heartbeat."""
        transport = make_transport(lambda r: httpx.Response(200, json={}))
        transport.request("GET", "/x", base_url="https://relay.test", pool="train")
        transport.request("GET", "/x", base_url="https://relay.test", pool="control")
        assert len(transport._clients) == 2
        transport.close()

    def test_close_is_idempotent_and_blocks_reuse(self):
        transport = make_transport(lambda r: httpx.Response(200, json={}))
        transport.request("GET", "/x", base_url="https://relay.test")
        transport.close()
        transport.close()
        with pytest.raises(RuntimeError):
            transport.request("GET", "/x", base_url="https://relay.test")
