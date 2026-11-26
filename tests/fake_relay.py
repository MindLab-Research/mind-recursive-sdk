"""An in-process fake relay, served through httpx's MockTransport.

Implements enough of the contract to drive a real training loop: the two-phase
future protocol, protobuf bodies on ``forward_backward``, ``client/config``
negotiation, heartbeats and the platform reporter endpoints.

It records every request, so tests can assert on what the SDK actually sent --
headers, content types, chunk boundaries -- rather than only on return values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from mind_recursive._proto import schema as S
from mind_recursive._proto.codec import (
    field_bytes,
    field_double,
    field_message,
    field_string,
    field_varint,
)


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    content: bytes
    #: Raw query string, so a test can assert on paging without re-parsing.
    query: str = ""

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))


def encode_forward_backward_output(
    *, metrics: dict[str, float], per_datum_logprobs: list[list[float]] | None = None
) -> bytes:
    """Build a protobuf ``ForwardBackwardOutput``, as the server would."""
    out = bytearray()
    out += field_string(S.ForwardBackwardOutput.LOSS_FN_OUTPUT_TYPE, "tinker.types:ArrayRecord")

    record = bytearray()
    if per_datum_logprobs is not None:
        flat: list[float] = []
        offsets = [0]
        for row in per_datum_logprobs:
            flat.extend(row)
            offsets.append(len(flat) * 4)  # float32 byte offsets
        values = np.asarray(flat, dtype=np.float32)
        batched = bytearray()
        batched += field_bytes(S.BatchedTensor.DATA, values.tobytes())
        batched += field_bytes(
            S.BatchedTensor.OFFSETS, np.asarray(offsets, dtype=np.int64).tobytes()
        )
        batched += field_varint(S.BatchedTensor.DTYPE, S.DTYPE_FLOAT32)
        entry = field_string(S.MapEntry.KEY, "logprobs") + field_message(
            S.MapEntry.VALUE, bytes(batched)
        )
        record += field_message(S.ArrayRecord.FIELDS, entry)
        record += field_varint(S.ArrayRecord.NUM_DATUMS, len(per_datum_logprobs))
    else:
        record += field_varint(S.ArrayRecord.NUM_DATUMS, 0)
    out += field_message(S.ForwardBackwardOutput.LOSS_FN_OUTPUTS, bytes(record))

    for key, value in metrics.items():
        entry = field_string(S.MapEntry.KEY, key) + field_double(S.MapEntry.VALUE, value)
        out += field_message(S.ForwardBackwardOutput.METRICS, entry)
    return bytes(out)


def encode_topk_logprobs(*, k: int, length: int) -> bytes:
    """Build a ``TopkLogprobs`` block with recognisable filler values.

    Cell ``(i, j)`` gets token id ``i * k + j`` and logprob ``-(i * k + j) / 10``,
    so a test can tell row-major order from column-major.
    """
    cells = k * length
    ids = np.arange(cells, dtype=np.int32)
    logprobs = (-ids / 10.0).astype(np.float32)
    out = bytearray()
    out += field_bytes(S.TopkLogprobs.TOKEN_IDS, ids.tobytes())
    out += field_bytes(S.TopkLogprobs.LOGPROBS, logprobs.tobytes())
    out += field_varint(S.TopkLogprobs.K, k)
    out += field_varint(S.TopkLogprobs.LENGTH, length)
    return bytes(out)


def encode_sample_response(
    *,
    sequences: list[tuple[list[int], list[float] | None, int]],
    cache_hits: int = 0,
    prompt_logprobs: list[float] | None = None,
    topk_sample: tuple[int, int] | None = None,
    topk_prompt: tuple[int, int] | None = None,
) -> bytes:
    """Build a protobuf ``SampleResponse``.

    ``topk_sample`` and ``topk_prompt`` are ``(k, length)`` pairs; passing one
    attaches a filled top-k block to every sequence, or to the prompt.
    """
    out = bytearray()
    for tokens, logprobs, stop_reason in sequences:
        seq = bytearray()
        seq += field_varint(S.SampledSequence.STOP_REASON, stop_reason)
        seq += field_bytes(
            S.SampledSequence.TOKENS, np.asarray(tokens, dtype=np.int32).tobytes()
        )
        if logprobs is not None:
            seq += field_bytes(
                S.SampledSequence.LOGPROBS,
                np.asarray(logprobs, dtype=np.float32).tobytes(),
            )
        if topk_sample is not None:
            seq += field_message(
                S.SampledSequence.TOPK_SAMPLED_LOGPROBS,
                encode_topk_logprobs(k=topk_sample[0], length=topk_sample[1]),
            )
        out += field_message(S.SampleResponse.SEQUENCES, bytes(seq))
    if prompt_logprobs is not None:
        out += field_bytes(
            S.SampleResponse.PROMPT_LOGPROBS,
            np.asarray(prompt_logprobs, dtype=np.float32).tobytes(),
        )
    if topk_prompt is not None:
        out += field_message(
            S.SampleResponse.TOPK_PROMPT_LOGPROBS,
            encode_topk_logprobs(k=topk_prompt[0], length=topk_prompt[1]),
        )
    if cache_hits:
        out += field_varint(S.SampleResponse.PROMPT_CACHE_HIT_TOKENS, cache_hits)
    return bytes(out)


@dataclass
class FakeRelay:
    """A scriptable MinT relay.

    Attributes:
        client_config: what ``client/config`` returns.
        pending_polls: how many ``try_again`` responses to emit per request_id
            before resolving, exercising the polling path.
        requests: every request received, in order.
        fail_paths: ``{path: (status, body)}`` to force failures.
    """

    client_config: dict[str, Any] = field(
        default_factory=lambda: {
            "pjwt_auth_enabled": False,
            "credential_default_source": "api_key",
            "fwdbwd_max_chunk_len": 1024,
            "fwdbwd_max_chunk_bytes_count": 50_000_000,
            "proto_compress_fwdbwd": False,
            "sample_join_sampling_session": True,
        }
    )
    capabilities: dict[str, Any] = field(
        default_factory=lambda: {
            "supported_models": [
                {
                    "model_name": "Qwen/Qwen3.8-27B",
                    "max_context_length": 131072,
                    "trainable": True,
                    "sampleable": True,
                }
            ]
        }
    )
    pending_polls: int = 0
    #: HTTP status used for ``try_again``. Real MinT uses 408; the inference service uses
    #: 200. Tests can pin either side of the difference.
    pending_status: int = 200
    metrics: dict[str, float] = field(
        default_factory=lambda: {"loss:mean": 0.5, "num_tokens:sum": 128.0}
    )
    per_datum_logprobs: list[list[float]] | None = None
    #: Fixed token groups for ``asample``; ``None`` varies them per call.
    sample_token_groups: list[list[int]] | None = None
    #: Texts to tokenise for ``asample`` (needs ``sample_tokenizer``). Lets a
    #: reward function see plausible completions instead of random token ids,
    #: which is what GRPO needs in order to produce non-zero variance.
    sample_texts: list[str] | None = None
    sample_tokenizer: Any = None
    fail_paths: dict[str, tuple[int, Any]] = field(default_factory=dict)
    #: Same shape as ``fail_paths``, but each entry is consumed by the first
    #: matching request. Used to exercise the retry path: fail once, succeed.
    fail_once_paths: dict[str, tuple[int, Any]] = field(default_factory=dict)
    requests: list[RecordedRequest] = field(default_factory=list)

    _counter: int = 0
    _futures: dict[str, dict[str, Any]] = field(default_factory=dict)
    _poll_counts: dict[str, int] = field(default_factory=dict)

    # -- helpers --------------------------------------------------------------
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def requests_to(self, path: str) -> list[RecordedRequest]:
        return [r for r in self.requests if r.path == path]

    def count(self, path: str) -> int:
        return len(self.requests_to(path))

    def last(self, path: str) -> RecordedRequest:
        """The most recent request to ``path``.

        Raises rather than returning ``None``: a test asserting on a request
        that never happened should fail loudly, not on an attribute error.
        """
        matches = self.requests_to(path)
        if not matches:
            raise AssertionError(
                f"no request to {path}; saw {sorted({r.path for r in self.requests})}"
            )
        return matches[-1]

    # -- the handler ----------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = request.content
        self.requests.append(
            RecordedRequest(
                method=request.method,
                path=path,
                headers={k.lower(): v for k, v in request.headers.items()},
                content=body,
                query=request.url.query.decode("utf-8"),
            )
        )

        if path in self.fail_paths:
            status, payload = self.fail_paths[path]
            if isinstance(payload, (bytes, bytearray)):
                return httpx.Response(status, content=bytes(payload))
            return httpx.Response(status, json=payload)

        if path in self.fail_once_paths:
            status, payload = self.fail_once_paths.pop(path)
            if isinstance(payload, (bytes, bytearray)):
                return httpx.Response(status, content=bytes(payload))
            return httpx.Response(status, json=payload)

        # -- report sink (session key, no execution id) -----------------------
        if request.method == "POST" and path.startswith(
            "/api/mindrsi/v1/service/report/"
        ) and path.rsplit("/", 1)[-1] in (
            "log",
            "events",
            "metric",
            "artifact",
            "summary",
        ):
            return httpx.Response(200, json={})

        # -- service segment (under /service) --------------------------------
        if path == "/api/mindrsi/v1/service/get_server_capabilities":
            return httpx.Response(200, json=self.capabilities)
        if path == "/api/mindrsi/v1/service/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if path in (
            "/api/mindrsi/v1/service/client/config",
            "/api/mindrsi/v1/service/client/dynamic_config",
        ):
            return httpx.Response(200, json=self.client_config)

        # -- session lifecycle (under /session) -------------------------------
        if path == "/api/mindrsi/v1/service/session/create":
            return httpx.Response(200, json={"session_id": "sess-0001", "type": "create_session"})
        if path.startswith("/api/mindrsi/v1/service/session/") and path.endswith("/heartbeat"):
            return httpx.Response(200, json={"type": "session_heartbeat"})
        if path.startswith("/api/mindrsi/v1/service/session/") and path.endswith("/finish"):
            return httpx.Response(200, json={})

        # -- trainer creation (under /service/train): the platform service
        # records the trainer and forwards to the training engine; the id
        # resolves on the service retrieve_future route.
        if path == "/api/mindrsi/v1/service/train/create_model":
            request_id = self._next_id("req-model")
            self._futures[request_id] = {
                "model_id": f"{json.loads(body)['session_id']}_0",
                "type": "create_model",
            }
            return httpx.Response(200, json={"request_id": request_id})

        if path == "/api/mindrsi/v1/service/train/create_model_from_state":
            request_id = self._next_id("req-model")
            self._futures[request_id] = {
                "model_id": f"{json.loads(body)['session_id']}_0",
                "type": "create_model_from_state",
            }
            return httpx.Response(200, json={"request_id": request_id})

        # -- training plane (under /train) ------------------------------------
        if path == "/api/mindrsi/v1/train/forward_backward":
            request_id = self._next_id("req-fwdbwd")
            self._futures[request_id] = {
                "__proto__": encode_forward_backward_output(
                    metrics=dict(self.metrics),
                    per_datum_logprobs=self.per_datum_logprobs,
                )
            }
            return httpx.Response(200, json={"request_id": request_id})

        if path == "/api/mindrsi/v1/train/optim_step":
            request_id = self._next_id("req-optim")
            self._futures[request_id] = {"type": "optim_step", "grad_norm": 1.25}
            return httpx.Response(200, json={"request_id": request_id})

        if path in (
            "/api/mindrsi/v1/train/save_weights",
            "/api/mindrsi/v1/train/save_weights_for_sampler",
        ):
            payload = json.loads(body) if body else {}
            name = str(payload.get("path") or "")
            # Both named and ephemeral saves send a path and get a mint:// path
            # back; the ephemeral export uses a per-step name "ephemeral-<seq>".
            if path.endswith("save_weights_for_sampler") and (
                name in ("", ".", "..")
                or not all(
                    ch.isascii() and (ch.isalnum() or ch in "-_.") for ch in name
                )
            ):
                return httpx.Response(
                    400,
                    json={"error": "path must be a non-empty URL-safe name"},
                )
            request_id = self._next_id("req-save")
            kind = "sampler" if "sampler" in path else "state"
            self._futures[request_id] = {
                "path": f"mint://run-abc/{kind}/{name or 'step-1'}",
                "type": "save_weights",
            }
            return httpx.Response(200, json={"request_id": request_id})

        if path == "/api/mindrsi/v1/train/load_weights":
            request_id = self._next_id("req-load")
            self._futures[request_id] = {"type": "load_weights"}
            return httpx.Response(200, json={"request_id": request_id})

        if path == "/api/mindrsi/v1/service/sample/create_sampling_session":
            # Every sampler is created on the service plane: the recursive
            # server opens it under its inference-side session (and deploys trained
            # weights first), so this returns a future that resolves to the
            # sampler id on the service retrieve_future route.
            request_id = self._next_id("req-samp-sess")
            self._futures[request_id] = {
                "sampling_session_id": self._next_id("samp-sess")
            }
            return httpx.Response(200, json={"request_id": request_id})

        if path == "/api/mindrsi/v1/sample/asample":
            request_id = self._next_id("req-sample")
            requested = 1
            try:
                requested = max(1, int(json.loads(body).get("num_samples") or 1))
            except (ValueError, AttributeError):
                pass
            if self.sample_texts is not None and self.sample_tokenizer is not None:
                groups = [
                    list(
                        self.sample_tokenizer(text, add_special_tokens=False)["input_ids"]
                    )
                    for text in self.sample_texts
                ]
            elif self.sample_token_groups is not None:
                groups = self.sample_token_groups
            else:
                # Vary the samples so reward functions can discriminate and
                # GRPO's group normalisation has non-zero variance.
                groups = [
                    [self._counter * 10 + i, 6 + i, 7 + i] for i in range(requested)
                ]
            sequences = [
                (tokens, [-0.1 * (i + 1)] * len(tokens), S.STOP_REASON_STOP)
                for i, tokens in enumerate(groups[:requested])
            ]
            # Honour the request's logprob flags, so a test sees the same
            # opt-in behaviour the real server has: prompt-side data only
            # comes back when asked for.
            try:
                asked = json.loads(body)
            except ValueError:
                asked = {}
            prompt_len = len(
                (asked.get("prompt") or {}).get("chunks", [{}])[0].get("tokens") or []
            )
            last_n = asked.get("prompt_logprobs_last_n")
            scored = min(last_n, prompt_len) if last_n else prompt_len
            topk_sample_k = int(asked.get("topk_sample_logprobs") or 0)
            topk_prompt_k = int(asked.get("topk_prompt_logprobs") or 0)
            self._futures[request_id] = {
                "__proto__": encode_sample_response(
                    sequences=sequences,
                    cache_hits=4,
                    prompt_logprobs=(
                        [-0.2 * (i + 1) for i in range(scored)]
                        if asked.get("prompt_logprobs")
                        else None
                    ),
                    topk_sample=(
                        (topk_sample_k, len(sequences[0][0]))
                        if topk_sample_k and sequences
                        else None
                    ),
                    topk_prompt=(
                        (topk_prompt_k, scored or prompt_len) if topk_prompt_k else None
                    ),
                )
            }
            return httpx.Response(200, json={"request_id": request_id})

        if path in (
            "/api/mindrsi/v1/train/retrieve_future",
            "/api/mindrsi/v1/sample/retrieve_future",
            "/api/mindrsi/v1/service/retrieve_future",
        ):
            return self._retrieve_future(json.loads(body)["request_id"])

        if path == "/api/mindrsi/v1/train/get_info":
            return httpx.Response(200, json={"model_data": {"base_model": "Qwen/Qwen3.8-27B"}})

        return httpx.Response(404, json={"error": f"no fake route for {path}", "category": "user"})

    def _retrieve_future(self, request_id: str) -> httpx.Response:
        """Emit ``try_again`` until the scripted budget is spent, then resolve."""
        seen = self._poll_counts.get(request_id, 0)
        self._poll_counts[request_id] = seen + 1
        if seen < self.pending_polls:
            return httpx.Response(
                self.pending_status,
                json={
                    "type": "try_again",
                    "request_id": request_id,
                    "queue_state": "active",
                },
            )
        payload = self._futures.get(request_id)
        if payload is None:
            return httpx.Response(404, json={"error": "unknown request_id", "category": "user"})
        if "__proto__" in payload:
            return httpx.Response(
                200,
                content=payload["__proto__"],
                headers={"Content-Type": "application/x-protobuf"},
            )
        return httpx.Response(200, json=payload)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _mock_client_factory(
    transport: Any, relay: FakeRelay
) -> Any:
    """A ``Transport._client`` replacement handing out MockTransport clients.

    Every pool gets one, so the real header, retry and error-mapping code
    paths stay intact -- the things most worth testing.
    """

    def build(pool: str, base_url: str) -> httpx.Client:
        key = f"{pool}:{base_url}"
        client = transport._clients.get(key)
        if client is None:
            client = httpx.Client(
                base_url=base_url,
                transport=relay.transport(),
                timeout=httpx.Timeout(5.0, connect=5.0),
                follow_redirects=False,
            )
            transport._clients[key] = client
        return client

    return build


def patch_user(client: Any, relay: FakeRelay) -> None:
    """Route a ``UserClient``'s transport at the fake relay.

    Works because constructing a ``UserClient`` sends nothing, so there is
    still a "before the first request" to patch in. A ``ServiceClient`` opens
    its session in the constructor -- being a session is what the class is --
    so it must be built through :func:`build_service` instead.
    """
    transport = client.transport
    client._patched_original_client = transport._client
    transport._client = _mock_client_factory(transport, relay)  # type: ignore[method-assign]


def build_service(relay: FakeRelay, **kwargs: Any) -> Any:
    """Construct a ``ServiceClient`` whose very first request hits ``relay``.

    ``ServiceClient`` negotiates config and opens its session inside
    ``__init__`` -- being a session is what the class is -- so a test cannot
    patch the transport afterwards. Patching ``Transport._client`` on the
    class for the duration of construction is the narrowest way in: the
    instance keeps the patched factory, so later requests are routed too.

    Defaults ``heartbeat_seconds`` high to keep the background thread out of
    the way; pass it explicitly to test the heartbeat itself.
    """
    import mind_recursive
    from mind_recursive._transport import Transport

    kwargs.setdefault("api_key", "sk-test-key")
    kwargs.setdefault("base_url", "https://relay.test")
    kwargs.setdefault("heartbeat_seconds", 3600)

    original = Transport._client
    holder: dict[str, Any] = {}

    def patched(self: Any, pool: str, base_url: str) -> httpx.Client:
        build = holder.get(id(self))
        if build is None:
            build = _mock_client_factory(self, relay)
            holder[id(self)] = build
        return build(pool, base_url)

    Transport._client = patched  # type: ignore[method-assign]
    try:
        service = mind_recursive.ServiceClient(**kwargs)
    finally:
        Transport._client = original  # type: ignore[method-assign]

    # The instance keeps the factory the class-level patch handed it, so
    # requests after construction go to the relay too.
    transport = service.transport
    transport._client = _mock_client_factory(transport, relay)  # type: ignore[method-assign]
    return service


def _mock_async_client_factory(transport: Any, relay: FakeRelay) -> Any:
    """Async counterpart of :func:`_mock_client_factory`."""

    def build(pool: str, base_url: str) -> httpx.AsyncClient:
        key = f"{pool}:{base_url}"
        client = transport._clients.get(key)
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url,
                transport=relay.transport(),
                timeout=httpx.Timeout(5.0, connect=5.0),
                follow_redirects=False,
            )
            transport._clients[key] = client
        return client

    return build


def build_async_service(relay: FakeRelay, **kwargs: Any) -> Any:
    """A ``ServiceClient`` whose async transport also hits ``relay``.

    The session is still opened over the sync transport (the constructor owns
    it); every ``*_async`` call goes through a real ``httpx.AsyncClient``
    backed by the same MockTransport, so the async retry/future paths are
    exercised rather than bypassed.
    """
    from mind_recursive._transport import AsyncTransport

    service = build_service(relay, **kwargs)
    transport = AsyncTransport(service.config)
    transport._client = _mock_async_client_factory(transport, relay)  # type: ignore[method-assign]
    service._async_transport = transport
    return service


def patch_user_async(client: Any, relay: FakeRelay) -> None:
    """Route a ``UserClient``'s async transport at the fake relay."""
    from mind_recursive._transport import AsyncTransport

    transport = AsyncTransport(client.config)
    transport._client = _mock_async_client_factory(transport, relay)  # type: ignore[method-assign]
    client._async_transport = transport
