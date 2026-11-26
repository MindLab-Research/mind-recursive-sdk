"""HTTP transport: auth headers, retries, error mapping, connection pools.

Three things here are load-bearing:

**Auth.** The Mind Recursive API key goes into ``X-API-Key`` verbatim. No prefix check, no
disguise. The stock Tinker SDK validates ``tml-``/``eyJ`` locally in
Tinker's auth token provider, which forced callers to
subclass ``str`` and override ``startswith`` to smuggle a ``sk-`` key past it.
Owning the transport is what makes that patch unnecessary.

**Connection pools.** Upstream separates training traffic from everything else
(``ClientConnectionPoolType.TRAIN``). We do the same: a long-lived pool for
training, a separate one for control and metadata calls, so a stalled
``forward_backward`` cannot starve a heartbeat.

**Byte fidelity.** ``forward_backward`` sends protobuf and its future resolves
to protobuf, so bodies are passed as raw bytes and responses are never
re-encoded. Decompression is disabled for the same reason the relay's own
proxy disables it: a response may be binary and must arrive byte-exact.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Mapping

import httpx

from ._config import Config
from ._errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
    status_to_error,
)

__all__ = ["Transport", "AsyncTransport", "JSON_CONTENT_TYPE", "PROTO_CONTENT_TYPE"]

log = logging.getLogger("mind_recursive.transport")

JSON_CONTENT_TYPE = "application/json"
PROTO_CONTENT_TYPE = "application/x-protobuf"

#: Retry on transport faults, 429, and 5xx. Never on 4xx (other than 429):
#: replaying a rejected request only burns quota.
_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 507, 509})

_BACKOFF_BASE = 0.5
_BACKOFF_CAP = 30.0


def _backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with full jitter; ``Retry-After`` wins when sent."""
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, _BACKOFF_CAP)
    ceiling = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)
    return random.uniform(0, ceiling)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None  # HTTP-date form; fall back to our own backoff


class Transport:
    """Synchronous HTTP transport against one relay origin.

    Not thread-safe for configuration changes, but safe for concurrent
    requests: ``httpx.Client`` is, and we keep no per-request state.
    """

    def __init__(self, config: Config, *, user_agent: str | None = None) -> None:
        self._config = config
        self._user_agent = user_agent or "mind-recursive-python"
        self._clients: dict[str, httpx.Client] = {}
        self._closed = False

    # -- headers --------------------------------------------------------------
    def _base_headers(self) -> dict[str, str]:
        """Headers on every request.

        ``X-API-Key`` carries the raw key. We deliberately do not also send
        ``Authorization: Bearer``: the relay accepts either, and sending one
        credential by one name keeps 401s unambiguous.
        """
        return {
            "X-API-Key": self._config.api_key,
            "User-Agent": self._user_agent,
            "Accept": JSON_CONTENT_TYPE,
        }

    # -- client pools ---------------------------------------------------------
    def _client(self, pool: str, base_url: str) -> httpx.Client:
        """Get or build the client for a pool.

        ``pool`` is the isolation key -- ``"train"`` for the training hot path,
        ``"control"`` for everything else.
        """
        if self._closed:
            raise RuntimeError("transport is closed")
        key = f"{pool}:{base_url}"
        client = self._clients.get(key)
        if client is None:
            client = httpx.Client(
                base_url=base_url,
                timeout=httpx.Timeout(
                    self._config.timeout, connect=self._config.connect_timeout
                ),
                # trust_env=False: a local SOCKS proxy inherited from the
                # ambient environment has hijacked this traffic before.
                trust_env=self._config.trust_env,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
                headers={"Accept-Encoding": "identity"},
            )
            self._clients[key] = client
        return client

    # -- requests -------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        base_url: str,
        pool: str = "control",
        json_body: Any | None = None,
        content: bytes | None = None,
        content_type: str | None = None,
        params: Mapping[str, Any] | None = None,
        accept: str = JSON_CONTENT_TYPE,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        allow_statuses: frozenset[int] | None = None,
    ) -> httpx.Response:
        """Issue a request, retrying retryable failures.

        Exactly one of ``json_body`` or ``content`` may be given. ``content``
        is sent verbatim, which is what the protobuf path needs. ``params``
        becomes the query string; the relay appends it to the upstream URL
        unchanged.

        ``allow_statuses`` returns those HTTP statuses to the caller instead of
        treating them as errors. ``retrieve_future`` needs this for 408: there,
        408 is the normal "still pending" signal, not a retryable failure.

        Returns:
            The successful response.

        Raises:
            APIError: a non-retryable status, or the retry budget ran out.
            APIConnectionError: the request never produced a response.
        """
        if self._closed:
            raise RuntimeError("transport is closed")
        if json_body is not None and content is not None:
            raise ValueError("pass json_body or content, not both")

        headers = self._base_headers()
        headers["Accept"] = accept
        if content is not None:
            headers["Content-Type"] = content_type or PROTO_CONTENT_TYPE
            body = content
        elif json_body is not None:
            headers["Content-Type"] = JSON_CONTENT_TYPE
            # separators: compact, and stable for any fingerprinting upstream.
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
        else:
            body = None
        if extra_headers:
            headers.update(extra_headers)

        budget = self._config.max_retries if max_retries is None else max_retries
        client = self._client(pool, base_url)
        request_timeout = (
            httpx.Timeout(timeout, connect=self._config.connect_timeout)
            if timeout is not None
            else httpx.USE_CLIENT_DEFAULT
        )

        last_error: Exception | None = None
        for attempt in range(budget + 1):
            try:
                response = client.request(
                    method,
                    path,
                    content=body,
                    params=params,
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = APITimeoutError(f"{method} {path} timed out", path=path)
                if attempt >= budget:
                    raise last_error from exc
            except httpx.HTTPError as exc:
                last_error = APIConnectionError(
                    f"{method} {path} failed to reach the relay: {exc}", path=path
                )
                if attempt >= budget:
                    raise last_error from exc
            else:
                if response.status_code < 400 or (
                    allow_statuses and response.status_code in allow_statuses
                ):
                    return response
                if response.status_code in _RETRY_STATUSES and attempt < budget:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    delay = _backoff_delay(attempt, retry_after)
                    log.debug(
                        "retrying %s %s after status %s in %.2fs (attempt %d/%d)",
                        method, path, response.status_code, delay, attempt + 1, budget,
                    )
                    response.close()
                    time.sleep(delay)
                    continue
                raise self._to_error(response, method, path)

            # Transport fault with budget remaining.
            time.sleep(_backoff_delay(attempt))

        raise last_error or APIError(f"{method} {path} exhausted its retry budget", path=path)

    def _to_error(self, response: httpx.Response, method: str, path: str) -> APIError:
        """Turn an error response into the right exception.

        Two envelope shapes exist on these endpoints: the Tinker-style shape
        ``{"error": ..., "category": ...}`` and the relay's ``{"message": ...}``.
        We read both so the message is never swallowed.
        """
        raw = b""
        try:
            raw = response.content
        except Exception:  # pragma: no cover - body already consumed/aborted
            pass

        message = ""
        category: str | None = None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict):
            for key in ("error", "message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    message = value.strip()
                    break
            raw_category = payload.get("category")
            if isinstance(raw_category, str):
                category = raw_category
        if not message:
            snippet = raw[:200].decode("utf-8", "replace").strip()
            message = snippet or f"HTTP {response.status_code}"

        status = response.status_code
        if status == 404:
            # The single most common 404 is a base URL carrying the old plane
            # suffix or a full endpoint, not an absent resource. Say so, since
            # the bare status sends people hunting the wrong problem.
            message = (
                f"{message} -- check the base URL is the relay origin; the SDK "
                f"appends the fixed /api/mindrsi/v1 root itself "
                f"(requested {base_hint(response)})"
            )
        if status == 429:
            return RateLimitError(
                message,
                status=status,
                category=category,
                path=path,
                body=raw,
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            )
        return status_to_error(status)(
            message, status=status, category=category, path=path, body=raw
        )

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
        self._closed = True

    def __enter__(self) -> Transport:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class AsyncTransport(Transport):
    """Async twin of :class:`Transport`, using ``httpx.AsyncClient``.

    Same headers, pools, retry budget and error mapping as the sync transport;
    only the waiting is different. The backend sees identical requests.
    """

    def __init__(self, config: Config, *, user_agent: str | None = None) -> None:
        super().__init__(config, user_agent=user_agent)
        self._clients = {}

    def _client(self, pool: str, base_url: str) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("transport is closed")
        key = f"{pool}:{base_url}"
        client = self._clients.get(key)
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(
                    self._config.timeout, connect=self._config.connect_timeout
                ),
                trust_env=self._config.trust_env,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
                headers={"Accept-Encoding": "identity"},
            )
            self._clients[key] = client
        return client

    async def request(  # type: ignore[override]
        self,
        method: str,
        path: str,
        *,
        base_url: str,
        pool: str = "control",
        json_body: Any | None = None,
        content: bytes | None = None,
        content_type: str | None = None,
        params: Mapping[str, Any] | None = None,
        accept: str = JSON_CONTENT_TYPE,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        allow_statuses: frozenset[int] | None = None,
    ) -> httpx.Response:
        if self._closed:
            raise RuntimeError("transport is closed")
        if json_body is not None and content is not None:
            raise ValueError("pass json_body or content, not both")

        headers = self._base_headers()
        headers["Accept"] = accept
        if content is not None:
            headers["Content-Type"] = content_type or PROTO_CONTENT_TYPE
            body = content
        elif json_body is not None:
            headers["Content-Type"] = JSON_CONTENT_TYPE
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
        else:
            body = None
        if extra_headers:
            headers.update(extra_headers)

        budget = self._config.max_retries if max_retries is None else max_retries
        client = self._client(pool, base_url)
        request_timeout = (
            httpx.Timeout(timeout, connect=self._config.connect_timeout)
            if timeout is not None
            else httpx.USE_CLIENT_DEFAULT
        )

        last_error: Exception | None = None
        for attempt in range(budget + 1):
            try:
                response = await client.request(
                    method,
                    path,
                    content=body,
                    params=params,
                    headers=headers,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException as exc:
                last_error = APITimeoutError(f"{method} {path} timed out", path=path)
                if attempt >= budget:
                    raise last_error from exc
            except httpx.HTTPError as exc:
                last_error = APIConnectionError(
                    f"{method} {path} failed to reach the relay: {exc}", path=path
                )
                if attempt >= budget:
                    raise last_error from exc
            else:
                if response.status_code < 400 or (
                    allow_statuses and response.status_code in allow_statuses
                ):
                    return response
                if response.status_code in _RETRY_STATUSES and attempt < budget:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    delay = _backoff_delay(attempt, retry_after)
                    log.debug(
                        "retrying %s %s after status %s in %.2fs (attempt %d/%d)",
                        method, path, response.status_code, delay, attempt + 1, budget,
                    )
                    await response.aclose()
                    await asyncio.sleep(delay)
                    continue
                raise self._to_error(response, method, path)

            await asyncio.sleep(_backoff_delay(attempt))

        raise last_error or APIError(f"{method} {path} exhausted its retry budget", path=path)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
        self._closed = True

    def close(self) -> None:  # pragma: no cover - async callers use aclose()
        raise RuntimeError("AsyncTransport must be closed with await aclose()")


def base_hint(response: httpx.Response) -> str:
    """The absolute URL a response came from, for error messages."""
    try:
        return str(response.request.url)
    except Exception:  # pragma: no cover
        return "unknown URL"
