"""The two-phase future protocol.

``create_model``, ``forward_backward``, ``optim_step``, ``save_weights`` and
friends do not return results. They return a ``request_id``, and the result is
fetched by polling ``POST /api/v1/retrieve_future``. This is not an optional
optimisation -- it is the only way to get a result. Observed on the wire::

    POST /api/v1/create_model
    RESP {"request_id":"01a1d90a-06d8-75e0-8f5e-fa1cc9946e11"}

    POST /api/v1/retrieve_future
    REQ  {"request_id":"01a1d90a-...","allow_metadata_only":true}
    RESP {"model_id":"01a1d90a-054d-76cd-bb63-fd038d4d9897_0"}

Polling resolves to one of three things:

* a ``try_again`` envelope -- still pending, carrying a ``queue_state`` that
  distinguishes a working queue from one paused on capacity or rate limit;
* a ``request_failed`` envelope -- the work failed, surfaced as
  :class:`RequestFailedError` with the server's category and message. This is
  how backend constraints arrive, e.g. the FSDP2 Adam-parameter rejection;
* a real payload, JSON or protobuf depending on the request.

``APIFuture.result()`` hides the polling but never hides those outcomes.
"""

from __future__ import annotations

import logging
import random
import asyncio
import threading
import time
from typing import Any, Awaitable, Callable, Generic, TypeVar

from ._errors import FutureTimeoutError, RequestFailedError

__all__ = ["APIFuture", "FutureResolver", "QUEUE_STATES"]

log = logging.getLogger("mind_recursive.futures")

T = TypeVar("T")
U = TypeVar("U")

#: ``try_again.queue_state`` values. ``active`` means the request is simply
#: not done; the paused states mean the server is deliberately holding it.
QUEUE_STATES = ("active", "paused_capacity", "paused_rate_limit")

_POLL_MIN = 0.05
_POLL_MAX = 2.0
#: While the queue is paused there is no point polling fast.
_POLL_MAX_PAUSED = 10.0


def _poll_delay(attempt: int, queue_state: str | None) -> float:
    """Backoff between polls, slower when the server says it is paused."""
    ceiling = _POLL_MAX_PAUSED if (queue_state or "").startswith("paused") else _POLL_MAX
    target = min(_POLL_MIN * (2 ** min(attempt, 10)), ceiling)
    return random.uniform(_POLL_MIN, max(target, _POLL_MIN))


class FutureResolver:
    """Polls ``retrieve_future`` on behalf of :class:`APIFuture`.

    Injected rather than inherited so tests can drive futures without HTTP.
    """

    def __init__(
        self,
        poll: Callable[[str], dict[str, Any] | bytes],
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._poll = poll
        self._sleep = sleep
        self._clock = clock

    def resolve(
        self,
        request_id: str,
        *,
        timeout: float | None,
        parse: Callable[[dict[str, Any] | bytes], T],
    ) -> T:
        """Poll until the request resolves.

        Args:
            request_id: the id returned by the submitting call.
            timeout: wall-clock budget in seconds; ``None`` waits indefinitely.
                Training steps legitimately take minutes, so there is no
                default deadline.
            parse: turns the resolved payload into the caller's type.

        Raises:
            FutureTimeoutError: the budget expired while still pending.
            RequestFailedError: the server reported the work failed.
        """
        started = self._clock()
        attempt = 0
        queue_state: str | None = None

        while True:
            payload = self._poll(request_id)

            # Protobuf bodies are never envelopes; a binary body is the result.
            if isinstance(payload, (bytes, bytearray)):
                return parse(bytes(payload))

            kind = payload.get("type")
            if kind == "try_again":
                queue_state = payload.get("queue_state") or queue_state
            elif kind == "request_failed" or (
                "error" in payload and "category" in payload
            ):
                raise RequestFailedError(
                    str(payload.get("error", "request failed")),
                    category=payload.get("category"),
                    request_id=request_id,
                )
            else:
                return parse(payload)

            if timeout is not None and self._clock() - started >= timeout:
                raise FutureTimeoutError(request_id, timeout, queue_state)

            delay = _poll_delay(attempt, queue_state)
            if timeout is not None:
                remaining = timeout - (self._clock() - started)
                if remaining <= 0:
                    raise FutureTimeoutError(request_id, timeout, queue_state)
                delay = min(delay, remaining)
            self._sleep(delay)
            attempt += 1


class AsyncFutureResolver:
    """Async twin of :class:`FutureResolver`; polls with ``asyncio.sleep``."""

    def __init__(
        self,
        poll: Callable[[str], Awaitable[dict[str, Any] | bytes]],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._poll = poll
        self._sleep = sleep
        self._clock = clock

    async def resolve(
        self,
        request_id: str,
        *,
        timeout: float | None,
        parse: Callable[[dict[str, Any] | bytes], T],
    ) -> T:
        started = self._clock()
        attempt = 0
        queue_state: str | None = None

        while True:
            payload = await self._poll(request_id)
            if isinstance(payload, (bytes, bytearray)):
                return parse(bytes(payload))

            kind = payload.get("type")
            if kind == "try_again":
                queue_state = payload.get("queue_state") or queue_state
            elif kind == "request_failed" or (
                "error" in payload and "category" in payload
            ):
                raise RequestFailedError(
                    str(payload.get("error", "request failed")),
                    category=payload.get("category"),
                    request_id=request_id,
                )
            else:
                return parse(payload)

            if timeout is not None and self._clock() - started >= timeout:
                raise FutureTimeoutError(request_id, timeout, queue_state)

            delay = _poll_delay(attempt, queue_state)
            if timeout is not None:
                remaining = timeout - (self._clock() - started)
                if remaining <= 0:
                    raise FutureTimeoutError(request_id, timeout, queue_state)
                delay = min(delay, remaining)
            await self._sleep(delay)
            attempt += 1


class APIFuture(Generic[T]):
    """A pending server-side request.

    Call :meth:`result` to block until it resolves. The result is cached, so
    calling it twice is cheap and does not re-poll.

    Named and shaped like Tinker's ``APIFuture`` so recipe code ports, but it
    is our class.
    """

    __slots__ = ("_request_id", "_resolver", "_parse", "_value", "_done", "_lock", "_error")

    def __init__(
        self,
        request_id: str,
        resolver: FutureResolver,
        parse: Callable[[dict[str, Any] | bytes], T],
    ) -> None:
        self._request_id = request_id
        self._resolver = resolver
        self._parse = parse
        self._value: T | None = None
        self._error: Exception | None = None
        self._done = False
        self._lock = threading.Lock()

    @property
    def request_id(self) -> str:
        """The server-side id, useful in logs and support requests."""
        return self._request_id

    def done(self) -> bool:
        """Whether the result has already been fetched."""
        return self._done

    def result(self, timeout: float | None = None) -> T:
        """Block until the request resolves and return its payload.

        Args:
            timeout: seconds to wait. ``None`` (default) waits as long as it
                takes -- a training step is allowed to be slow.

        Raises:
            FutureTimeoutError: still pending when the budget expired.
            RequestFailedError: the server reported failure.
        """
        with self._lock:
            if self._done:
                if self._error is not None:
                    raise self._error
                return self._value  # type: ignore[return-value]
            try:
                value = self._resolver.resolve(
                    self._request_id, timeout=timeout, parse=self._parse
                )
            except FutureTimeoutError:
                # A timeout is not terminal: the request is still running and
                # a later, more patient call can still collect it.
                raise
            except Exception as exc:
                self._error = exc
                self._done = True
                raise
            self._value = value
            self._done = True
            return value

    def map(self, fn: Callable[[T], U]) -> APIFuture[U]:
        """A future that applies ``fn`` to this future's result.

        Used by compatibility shells that have to reshape a payload without
        issuing a second request -- for example
        :meth:`mind_recursive.Sampler.compute_logprobs` extracts
        ``SampleResponse.prompt_logprobs`` out of the sampling future.
        """
        return APIFuture(
            self._request_id,
            self._resolver,
            lambda payload: fn(self._parse(payload)),
        )

    def __repr__(self) -> str:
        state = "done" if self._done else "pending"
        return f"<APIFuture {self._request_id} {state}>"
