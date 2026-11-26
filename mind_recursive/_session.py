"""Session negotiation, heartbeat, and the server's behaviour flags.

Two pieces the stock SDK gets for free and we must implement:

**``client/config`` is a set of instructions, not information.** The server
tells the client how to behave: how large a chunk may be, whether to compress,
how many concurrent samples to allow. Ignoring it is not "using defaults", it
is violating a negotiated contract. Observed on alpha::

    {"pjwt_auth_enabled":false, "credential_default_source":"api_key",
     "parallel_fwdbwd_chunks":true, "proto_write_fwdbwd":false,
     "proto_compress_fwdbwd":false, "fwd_via_fwdbwd":false,
     "fwdbwd_max_chunk_len":1024, "fwdbwd_max_chunk_bytes_count":50000000,
     "sample_max_concurrent_requests":2000, "sample_join_sampling_session":true,
     "use_pyqwest_transport":false}

``pjwt_auth_enabled: false`` is also the root cause of a dead end documented
in the design: it is why ``TINKER_CREDENTIAL_CMD`` cannot work against this platform.

**Heartbeats are mandatory.** Without ``session_heartbeat`` the server reclaims
the session's resources as idle, killing an in-flight training run. The
heartbeat runs on a daemon thread so it cannot keep a process alive, and it
stops before the session is finished.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["ClientConfig", "Heartbeat", "DEFAULT_HEARTBEAT_SECONDS"]

log = logging.getLogger("mind_recursive.session")

#: Upstream heartbeats roughly every 10s. The server's idle reclaim window is
#: larger than this, so the interval has headroom for one missed beat.
DEFAULT_HEARTBEAT_SECONDS = 10.0


@dataclass
class ClientConfig:
    """Server-negotiated behaviour flags.

    Unknown keys are kept in ``extra`` rather than dropped, so a server that
    adds a flag does not lose it on the way through an older SDK.
    """

    #: Whether the server does JWT exchange. False on Mind Recursive: it only accepts
    #: API keys, which is why credential-command auth cannot work here.
    pjwt_auth_enabled: bool = False
    credential_default_source: str = "api_key"

    #: Chunking budget for forward_backward. Exceeding either limit earns a
    #: 413 rather than a helpful message, so the client must split batches.
    fwdbwd_max_chunk_len: int = 1024
    fwdbwd_max_chunk_bytes_count: int = 50_000_000
    parallel_fwdbwd_chunks: bool = True

    #: Whether to zstd-compress forward_backward bodies. Server's call.
    proto_compress_fwdbwd: bool = False
    proto_write_fwdbwd: bool = False
    #: Whether a plain forward must be routed through forward_backward.
    fwd_via_fwdbwd: bool = False

    sample_max_concurrent_requests: int = 2000
    sample_join_sampling_session: bool = True
    sample_no_retries: bool = False

    extra: dict[str, Any] = field(default_factory=dict)

    _KNOWN = (
        "pjwt_auth_enabled",
        "credential_default_source",
        "fwdbwd_max_chunk_len",
        "fwdbwd_max_chunk_bytes_count",
        "parallel_fwdbwd_chunks",
        "proto_compress_fwdbwd",
        "proto_write_fwdbwd",
        "fwd_via_fwdbwd",
        "sample_max_concurrent_requests",
        "sample_join_sampling_session",
        "sample_no_retries",
    )

    @classmethod
    def from_wire(cls, payload: dict[str, Any] | None) -> ClientConfig:
        """Build from a ``client/config`` response, tolerating absent keys.

        A missing flag keeps our default; a malformed value is ignored with a
        warning rather than raising, because a surprising config value should
        not stop a training run from starting.
        """
        if not payload:
            return cls()
        known: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in payload.items():
            if key in cls._KNOWN:
                known[key] = value
            else:
                extra[key] = value

        config = cls(extra=extra)
        for key, value in known.items():
            current = getattr(config, key)
            try:
                if isinstance(current, bool):
                    coerced: Any = bool(value)
                elif isinstance(current, int):
                    coerced = int(value)
                else:
                    coerced = value
            except (TypeError, ValueError):
                log.warning("ignoring malformed client/config value %s=%r", key, value)
                continue
            setattr(config, key, coerced)
        return config

    def chunk_budget(self) -> tuple[int, int]:
        """``(max_datums_per_chunk, max_bytes_per_chunk)``."""
        return self.fwdbwd_max_chunk_len, self.fwdbwd_max_chunk_bytes_count


class Heartbeat:
    """Background keep-alive for one session.

    Runs on a daemon thread: a forgotten heartbeat must never prevent a process
    from exiting. Failures are logged and retried on the next tick -- a single
    missed beat is survivable, and raising from a background thread would be
    both unhelpful and unobservable.

    The exception is a *terminal* failure, such as the server answering 410
    Gone because it reclaimed the session. Retrying that forever would be noise
    on the network and in the log, so an exception listed in ``stop_on`` ends
    the loop instead.
    """

    def __init__(
        self,
        beat: Callable[[], None],
        *,
        interval: float = DEFAULT_HEARTBEAT_SECONDS,
        name: str = "mind-recursive-heartbeat",
        stop_on: tuple[type[BaseException], ...] = (),
    ) -> None:
        if not (interval > 0) or interval != interval or interval == float("inf"):
            raise ValueError(f"heartbeat interval must be a finite positive number, got {interval!r}")
        self._beat = beat
        self._interval = interval
        self._name = name
        self._stop_on = stop_on
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_failures = 0
        self._terminal_error: BaseException | None = None

    def start(self) -> None:
        """Start beating. Idempotent."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Wait first: the session was just created, so it is already fresh.
        while not self._stop.wait(self._interval):
            try:
                self._beat()
                self._consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - a heartbeat must not crash
                if self._stop_on and isinstance(exc, self._stop_on):
                    self._terminal_error = exc
                    log.info("session heartbeat stopped: %s", exc)
                    return
                self._consecutive_failures += 1
                log.warning(
                    "session heartbeat failed (%d consecutive): %s",
                    self._consecutive_failures,
                    exc,
                )

    def stop(self, timeout: float = 2.0) -> None:
        """Stop beating and join the thread.

        Must be called before finishing a session: the ordering (stop, then
        report the terminal state) is the same constraint the training
        package's reporter observes.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def consecutive_failures(self) -> int:
        """Failed beats since the last success -- surfaced in ``sys.*`` metrics."""
        return self._consecutive_failures

    @property
    def terminal_error(self) -> BaseException | None:
        """The ``stop_on`` exception that ended the loop, if one did."""
        return self._terminal_error

    def __enter__(self) -> Heartbeat:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
