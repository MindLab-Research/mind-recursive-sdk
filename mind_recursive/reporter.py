"""``Reporter``: the HTTP write path behind ``sc.log/metric/...``.

Reports are sent over the client's own transport to the report base,
``{origin}/api/mindrsi/v1/service/report/{session_id}/{events,log,artifact,summary}`` --
the shapes the reporting service exposes, mounted where a relay user
can reach them. The key is the SDK's own ``session_id``; the SDK has no
"execution" concept. There is no local-only mode and no opt-in switch: a
reporting call is an HTTP call, using the same connection, credentials and
retry policy as the rest of the run.

It stays best-effort: a reporting failure is logged and swallowed, because a
metrics endpoint being down must not kill a GPU training run. Shapes are
validated against the upstream contract before the request, so a payload that
would be rejected leaves a clear local warning instead of an opaque 400.

Reading reports back is a console/platform concern, not part of this SDK.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from . import _protocol as P
from ._console import emit

__all__ = ["Reporter", "SYS_PREFIX"]

log = logging.getLogger("mind_recursive.reporter")

#: Namespace for metrics the SDK reports on its own behalf. Keeping these
#: separate matters: a user metric colliding with a built-in would hit the
#: server's type lock and get the whole report rejected.
SYS_PREFIX = "sys."

#: ``kind`` -> full report path. ``log`` and ``event`` share ``/events``: a
#: log line *is* an event upstream. Metrics go to ``/log``, which is the
#: upstream listener's name for it.
_ROUTES = {
    "event": P.PATH_REPORT_EVENTS,
    "log": P.PATH_REPORT_EVENTS,
    "metric": P.PATH_REPORT_METRIC,
    "artifact": P.PATH_REPORT_ARTIFACT,
    "summary": P.PATH_REPORT_SUMMARY,
}

#: Reporting is not the training path: a report may wait a little, but a dead
#: endpoint must not hold a step for minutes.
REPORT_TIMEOUT = 10.0
REPORT_MAX_RETRIES = 1

_LEVELS = ("debug", "info", "warning", "error")

#: ``level`` as spelled on the wire -> ``logging`` level for the terminal echo.
_TERMINAL_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _echo_log(message: str, level: str, step: int | None) -> None:
    """Echo a user log line to the terminal, beside the web report."""
    prefix = f"step={step} " if step is not None else ""
    emit(f"log: {prefix}{message}", level=_TERMINAL_LEVELS.get(level, logging.INFO))


def _echo_metric(metrics: dict[str, Any], step: int, stage: str | None) -> None:
    label = f" [{stage}]" if stage else ""
    emit(f"metric{label} step={step}: {metrics}")


def _echo_artifact(uri: str, artifact_type: str, name: str | None) -> None:
    suffix = f" (name={name})" if name else ""
    emit(f"artifact {artifact_type}: {uri}{suffix}")

#: Upstream limits, enforced here so a shape that the reporting service would
#: reject does not pass silently. Sources: event field caps in
#: ``internal/biz/biz_execution/report_events.go:26-40``; the artifact byte cap
#: is ``REPORT_MAX_ARTIFACT_BYTES``, default 8 KiB.
MAX_EVENT_ID_CHARS = 64
MAX_EVENT_TYPE_BYTES = 128
MAX_EVENT_CODE_CHARS = 128
MAX_EVENT_MESSAGE_CHARS = 2048


def _now_ms() -> int:
    return int(time.time() * 1000)


class Reporter:
    """The reporting surface behind ``ServiceClient.log/metric/...``.

    One reporter per client, owned by the client itself. It is not created by
    the caller and not exposed on the package surface: there is exactly one
    writer for a session, and its methods are the ``ServiceClient``'s methods.

    Every method is an HTTP write (see the module docstring), best-effort and
    never raising: reporting must not break training.

    Example::

        sc = mind_recursive.ServiceClient()
        sc.log("starting on Qwen/Qwen3.8-27B")
        for step, batch in enumerate(batches):
            out = tc.forward_backward(batch).result()
            tc.optim_step().result()
            sc.metric({"loss": out.loss}, step=step, total=800)
        sc.summary(final_loss=0.42)
    """

    def __init__(self, *, service: Any) -> None:
        self._service = service
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def reporter_id(self) -> str | None:
        """``{session_id}:reporter``, or ``None`` once the client is closed.

        Derived the same way a model id is, so the single reporter is
        addressable as a resource inside its session rather than anonymous.
        """
        session_id = getattr(self._service, "session_id", None)
        if session_id is None:
            return None
        return f"{session_id}:reporter"

    # -- reporting ------------------------------------------------------------
    def log(
        self,
        message: str,
        *,
        level: str = "info",
        step: int | None = None,
    ) -> None:
        """Record one runtime log line. Sugar over :meth:`event` with ``type="log"``.

        Args:
            message: human-readable text.
            level: ``debug`` | ``info`` | ``warning`` | ``error``.
            step: optional step number, to line the log up with metrics.
        """
        try:
            message = str(message)
            _echo_log(message, level, step)
            self.event("log", level=level, message=message, step=step)
        except Exception as exc:  # noqa: BLE001 - reporting must not break training
            log.warning("reporter.log failed (ignored): %s", exc)

    def event(
        self,
        type: str,  # noqa: A002 - the contract field is named "type"
        *,
        level: str = "info",
        code: str | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        """Record a structured timeline event.

        ``log`` is a special case: ``log(text)`` is ``event("log", message=text)``.

        Args:
            type: dotted open-set name, e.g. ``upstream.retrying``. Upstream
                requires ``^[a-z0-9_.]+$``, at most 128 bytes.
            level: ``debug`` | ``info`` | ``warning`` | ``error``.
            code: machine-readable reason code, e.g. ``429``.
            message: human-readable text.
            data: structured extras (must be a JSON object, not a list).
            step: optional step number, to line the event up with metrics.
        """
        try:
            with self._lock:
                self._seq += 1
                seq = self._seq
            event: dict[str, Any] = {
                "id": f"{seq:08d}-{_now_ms()}"[:MAX_EVENT_ID_CHARS],
                "seq": seq,
                "ts": _now_ms(),
                "type": str(type)[:MAX_EVENT_TYPE_BYTES],
                "level": level if level in _LEVELS else "info",
            }
            if message is not None:
                event["message"] = str(message)[:MAX_EVENT_MESSAGE_CHARS]
            if code:
                event["code"] = str(code)[:MAX_EVENT_CODE_CHARS]
            if data:
                # Upstream rejects a non-object ``data`` for the whole batch.
                if not isinstance(data, dict):
                    raise TypeError(
                        f"event data must be a dict, got {data.__class__.__name__}"
                    )
                event["data"] = data
            if step is not None:
                event["step"] = int(step)
            self._record("event", event)
        except Exception as exc:  # noqa: BLE001 - reporting must not break training
            log.warning("reporter.event failed (ignored): %s", exc)

    def metric(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        stage: str | None = None,
        total: int | None = None,
        timings: dict[str, float] | None = None,
        counters: dict[str, int] | None = None,
    ) -> None:
        """Record one step's metrics.

        Values follow the server's metric type system: scalars, or objects
        carrying ``_type``. A key's type is locked by its first non-null value
        upstream, so a later type change would be rejected there.

        Args:
            metrics: this step's values. ``None`` values are dropped rather
                than sent, since a null would otherwise occupy a key without
                locking its type.
            step: the step number. Points are upserted by step.
            stage: optional label such as ``training`` or ``evaling``.
            total: revised total step count, for progress display.
            timings: named durations for this step. Emitted as the
                ``sys.step_seconds`` distribution.
            counters: named counters for this step. Emitted as ``sys.<name>``.
        """
        try:
            payload = {k: v for k, v in metrics.items() if v is not None}
            if timings:
                labels = sorted(timings)
                payload[f"{SYS_PREFIX}step_seconds"] = {
                    "_type": "distribution",
                    "labels": labels,
                    "values": [round(max(timings[k], 0.0), 6) for k in labels],
                    "unit": "s",
                    "_render": "stack",
                }
            for name, value in (counters or {}).items():
                payload[f"{SYS_PREFIX}{name}"] = value
            body: dict[str, Any] = {"ts": _now_ms(), "step": int(step), "metrics": payload}
            if stage is not None:
                body["stage"] = stage
            if total is not None:
                body["total"] = total
            _echo_metric(payload, step, stage)
            self._record("metric", body)
        except Exception as exc:  # noqa: BLE001 - reporting must not break training
            log.warning("reporter.metric failed (ignored): %s", exc)

    def artifact(
        self,
        uri: str,
        *,
        type: str,  # noqa: A002 - the contract field is named "type"
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a produced artefact by reference. Append-only.

        Only the pointer is kept; the bytes are never fetched. ``uri`` must
        carry a scheme -- ``mint://``, ``oss://``, ``hf://``, ``https://``.
        Upstream rejects a bare string, ``javascript:`` and ``data:``.
        """
        try:
            text = str(uri)
            if "://" not in text:
                raise ValueError(f"artifact uri needs a scheme: {text!r}")
            body: dict[str, Any] = {"ts": _now_ms(), "type": type, "uri": text}
            for key, value in (
                ("name", name),
                ("version", version),
                ("description", description),
                ("metadata", metadata),
            ):
                if value is not None:
                    body[key] = value
            _echo_artifact(text, type, name)
            self._record("artifact", body)
        except Exception as exc:  # noqa: BLE001
            log.warning("reporter.artifact failed (ignored): %s", exc)

    def summary(self, mapping: dict[str, Any] | None = None, /, **kwargs: Any) -> None:
        """Record terminal scalars. Merged by key, last write wins.

        ``summary(final_loss=0.4)`` and ``summary({"final_loss": 0.4})`` are
        equivalent and may be mixed.
        """
        try:
            merged: dict[str, Any] = {}
            if mapping:
                merged.update(mapping)
            merged.update(kwargs)
            self._record("summary", {"ts": _now_ms(), "summary": merged})
        except Exception as exc:  # noqa: BLE001
            log.warning("reporter.summary failed (ignored): %s", exc)

    # -- sink -----------------------------------------------------------------
    def _record(self, kind: str, body: dict[str, Any]) -> None:
        """POST one report through the service's own transport.

        Reusing the transport keeps headers, retry policy, pooling and timeouts
        identical to the rest of the run, so a reporting call cannot be the one
        request that behaves differently. ``require_session`` only rules out a
        closed client here: a report belongs to a session, and the client has
        held one since construction.

        Failures are logged and swallowed: reporting must never break training.
        """
        try:
            path = _ROUTES[kind]
            session_id = self._service.require_session()
            payload = (
                {"events": [body]} if kind in ("event", "log") else dict(body)
            )
            self._service.transport.request(
                "POST",
                path.format(session_id=session_id),
                base_url=self._service.config.report_base_url,
                pool="control",
                json_body=payload,
                timeout=REPORT_TIMEOUT,
                max_retries=REPORT_MAX_RETRIES,
            )
        except Exception as exc:  # noqa: BLE001 - reporting must not break training
            log.warning("report %s failed (ignored): %s", kind, exc)

    def __repr__(self) -> str:
        return f"<Reporter {self.reporter_id or '(no session)'}>"
