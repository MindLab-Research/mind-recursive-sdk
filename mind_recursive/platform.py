"""Platform resources: datasets, managed training, executions, model versions.

Everything here hangs off :class:`~mind_recursive.user.UserClient` (``uc.datasets``,
``uc.training``, ...) because none of it needs a training session. The calls go
to the platform API at ``{origin}/api/mint/cli/v1``.

That contract differs from the training planes in three ways, all handled here
so callers never see them:

* **Envelope.** Success is HTTP 200 with ``{"code": 200, "metadata": ...}``;
  a business error is also HTTP 200, with another ``code``. The relay's own
  rejections are ``{"error": {"code", "message"}, "request_id"}`` on a 4xx.
* **int64 as strings.** protojson writes int64 as ``"123"``; the dataclasses
  below convert them to ``int``.
* **Retries.** Only GETs and writes carrying an ``Idempotency-Key`` are
  replayed. ``training.submit`` and ``datasets.upload`` generate a key per call
  and reuse it across their own retries; other writes are sent once.

Presigned object URLs (dataset upload and download) are fetched with a clean
client: no ``X-API-Key``, no redirects, and never logged or put in an error.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, Mapping, Sequence, TypeVar
from urllib.parse import quote, urlsplit

import httpx

from ._config import Config
from ._errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    ConfigurationError,
    ConflictError,
    FutureTimeoutError,
    MindRecursiveError,
    NotFoundError,
    PermissionDeniedError,
    status_to_error,
)
from ._transport import JSON_CONTENT_TYPE, Transport

__all__ = [
    "Account",
    "PlatformCapabilities",
    "Page",
    "Dataset",
    "DatasetFinding",
    "ValidationReport",
    "DatasetValidationError",
    "Execution",
    "ExecutionProgress",
    "MetricPoint",
    "MetricsPage",
    "MetricKey",
    "ExecutionArtifact",
    "ExecutionResult",
    "ExecutionEvent",
    "ModelVersion",
    "Deployment",
    "DeploymentError",
    "Datasets",
    "Training",
    "Models",
    "Deployments",
    "Chat",
    "ChatCompletions",
    "ResponseObject",
    "TERMINAL_STATES",
    "DEPLOYMENT_SETTLED_STATES",
]

log = logging.getLogger("mind_recursive.platform")

T = TypeVar("T")

#: Execution states after which nothing changes any more.
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})

_CODE_OK = 200
_CODE_PERMISSION_DENIED = 100015
_CODE_DATASET_NOT_FOUND = 140056
_CODE_DATASET_REFERENCED = 140058


# -- wire helpers -------------------------------------------------------------
def _int(value: Any) -> int | None:
    """protojson int64: a string, a number, or absent."""
    if value is None or value == "":
        return None
    return int(value)


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value != "" else None


def _seg(value: str) -> str:
    """One escaped path segment; a ``/`` in an id must not change the route."""
    if not isinstance(value, str) or not value:
        raise ValueError("resource id must be a non-empty string")
    return quote(value, safe="")


def _business_error(code: int, message: str, path: str, status: int) -> APIError:
    """Map a server business code onto the SDK's exception classes."""
    text = message or f"server error {code}"
    kwargs: dict[str, Any] = {"status": status, "path": path, "code": code}
    if code == _CODE_PERMISSION_DENIED or code == 403:
        return PermissionDeniedError(text, **kwargs)
    if 100000 <= code < 110000:
        return AuthenticationError(text, **kwargs)
    if code == _CODE_DATASET_NOT_FOUND:
        return NotFoundError(text, **kwargs)
    if code == _CODE_DATASET_REFERENCED:
        return ConflictError(text, **kwargs)
    return APIError(text, **kwargs)


# -- data types ---------------------------------------------------------------
@dataclass
class Page(Generic[T]):
    """One page of a list call. ``total`` counts every match, not this page."""

    items: list[T] = field(default_factory=list)
    total: int | None = None

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class Account:
    """``GET /me``: who the key belongs to, as the server sees it."""

    user_id: str | None = None
    role: str | None = None
    channel: str | None = None
    admission_create: bool = False
    admission_retry: bool = False
    admission_continue: bool = False

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> Account:
        return cls(
            user_id=_str(d.get("user_id")),
            role=_str(d.get("role")),
            channel=_str(d.get("channel")),
            admission_create=bool(d.get("admission_create")),
            admission_retry=bool(d.get("admission_retry")),
            admission_continue=bool(d.get("admission_continue")),
        )


@dataclass
class PlatformCapabilities:
    """``GET /capabilities``: which platform features this deployment enables."""

    training: bool = False
    retry: bool = False
    deployment: bool = False
    organization_sharing: bool = False
    evaluate: bool = False
    contract_version: str | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> PlatformCapabilities:
        return cls(
            training=bool(d.get("training")),
            retry=bool(d.get("retry")),
            deployment=bool(d.get("deployment")),
            organization_sharing=bool(d.get("organization_sharing")),
            evaluate=bool(d.get("evaluate")),
            contract_version=_str(d.get("contract_version")),
        )


@dataclass
class Dataset:
    """One registered dataset version.

    ``status`` is ``uploading`` / ``building`` while the server validates, then
    ``available`` or ``failed`` (with ``failure_reason``).
    """

    dataset_version: str
    type: str | None = None
    status: str | None = None
    failure_reason: str | None = None
    owner_uid: str | None = None
    digest: str | None = None
    size_bytes: int | None = None
    original_filename: str | None = None
    note: str | None = None
    rows: int | None = None
    messages: int | None = None
    assistant_messages: int | None = None
    tool_calls: int | None = None
    max_messages: int | None = None
    #: Milliseconds since the epoch.
    created_at: int | None = None

    @property
    def available(self) -> bool:
        return self.status == "available"

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> Dataset:
        return cls(
            dataset_version=d.get("dataset_version") or "",
            type=_str(d.get("type")),
            status=_str(d.get("status")),
            failure_reason=_str(d.get("failure_reason")),
            owner_uid=_str(d.get("owner_uid")),
            digest=_str(d.get("digest")),
            size_bytes=_int(d.get("size_bytes")),
            original_filename=_str(d.get("original_filename")),
            note=_str(d.get("note")),
            rows=_int(d.get("rows")),
            messages=_int(d.get("messages")),
            assistant_messages=_int(d.get("assistant_messages")),
            tool_calls=_int(d.get("tool_calls")),
            max_messages=_int(d.get("max_messages")),
            created_at=_int(d.get("created_at")),
        )


@dataclass
class DatasetFinding:
    """One validation finding. ``severity`` is ``fatal`` or ``warn``."""

    severity: str | None = None
    layer: str | None = None
    rule: str | None = None
    line: int | None = None
    problem_id: str | None = None
    detail: str | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> DatasetFinding:
        return cls(
            severity=_str(d.get("severity")),
            layer=_str(d.get("layer")),
            rule=_str(d.get("rule")),
            line=_int(d.get("line")),
            problem_id=_str(d.get("problem_id")),
            detail=_str(d.get("detail")),
        )


@dataclass
class ValidationReport:
    """The persisted validation report of a dataset, one page of findings."""

    dataset_version: str
    status: str | None = None
    validator_revision: str | None = None
    findings: list[DatasetFinding] = field(default_factory=list)
    fatal_count: int = 0
    warn_count: int = 0
    total: int = 0

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> ValidationReport:
        return cls(
            dataset_version=d.get("dataset_version") or "",
            status=_str(d.get("status")),
            validator_revision=_str(d.get("validator_revision")),
            findings=[DatasetFinding.from_wire(f) for f in d.get("findings") or []],
            fatal_count=_int(d.get("fatal_count")) or 0,
            warn_count=_int(d.get("warn_count")) or 0,
            total=_int(d.get("total")) or 0,
        )


class DatasetValidationError(MindRecursiveError):
    """An uploaded dataset finished validation as ``failed``.

    ``dataset`` carries the server's record, including ``failure_reason``. Use
    ``uc.datasets.validation_report(version)`` for the per-line findings.
    """

    def __init__(self, dataset: Dataset):
        reason = dataset.failure_reason or "validation failed"
        super().__init__(f"dataset {dataset.dataset_version} failed: {reason}")
        self.dataset = dataset


@dataclass
class ExecutionProgress:
    """The latest progress snapshot embedded in an execution."""

    step: int | None = None
    total: int | None = None
    loss: str | None = None
    stage: str | None = None
    updated_at: int | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any] | None) -> ExecutionProgress | None:
        if not d:
            return None
        return cls(
            step=_int(d.get("step")),
            total=_int(d.get("total")),
            loss=_str(d.get("loss")),
            stage=_str(d.get("stage")),
            updated_at=_int(d.get("updated_at")),
        )


@dataclass
class Execution:
    """One managed job (training run). ``state`` is the scheduler's truth.

    ``version_id`` is set once a training run has registered its output model
    version; pass it to ``uc.models.get`` or ``uc.deployments.create``.
    """

    execution_id: str
    kind: str | None = None
    state: str | None = None
    display_name: str | None = None
    owner_uid: str | None = None
    cancel_requested: bool = False
    result_reason: str | None = None
    result_exit_code: int | None = None
    submitted_at: int | None = None
    started_at: int | None = None
    finished_at: int | None = None
    updated_at: int | None = None
    progress: ExecutionProgress | None = None
    dataset_version: str | None = None
    base_model: str | None = None
    retry_of: str | None = None
    output_registration: str | None = None
    train_kind: str | None = None
    package_version: str | None = None
    version_id: str | None = None
    base_version_id: str | None = None
    admission_status: str | None = None
    #: The full server record, for fields not named above (``history`` etc.).
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> Execution:
        return cls(
            execution_id=d.get("execution_id") or "",
            kind=_str(d.get("kind")),
            state=_str(d.get("state")),
            display_name=_str(d.get("display_name")),
            owner_uid=_str(d.get("owner_uid")),
            cancel_requested=bool(d.get("cancel_requested")),
            result_reason=_str(d.get("result_reason")),
            result_exit_code=_int(d.get("result_exit_code")),
            submitted_at=_int(d.get("submitted_at")),
            started_at=_int(d.get("started_at")),
            finished_at=_int(d.get("finished_at")),
            updated_at=_int(d.get("updated_at")),
            progress=ExecutionProgress.from_wire(d.get("progress")),
            dataset_version=_str(d.get("dataset_version")),
            base_model=_str(d.get("base_model")),
            retry_of=_str(d.get("retry_of")),
            output_registration=_str(d.get("output_registration")),
            train_kind=_str(d.get("frozen_train_kind")),
            package_version=_str(d.get("frozen_package_version")),
            version_id=_str(d.get("version_id")),
            base_version_id=_str(d.get("base_version_id")),
            admission_status=_str(d.get("admission_status")),
            raw=dict(d),
        )


@dataclass
class MetricPoint:
    """One reported metric row. ``metrics`` values are kept as sent."""

    ts: int | None = None
    stage: str | None = None
    step: int | None = None
    total: int | None = None
    loss: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> MetricPoint:
        return cls(
            ts=_int(d.get("ts")),
            stage=_str(d.get("stage")),
            step=_int(d.get("step")),
            total=_int(d.get("total")),
            loss=_str(d.get("loss")),
            metrics=dict(d.get("metrics") or {}),
        )


@dataclass
class MetricsPage:
    """One page of metrics.

    ``caught_up`` means "no more rows right now", not "the run has finished";
    keep ``next_cursor`` and poll again while the execution is running.
    """

    points: list[MetricPoint] = field(default_factory=list)
    next_cursor: str | None = None
    caught_up: bool = False


@dataclass
class MetricKey:
    key: str
    type: str | None = None
    kind: str | None = None
    first_step: int | None = None
    last_step: int | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> MetricKey:
        return cls(
            key=d.get("key") or "",
            type=_str(d.get("type")),
            kind=_str(d.get("kind")),
            first_step=_int(d.get("first_step")),
            last_step=_int(d.get("last_step")),
        )


@dataclass
class ExecutionArtifact:
    type: str | None = None
    uri: str | None = None
    name: str | None = None
    version: str | None = None
    description: str | None = None
    metadata: Any = None
    ts: int | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> ExecutionArtifact:
        return cls(
            type=_str(d.get("type")),
            uri=_str(d.get("uri")),
            name=_str(d.get("name")),
            version=_str(d.get("version")),
            description=_str(d.get("description")),
            metadata=d.get("metadata"),
            ts=_int(d.get("ts")),
        )


@dataclass
class ExecutionResult:
    """What the algorithm reported: artifacts, summary, error.

    ``status`` is the algorithm's own report and does not replace
    :attr:`Execution.state`.
    """

    status: str | None = None
    artifacts: list[ExecutionArtifact] = field(default_factory=list)
    data_summary: Any = None
    error_code: str | None = None
    error_message: str | None = None
    sealed: bool = False
    stage: str | None = None
    step: int | None = None
    total: int | None = None
    last_event_at: int | None = None
    source: str | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> ExecutionResult:
        return cls(
            status=_str(d.get("status")),
            artifacts=[ExecutionArtifact.from_wire(a) for a in d.get("artifacts") or []],
            data_summary=d.get("data_summary"),
            error_code=_str(d.get("error_code")),
            error_message=_str(d.get("error_message")),
            sealed=bool(d.get("sealed")),
            stage=_str(d.get("stage")),
            step=_int(d.get("step")),
            total=_int(d.get("total")),
            last_event_at=_int(d.get("last_event_at")),
            source=_str(d.get("source")),
        )


@dataclass
class ExecutionEvent:
    """One structured event. ``data_json`` stays a JSON string, as sent."""

    id: str | None = None
    seq: int | None = None
    ts: int | None = None
    type: str | None = None
    level: str | None = None
    code: str | None = None
    message: str | None = None
    data_json: str | None = None
    received_at: int | None = None

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> ExecutionEvent:
        return cls(
            id=_str(d.get("id")),
            seq=_int(d.get("seq")),
            ts=_int(d.get("ts")),
            type=_str(d.get("type")),
            level=_str(d.get("level")),
            code=_str(d.get("code")),
            message=_str(d.get("message")),
            data_json=_str(d.get("data_json")),
            received_at=_int(d.get("received_at")),
        )


@dataclass
class ModelVersion:
    """A registered model version (the output of a managed training run)."""

    version_id: str
    owner_uid: str | None = None
    execution_id: str | None = None
    train_kind: str | None = None
    dataset_version: str | None = None
    train_data_rows: int | None = None
    package_version: str | None = None
    parent_id: str | None = None
    base_model: str | None = None
    base_model_revision: str | None = None
    #: Training upstream frozen on the version: ``mint`` or ``tinker``.
    upstream: str | None = None
    created_at: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> ModelVersion:
        return cls(
            version_id=d.get("version_id") or "",
            owner_uid=_str(d.get("owner_uid")),
            execution_id=_str(d.get("execution_id")),
            train_kind=_str(d.get("train_kind")),
            dataset_version=_str(d.get("dataset_version")),
            train_data_rows=_int(d.get("train_data_rows")),
            package_version=_str(d.get("package_version")),
            parent_id=_str(d.get("parent_id")),
            base_model=_str(d.get("base_model")),
            base_model_revision=_str(d.get("base_model_revision")),
            upstream=_str(d.get("upstream")),
            created_at=_int(d.get("created_at")),
            raw=dict(d),
        )


# -- request layer ------------------------------------------------------------
class _Platform:
    """Calls against the CLI entry: envelope unwrapping and error mapping.

    Reuses the owning client's :class:`Transport` (same key header, pools and
    connection-fault retries). HTTP status retries are left to the transport,
    except that non-idempotent writes get ``max_retries=0`` so a lost response
    cannot submit twice.
    """

    def __init__(self, config: Config, transport: Transport, owner: Any = None) -> None:
        self._config = config
        self._transport = transport
        self._owner = owner

    @property
    def base_url(self) -> str:
        if not self._config.platform_base_url:
            raise ConfigurationError(
                "this Config has no platform_base_url; build it with build_config()"
            )
        return self._config.platform_base_url

    def call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Send one request and return the envelope's ``metadata``."""
        if getattr(self._owner, "_closed", False):
            raise ConfigurationError("this UserClient has been closed")
        replayable = method == "GET" or idempotency_key is not None
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        query = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        try:
            response = self._transport.request(
                method,
                path,
                base_url=self.base_url,
                json_body=json_body,
                params=query or None,
                accept=JSON_CONTENT_TYPE,
                extra_headers=headers,
                timeout=timeout,
                max_retries=None if replayable else 0,
            )
        except APIError as exc:
            raise _entry_error(exc, path) from exc
        return _unwrap(response, path)


def _entry_error(exc: APIError, path: str) -> APIError:
    """Re-read a 4xx raised by the transport as a relay entry rejection.

    The relay answers ``{"error": {"code": "...", "message": "..."}}``; the
    transport only looks for a string ``error``, so the code would be lost.
    """
    if not exc.body:
        return exc
    try:
        payload = json.loads(exc.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return exc
    if not isinstance(payload, dict):
        return exc
    err = payload.get("error")
    if isinstance(err, dict) and isinstance(err.get("code"), str):
        code = err["code"]
        message = err.get("message") or code
        cls = type(exc)
        if code.startswith("credential_"):
            cls = AuthenticationError
        elif code in ("access_forbidden", "admission_denied"):
            cls = PermissionDeniedError
        elif code == "route_not_allowed":
            message = (
                f"{message} -- the relay CLI entry does not expose {path}"
            )
        return cls(
            f"{code}: {message}",
            status=exc.status,
            path=path,
            body=exc.body,
            code=code,
            request_id=_str(payload.get("request_id")),
        )
    if isinstance(payload.get("code"), int) and payload["code"] != _CODE_OK:
        return _business_error(
            payload["code"], str(payload.get("message") or ""), path, exc.status or 0
        )
    return exc


def _unwrap(response: httpx.Response, path: str) -> Any:
    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError(
            f"non-JSON response from the CLI entry (HTTP {response.status_code})",
            status=response.status_code,
            path=path,
        ) from exc
    if not isinstance(payload, dict) or "code" not in payload:
        raise APIError(
            "response is not a {code, message, metadata} envelope; check that "
            "base_url is the relay origin",
            status=response.status_code,
            path=path,
        )
    code = payload.get("code")
    if code != _CODE_OK:
        raise _business_error(
            int(code) if isinstance(code, (int, str)) and str(code).isdigit() else -1,
            str(payload.get("message") or ""),
            path,
            response.status_code,
        )
    return payload.get("metadata")


def _obj(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise APIError(f"expected an object in metadata, got {type(value).__name__}", path=path)
    return value


def _clean_http(config: Config, *, timeout: float | None) -> httpx.Client:
    """A client for presigned object URLs: no credentials, no redirects."""
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=config.connect_timeout),
        trust_env=config.trust_env,
        follow_redirects=False,
    )


def _check_presigned(url: Any) -> str:
    if not isinstance(url, str):
        raise APIError("server returned no object URL")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username:
        # Never echo the URL itself: it carries a signature.
        raise APIError("server returned an invalid object URL")
    return url


# -- resources ----------------------------------------------------------------
class Datasets:
    """``uc.datasets``: upload, inspect, download and delete datasets.

    Example::

        ds = uc.datasets.upload("train.jsonl", type="sft", note="v1")
        print(ds.dataset_version, ds.rows)
        uc.datasets.download(ds.dataset_version, "copy.jsonl")
    """

    def __init__(self, api: _Platform) -> None:
        self._api = api

    def list(
        self, *, type: str | None = None, page: int | None = None, page_size: int | None = None
    ) -> Page[Dataset]:
        """Your datasets, newest first. ``type`` filters to ``sft`` or ``grpo``."""
        path = "/datasets"
        d = _obj(
            self._api.call("GET", path, params={"type": type, "page": page, "page_size": page_size}),
            path,
        )
        return Page([Dataset.from_wire(x) for x in d.get("list") or []], _int(d.get("total")))

    def get(self, dataset_version: str, *, owner_uid: str | None = None) -> Dataset:
        """One dataset. ``owner_uid`` reads a dataset shared with you."""
        path = f"/datasets/{_seg(dataset_version)}"
        return Dataset.from_wire(
            _obj(self._api.call("GET", path, params={"owner_uid": owner_uid}), path)
        )

    def rows(
        self,
        dataset_version: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        owner_uid: str | None = None,
    ) -> list[Any]:
        """A slice of the dataset's rows, each as the decoded JSON object."""
        path = f"/datasets/{_seg(dataset_version)}/rows"
        d = _obj(
            self._api.call(
                "GET", path, params={"offset": offset, "limit": limit, "owner_uid": owner_uid}
            ),
            path,
        )
        return list(d.get("rows") or [])

    def validation_report(
        self,
        dataset_version: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        owner_uid: str | None = None,
    ) -> ValidationReport:
        """The persisted validation findings, one page at a time."""
        path = f"/datasets/{_seg(dataset_version)}/validation-report"
        d = _obj(
            self._api.call(
                "GET", path, params={"offset": offset, "limit": limit, "owner_uid": owner_uid}
            ),
            path,
        )
        return ValidationReport.from_wire(d)

    def update_note(self, dataset_version: str, note: str) -> Dataset:
        """Change the note (at most 80 characters). Nothing else is editable."""
        path = f"/datasets/{_seg(dataset_version)}"
        return Dataset.from_wire(
            _obj(self._api.call("PATCH", path, json_body={"note": note}), path)
        )

    def delete(self, dataset_version: str) -> None:
        """Delete a dataset. Raises :class:`ConflictError` while a run uses it."""
        self._api.call("DELETE", f"/datasets/{_seg(dataset_version)}")

    def upload(
        self,
        file: str | os.PathLike[str],
        *,
        type: str = "sft",
        dataset_version: str | None = None,
        note: str | None = None,
        idempotency_key: str | None = None,
        wait: bool = True,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> Dataset:
        """Upload a local JSONL file and register it as a dataset.

        Three steps, as the CLI does them: register the upload intent, PUT the
        file straight to object storage with the returned presigned URL, then
        tell the server it is in place. The server validates in the background;
        with ``wait=True`` (the default) this polls until the dataset is
        ``available`` and raises :class:`DatasetValidationError` if it ends
        ``failed``.

        Args:
            file: local path.
            type: ``sft`` or ``grpo``.
            dataset_version: your own version name; the server picks one if omitted.
            note: free-text note.
            idempotency_key: reuse to retry the *same* upload safely. A fresh
                key is generated per call when omitted.
            wait: poll until validation finishes.
            timeout: seconds to wait for validation.
            poll_interval: initial poll interval; doubles up to 30s.
        """
        path_str = os.fspath(file)
        size = os.path.getsize(path_str)
        key = idempotency_key or f"sdk-{uuid.uuid4()}"
        intent_path = "/datasets/uploads"
        intent = _obj(
            self._api.call(
                "POST",
                intent_path,
                json_body={
                    "type": type,
                    "dataset_version": dataset_version or "",
                    "note": note or "",
                    "original_filename": os.path.basename(path_str),
                },
                idempotency_key=key,
            ),
            intent_path,
        )
        version = intent.get("dataset_version") or ""
        if not version:
            raise APIError("upload intent returned no dataset_version", path=intent_path)
        target = intent.get("upload")
        max_bytes = _int(intent.get("max_bytes"))
        if target:
            if max_bytes and size > max_bytes:
                raise ConfigurationError(
                    f"{path_str} is {size} bytes; the server accepts at most {max_bytes}"
                )
            self._put_object(target, path_str, size)
            complete = f"/datasets/{_seg(version)}/upload-complete"
            self._api.call(
                "POST",
                complete,
                json_body={"dataset_version": version, "size": size},
                idempotency_key=f"{key}-complete",
            )
        # No target: this key's intent was already consumed (a retried upload);
        # the dataset exists, so only its status is left to read.
        if not wait:
            return self.get(version)
        return self.wait(version, timeout=timeout, poll_interval=poll_interval)

    def wait(
        self, dataset_version: str, *, timeout: float = 600.0, poll_interval: float = 1.0
    ) -> Dataset:
        """Poll until validation finishes. Raises on ``failed`` or timeout."""
        deadline = time.monotonic() + timeout
        delay = max(poll_interval, 0.05)
        while True:
            ds = self.get(dataset_version)
            if ds.status == "available":
                return ds
            if ds.status == "failed":
                raise DatasetValidationError(ds)
            if time.monotonic() >= deadline:
                raise FutureTimeoutError(dataset_version, timeout, ds.status)
            time.sleep(min(delay, max(deadline - time.monotonic(), 0)))
            delay = min(delay * 2, 30.0)

    def download(
        self, dataset_version: str, destination: str | os.PathLike[str], *, timeout: float = 600.0
    ) -> str:
        """Download an ``available`` dataset to a local file.

        Refuses to overwrite an existing path. Writes to a temporary file in the
        same directory and links it into place, so a failed transfer leaves
        nothing behind. Returns the destination path.
        """
        dest = os.fspath(destination)
        if os.path.lexists(dest):
            raise FileExistsError(f"download destination already exists: {dest}")
        path = f"/datasets/{_seg(dataset_version)}/download-url"
        d = _obj(self._api.call("GET", path), path)
        url = _check_presigned(d.get("url"))

        directory = os.path.dirname(os.path.abspath(dest))
        fd, tmp = tempfile.mkstemp(prefix=".mind-recursive-dataset-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as out, _clean_http(
                self._api._config, timeout=timeout
            ) as http:
                try:
                    with http.stream("GET", url) as resp:
                        if resp.status_code != 200:
                            raise APIError(
                                f"object download returned HTTP {resp.status_code}",
                                status=resp.status_code,
                                path=path,
                            )
                        expected = resp.headers.get("Content-Length")
                        written = 0
                        for chunk in resp.iter_raw():
                            out.write(chunk)
                            written += len(chunk)
                except httpx.HTTPError as exc:
                    raise APIConnectionError("object download failed", path=path) from exc
                if expected is not None and written != int(expected):
                    raise APIError("object download body length mismatch", path=path)
                out.flush()
                os.fsync(out.fileno())
            # link() fails if dest appeared meanwhile; rename() would replace it.
            os.link(tmp, dest)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
        return dest

    def _put_object(self, target: Mapping[str, Any], file: str, size: int) -> None:
        method = (target.get("method") or "PUT").upper()
        if method != "PUT":
            raise APIError(f"unsupported upload method {method!r}")
        url = _check_presigned(target.get("url"))
        # Only the headers the signature covers. Adding Content-Type breaks
        # OSS V4 signatures, so it is deliberately not set.
        headers = {str(k): str(v) for k, v in (target.get("headers") or {}).items()}
        headers["Content-Length"] = str(size)
        with open(file, "rb") as fh, _clean_http(self._api._config, timeout=None) as http:
            try:
                resp = http.request("PUT", url, content=_iter_file(fh), headers=headers)
            except httpx.HTTPError as exc:
                raise APIConnectionError("dataset object upload failed") from exc
        if not 200 <= resp.status_code < 300:
            raise APIError(
                f"dataset object upload returned HTTP {resp.status_code}",
                status=resp.status_code,
            )


def _iter_file(fh: Any, chunk: int = 1 << 20) -> Iterator[bytes]:
    while True:
        block = fh.read(chunk)
        if not block:
            return
        yield block


class Training:
    """``uc.training``: managed training runs and their records.

    A managed run executes on the platform: you give a dataset, a base model
    and a config, and get back an :class:`Execution`. Everything about the run
    afterwards -- listing your runs, waiting, cancelling, progress, metrics,
    events -- is here too. The config is validated on the server against the
    algorithm package's schema.

    Example::

        run = uc.training.submit(
            dataset_version=ds.dataset_version,
            base_model="Qwen/Qwen3.8-27B",
            config={"learning_rate": 1e-4, "num_epochs": 1},
        )
        run = uc.training.wait(run.execution_id)
        print(run.state, run.version_id)

        for r in uc.training.list(state="running"):
            print(r.execution_id, r.progress)
        for p in uc.training.iter_metrics(run.execution_id, keys=["loss"]):
            print(p.step, p.metrics)
    """

    def __init__(self, api: _Platform) -> None:
        self._api = api

    def submit(
        self,
        *,
        dataset_version: str | None = None,
        base_model: str | None = None,
        config: Mapping[str, Any] | None = None,
        kind: str = "sft",
        display_name: str | None = None,
        package_version: str | None = None,
        base_version_id: str | None = None,
        dataset_owner_uid: str | None = None,
        base_version_owner_uid: str | None = None,
        idempotency_key: str | None = None,
    ) -> Execution:
        """Start a managed training run.

        Args:
            dataset_version: the dataset to train on.
            base_model: base model name; omit when continuing ``base_version_id``.
            config: hyperparameters, as the algorithm package's schema defines them.
            kind: training kind, ``sft`` or ``grpo``.
            display_name: shown in run lists.
            package_version: pin an algorithm package; default is the newest stable.
            base_version_id: continue training from this model version.
            dataset_owner_uid / base_version_owner_uid: use resources shared with you.
            idempotency_key: resubmitting with the same key returns the same run.
                Generated per call when omitted, and reused on network retries.
        """
        body = {
            "dataset_version": dataset_version,
            "base_model": base_model,
            "config": dict(config) if config is not None else None,
            "display_name": display_name,
            "train_kind": kind,
            "base_version_id": base_version_id,
            "package_version": package_version,
            "dataset_owner_uid": dataset_owner_uid,
            "base_version_owner_uid": base_version_owner_uid,
        }
        path = "/training/runs"
        d = self._api.call(
            "POST",
            path,
            json_body={k: v for k, v in body.items() if v not in (None, "")},
            idempotency_key=idempotency_key or str(uuid.uuid4()),
        )
        return Execution.from_wire(_obj(d, path))

    def retry(
        self,
        execution_id: str,
        *,
        continue_training: bool = False,
        config: Mapping[str, Any] | None = None,
        display_name: str | None = None,
        idempotency_key: str | None = None,
    ) -> Execution:
        """Retry a finished run, as a new execution.

        ``continue_training=True`` resumes from the run's last checkpoint
        instead of starting over (the package must be resumable).
        """
        body: dict[str, Any] = {}
        if continue_training:
            body["continue_training"] = True
        if config is not None:
            body["config"] = dict(config)
        if display_name:
            body["display_name"] = display_name
        path = f"/training/runs/{_seg(execution_id)}/retry"
        d = self._api.call(
            "POST", path, json_body=body, idempotency_key=idempotency_key or str(uuid.uuid4())
        )
        return Execution.from_wire(_obj(d, path))


    def list(
        self,
        *,
        state: str | None = None,
        updated_since: int | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> Page[Execution]:
        """Your training runs, newest first.

        Args:
            state: ``pending`` / ``running`` / ``succeeded`` / ``failed`` / ``cancelled``.
            updated_since: milliseconds; only rows updated at or after it.
            page / page_size: 1-based page; the server defaults to 20 per page, max 500.
        """
        path = "/executions"
        d = _obj(
            self._api.call(
                "GET",
                path,
                params={
                    "kind": "training",
                    "state": state,
                    "updated_since": updated_since,
                    "page": page,
                    "page_size": page_size,
                },
            ),
            path,
        )
        return Page([Execution.from_wire(x) for x in d.get("list") or []], _int(d.get("total")))

    def get(self, execution_id: str) -> Execution:
        path = f"/executions/{_seg(execution_id)}"
        return Execution.from_wire(_obj(self._api.call("GET", path), path))

    def cancel(self, execution_id: str) -> Execution:
        """Request cancellation.

        The returned execution may still be ``running`` with
        ``cancel_requested=True``; the scheduler moves it to ``cancelled``.
        Use :meth:`wait` to see the end.
        """
        path = f"/executions/{_seg(execution_id)}/cancel"
        d = self._api.call("POST", path, json_body={"execution_id": execution_id})
        return Execution.from_wire(_obj(d, path))

    def rename(self, execution_id: str, display_name: str) -> Execution:
        """Change the display name (1-60 visible characters)."""
        path = f"/executions/{_seg(execution_id)}/rename"
        d = self._api.call(
            "POST", path, json_body={"execution_id": execution_id, "display_name": display_name}
        )
        return Execution.from_wire(_obj(d, path))

    def delete(self, execution_id: str) -> None:
        """Delete a finished execution record."""
        self._api.call("DELETE", f"/executions/{_seg(execution_id)}")

    def wait(
        self,
        execution_id: str,
        *,
        timeout: float | None = None,
        poll_interval: float = 10.0,
    ) -> Execution:
        """Poll until the execution reaches a terminal state and return it.

        Does not raise on ``failed`` or ``cancelled``: check ``state`` (and
        ``result_reason``) on the returned object. Raises
        :class:`FutureTimeoutError` if ``timeout`` seconds pass first.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            run = self.get(execution_id)
            if run.terminal:
                return run
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise FutureTimeoutError(execution_id, timeout or 0.0, run.state)
                time.sleep(min(poll_interval, left))
            else:
                time.sleep(poll_interval)

    def progress(self, execution_id: str, *, tail: int | None = None) -> list[str]:
        """The last ``tail`` structured progress lines, oldest first.

        These are reported progress rows, not the job's raw stdout/stderr.
        """
        path = f"/executions/{_seg(execution_id)}/progress"
        d = _obj(self._api.call("GET", path, params={"tail": tail}), path)
        return [str(line) for line in d.get("lines") or []]

    def metrics(
        self,
        execution_id: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        keys: Sequence[str] | None = None,
        include_snapshots: bool = False,
    ) -> MetricsPage:
        """One page of reported metrics. Pass ``next_cursor`` back for the next."""
        path = f"/executions/{_seg(execution_id)}/metrics"
        params: dict[str, Any] = {"cursor": cursor, "limit": limit}
        if keys:
            params["keys"] = list(keys)
        if include_snapshots:
            params["include_snapshots"] = "true"
        d = _obj(self._api.call("GET", path, params=params), path)
        return MetricsPage(
            points=[MetricPoint.from_wire(p) for p in d.get("points") or []],
            next_cursor=_str(d.get("next_cursor")),
            caught_up=bool(d.get("caught_up")),
        )

    def iter_metrics(
        self,
        execution_id: str,
        *,
        keys: Sequence[str] | None = None,
        limit: int | None = None,
        include_snapshots: bool = False,
    ) -> Iterator[MetricPoint]:
        """Every metric reported so far, following the cursor until caught up.

        Stops at "caught up", which for a running execution only means "no
        newer rows yet".
        """
        cursor: str | None = None
        while True:
            page = self.metrics(
                execution_id,
                cursor=cursor,
                limit=limit,
                keys=keys,
                include_snapshots=include_snapshots,
            )
            yield from page.points
            if page.caught_up or not page.next_cursor or page.next_cursor == cursor:
                return
            cursor = page.next_cursor

    def metric_keys(self, execution_id: str) -> list[MetricKey]:
        """Which metric keys the run reported, their type and step range."""
        path = f"/executions/{_seg(execution_id)}/metric-keys"
        d = _obj(self._api.call("GET", path), path)
        return [MetricKey.from_wire(k) for k in d.get("keys") or []]

    def result(self, execution_id: str) -> ExecutionResult:
        """What the algorithm reported: artifacts, data summary, error."""
        path = f"/executions/{_seg(execution_id)}/result"
        return ExecutionResult.from_wire(_obj(self._api.call("GET", path), path))

    def events(
        self, execution_id: str, *, page: int | None = None, page_size: int | None = None
    ) -> Page[ExecutionEvent]:
        """Structured events, newest received first."""
        path = f"/executions/{_seg(execution_id)}/events"
        d = _obj(self._api.call("GET", path, params={"page": page, "page_size": page_size}), path)
        return Page(
            [ExecutionEvent.from_wire(e) for e in d.get("events") or []], _int(d.get("total"))
        )


class Models:
    """``uc.models``: your models, i.e. the model versions managed training produced.

    Each successful run registers one, identified by ``version_id``: which
    dataset, base model and run (``execution_id``) made it, and which version
    it was continued from (``parent_id``). It is what ``uc.deployments.create``
    deploys and what ``uc.training.submit(base_version_id=...)`` continues.
    """

    def __init__(self, api: _Platform) -> None:
        self._api = api

    def list(self, *, page: int | None = None, page_size: int | None = None) -> Page[ModelVersion]:
        path = "/versions"
        d = _obj(self._api.call("GET", path, params={"page": page, "page_size": page_size}), path)
        return Page([ModelVersion.from_wire(v) for v in d.get("list") or []], _int(d.get("total")))

    def get(self, version_id: str, *, owner_uid: str | None = None) -> ModelVersion:
        path = f"/versions/{_seg(version_id)}"
        return ModelVersion.from_wire(
            _obj(self._api.call("GET", path, params={"owner_uid": owner_uid}), path)
        )

    def lineage(self, version_id: str, *, owner_uid: str | None = None) -> list[ModelVersion]:
        """The parent chain of a version, as the server orders it."""
        path = f"/versions/{_seg(version_id)}/lineage"
        d = _obj(self._api.call("GET", path, params={"owner_uid": owner_uid}), path)
        return [ModelVersion.from_wire(v) for v in d.get("chain") or []]

    def delete(self, version_id: str) -> None:
        self._api.call("DELETE", f"/versions/{_seg(version_id)}")


# -- deployments --------------------------------------------------------------
#: Deployment states after which polling stops.
DEPLOYMENT_SETTLED_STATES = frozenset({"ready", "stopped", "unknown"})


@dataclass
class Deployment:
    """One dedicated deployment of a model version.

    ``state`` is ``pending`` / ``ready`` / ``stopped`` / ``unknown``. A
    deployment is only callable once ``state == "ready"``; then
    ``public_model_id`` is the ``model`` name for the relay's inference API
    (``uc.chat.completions.create``).
    """

    deployment_id: str
    owner_uid: str | None = None
    version_id: str | None = None
    version_owner_uid: str | None = None
    display_name: str | None = None
    public_model_id: str | None = None
    base_model: str | None = None
    source: str | None = None
    train_kind: str | None = None
    #: ``active`` or ``stopped``: what was asked for.
    desired_state: str | None = None
    #: ``pending`` / ``ready`` / ``stopped`` / ``unknown``: what is observed.
    state: str | None = None
    loaded: bool = False
    healthy: bool = False
    route_ready: bool = False
    resources_released: bool = False
    route_removed: bool = False
    is_queued: bool = False
    created_at: int | None = None
    updated_at: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @classmethod
    def from_wire(cls, d: Mapping[str, Any]) -> Deployment:
        return cls(
            deployment_id=d.get("deployment_id") or "",
            owner_uid=_str(d.get("owner_uid")),
            version_id=_str(d.get("version_id")),
            version_owner_uid=_str(d.get("version_owner_uid")),
            display_name=_str(d.get("display_name")),
            public_model_id=_str(d.get("public_model_id")),
            base_model=_str(d.get("base_model")),
            source=_str(d.get("source")),
            train_kind=_str(d.get("train_kind")),
            desired_state=_str(d.get("desired_state")),
            state=_str(d.get("state")),
            loaded=bool(d.get("loaded")),
            healthy=bool(d.get("healthy")),
            route_ready=bool(d.get("route_ready")),
            resources_released=bool(d.get("resources_released")),
            route_removed=bool(d.get("route_removed")),
            is_queued=bool(d.get("is_queued")),
            created_at=_int(d.get("created_at")),
            updated_at=_int(d.get("updated_at")),
            raw=dict(d),
        )


class DeploymentError(MindRecursiveError):
    """A deployment settled in a state other than the one waited for."""

    def __init__(self, deployment: Deployment, wanted: str):
        super().__init__(
            f"deployment {deployment.deployment_id} is {deployment.state!r}, wanted {wanted!r}"
        )
        self.deployment = deployment


class Deployments:
    """``uc.deployments``: deploy model versions for inference.

    Example::

        dep = uc.deployments.create(version_id=run.version_id, display_name="sft-v1")
        dep = uc.deployments.wait(dep.deployment_id)          # until ready
        reply = uc.chat.completions.create(
            model=dep.public_model_id, messages=[{"role": "user", "content": "hi"}]
        )
        uc.deployments.stop(dep.deployment_id)                # releases the GPU
    """

    def __init__(self, api: _Platform) -> None:
        self._api = api

    def create(
        self,
        *,
        version_id: str,
        display_name: str | None = None,
        version_owner_uid: str | None = None,
        source: str | None = None,
    ) -> Deployment:
        """Deploy a model version. Returns at once, usually ``pending``.

        Args:
            version_id: a model version you own or that is shared with you.
            display_name: shown in lists.
            version_owner_uid: the version's owner, for a shared version.
            source: ``"lora"`` (default) or ``"tinker"``.

        Sent once, never retried: a lost response could otherwise start two
        deployments. Check :meth:`list` before calling again.
        """
        body = {
            "version_id": version_id,
            "display_name": display_name or "",
            "version_owner_uid": version_owner_uid,
            "source": source,
        }
        path = "/deployments"
        d = self._api.call(
            "POST", path, json_body={k: v for k, v in body.items() if v is not None}
        )
        return Deployment.from_wire(_obj(d, path))

    def list(
        self, *, state: str | None = None, page: int | None = None, page_size: int | None = None
    ) -> Page[Deployment]:
        """Your deployments. ``state`` filters to pending/ready/stopped/unknown."""
        path = "/deployments"
        d = _obj(
            self._api.call(
                "GET", path, params={"state": state, "page": page, "page_size": page_size}
            ),
            path,
        )
        return Page([Deployment.from_wire(x) for x in d.get("list") or []], _int(d.get("total")))

    def get(self, deployment_id: str) -> Deployment:
        path = f"/deployments/{_seg(deployment_id)}"
        return Deployment.from_wire(_obj(self._api.call("GET", path), path))

    def stop(self, deployment_id: str) -> Deployment:
        """Stop: release resources and remove the route. Keeps the deployment id.

        In-flight requests are not guaranteed to be cut off at once.
        """
        path = f"/deployments/{_seg(deployment_id)}/stop"
        return Deployment.from_wire(_obj(self._api.call("POST", path, json_body={}), path))

    def start(self, deployment_id: str) -> Deployment:
        """Start a stopped deployment again, under the same id."""
        path = f"/deployments/{_seg(deployment_id)}/start"
        return Deployment.from_wire(_obj(self._api.call("POST", path, json_body={}), path))

    def delete(self, deployment_id: str) -> Deployment:
        """Delete the deployment record."""
        path = f"/deployments/{_seg(deployment_id)}"
        return Deployment.from_wire(_obj(self._api.call("DELETE", path), path))

    def wait(
        self,
        deployment_id: str,
        *,
        until: str = "ready",
        timeout: float | None = 1800.0,
        poll_interval: float = 10.0,
    ) -> Deployment:
        """Poll until the deployment reaches ``until`` (``ready`` or ``stopped``).

        Raises :class:`DeploymentError` on ``unknown`` (the server cannot prove
        either outcome) or when the deployment was switched the other way, and
        :class:`FutureTimeoutError` on timeout.
        """
        if until not in ("ready", "stopped"):
            raise ValueError("until must be 'ready' or 'stopped'")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            dep = self.get(deployment_id)
            if dep.state == until:
                return dep
            # Settled the other way *and* that is what is now desired: waiting
            # longer cannot help. Right after start()/stop() the observed state
            # still lags the desired one, so a lagging state alone is not final.
            wanted_desired = "active" if until == "ready" else "stopped"
            if dep.state == "unknown" or (
                dep.state in DEPLOYMENT_SETTLED_STATES and dep.desired_state not in (None, wanted_desired)
            ):
                raise DeploymentError(dep, until)
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise FutureTimeoutError(deployment_id, timeout or 0.0, dep.state)
                time.sleep(min(poll_interval, left))
            else:
                time.sleep(poll_interval)


# -- inference ----------------------------------------------------------------
class ResponseObject(dict):
    """A decoded JSON object, readable like the OpenAI SDK's response objects.

    ``resp.choices[0].message.content`` and ``resp["choices"][0]["message"]
    ["content"]`` both work; it is a plain ``dict`` underneath, so ``json.dumps``
    and ``==`` behave as expected. A missing key reads as ``None`` (as optional
    fields do in those SDKs, e.g. a final stream chunk's ``delta.content``).
    Keys that collide with ``dict`` methods (``items``, ``keys``, ``get``...)
    must be read with ``[]``.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name)


class _InferenceHTTP:
    """POSTs to the relay's public inference API at the origin root.

    Not the CLI entry: the platform service answers 410 for inference arriving
    on the CLI entry. Never retried and never redirected, because a lost
    response may follow a billable generation.
    """

    def __init__(self, api: _Platform) -> None:
        self._api = api

    def post(self, path: str, body: dict[str, Any], *, stream: bool, timeout: float) -> Any:
        if stream:
            body["stream"] = True
            # Headers are built here, not inside the generator, so a closed
            # client fails at the call rather than at the first iteration.
            return self._stream(path, body, timeout, self._headers("text/event-stream"))
        with self._client(timeout) as http:
            try:
                resp = http.post(path, json=body, headers=self._headers("application/json"))
            except httpx.TimeoutException as exc:
                raise APITimeoutError(f"POST {path} timed out", path=path) from exc
            except httpx.HTTPError as exc:
                raise APIConnectionError(f"POST {path} failed: {exc}", path=path) from exc
            if resp.status_code >= 300:
                raise _inference_error(resp.status_code, resp.content, path)
            try:
                return resp.json(object_hook=ResponseObject)
            except ValueError as exc:
                raise APIError("inference response is not JSON", path=path) from exc

    def _stream(
        self, path: str, body: dict[str, Any], timeout: float, headers: dict[str, str]
    ) -> Iterator[Any]:
        with self._client(timeout) as http:
            try:
                with http.stream("POST", path, json=body, headers=headers) as resp:
                    if resp.status_code >= 300:
                        raise _inference_error(resp.status_code, resp.read(), path)
                    yield from _iter_sse(resp.iter_lines())
            except httpx.TimeoutException as exc:
                raise APITimeoutError(f"POST {path} timed out", path=path) from exc
            except httpx.HTTPError as exc:
                raise APIConnectionError(f"POST {path} failed: {exc}", path=path) from exc

    def _headers(self, accept: str) -> dict[str, str]:
        if self._api._owner is not None and getattr(self._api._owner, "_closed", False):
            raise ConfigurationError("this UserClient has been closed")
        return {
            "X-API-Key": self._api._config.api_key,
            "Accept": accept,
            "User-Agent": "mind-recursive-python",
        }

    def _client(self, timeout: float) -> httpx.Client:
        cfg = self._api._config
        return httpx.Client(
            base_url=cfg.origin,
            timeout=httpx.Timeout(timeout, connect=cfg.connect_timeout),
            trust_env=cfg.trust_env,
            # A 307/308 would replay a billable POST.
            follow_redirects=False,
        )


class ChatCompletions:
    """``uc.chat.completions``: the OpenAI chat format."""

    def __init__(self, http: _InferenceHTTP) -> None:
        self._http = http

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        stream: bool = False,
        timeout: float = 600.0,
        **params: Any,
    ) -> Any:
        """``POST {origin}/v1/chat/completions``, called like OpenAI's.

        Args:
            model: the deployment's :attr:`Deployment.public_model_id`.
            messages: OpenAI chat messages.
            stream: return an iterator of chunks instead of one response.
            timeout: seconds for the whole request.
            **params: anything else OpenAI's ``create`` takes (``max_tokens``,
                ``temperature``, ``tools``...), sent in the body as given.

        Returns:
            A :class:`ResponseObject`, e.g. ``resp.choices[0].message.content``;
            with ``stream=True``, an iterator of chunk objects, e.g.
            ``chunk.choices[0].delta.content``.
        """
        body = {"model": model, "messages": [dict(m) for m in messages], **params}
        return self._http.post("/v1/chat/completions", body, stream=stream, timeout=timeout)


class Chat:
    """``uc.chat``: holds :attr:`completions`, as in the OpenAI SDK."""

    def __init__(self, http: _InferenceHTTP) -> None:
        self.completions = ChatCompletions(http)


def _iter_sse(lines: Iterator[str]) -> Iterator[Any]:
    """Decode ``data:`` lines of an SSE stream into JSON objects.

    Multi-line ``data`` fields are joined; ``[DONE]`` ends the stream; comment
    and ``event:`` lines are skipped (the event type is also inside the JSON
    in the OpenAI format).
    """
    buf: list[str] = []
    for line in lines:
        if line == "":
            if buf:
                data = "\n".join(buf)
                buf = []
                if data.strip() == "[DONE]":
                    return
                yield json.loads(data, object_hook=ResponseObject)
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" "))
    if buf:
        data = "\n".join(buf)
        if data.strip() != "[DONE]":
            yield json.loads(data, object_hook=ResponseObject)


def _inference_error(status: int, raw: bytes, path: str) -> APIError:
    """OpenAI-style ``{"error": {"message", "type", "code"}}`` or relay ``{"code", "message"}``."""
    message, code = "", None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or "")
            code = err.get("code") or err.get("type")
        elif isinstance(err, str):
            message = err
        if not message and isinstance(payload.get("message"), str):
            message = payload["message"]
            code = payload.get("code")
    if not message:
        message = raw[:200].decode("utf-8", "replace").strip() or f"HTTP {status}"
    return status_to_error(status)(
        message, status=status, path=path, body=raw, code=code,
        request_id=_str(payload.get("request_id")) if isinstance(payload, dict) else None,
    )
