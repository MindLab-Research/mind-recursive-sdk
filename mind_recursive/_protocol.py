"""Wire constants and payload parsers.

Paths, header names and the small parse functions that turn JSON envelopes into
our types. Kept in one module so the whole surface the server sees is auditable
in a single place.

Note the plane split: training *and sampling* paths are relative to
``{origin}/train`` -- the relay's ``/train/**`` handler is a general
pass-through and the training service implements sampling itself. Platform paths are
relative to ``{origin}/api/mint/cli/v1``, which the relay remounts under the
platform service's ``/api/v1``. ``_config`` derives both, so nothing here needs
to know the origin.
"""

from __future__ import annotations

from typing import Any

from ._errors import ConfigurationError
from .types import ForwardBackwardOutput, SampleResponse

__all__ = [
    "PATH_CAPABILITIES",
    "PATH_HEALTHZ",
    "PATH_CLIENT_CONFIG",
    "PATH_CLIENT_DYNAMIC_CONFIG",
    "PATH_CREATE_SESSION",
    "PATH_HEARTBEAT",
    "PATH_FINISH_SESSION",
    "PATH_CREATE_MODEL",
    "PATH_CREATE_MODEL_FROM_STATE",
    "PATH_FORWARD_BACKWARD",
    "PATH_OPTIM_STEP",
    "PATH_GET_INFO",
    "PATH_SAVE_WEIGHTS",
    "PATH_SAVE_WEIGHTS_FOR_SAMPLER",
    "PATH_LOAD_WEIGHTS",
    "PATH_RETRIEVE_FUTURE",
    "PATH_SAMPLE_RETRIEVE_FUTURE",
    "PATH_SERVICE_RETRIEVE_FUTURE",
    "PATH_SERVICE_CREATE_SAMPLING_SESSION",
    "PATH_ASAMPLE",
    "PATH_TELEMETRY",
    "PATH_REPORT_EVENTS",
    "PATH_REPORT_METRIC",
    "PATH_REPORT_ARTIFACT",
    "PATH_REPORT_SUMMARY",
    "PROTO_CONTENT_TYPE",
    "ACCEPT_PROTO_OR_JSON",
    "parse_model_id",
    "parse_weights_path",
    "parse_sampling_session_id",
    "parse_optim_step",
    "parse_load_weights",
    "parse_forward_backward_output",
    "parse_sample_response",
]

# Every path is relative to one of five bases (three destination platforms:
# /train -> the training service, /sample -> the inference service, /service -> platform service,
# which also owns /service/session and /service/report). The base already contains the
# fixed `/api/mindrsi/v1` root and the segment, so a path never repeats
# `/api/v1`, `/api/mint/cli/v1` or `/v1`.

# -- session segment (relative to {origin}/api/mindrsi/v1/service/session) ----------------
#: Session lifecycle is client-level, not training-level: the container is
#: opened before a trainer or sampler exists and both live inside it.
PATH_CREATE_SESSION = "/create"
PATH_HEARTBEAT = "/{session_id}/heartbeat"
PATH_FINISH_SESSION = "/{session_id}/finish"

# -- service segment (relative to {origin}/api/mindrsi/v1/service) ----------------
#: Connectivity and negotiation. Asking what the backend serves, or how a
#: client must behave, is not a training call and needs no session.
PATH_CAPABILITIES = "/get_server_capabilities"
PATH_HEALTHZ = "/healthz"
PATH_CLIENT_CONFIG = "/client/config"
PATH_CLIENT_DYNAMIC_CONFIG = "/client/dynamic_config"
PATH_TELEMETRY = "/telemetry"

# -- train plane (relative to {origin}/api/mindrsi/v1/train) ----------------------
#: ``create_model`` / ``create_model_from_state`` are sent on the
#: ``/service/train`` base (the platform service owns trainer lifecycle); the
#: leaf names are unchanged.
PATH_CREATE_MODEL = "/create_model"
#: Create the resource and load a saved training state in one submission. The
#: server resolves ``state_path`` to a ready training-state artifact; this is
#: the atomic counterpart of ``create_model`` + ``load_weights``.
PATH_CREATE_MODEL_FROM_STATE = "/create_model_from_state"
PATH_FORWARD_BACKWARD = "/forward_backward"
PATH_OPTIM_STEP = "/optim_step"
PATH_GET_INFO = "/get_info"
PATH_SAVE_WEIGHTS = "/save_weights"
PATH_SAVE_WEIGHTS_FOR_SAMPLER = "/save_weights_for_sampler"
PATH_LOAD_WEIGHTS = "/load_weights"
PATH_RETRIEVE_FUTURE = "/retrieve_future"

# -- sample plane (relative to {origin}/api/mindrsi/v1/sample) --------------------
PATH_ASAMPLE = "/asample"
#: Create a sampler. Every sampler is created on the service base: the
#: platform service opens it under the inference-side session it manages for the
#: Service Session (deploying trained weights first), so this returns a
#: ``request_id`` and resolves via the service plane's ``retrieve_future``.
PATH_SERVICE_CREATE_SAMPLING_SESSION = "/create_sampling_session"
#: Sampling futures are polled on the sample base. The wire path is the same as
#: the training one; the base is what tells the relay which plane produced it.
PATH_SAMPLE_RETRIEVE_FUTURE = "/retrieve_future"
#: A service-plane future (temporary-deployment sampler create) is polled on
#: the service base: ``{origin}/api/mindrsi/v1/service/retrieve_future``.
PATH_SERVICE_RETRIEVE_FUTURE = "/retrieve_future"

# -- report plane (relative to {origin}/api/mindrsi/v1/service/report) --------------------
#: The reporting service's write routes, re-mounted at the report
#: base and keyed by the SDK's own ``session_id``. The SDK has no "execution"
#: concept and never sends an ``execution_id``; the relay maps the session to
#: whatever record it keeps. They are the contract the relay implements; the
#: production relay does not serve them yet, so a report is best-effort and a
#: 404 is a warning.
#:
#: ``log`` and ``event`` share ``/events`` because a log line *is* an event
#: upstream; metrics go to ``/log``, which is the listener's name for them.
PATH_REPORT_EVENTS = "/{session_id}/events"
PATH_REPORT_METRIC = "/{session_id}/log"
PATH_REPORT_ARTIFACT = "/{session_id}/artifact"
PATH_REPORT_SUMMARY = "/{session_id}/summary"

# -- headers ------------------------------------------------------------------
#: There is no custom-header constant here because the SDK sends none: every
#: request carries ``X-API-Key`` and ``User-Agent`` and nothing else.
#: Attribution rides in ``create_session``'s ``client_id`` field, because
#: "these sessions come from one process" is a property of the session, not of
#: each request. Business labels such as "algorithm" or "rollout vs eval" are
#: not request attributes either: billing attributes by model and by operation,
#: so a per-request label would invent a dimension nothing reads. Free-form
#: tags belong in ``user_metadata`` on the resource.

#: What a sampling session's weights are bound to. ``snapshot`` is fixed
#: weights; ``inflight`` tracks a live training model, so samples follow the
#: weights as training steps (on-policy RL). This is a real wire field, not an
#: attribution label: the training service consumes it.
SAMPLER_KINDS = ("snapshot", "inflight")

PROTO_CONTENT_TYPE = "application/x-protobuf"
#: forward_backward futures resolve to protobuf; everything else to JSON. We
#: advertise both and branch on the response's Content-Type.
ACCEPT_PROTO_OR_JSON = "application/x-protobuf, application/json"


def _require(payload: dict[str, Any], key: str, what: str) -> Any:
    value = payload.get(key)
    if value in (None, ""):
        raise ConfigurationError(f"{what} response is missing {key!r}: {payload!r}")
    return value


def parse_model_id(payload: dict[str, Any] | bytes) -> str:
    """Pull ``model_id`` out of a resolved ``create_model`` future."""
    if isinstance(payload, (bytes, bytearray)):
        raise ConfigurationError("create_model resolved to a binary body")
    return str(_require(payload, "model_id", "create_model"))


class CheckpointPath(str):
    """A save path that also answers to upstream's ``.path`` spelling.

    Tinker resolves save futures to a response object with a ``path`` field;
    this SDK resolves them to the path string itself. A ``str`` subclass keeps
    both call shapes working without changing what the value is.
    """

    @property
    def path(self) -> str:
        return str(self)


def parse_weights_path(payload: dict[str, Any] | bytes) -> str:
    """Pull the ``mint://`` path out of a resolved save-weights future."""
    if isinstance(payload, (bytes, bytearray)):
        raise ConfigurationError("save_weights resolved to a binary body")
    return CheckpointPath(_require(payload, "path", "save_weights"))


def parse_sampling_session_id(payload: dict[str, Any] | bytes) -> str:
    """Pull ``sampling_session_id`` out of a create_sampling_session result.

    Handles both shapes: the relay-direct create returns it on the call, and
    the service-plane create resolves a future to the same object.
    """
    if isinstance(payload, (bytes, bytearray)):
        raise ConfigurationError(
            "create_sampling_session resolved to a binary body"
        )
    return str(_require(payload, "sampling_session_id", "create_sampling_session"))


def parse_optim_step(payload: dict[str, Any] | bytes) -> dict[str, Any]:
    """``optim_step`` resolves to a small JSON object (may be empty)."""
    if isinstance(payload, (bytes, bytearray)):
        raise ConfigurationError("optim_step resolved to a binary body")
    return payload


def parse_load_weights(payload: dict[str, Any] | bytes) -> dict[str, Any]:
    """``load_weights`` resolves to a small JSON object.

    Separate from :func:`parse_optim_step` even though the shape is the same
    today: upstream models them as two types (``LoadWeightsResponse`` carries
    ``path`` and ``model_id``, ``OptimStepResponse`` carries ``metrics``), so
    sharing one parser names the wrong thing and would hide the day they
    diverge.
    """
    if isinstance(payload, (bytes, bytearray)):
        raise ConfigurationError("load_weights resolved to a binary body")
    return payload


def parse_forward_backward_output(
    payload: dict[str, Any] | bytes,
) -> ForwardBackwardOutput:
    """Decode a resolved ``forward_backward`` future.

    Normally protobuf. A JSON body is still accepted: the server may fall back
    to it, and refusing to read a well-formed result would be gratuitous.
    """
    from ._proto import decode_forward_backward_output

    if isinstance(payload, (bytes, bytearray)):
        return decode_forward_backward_output(bytes(payload))

    metrics = {
        key: float(value)
        for key, value in (payload.get("metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    return ForwardBackwardOutput(
        loss_fn_output_type=str(payload.get("loss_fn_output_type") or "ArrayRecord"),
        loss_fn_outputs=[],
        metrics=metrics,
    )


def parse_sample_response(payload: dict[str, Any] | bytes) -> SampleResponse:
    """Decode a resolved ``asample`` future (protobuf or JSON)."""
    from ._proto import decode_sample_response
    from .types import SampledSequence

    if isinstance(payload, (bytes, bytearray)):
        return decode_sample_response(bytes(payload))

    sequences = []
    for item in payload.get("sequences") or []:
        sequences.append(
            SampledSequence(
                tokens=[int(t) for t in item.get("tokens") or []],
                stop_reason=item.get("stop_reason") or "stop",
                logprobs=item.get("logprobs"),
                sequence_id=item.get("sequence_id"),
                topk_logprobs=_parse_topk(item.get("topk_sampled_logprobs")),
            )
        )
    return SampleResponse(
        sequences=sequences,
        prompt_logprobs=payload.get("prompt_logprobs"),
        prompt_cache_hit_tokens=int(payload.get("prompt_cache_hit_tokens") or 0),
        topk_prompt_logprobs=_parse_topk(payload.get("topk_prompt_logprobs")),
        target_prompt_logprobs=_parse_tensor(payload.get("target_prompt_logprobs")),
        prompt_alt_tokens=_parse_prompt_alt_tokens(payload.get("prompt_alt_tokens")),
    )


def _parse_topk(raw: Any) -> Any:
    """Build a ``TopkLogprobs`` from its JSON object, if present."""
    if not raw:
        return None
    from .types import TopkLogprobs

    k = int(raw.get("k") or 0)
    token_ids = [int(t) for t in raw.get("token_ids") or []]
    length = int(raw.get("length") or 0)
    if length == 0 and k:
        length = len(token_ids) // k
    return TopkLogprobs(
        token_ids=token_ids,
        logprobs=[float(x) for x in raw.get("logprobs") or []],
        k=k,
        length=length,
    )


def _parse_tensor(raw: Any) -> Any:
    """Build a ``TensorData`` from the JSON tensor shape, if present."""
    if not raw:
        return None
    import numpy as np

    from .types import TensorData

    dtype = raw.get("dtype") or "float32"
    return TensorData(
        data=np.asarray(raw.get("data") or []),
        dtype=dtype,
        shape=raw.get("shape"),
        sparse_crow_indices=raw.get("sparse_crow_indices"),
        sparse_col_indices=raw.get("sparse_col_indices"),
    )


def _parse_prompt_alt_tokens(raw: Any) -> Any:
    if not raw:
        return None
    from .types import PromptAltTokens

    return PromptAltTokens(
        tokens=_parse_tensor(raw.get("tokens")),
        logprobs=_parse_tensor(raw.get("logprobs")),
    )
