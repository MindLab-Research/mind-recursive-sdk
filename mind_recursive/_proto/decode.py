"""Decode protobuf responses: ``ForwardBackwardOutput`` and ``SampleResponse``.

The subtle part is ``BatchedTensor``: the server packs every datum's values
into one buffer plus an int64 ``offsets`` array. Those offsets are **byte**
offsets into the pre-widening stream, so when bfloat16 is widened to float32
or int32 to int64 the offsets still have to be divided by the *original*
itemsize. Getting this wrong silently misaligns per-datum logprobs.

Unknown fields are skipped rather than rejected, so a server that adds a field
does not break older SDKs.
"""

from __future__ import annotations

import struct

import numpy as np

from ..types import (
    ForwardBackwardOutput,
    PromptAltTokens,
    SampledSequence,
    SampleResponse,
    TensorData,
    TopkLogprobs,
)
from . import schema as S
from .codec import WIRE_BYTES, decode_varint, iter_fields

__all__ = [
    "decode_forward_backward_output",
    "decode_sample_response",
    "decode_tensor",
    "decode_topk_logprobs",
]


def _unpack_varints(buf: bytes) -> list[int]:
    """Read a packed repeated varint field."""
    out: list[int] = []
    pos = 0
    while pos < len(buf):
        value, pos = decode_varint(buf, pos)
        out.append(value)
    return out


def _map_entry(buf: bytes) -> tuple[str, bytes | float | int | None]:
    """Decode one map entry into ``(key, raw_value)``."""
    key = ""
    value: bytes | float | int | None = None
    for number, wire, raw in iter_fields(buf):
        if number == S.MapEntry.KEY and wire == WIRE_BYTES:
            key = bytes(raw).decode("utf-8")
        elif number == S.MapEntry.VALUE:
            value = struct.unpack("<d", raw)[0] if wire == 1 else raw
    return key, value


def decode_tensor(buf: bytes) -> TensorData:
    """Decode a proto ``Tensor`` into a ``TensorData``, preserving sparsity."""
    dense: bytes | None = None
    sparse: bytes | None = None
    dtype_enum = S.DTYPE_FLOAT32
    shape: list[int] = []

    for number, wire, raw in iter_fields(buf):
        if number == S.Tensor.DENSE and wire == WIRE_BYTES:
            dense = bytes(raw)
        elif number == S.Tensor.SPARSE_CSR and wire == WIRE_BYTES:
            sparse = bytes(raw)
        elif number == S.Tensor.DTYPE:
            dtype_enum = int(raw)
        elif number == S.Tensor.SHAPE:
            # Packed when emitted by the generated encoder, but a conformant
            # writer may also emit one varint per element.
            shape.extend(_unpack_varints(bytes(raw)) if wire == WIRE_BYTES else [int(raw)])

    np_name = S.DTYPE_TO_NUMPY.get(dtype_enum)
    public = S.DTYPE_TO_PUBLIC.get(dtype_enum)
    if np_name is None or public is None:
        raise ValueError(f"unsupported proto dtype: {dtype_enum}")

    if sparse is not None:
        values = b""
        crow = b""
        col = b""
        for number, wire, raw in iter_fields(sparse):
            if number == S.SparseCsr.VALUES:
                values = bytes(raw)
            elif number == S.SparseCsr.CROW_INDICES:
                crow = bytes(raw)
            elif number == S.SparseCsr.COL_INDICES:
                col = bytes(raw)
        return TensorData(
            data=np.frombuffer(values, dtype=np_name).copy(),
            dtype=public,  # type: ignore[arg-type]
            shape=shape or None,
            sparse_crow_indices=np.frombuffer(crow, dtype=np.int64).tolist(),
            sparse_col_indices=np.frombuffer(col, dtype=np.int64).tolist(),
        )

    arr = np.frombuffer(dense or b"", dtype=np_name).copy()
    arr = _widen(arr, dtype_enum)
    return TensorData(data=arr, dtype=public, shape=shape or None)  # type: ignore[arg-type]


def _widen(arr: np.ndarray, dtype_enum: int) -> np.ndarray:
    """Widen a wire dtype to its public counterpart, preserving element count.

    bfloat16 is the upper 16 bits of a float32, so shifting left by 16 and
    viewing as float32 is bit-exact for finite values.
    """
    if dtype_enum == S.DTYPE_BFLOAT16:
        return (arr.astype(np.uint32) << 16).view(np.float32)
    if dtype_enum == S.DTYPE_INT32:
        return arr.astype(np.int64)
    return arr


def _decode_batched_tensor(buf: bytes) -> tuple[list[np.ndarray], str, list[int]]:
    """Slice a ``BatchedTensor`` into per-datum flat arrays.

    Returns ``(per_datum_arrays, public_dtype, trailing_shape)``.
    """
    data = b""
    offsets_raw = b""
    dtype_enum = S.DTYPE_FLOAT32
    trailing: list[int] = []

    for number, wire, raw in iter_fields(buf):
        if number == S.BatchedTensor.DATA:
            data = bytes(raw)
        elif number == S.BatchedTensor.OFFSETS:
            offsets_raw = bytes(raw)
        elif number == S.BatchedTensor.DTYPE:
            dtype_enum = int(raw)
        elif number == S.BatchedTensor.TRAILING_SHAPE:
            trailing.extend(
                _unpack_varints(bytes(raw)) if wire == WIRE_BYTES else [int(raw)]
            )

    np_name = S.DTYPE_TO_NUMPY.get(dtype_enum)
    public = S.DTYPE_TO_PUBLIC.get(dtype_enum)
    if np_name is None or public is None:
        raise ValueError(f"unsupported proto DType in BatchedTensor: {dtype_enum}")

    offsets = np.frombuffer(offsets_raw, dtype=np.int64)
    buf_arr = np.frombuffer(data, dtype=np_name)
    # Capture the itemsize *before* widening: offsets index the original stream.
    src_itemsize = np.dtype(np_name).itemsize
    buf_arr = _widen(buf_arr, dtype_enum)

    per_datum: list[np.ndarray] = []
    for i in range(max(len(offsets) - 1, 0)):
        start = int(offsets[i]) // src_itemsize
        end = int(offsets[i + 1]) // src_itemsize
        per_datum.append(buf_arr[start:end])
    return per_datum, public, trailing


def decode_forward_backward_output(buf: bytes) -> ForwardBackwardOutput:
    """Decode a ``ForwardBackwardOutput``.

    Handles any number of ``ArrayRecord`` chunks. Field names are invariant
    across chunks, but dtype and trailing shape are per-chunk: ragged outputs
    (variable-K loss fields) arrive as several internally-rectangular chunks.
    """
    output_type = ""
    records: list[bytes] = []
    metrics: dict[str, float] = {}

    for number, _wire, raw in iter_fields(buf):
        if number == S.ForwardBackwardOutput.LOSS_FN_OUTPUT_TYPE:
            output_type = bytes(raw).decode("utf-8")
        elif number == S.ForwardBackwardOutput.LOSS_FN_OUTPUTS:
            records.append(bytes(raw))
        elif number == S.ForwardBackwardOutput.METRICS:
            key, value = _map_entry(bytes(raw))
            if key:
                metrics[key] = float(value) if isinstance(value, (int, float)) else 0.0

    # Bare class name, matching the JSON response shape.
    class_name = output_type.rsplit(":", 1)[-1] if output_type else "ArrayRecord"
    if not records:
        return ForwardBackwardOutput(
            loss_fn_output_type=class_name, loss_fn_outputs=[], metrics=metrics
        )

    per_field_arrays: dict[str, list[np.ndarray]] = {}
    per_field_meta: dict[str, list[tuple[str, list[int]]]] = {}
    num_datums = 0

    for record in records:
        for number, wire, raw in iter_fields(record):
            if number == S.ArrayRecord.FIELDS:
                name, payload = _map_entry(bytes(raw))
                if not name or not isinstance(payload, (bytes, bytearray)):
                    continue
                arrays, dtype, trailing = _decode_batched_tensor(bytes(payload))
                per_field_arrays.setdefault(name, []).extend(arrays)
                per_field_meta.setdefault(name, []).extend(
                    [(dtype, trailing)] * len(arrays)
                )
            elif number == S.ArrayRecord.NUM_DATUMS:
                num_datums += int(raw)

    # num_datums is authoritative: it survives server-side filters that strip
    # every field (e.g. drop_fwdbwd_logprobs). Fall back to the first field's
    # count for messages predating it.
    if num_datums == 0 and per_field_arrays:
        num_datums = len(next(iter(per_field_arrays.values())))

    outputs: list[dict[str, TensorData]] = []
    for i in range(num_datums):
        datum: dict[str, TensorData] = {}
        for name, arrays in per_field_arrays.items():
            if i >= len(arrays):
                continue
            dtype, trailing = per_field_meta[name][i]
            arr = arrays[i]
            trailing_elems = 1
            for dim in trailing:
                trailing_elems *= dim
            leading = arr.size // trailing_elems if trailing_elems else 0
            datum[name] = TensorData(
                data=arr, dtype=dtype, shape=[leading, *trailing]  # type: ignore[arg-type]
            )
        outputs.append(datum)

    return ForwardBackwardOutput(
        loss_fn_output_type=class_name, loss_fn_outputs=outputs, metrics=metrics
    )


def decode_topk_logprobs(buf: bytes) -> TopkLogprobs:
    """Decode a ``TopkLogprobs`` block.

    ``token_ids`` is int32 and ``logprobs`` float32, both row-major over
    ``length`` positions by ``k`` ranks.
    """
    token_ids: list[int] = []
    logprobs: list[float] = []
    k = 0
    length = 0

    for number, _wire, raw in iter_fields(buf):
        if number == S.TopkLogprobs.TOKEN_IDS:
            token_ids = np.frombuffer(bytes(raw), dtype=np.int32).tolist()
        elif number == S.TopkLogprobs.LOGPROBS:
            logprobs = np.frombuffer(bytes(raw), dtype=np.float32).tolist()
        elif number == S.TopkLogprobs.K:
            k = int(raw)
        elif number == S.TopkLogprobs.LENGTH:
            length = int(raw)

    # length is derivable, but only the server knows it when k is 0.
    if length == 0 and k:
        length = len(token_ids) // k
    return TopkLogprobs(token_ids=token_ids, logprobs=logprobs, k=k, length=length)


def _decode_prompt_alt_tokens(buf: bytes) -> PromptAltTokens:
    tokens: TensorData | None = None
    logprobs: TensorData | None = None
    for number, _wire, raw in iter_fields(buf):
        if number == S.PromptAltTokens.TOKENS:
            tokens = decode_tensor(bytes(raw))
        elif number == S.PromptAltTokens.LOGPROBS:
            logprobs = decode_tensor(bytes(raw))
    return PromptAltTokens(tokens=tokens, logprobs=logprobs)


def decode_sample_response(buf: bytes) -> SampleResponse:
    """Decode a ``SampleResponse``."""
    sequences: list[SampledSequence] = []
    prompt_logprobs: list[float] | None = None
    cache_hits = 0
    topk_prompt: TopkLogprobs | None = None
    target_prompt: TensorData | None = None
    alt_tokens: PromptAltTokens | None = None

    for number, _wire, raw in iter_fields(buf):
        if number == S.SampleResponse.SEQUENCES:
            sequences.append(_decode_sampled_sequence(bytes(raw)))
        elif number == S.SampleResponse.PROMPT_LOGPROBS:
            prompt_logprobs = np.frombuffer(bytes(raw), dtype=np.float32).tolist()
        elif number == S.SampleResponse.PROMPT_CACHE_HIT_TOKENS:
            cache_hits = int(raw)
        elif number == S.SampleResponse.TOPK_PROMPT_LOGPROBS:
            topk_prompt = decode_topk_logprobs(bytes(raw))
        elif number == S.SampleResponse.TARGET_PROMPT_LOGPROBS:
            target_prompt = decode_tensor(bytes(raw))
        elif number == S.SampleResponse.PROMPT_ALT_TOKENS:
            alt_tokens = _decode_prompt_alt_tokens(bytes(raw))

    return SampleResponse(
        sequences=sequences,
        prompt_logprobs=prompt_logprobs,
        prompt_cache_hit_tokens=cache_hits,
        topk_prompt_logprobs=topk_prompt,
        target_prompt_logprobs=target_prompt,
        prompt_alt_tokens=alt_tokens,
    )


def _decode_sampled_sequence(buf: bytes) -> SampledSequence:
    stop_reason = "stop"
    tokens: list[int] = []
    logprobs: list[float] | None = None
    topk: TopkLogprobs | None = None

    for number, _wire, raw in iter_fields(buf):
        if number == S.SampledSequence.STOP_REASON:
            mapped = S.STOP_REASON_TO_STR.get(int(raw))
            if mapped is None:
                raise ValueError(f"unknown stop_reason enum value: {int(raw)}")
            stop_reason = mapped
        elif number == S.SampledSequence.TOKENS:
            tokens = np.frombuffer(bytes(raw), dtype=np.int32).tolist()
        elif number == S.SampledSequence.LOGPROBS:
            data = bytes(raw)
            logprobs = np.frombuffer(data, dtype=np.float32).tolist() if data else None
        elif number == S.SampledSequence.TOPK_SAMPLED_LOGPROBS:
            topk = decode_topk_logprobs(bytes(raw))

    return SampledSequence(
        tokens=tokens,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        logprobs=logprobs,
        topk_logprobs=topk,
    )
