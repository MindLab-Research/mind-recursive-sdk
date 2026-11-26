"""Encode ``ForwardBackwardRequest`` to protobuf bytes.

This is the training hot path: it runs once per step, with the whole batch in
the body. Correctness here is non-negotiable -- a mis-encoded tensor trains on
garbage without raising -- so the encoding of every field is spelled out
explicitly against ``schema`` rather than inferred.

Byte-level conventions copied from the upstream encoder
(``tinker/proto/request_conv.py``), because the server's decoder assumes them:

* ``EncodedTextChunk.tokens``: **int32** little-endian, not int64.
* ``Tensor.dense``: contiguous bytes in the tensor's declared dtype.
* ``SparseCsr`` indices: int64 regardless of the value dtype.
* ``loss_fn_config``: numbers are dual-written to the legacy map (field 5) and
  the v2 map (field 7); strings go only to v2. Old servers read field 5, new
  servers prefer field 7.
"""

from __future__ import annotations

import numpy as np

from ..types import (
    Datum,
    DmelChunk,
    EncodedTextChunk,
    ImageChunk,
    ModelInputChunk,
    TensorData,
)
from . import schema as S
from .codec import (
    field_bytes,
    field_double,
    field_message,
    field_string,
    field_varint,
    packed_varints,
)

__all__ = ["encode_forward_backward_request", "encode_tensor", "encode_datum"]


def encode_tensor(td: TensorData) -> bytes:
    """Serialise a ``TensorData`` as a proto ``Tensor``."""
    proto_dtype = S.PUBLIC_TO_DTYPE.get(td.dtype)
    if proto_dtype is None:
        raise ValueError(f"unsupported TensorData dtype for the wire: {td.dtype!r}")

    out = bytearray()
    if td.is_sparse:
        # Sparse CSR: values keep the declared dtype, indices are always int64.
        csr = bytearray()
        csr += field_bytes(S.SparseCsr.VALUES, np.ascontiguousarray(td.data).tobytes())
        csr += field_bytes(
            S.SparseCsr.CROW_INDICES,
            np.asarray(td.sparse_crow_indices, dtype=np.int64).tobytes(),
        )
        csr += field_bytes(
            S.SparseCsr.COL_INDICES,
            np.asarray(td.sparse_col_indices, dtype=np.int64).tobytes(),
        )
        out += field_message(S.Tensor.SPARSE_CSR, bytes(csr))
    else:
        out += field_bytes(S.Tensor.DENSE, np.ascontiguousarray(td.data).tobytes())

    out += field_varint(S.Tensor.DTYPE, proto_dtype)
    if td.shape:
        out += packed_varints(S.Tensor.SHAPE, [int(s) for s in td.shape])
    return bytes(out)


def _map_entry(number: int, key: str, value_payload: bytes) -> bytes:
    """One ``map<string, Message>`` entry."""
    entry = field_string(S.MapEntry.KEY, key) + field_message(
        S.MapEntry.VALUE, value_payload
    )
    return field_message(number, entry)


def _map_entry_double(number: int, key: str, value: float) -> bytes:
    """One ``map<string, double>`` entry."""
    entry = field_string(S.MapEntry.KEY, key) + field_double(S.MapEntry.VALUE, value)
    return field_message(number, entry)


def encode_chunk(chunk: ModelInputChunk) -> bytes:
    """The ``Chunk`` message body: its single oneof arm, with no outer tag.

    The caller wraps this in ``Datum.model_input``, so tagging happens at
    exactly one site.
    """
    if isinstance(chunk, EncodedTextChunk):
        tokens = np.asarray(chunk.tokens, dtype=np.int32).tobytes()
        return field_message(
            S.Chunk.ENCODED_TEXT, field_bytes(S.EncodedTextChunk.TOKENS, tokens)
        )
    if isinstance(chunk, ImageChunk):
        inner = bytearray()
        inner += field_bytes(S.ImageChunk.DATA, chunk.data)
        inner += field_string(S.ImageChunk.FORMAT, chunk.format)
        if chunk.expected_tokens is not None:
            inner += field_varint(S.ImageChunk.EXPECTED_TOKENS, int(chunk.expected_tokens))
        return field_message(S.Chunk.IMAGE, bytes(inner))
    if isinstance(chunk, DmelChunk):
        return field_message(S.Chunk.DMEL, field_bytes(S.DmelChunk.DMEL, chunk.dmel))
    raise ValueError(f"unsupported model input chunk type: {type(chunk).__name__}")


def encode_datum(datum: Datum) -> bytes:
    """Serialise one ``Datum``."""
    out = bytearray()
    for chunk in datum.model_input.chunks:
        out += field_message(S.Datum.MODEL_INPUT, encode_chunk(chunk))
    for name, td in datum.loss_fn_inputs.items():
        out += _map_entry(S.Datum.LOSS_FN_INPUTS, name, encode_tensor(td))
    return bytes(out)


def encode_forward_backward_request(
    *,
    model_id: str,
    seq_id: int,
    data: list[Datum],
    loss_fn: str,
    loss_fn_config: dict[str, float | str] | None = None,
    forward_only: bool = False,
) -> bytes:
    """Serialise a complete ``ForwardBackwardRequest``.

    Args:
        model_id: the training model this step applies to.
        seq_id: monotonic step sequence within the session. Upstream always
            sets it (>= 1); a 0 trips the server's range check.
        data: the batch.
        loss_fn: one of the server's loss function names.
        loss_fn_config: optional loss parameters. Numbers are dual-written to
            the legacy and v2 maps; strings exist only on v2.
        forward_only: run the forward pass without accumulating gradients.
    """
    out = bytearray()
    out += field_string(S.ForwardBackwardRequest.MODEL_ID, model_id)
    out += field_varint(S.ForwardBackwardRequest.SEQ_ID, int(seq_id))
    for datum in data:
        out += field_message(S.ForwardBackwardRequest.DATA, encode_datum(datum))
    out += field_string(S.ForwardBackwardRequest.LOSS_FN, loss_fn)

    if loss_fn_config:
        for key, value in loss_fn_config.items():
            if isinstance(value, str):
                v2 = field_string(S.LossConfigValue.TEXT, value)
                out += _map_entry(S.ForwardBackwardRequest.LOSS_FN_CONFIG_V2, key, v2)
            else:
                number = float(value)
                out += _map_entry_double(
                    S.ForwardBackwardRequest.LOSS_FN_CONFIG, key, number
                )
                v2 = field_double(S.LossConfigValue.NUMBER, number)
                out += _map_entry(S.ForwardBackwardRequest.LOSS_FN_CONFIG_V2, key, v2)

    if forward_only:
        out += field_varint(S.ForwardBackwardRequest.FORWARD_ONLY, 1)
    return bytes(out)
