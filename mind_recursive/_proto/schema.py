"""Field numbers and enum values for the public Tinker protobuf schema.

Transcribed from ``tinker/proto/tinker_public_pb2`` (tinker 0.39.0) by
reflecting over the generated descriptors. This is the one place in the SDK
that mirrors an upstream schema, and it is deliberately data-only: the schema
is part of the *on-wire protocol*, which we must match regardless of whether
we depend on upstream client code.

``tests/test_proto_golden.py`` re-derives these from the generated module when
it is importable, so drift fails the test rather than corrupting a training run.
"""

from __future__ import annotations

__all__ = [
    "DTYPE_UNSPECIFIED",
    "DTYPE_FLOAT32",
    "DTYPE_INT64",
    "DTYPE_INT32",
    "DTYPE_BFLOAT16",
    "STOP_REASON_STOP",
    "STOP_REASON_LENGTH",
    "DTYPE_TO_NUMPY",
    "DTYPE_TO_PUBLIC",
    "PUBLIC_TO_DTYPE",
    "STOP_REASON_TO_STR",
]

# -- enum DType ---------------------------------------------------------------
DTYPE_UNSPECIFIED = 0
DTYPE_FLOAT32 = 1
DTYPE_INT64 = 2
DTYPE_INT32 = 3
DTYPE_BFLOAT16 = 4

# -- enum StopReason ----------------------------------------------------------
STOP_REASON_STOP = 0
STOP_REASON_LENGTH = 1

STOP_REASON_TO_STR: dict[int, str] = {
    STOP_REASON_STOP: "stop",
    STOP_REASON_LENGTH: "length",
}

#: Proto dtype -> numpy dtype string for ``frombuffer``. bfloat16 arrives as
#: uint16 bytes and is widened to float32 by the decoder.
DTYPE_TO_NUMPY: dict[int, str] = {
    DTYPE_FLOAT32: "float32",
    DTYPE_INT64: "int64",
    DTYPE_INT32: "int32",
    DTYPE_BFLOAT16: "uint16",
}

#: Proto dtype -> public TensorDtype. Integer widths collapse to int64 and
#: bfloat16 collapses to float32, matching the JSON response shape.
DTYPE_TO_PUBLIC: dict[int, str] = {
    DTYPE_FLOAT32: "float32",
    DTYPE_BFLOAT16: "float32",
    DTYPE_INT64: "int64",
    DTYPE_INT32: "int64",
}

#: Public TensorDtype -> proto dtype. Only these two are exposed to users.
PUBLIC_TO_DTYPE: dict[str, int] = {
    "float32": DTYPE_FLOAT32,
    "int64": DTYPE_INT64,
}


class Tensor:
    DENSE = 1          # bytes
    SPARSE_CSR = 2     # message SparseCsr
    DTYPE = 3          # enum DType
    SHAPE = 4          # repeated int64


class SparseCsr:
    VALUES = 1         # bytes
    CROW_INDICES = 2   # bytes
    COL_INDICES = 3    # bytes


class EncodedTextChunk:
    TOKENS = 1         # bytes (int32-packed)


class ImageChunk:
    DATA = 1           # bytes
    FORMAT = 2         # string
    EXPECTED_TOKENS = 3  # int32
    OVERRIDE_DIMENSIONS = 4  # message ImageDimensions


class ImageDimensions:
    WIDTH = 1          # uint32
    HEIGHT = 2         # uint32


class DmelChunk:
    DMEL = 1           # bytes


class Chunk:
    """oneof: exactly one of these is set per chunk."""

    ENCODED_TEXT = 1   # message EncodedTextChunk
    IMAGE = 2          # message ImageChunk
    DMEL = 3           # message DmelChunk


class Datum:
    MODEL_INPUT = 1          # repeated Chunk
    LOSS_FN_INPUTS = 2       # map<string, Tensor>
    MODEL_INPUT_SPANS = 3    # repeated ProvenanceSpan
    LOSS_FN_INPUT_SPANS = 4  # repeated ProvenanceSpan


class ForwardBackwardRequest:
    MODEL_ID = 1             # string
    SEQ_ID = 2               # int32
    DATA = 3                 # repeated Datum
    LOSS_FN = 4              # string
    LOSS_FN_CONFIG = 5       # map<string, double>      (legacy)
    FORWARD_ONLY = 6         # bool
    LOSS_FN_CONFIG_V2 = 7    # map<string, LossConfigValue>


class LossConfigValue:
    """oneof: number or text."""

    NUMBER = 1         # double
    TEXT = 2           # string


class BatchedTensor:
    DATA = 1           # bytes
    OFFSETS = 2        # bytes (int64-packed byte offsets)
    DTYPE = 3          # enum DType
    TRAILING_SHAPE = 4  # repeated int64


class ArrayRecord:
    TYPE_TAG = 1       # string
    FIELDS = 2         # map<string, BatchedTensor>
    NUM_DATUMS = 3     # int64


class ForwardBackwardOutput:
    LOSS_FN_OUTPUT_TYPE = 1  # string
    LOSS_FN_OUTPUTS = 2      # repeated ArrayRecord
    METRICS = 3              # map<string, double>


class SampledSequence:
    STOP_REASON = 1          # enum StopReason
    TOKENS = 2               # bytes (int32-packed)
    LOGPROBS = 3             # bytes (float32-packed)
    TOPK_SAMPLED_LOGPROBS = 4  # message TopkLogprobs


class SampleResponse:
    SEQUENCES = 1                # repeated SampledSequence
    PROMPT_LOGPROBS = 2          # bytes (float32-packed)
    TOPK_PROMPT_LOGPROBS = 3     # message TopkLogprobs
    PROMPT_CACHE_HIT_TOKENS = 4  # int32
    TARGET_PROMPT_LOGPROBS = 5   # message Tensor
    PROMPT_ALT_TOKENS = 6        # message PromptAltTokens


class TopkLogprobs:
    TOKEN_IDS = 1      # bytes (int32-packed, n*k)
    LOGPROBS = 2       # bytes (float32-packed, n*k)
    K = 3              # uint32
    LENGTH = 4         # uint32


class PromptAltTokens:
    TOKENS = 1         # message Tensor (int64, [len(prompt)-1, k])
    LOGPROBS = 2       # message Tensor (float32, same shape)


class MapEntry:
    """Every protobuf map entry is a sub-message with these two fields."""

    KEY = 1
    VALUE = 2
