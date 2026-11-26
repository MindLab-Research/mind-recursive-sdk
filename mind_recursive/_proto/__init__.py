"""Hand-written protobuf codec for the two binary messages on the wire.

Only ``forward_backward`` (request + its future response) and ``sample``
responses are protobuf; everything else is JSON. See ``codec`` for the wire
primitives, ``schema`` for field numbers, and ``encode``/``decode`` for the
message mappings.
"""

from .decode import (
    decode_forward_backward_output,
    decode_sample_response,
    decode_tensor,
)
from .encode import encode_datum, encode_forward_backward_request, encode_tensor

__all__ = [
    "encode_forward_backward_request",
    "encode_datum",
    "encode_tensor",
    "decode_forward_backward_output",
    "decode_sample_response",
    "decode_tensor",
]
