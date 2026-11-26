"""Minimal protobuf wire codec.

``forward_backward`` is protobuf-only upstream -- ``tinker/resources/training.py``
hardcodes ``Content-Type: application/x-protobuf`` with no JSON branch -- and its
future resolves to a protobuf body too. Everything else on the wire is JSON.

Rather than depend on ``protobuf`` plus generated modules, we encode and decode
these two messages by hand. The schema is small, frozen by the server, and
exercised on every training step, so a few hundred lines of explicit wire
handling is cheaper than a C-extension dependency.

Field numbers below are transcribed from ``tinker/proto/tinker_public_pb2``
(tinker 0.39.0) and pinned by ``tests/test_proto_golden.py``, which asserts our
bytes match the generated encoder when it is importable.

Wire types used here::

    0  varint          int32/int64/uint32/uint64/bool/enum
    1  fixed64         double
    2  length-delim    string/bytes/embedded message/packed repeated
    5  fixed32         float
"""

from __future__ import annotations

import struct
from typing import Iterator

__all__ = [
    "encode_varint",
    "decode_varint",
    "tag",
    "field_varint",
    "field_bytes",
    "field_string",
    "field_double",
    "field_message",
    "iter_fields",
    "WIRE_VARINT",
    "WIRE_FIXED64",
    "WIRE_BYTES",
    "WIRE_FIXED32",
]

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_BYTES = 2
WIRE_FIXED32 = 5


def encode_varint(value: int) -> bytes:
    """Base-128 varint. Negative values use two's complement over 64 bits."""
    if value < 0:
        value += 1 << 64
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Read one varint. Returns ``(value, new_pos)``."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint overflows 64 bits")


def tag(number: int, wire_type: int) -> bytes:
    return encode_varint((number << 3) | wire_type)


def field_varint(number: int, value: int) -> bytes:
    return tag(number, WIRE_VARINT) + encode_varint(value)


def field_bytes(number: int, value: bytes) -> bytes:
    return tag(number, WIRE_BYTES) + encode_varint(len(value)) + value


def field_string(number: int, value: str) -> bytes:
    return field_bytes(number, value.encode("utf-8"))


def field_double(number: int, value: float) -> bytes:
    return tag(number, WIRE_FIXED64) + struct.pack("<d", value)


def field_message(number: int, payload: bytes) -> bytes:
    """Embed a sub-message. Equivalent to ``field_bytes`` but clearer at call sites."""
    return field_bytes(number, payload)


def packed_varints(number: int, values: list[int]) -> bytes:
    """Packed repeated varint field (proto3 default for scalar repeats)."""
    if not values:
        return b""
    body = b"".join(encode_varint(v) for v in values)
    return field_bytes(number, body)


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int | float]]:
    """Walk a serialised message, yielding ``(field_number, wire_type, value)``.

    Length-delimited fields yield ``bytes``; varints yield ``int``; fixed32
    and fixed64 yield the raw 4 or 8 bytes so the caller decides between
    float/double and fixed-width integers. Unknown fields are yielded too,
    which is how forward compatibility is preserved: callers skip what they
    do not recognise instead of failing.
    """
    pos = 0
    end = len(buf)
    while pos < end:
        key, pos = decode_varint(buf, pos)
        number, wire_type = key >> 3, key & 0x07
        if wire_type == WIRE_VARINT:
            value, pos = decode_varint(buf, pos)
            yield number, wire_type, value
        elif wire_type == WIRE_BYTES:
            length, pos = decode_varint(buf, pos)
            if pos + length > end:
                raise ValueError("truncated length-delimited field")
            yield number, wire_type, buf[pos : pos + length]
            pos += length
        elif wire_type == WIRE_FIXED64:
            if pos + 8 > end:
                raise ValueError("truncated fixed64")
            yield number, wire_type, buf[pos : pos + 8]
            pos += 8
        elif wire_type == WIRE_FIXED32:
            if pos + 4 > end:
                raise ValueError("truncated fixed32")
            yield number, wire_type, buf[pos : pos + 4]
            pos += 4
        elif wire_type == 3:  # deprecated start-group
            raise ValueError("group wire type is not supported")
        elif wire_type == 4:  # deprecated end-group
            raise ValueError("unexpected end-group")
        else:
            raise ValueError(f"unknown wire type {wire_type}")
