"""Codec self-consistency, needing no upstream package.

``test_proto_golden.py`` proves we agree with the generated encoder. These
tests cover the primitives and the edge cases that are easier to construct by
hand: varint boundaries, offset slicing, unknown-field tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from mind_recursive._proto import schema as S
from mind_recursive._proto.codec import (
    decode_varint,
    encode_varint,
    field_bytes,
    field_message,
    field_string,
    field_varint,
    iter_fields,
    packed_varints,
)
from mind_recursive._proto.decode import decode_forward_backward_output, decode_tensor
from mind_recursive._proto.encode import encode_forward_backward_request, encode_tensor
from mind_recursive.types import Datum, EncodedTextChunk, ModelInput, TensorData


class TestVarint:
    @pytest.mark.parametrize(
        "value",
        [0, 1, 127, 128, 255, 256, 16383, 16384, 2**31 - 1, 2**32, 2**63 - 1],
    )
    def test_round_trip(self, value):
        assert decode_varint(encode_varint(value), 0) == (value, len(encode_varint(value)))

    def test_boundary_widths(self):
        """7-bit group boundaries are where hand-rolled varints break."""
        assert len(encode_varint(127)) == 1
        assert len(encode_varint(128)) == 2
        assert len(encode_varint(16383)) == 2
        assert len(encode_varint(16384)) == 3

    def test_negative_uses_twos_complement(self):
        """Proto encodes negative int32/int64 as 10-byte two's complement."""
        encoded = encode_varint(-1)
        assert len(encoded) == 10
        assert decode_varint(encoded, 0)[0] == 2**64 - 1

    def test_truncated_varint_is_rejected(self):
        with pytest.raises(ValueError, match="truncated"):
            decode_varint(b"\x80\x80", 0)

    def test_overlong_varint_is_rejected(self):
        with pytest.raises(ValueError, match="overflow"):
            decode_varint(b"\x80" * 11, 0)


class TestFieldIteration:
    def test_mixed_wire_types_round_trip(self):
        buf = (
            field_varint(1, 150)
            + field_string(2, "hello")
            + field_bytes(3, b"\x00\xff")
            + packed_varints(4, [1, 2, 300])
        )
        fields = list(iter_fields(buf))
        assert fields[0] == (1, 0, 150)
        assert fields[1][2] == b"hello"
        assert fields[2][2] == b"\x00\xff"
        assert fields[3][2] == encode_varint(1) + encode_varint(2) + encode_varint(300)

    def test_unknown_fields_are_yielded_not_rejected(self):
        """Forward compatibility: a new server field must not break old SDKs."""
        buf = field_varint(1, 5) + field_varint(99, 7)
        numbers = [num for num, _w, _v in iter_fields(buf)]
        assert numbers == [1, 99]

    def test_empty_packed_field_is_omitted(self):
        assert packed_varints(4, []) == b""

    def test_truncated_length_delimited_is_rejected(self):
        with pytest.raises(ValueError, match="truncated"):
            list(iter_fields(b"\x12\x05ab"))  # claims 5 bytes, supplies 2

    def test_group_wire_types_are_rejected(self):
        with pytest.raises(ValueError, match="group"):
            list(iter_fields(b"\x0b"))


class TestTensorRoundTrip:
    @pytest.mark.parametrize(
        "values,dtype",
        [
            ([1.0, 2.5, -3.25], "float32"),
            ([1, 2, 3, 2**40], "int64"),
            ([], "float32"),
        ],
    )
    def test_dense_round_trip(self, values, dtype):
        original = TensorData(np.asarray(values, dtype=dtype), dtype)
        decoded = decode_tensor(encode_tensor(original))
        assert decoded.dtype == dtype
        np.testing.assert_allclose(decoded.data, original.data)

    def test_sparse_round_trip_stays_sparse(self):
        original = TensorData(
            np.array([1.5, 2.5], np.float32),
            "float32",
            shape=[2, 4],
            sparse_crow_indices=[0, 1, 2],
            sparse_col_indices=[0, 3],
        )
        decoded = decode_tensor(encode_tensor(original))
        assert decoded.is_sparse
        assert decoded.sparse_crow_indices == [0, 1, 2]
        assert decoded.sparse_col_indices == [0, 3]
        assert decoded.shape == [2, 4]

    def test_dtype_is_coerced_on_construction(self):
        """Declaring int64 must produce int64 bytes, not float bytes."""
        tensor = TensorData(np.array([1.0, 2.0], np.float64), "int64")
        assert tensor.data.dtype == np.int64
        assert decode_tensor(encode_tensor(tensor)).data.tolist() == [1, 2]

    def test_unsupported_dtype_is_rejected(self):
        with pytest.raises(ValueError):
            TensorData(np.array([1], np.int8), "int8")  # type: ignore[arg-type]

    def test_half_specified_sparsity_is_rejected(self):
        with pytest.raises(ValueError):
            TensorData(np.array([1.0], np.float32), "float32", sparse_crow_indices=[0, 1])


class TestForwardBackwardRequest:
    def test_tokens_are_encoded_as_int32(self):
        """int64 here would silently shift every token id."""
        raw = encode_forward_backward_request(
            model_id="m",
            seq_id=1,
            data=[Datum(model_input=ModelInput.from_ints([1, 2, 3]))],
            loss_fn="cross_entropy",
        )
        datum = next(
            value
            for num, _w, value in iter_fields(raw)
            if num == S.ForwardBackwardRequest.DATA
        )
        chunk = next(
            value for num, _w, value in iter_fields(datum) if num == S.Datum.MODEL_INPUT
        )
        text = next(
            value for num, _w, value in iter_fields(chunk) if num == S.Chunk.ENCODED_TEXT
        )
        tokens = next(
            value
            for num, _w, value in iter_fields(text)
            if num == S.EncodedTextChunk.TOKENS
        )
        assert len(tokens) == 12  # 3 tokens x 4 bytes
        np.testing.assert_array_equal(np.frombuffer(tokens, np.int32), [1, 2, 3])

    def test_multiple_chunks_stay_separate(self):
        raw = encode_forward_backward_request(
            model_id="m",
            seq_id=1,
            data=[
                Datum(
                    model_input=ModelInput(
                        chunks=[EncodedTextChunk([1]), EncodedTextChunk([2])]
                    )
                )
            ],
            loss_fn="cross_entropy",
        )
        datum = next(
            value
            for num, _w, value in iter_fields(raw)
            if num == S.ForwardBackwardRequest.DATA
        )
        chunks = [
            value for num, _w, value in iter_fields(datum) if num == S.Datum.MODEL_INPUT
        ]
        assert len(chunks) == 2

    def test_forward_only_is_omitted_when_false(self):
        """proto3 default: absent means false; emitting it is wasted bytes."""
        raw = encode_forward_backward_request(
            model_id="m",
            seq_id=1,
            data=[Datum(model_input=ModelInput.from_ints([1]))],
            loss_fn="cross_entropy",
            forward_only=False,
        )
        numbers = {num for num, _w, _v in iter_fields(raw)}
        assert S.ForwardBackwardRequest.FORWARD_ONLY not in numbers


class TestBatchedTensorSlicing:
    def _build(self, values: np.ndarray, offsets: list[int], dtype: int, num: int) -> bytes:
        batched = (
            field_bytes(S.BatchedTensor.DATA, values.tobytes())
            + field_bytes(
                S.BatchedTensor.OFFSETS, np.asarray(offsets, np.int64).tobytes()
            )
            + field_varint(S.BatchedTensor.DTYPE, dtype)
        )
        entry = field_string(S.MapEntry.KEY, "logprobs") + field_message(
            S.MapEntry.VALUE, batched
        )
        record = field_message(S.ArrayRecord.FIELDS, entry) + field_varint(
            S.ArrayRecord.NUM_DATUMS, num
        )
        return field_string(
            S.ForwardBackwardOutput.LOSS_FN_OUTPUT_TYPE, "ArrayRecord"
        ) + field_message(S.ForwardBackwardOutput.LOSS_FN_OUTPUTS, record)

    def test_offsets_are_byte_offsets_not_element_indices(self):
        """Treating these as element indices misaligns every datum."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], np.float32)
        raw = self._build(values, [0, 8, 20], S.DTYPE_FLOAT32, 2)
        out = decode_forward_backward_output(raw)
        np.testing.assert_allclose(out.loss_fn_outputs[0]["logprobs"].data, [1.0, 2.0])
        np.testing.assert_allclose(
            out.loss_fn_outputs[1]["logprobs"].data, [3.0, 4.0, 5.0]
        )

    def test_int32_widens_to_int64_without_shifting_offsets(self):
        values = np.array([10, 20, 30], np.int32)
        raw = self._build(values, [0, 4, 12], S.DTYPE_INT32, 2)
        out = decode_forward_backward_output(raw)
        assert out.loss_fn_outputs[0]["logprobs"].data.tolist() == [10]
        assert out.loss_fn_outputs[1]["logprobs"].data.tolist() == [20, 30]
        assert out.loss_fn_outputs[0]["logprobs"].dtype == "int64"

    def test_empty_output_is_handled(self):
        out = decode_forward_backward_output(b"")
        assert out.loss_fn_outputs == []
        assert out.metrics == {}
        assert out.loss is None
        assert out.num_tokens is None
