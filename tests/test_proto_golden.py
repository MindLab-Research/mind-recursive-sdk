"""Differential tests against the generated Tinker protobuf encoder.

Our codec is hand-written, so the only trustworthy check is that it produces
bytes the real generated encoder parses, and that it decodes bytes the real
encoder produced.

We load ``tinker_public_pb2`` **directly from its file** rather than importing
``tinker``. The generated module needs nothing but ``google.protobuf``, while
``import tinker`` executes the package ``__init__`` and drags in pydantic --
which fails against a different interpreter's native extensions. Loading the
file keeps this test runnable from our own venv with only ``protobuf`` present.

Point ``MIND_RECURSIVE_TINKER_PB2`` at the generated file, or
``MIND_RECURSIVE_TINKER_SITE_PACKAGES`` at a site-packages holding ``tinker``. When
neither is available the module skips: a public checkout does not ship the
tinker package, while a development box can still run the differential check
by setting one of the variables. Self-consistency coverage that needs no
upstream lives in ``test_proto_roundtrip.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mind_recursive._proto import schema as S
from mind_recursive._proto.decode import decode_forward_backward_output, decode_sample_response
from mind_recursive._proto.encode import encode_forward_backward_request
from mind_recursive.types import Datum, EncodedTextChunk, ModelInput, TensorData

_REL = "tinker/proto/tinker_public_pb2.py"


def _candidate_paths() -> list[Path]:
    """Where the generated module might live, most explicit first."""
    out: list[Path] = []
    direct = os.environ.get("MIND_RECURSIVE_TINKER_PB2")
    if direct:
        out.append(Path(direct))
    site = os.environ.get("MIND_RECURSIVE_TINKER_SITE_PACKAGES")
    if site:
        out.append(Path(site) / _REL)
    for entry in sys.path:
        if entry:
            out.append(Path(entry) / _REL)
    return out


def _load_generated_pb2():
    """Load the generated descriptors without importing the tinker package."""
    tried: list[str] = []
    for path in _candidate_paths():
        tried.append(str(path))
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_tinker_public_pb2_golden", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "DESCRIPTOR"):
            continue
        return module, str(path)
    raise AssertionError(
        "could not load the generated tinker_public_pb2. These differential "
        "tests guard a hand-written wire codec and must not be skipped.\n"
        "Set MIND_RECURSIVE_TINKER_PB2 to the generated file, or "
        "MIND_RECURSIVE_TINKER_SITE_PACKAGES to a site-packages containing tinker.\n"
        "Tried:\n  " + "\n  ".join(tried)
    )


if not any(path.is_file() for path in _candidate_paths()):
    pytest.skip(
        "tinker_public_pb2 not available; set MIND_RECURSIVE_TINKER_PB2 or "
        "MIND_RECURSIVE_TINKER_SITE_PACKAGES to run the differential protobuf tests",
        allow_module_level=True,
    )

public_pb, PB2_PATH = _load_generated_pb2()


def test_generated_schema_actually_loaded():
    """Guard the guard: prove we are diffing against real generated code."""
    assert "tinker_public_pb2" in PB2_PATH
    assert hasattr(public_pb, "ForwardBackwardRequest")
    assert hasattr(public_pb, "DESCRIPTOR")
    # A real descriptor pool, not a hand-written stand-in that would make asserts vacuous.
    assert "ForwardBackwardRequest" in public_pb.DESCRIPTOR.message_types_by_name
    msg = public_pb.ForwardBackwardRequest()
    msg.model_id = "probe"
    assert public_pb.ForwardBackwardRequest.FromString(msg.SerializeToString()).model_id == "probe"


def _sample_datums() -> list[Datum]:
    """A batch exercising multiple chunks, dtypes and a sparse tensor."""
    return [
        Datum(
            model_input=ModelInput.from_ints([1, 2, 3, 4]),
            loss_fn_inputs={
                "weights": TensorData(np.array([1.0, 1.0, 0.0, 0.5], np.float32), "float32"),
                "target_tokens": TensorData(np.array([2, 3, 4, 0], np.int64), "int64"),
            },
        ),
        Datum(
            model_input=ModelInput(
                chunks=[EncodedTextChunk(tokens=[7, 8]), EncodedTextChunk(tokens=[9])]
            ),
            loss_fn_inputs={
                "weights": TensorData(np.array([1.0, 1.0, 1.0], np.float32), "float32"),
            },
        ),
    ]


def test_schema_field_numbers_match_generated_descriptors():
    """Our transcribed field numbers must equal the generated schema's."""
    expected = {
        "ForwardBackwardRequest": {
            "model_id": S.ForwardBackwardRequest.MODEL_ID,
            "seq_id": S.ForwardBackwardRequest.SEQ_ID,
            "data": S.ForwardBackwardRequest.DATA,
            "loss_fn": S.ForwardBackwardRequest.LOSS_FN,
            "loss_fn_config": S.ForwardBackwardRequest.LOSS_FN_CONFIG,
            "forward_only": S.ForwardBackwardRequest.FORWARD_ONLY,
            "loss_fn_config_v2": S.ForwardBackwardRequest.LOSS_FN_CONFIG_V2,
        },
        "Datum": {
            "model_input": S.Datum.MODEL_INPUT,
            "loss_fn_inputs": S.Datum.LOSS_FN_INPUTS,
            "model_input_spans": S.Datum.MODEL_INPUT_SPANS,
            "loss_fn_input_spans": S.Datum.LOSS_FN_INPUT_SPANS,
        },
        "Chunk": {
            "encoded_text": S.Chunk.ENCODED_TEXT,
            "image": S.Chunk.IMAGE,
            "dmel": S.Chunk.DMEL,
        },
        "Tensor": {
            "dense": S.Tensor.DENSE,
            "sparse_csr": S.Tensor.SPARSE_CSR,
            "dtype": S.Tensor.DTYPE,
            "shape": S.Tensor.SHAPE,
        },
        "SparseCsr": {
            "values": S.SparseCsr.VALUES,
            "crow_indices": S.SparseCsr.CROW_INDICES,
            "col_indices": S.SparseCsr.COL_INDICES,
        },
        "BatchedTensor": {
            "data": S.BatchedTensor.DATA,
            "offsets": S.BatchedTensor.OFFSETS,
            "dtype": S.BatchedTensor.DTYPE,
            "trailing_shape": S.BatchedTensor.TRAILING_SHAPE,
        },
        "ArrayRecord": {
            "type_tag": S.ArrayRecord.TYPE_TAG,
            "fields": S.ArrayRecord.FIELDS,
            "num_datums": S.ArrayRecord.NUM_DATUMS,
        },
        "ForwardBackwardOutput": {
            "loss_fn_output_type": S.ForwardBackwardOutput.LOSS_FN_OUTPUT_TYPE,
            "loss_fn_outputs": S.ForwardBackwardOutput.LOSS_FN_OUTPUTS,
            "metrics": S.ForwardBackwardOutput.METRICS,
        },
        "SampleResponse": {
            "sequences": S.SampleResponse.SEQUENCES,
            "prompt_logprobs": S.SampleResponse.PROMPT_LOGPROBS,
            "prompt_cache_hit_tokens": S.SampleResponse.PROMPT_CACHE_HIT_TOKENS,
        },
        "SampledSequence": {
            "stop_reason": S.SampledSequence.STOP_REASON,
            "tokens": S.SampledSequence.TOKENS,
            "logprobs": S.SampledSequence.LOGPROBS,
        },
    }
    for message_name, fields in expected.items():
        descriptor = public_pb.DESCRIPTOR.message_types_by_name[message_name]
        actual = {f.name: f.number for f in descriptor.fields}
        for field_name, number in fields.items():
            assert actual[field_name] == number, (
                f"{message_name}.{field_name}: schema says {number}, "
                f"generated says {actual[field_name]}"
            )


def test_enum_values_match_generated():
    assert public_pb.DTYPE_FLOAT32 == S.DTYPE_FLOAT32
    assert public_pb.DTYPE_INT64 == S.DTYPE_INT64
    assert public_pb.DTYPE_INT32 == S.DTYPE_INT32
    assert public_pb.DTYPE_BFLOAT16 == S.DTYPE_BFLOAT16
    assert public_pb.STOP_REASON_STOP == S.STOP_REASON_STOP
    assert public_pb.STOP_REASON_LENGTH == S.STOP_REASON_LENGTH


def test_our_request_parses_as_generated_message():
    """The generated parser must read our bytes back field-for-field."""
    datums = _sample_datums()
    raw = encode_forward_backward_request(
        model_id="model-abc_0",
        seq_id=7,
        data=datums,
        loss_fn="cross_entropy",
    )

    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(raw)

    assert parsed.model_id == "model-abc_0"
    assert parsed.seq_id == 7
    assert parsed.loss_fn == "cross_entropy"
    assert parsed.forward_only is False
    assert len(parsed.data) == 2

    first = parsed.data[0]
    assert len(first.model_input) == 1
    tokens = np.frombuffer(first.model_input[0].encoded_text.tokens, dtype=np.int32)
    np.testing.assert_array_equal(tokens, [1, 2, 3, 4])

    weights = np.frombuffer(first.loss_fn_inputs["weights"].dense, dtype=np.float32)
    np.testing.assert_allclose(weights, [1.0, 1.0, 0.0, 0.5])
    assert first.loss_fn_inputs["weights"].dtype == public_pb.DTYPE_FLOAT32

    targets = np.frombuffer(first.loss_fn_inputs["target_tokens"].dense, dtype=np.int64)
    np.testing.assert_array_equal(targets, [2, 3, 4, 0])
    assert first.loss_fn_inputs["target_tokens"].dtype == public_pb.DTYPE_INT64

    # Two text chunks stay two chunks -- we must not silently merge them.
    assert len(parsed.data[1].model_input) == 2


def test_forward_only_and_loss_config_dual_write():
    """Numeric loss config lands in both the legacy and v2 maps; text only v2."""
    raw = encode_forward_backward_request(
        model_id="m",
        seq_id=1,
        data=_sample_datums()[:1],
        loss_fn="ppo",
        loss_fn_config={"clip": 0.2, "variant": "dual"},
        forward_only=True,
    )
    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(raw)

    assert parsed.forward_only is True
    assert parsed.loss_fn_config["clip"] == pytest.approx(0.2)
    assert "variant" not in parsed.loss_fn_config
    assert parsed.loss_fn_config_v2["clip"].number == pytest.approx(0.2)
    assert parsed.loss_fn_config_v2["variant"].text == "dual"


def test_sparse_tensor_matches_generated_layout():
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "advantages": TensorData(
                np.array([1.5, 2.5], np.float32),
                "float32",
                shape=[2, 4],
                sparse_crow_indices=[0, 1, 2],
                sparse_col_indices=[0, 3],
            )
        },
    )
    raw = encode_forward_backward_request(
        model_id="m", seq_id=1, data=[datum], loss_fn="importance_sampling"
    )
    parsed = public_pb.ForwardBackwardRequest()
    parsed.ParseFromString(raw)

    tensor = parsed.data[0].loss_fn_inputs["advantages"]
    assert tensor.HasField("sparse_csr")
    np.testing.assert_allclose(
        np.frombuffer(tensor.sparse_csr.values, dtype=np.float32), [1.5, 2.5]
    )
    np.testing.assert_array_equal(
        np.frombuffer(tensor.sparse_csr.crow_indices, dtype=np.int64), [0, 1, 2]
    )
    np.testing.assert_array_equal(
        np.frombuffer(tensor.sparse_csr.col_indices, dtype=np.int64), [0, 3]
    )
    assert list(tensor.shape) == [2, 4]


def test_we_decode_generated_forward_backward_output():
    """Build an output with the generated encoder; our decoder must read it."""
    msg = public_pb.ForwardBackwardOutput()
    msg.loss_fn_output_type = "tinker.types:ArrayRecord"
    msg.metrics["loss:mean"] = 0.5
    msg.metrics["num_tokens:sum"] = 1234.0

    record = msg.loss_fn_outputs.add()
    record.num_datums = 2
    # Two datums of float32 logprobs: 3 then 2 values. Offsets are byte offsets.
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], np.float32)
    field = record.fields["logprobs"]
    field.data = values.tobytes()
    field.offsets = np.array([0, 12, 20], np.int64).tobytes()
    field.dtype = public_pb.DTYPE_FLOAT32

    out = decode_forward_backward_output(msg.SerializeToString())

    assert out.loss_fn_output_type == "ArrayRecord"  # bare class name
    assert out.loss == pytest.approx(0.5)
    assert out.num_tokens == 1234
    assert len(out.loss_fn_outputs) == 2
    np.testing.assert_allclose(
        out.loss_fn_outputs[0]["logprobs"].data, [0.1, 0.2, 0.3], rtol=1e-6
    )
    np.testing.assert_allclose(
        out.loss_fn_outputs[1]["logprobs"].data, [0.4, 0.5], rtol=1e-6
    )


def test_we_decode_generated_bfloat16_output():
    """bfloat16 offsets index the 2-byte stream even though we widen to float32."""
    msg = public_pb.ForwardBackwardOutput()
    msg.loss_fn_output_type = "ArrayRecord"
    record = msg.loss_fn_outputs.add()
    record.num_datums = 2

    # bf16 for [1.0, 2.0, 4.0, 8.0] is the top 16 bits of each float32.
    f32 = np.array([1.0, 2.0, 4.0, 8.0], np.float32)
    bf16 = (f32.view(np.uint32) >> 16).astype(np.uint16)
    field = record.fields["logprobs"]
    field.data = bf16.tobytes()
    field.offsets = np.array([0, 4, 8], np.int64).tobytes()  # 2 bytes/elem
    field.dtype = public_pb.DTYPE_BFLOAT16

    out = decode_forward_backward_output(msg.SerializeToString())

    assert len(out.loss_fn_outputs) == 2
    np.testing.assert_allclose(out.loss_fn_outputs[0]["logprobs"].data, [1.0, 2.0])
    np.testing.assert_allclose(out.loss_fn_outputs[1]["logprobs"].data, [4.0, 8.0])
    assert out.loss_fn_outputs[0]["logprobs"].dtype == "float32"


def test_we_decode_generated_sample_response():
    msg = public_pb.SampleResponse()
    msg.prompt_cache_hit_tokens = 6
    msg.prompt_logprobs = np.array([-0.1, -0.2], np.float32).tobytes()

    seq = msg.sequences.add()
    seq.stop_reason = public_pb.STOP_REASON_LENGTH
    seq.tokens = np.array([11, 12, 13], np.int32).tobytes()
    seq.logprobs = np.array([-1.0, -2.0, -3.0], np.float32).tobytes()

    out = decode_sample_response(msg.SerializeToString())

    assert out.prompt_cache_hit_tokens == 6
    np.testing.assert_allclose(out.prompt_logprobs, [-0.1, -0.2], rtol=1e-6)
    assert len(out.sequences) == 1
    assert out.sequences[0].stop_reason == "length"
    assert out.sequences[0].tokens == [11, 12, 13]
    np.testing.assert_allclose(out.sequences[0].logprobs, [-1.0, -2.0, -3.0])


def test_we_decode_generated_topk_and_prompt_scoring_fields():
    """The logprob fields the generated encoder writes, decoded field for field.

    Hand-written decoding of a row-major N x K block is exactly the kind of
    thing that looks right and is transposed, so the generated encoder lays the
    bytes out and we read them back.
    """
    k, positions = 3, 2
    ids = np.arange(k * positions, dtype=np.int32)
    logprobs = (-ids / 10.0).astype(np.float32)

    msg = public_pb.SampleResponse()
    seq = msg.sequences.add()
    seq.stop_reason = public_pb.STOP_REASON_STOP
    seq.tokens = np.array([11, 12], np.int32).tobytes()
    seq.topk_sampled_logprobs.token_ids = ids.tobytes()
    seq.topk_sampled_logprobs.logprobs = logprobs.tobytes()
    seq.topk_sampled_logprobs.k = k
    seq.topk_sampled_logprobs.length = positions

    msg.topk_prompt_logprobs.token_ids = ids.tobytes()
    msg.topk_prompt_logprobs.logprobs = logprobs.tobytes()
    msg.topk_prompt_logprobs.k = k
    msg.topk_prompt_logprobs.length = positions

    msg.target_prompt_logprobs.dense = np.array([-0.9, -0.8], np.float32).tobytes()
    msg.target_prompt_logprobs.dtype = public_pb.DTYPE_FLOAT32
    msg.target_prompt_logprobs.shape.extend([2, 1])

    msg.prompt_alt_tokens.tokens.dense = np.array([4, 5], np.int64).tobytes()
    msg.prompt_alt_tokens.tokens.dtype = public_pb.DTYPE_INT64
    msg.prompt_alt_tokens.tokens.shape.extend([2, 1])
    msg.prompt_alt_tokens.logprobs.dense = np.array([-1.5, -1.6], np.float32).tobytes()
    msg.prompt_alt_tokens.logprobs.dtype = public_pb.DTYPE_FLOAT32
    msg.prompt_alt_tokens.logprobs.shape.extend([2, 1])

    out = decode_sample_response(msg.SerializeToString())

    sampled_topk = out.sequences[0].topk_logprobs
    assert (sampled_topk.k, sampled_topk.length) == (k, positions)
    # Row-major: position i, rank j is at i*k + j. A transpose fails here.
    assert [tid for tid, _ in sampled_topk.row(0)] == [0, 1, 2]
    assert [tid for tid, _ in sampled_topk.row(1)] == [3, 4, 5]
    np.testing.assert_allclose(
        [lp for _, lp in sampled_topk.row(1)], [-0.3, -0.4, -0.5], rtol=1e-6
    )

    assert out.topk_prompt_logprobs.k == k
    assert [tid for tid, _ in out.topk_prompt_logprobs.row(0)] == [0, 1, 2]

    assert out.target_prompt_logprobs.dtype == "float32"
    assert out.target_prompt_logprobs.shape == [2, 1]
    np.testing.assert_allclose(
        out.target_prompt_logprobs.data, [-0.9, -0.8], rtol=1e-6
    )

    assert out.prompt_alt_tokens.tokens.dtype == "int64"
    assert out.prompt_alt_tokens.tokens.data.tolist() == [4, 5]
    np.testing.assert_allclose(
        out.prompt_alt_tokens.logprobs.data, [-1.5, -1.6], rtol=1e-6
    )


def test_topk_length_is_derived_when_the_server_omits_it():
    """length is redundant with len(token_ids)/k, so it may be absent."""
    msg = public_pb.SampleResponse()
    seq = msg.sequences.add()
    seq.stop_reason = public_pb.STOP_REASON_STOP
    seq.topk_sampled_logprobs.token_ids = np.arange(6, dtype=np.int32).tobytes()
    seq.topk_sampled_logprobs.logprobs = np.zeros(6, np.float32).tobytes()
    seq.topk_sampled_logprobs.k = 3

    out = decode_sample_response(msg.SerializeToString())

    assert out.sequences[0].topk_logprobs.length == 2


def test_num_datums_survives_stripped_fields():
    """drop_fwdbwd_logprobs strips every field; num_datums still drives shape."""
    msg = public_pb.ForwardBackwardOutput()
    msg.loss_fn_output_type = "ArrayRecord"
    record = msg.loss_fn_outputs.add()
    record.num_datums = 3

    out = decode_forward_backward_output(msg.SerializeToString())
    assert len(out.loss_fn_outputs) == 3
    assert all(d == {} for d in out.loss_fn_outputs)
