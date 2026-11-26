"""Sampling's logprob surface, and where samplers come from.

Two promises are under test. First, logprobs for the *sampled* tokens arrive
without being asked for, because every policy-gradient loss needs them, while
everything about the *prompt* is opt-in since it costs extra prefill. Second, a
sampler can be built from a trainer -- including one bound to the live training
model, which is what an on-policy loop needs instead of a checkpoint per step.

These are wire-contract tests: they assert what leaves the SDK and what it
makes of the reply, against the fake relay rather than a mocked method.
"""

from __future__ import annotations

import numpy as np
import pytest

import mind_recursive
from tests.fake_relay import FakeRelay, build_service


@pytest.fixture
def relay():
    return FakeRelay()


@pytest.fixture
def service(relay):
    client = build_service(relay)
    yield client
    client.close()


@pytest.fixture
def sampler(service):
    return service.create_sampler(base_model="Qwen/Qwen3.8-27B")


def _last_sample_body(relay):
    return relay.requests_to("/api/mindrsi/v1/sample/asample")[-1].json()


class TestSampledLogprobsAreUnconditional:
    """The GRPO case: no flag, or the importance ratio cannot be formed."""

    def test_sequences_carry_logprobs_without_being_asked(self, sampler):
        response = sampler.sample([1, 2, 3]).result()

        sequence = response.sequences[0]
        assert sequence.logprobs is not None
        assert len(sequence.logprobs) == len(sequence.tokens)

    def test_no_logprob_flag_is_sent_by_default(self, sampler, relay):
        """A default call must not quietly buy prompt-side prefill work."""
        sampler.sample([1, 2, 3])

        body = _last_sample_body(relay)
        for key in (
            "prompt_logprobs",
            "prompt_logprobs_last_n",
            "topk_prompt_logprobs",
            "topk_sample_logprobs",
            "target_prompt_logprobs",
            "prompt_alt_tokens_k",
            "record_stability_info",
        ):
            assert key not in body
        # seq_id is always sent (the inference service requires it); see TestStabilityInfo.


class TestPromptLogprobsAreOptIn:
    def test_prompt_logprobs_stay_none_until_requested(self, sampler):
        assert sampler.sample([1, 2, 3]).result().prompt_logprobs is None

    def test_requesting_them_sends_the_flag_and_fills_the_field(self, sampler, relay):
        response = sampler.sample([1, 2, 3, 4], prompt_logprobs=True).result()

        assert _last_sample_body(relay)["prompt_logprobs"] is True
        assert response.prompt_logprobs is not None
        assert len(response.prompt_logprobs) == 4

    def test_last_n_scores_only_the_tail(self, sampler, relay):
        response = sampler.sample(
            [1, 2, 3, 4, 5], prompt_logprobs=True, prompt_logprobs_last_n=2
        ).result()

        assert _last_sample_body(relay)["prompt_logprobs_last_n"] == 2
        assert len(response.prompt_logprobs) == 2

    def test_last_n_without_the_flag_is_refused_locally(self, sampler):
        """It scores prompt tokens, so on its own it would be a silent no-op."""
        with pytest.raises(ValueError, match="needs"):
            sampler.sample([1, 2, 3], prompt_logprobs_last_n=2)

    def test_last_n_past_the_prompt_is_refused_locally(self, sampler):
        with pytest.raises(ValueError, match="at most"):
            sampler.sample([1, 2, 3], prompt_logprobs=True, prompt_logprobs_last_n=9)


class TestTopkLogprobs:
    def test_sampled_topk_decodes_row_major(self, sampler):
        """Position i, rank j sits at i*k + j -- not the transpose."""
        response = sampler.sample([1, 2, 3], topk_sample_logprobs=3).result()

        topk = response.sequences[0].topk_logprobs
        assert topk is not None
        assert topk.k == 3
        # The fake fills cell (i, j) with token id i*k + j.
        assert [tid for tid, _ in topk.row(0)] == [0, 1, 2]
        assert [tid for tid, _ in topk.row(1)] == [3, 4, 5]
        assert len(topk.to_rows()) == topk.length

    def test_prompt_topk_decodes(self, sampler, relay):
        response = sampler.sample(
            [1, 2, 3, 4], prompt_logprobs=True, topk_prompt_logprobs=2
        ).result()

        assert _last_sample_body(relay)["topk_prompt_logprobs"] == 2
        assert response.topk_prompt_logprobs.k == 2

    def test_topk_is_none_when_not_requested(self, sampler):
        response = sampler.sample([1, 2, 3]).result()

        assert response.sequences[0].topk_logprobs is None
        assert response.topk_prompt_logprobs is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"topk_sample_logprobs": -1},
            {"topk_prompt_logprobs": -1},
            {"prompt_alt_tokens_k": -1},
            {"seq_id": -1},
        ],
    )
    def test_negative_counts_are_refused_locally(self, sampler, kwargs):
        with pytest.raises(ValueError):
            sampler.sample([1, 2, 3], **kwargs)


class TestTargetPromptLogprobs:
    def test_the_tensor_travels_as_json_not_protobuf(self, sampler, relay):
        """Sampling requests are JSON; only forward_backward is protobuf."""
        ids = mind_recursive.TensorData(np.array([[5], [-1]], dtype=np.int64), "int64")

        sampler.sample([1, 2, 3], target_prompt_logprobs=ids)

        sent = _last_sample_body(relay)["target_prompt_logprobs"]
        assert sent == {"data": [5, -1], "dtype": "int64", "shape": [2, 1]}

    def test_a_sparse_tensor_keeps_its_csr_indices(self, sampler, relay):
        ids = mind_recursive.TensorData(
            np.array([5], dtype=np.int64),
            "int64",
            shape=[2, 1],
            sparse_crow_indices=[0, 0, 1],
            sparse_col_indices=[0],
        )

        sampler.sample([1, 2, 3], target_prompt_logprobs=ids)

        sent = _last_sample_body(relay)["target_prompt_logprobs"]
        assert sent["sparse_crow_indices"] == [0, 0, 1]
        assert sent["sparse_col_indices"] == [0]


class TestSamplerProvenance:
    """Where a sampler comes from, and what weights it is bound to."""

    def test_a_trainer_can_open_a_sampler_on_saved_weights(self, service):
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        sampler = trainer.create_sampler(model_path="mint://run/adapters/x")

        assert sampler.kind == "snapshot"

    def test_snapshot_without_a_path_is_refused_locally(self, service):
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        with pytest.raises(ValueError, match="model_path"):
            trainer.create_sampler()

    def test_inflight_binds_to_the_live_model_and_needs_no_path(
        self, service, relay
    ):
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        sampler = trainer.create_sampler(kind="inflight")

        # inflight samples a live training model, which the inference service has to deploy, so
        # it goes through the service plane, not the relay-direct sample path.
        assert not relay.requests_to("/api/mindrsi/v1/sample/create_sampling_session")
        body = relay.requests_to(
            "/api/mindrsi/v1/service/sample/create_sampling_session"
        )[-1].json()
        assert body["kind"] == "inflight"
        assert body["model_id"] == trainer.model_id
        assert "model_path" not in body
        assert sampler.kind == "inflight"

    def test_inflight_from_the_service_client_needs_a_model_id(self, service):
        """Only a trainer knows the model id, so this path must not guess."""
        with pytest.raises(ValueError, match="model_id"):
            service.create_sampler(kind="inflight")

    def test_inflight_refuses_fixed_weights_alongside_it(self, service):
        with pytest.raises(ValueError, match="fixed weights"):
            service.create_sampler(
                kind="inflight", model_id="mdl-1", model_path="mint://x"
            )

    def test_an_unknown_kind_is_refused(self, service):
        with pytest.raises(ValueError, match="kind must be one of"):
            service.create_sampler(base_model="Qwen/Qwen3.8-27B", kind="bogus")

    def test_save_weights_and_get_sampler_returns_a_live_sampler(self, service):
        """It blocks on the save, so there is no future left to await."""
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        sampler = trainer.save_weights_and_get_sampler("step-1")

        assert isinstance(sampler, mind_recursive.Sampler)
        assert sampler.model_path is not None
        assert sampler.kind == "snapshot"

    def test_a_sampler_from_a_trainer_shares_the_service_transport(self, service):
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

        sampler = trainer.create_sampler(kind="inflight")

        assert sampler._service is service
        assert not hasattr(sampler, "_transport")


class TestMultiTurnSeqId:
    def test_seq_id_is_allocated_per_sampler_when_omitted(self, sampler, relay):
        """the inference service rejects an asample without a non-negative seq_id."""
        sampler.sample([1, 2, 3])
        assert _last_sample_body(relay)["seq_id"] == 0
        sampler.sample([1, 2, 3])
        assert _last_sample_body(relay)["seq_id"] == 1

    def test_seq_id_is_sent_when_given(self, sampler, relay):
        """Multi-turn callers need the request id to be deterministic."""
        sampler.sample([1, 2, 3], seq_id=7)

        assert _last_sample_body(relay)["seq_id"] == 7


class TestStabilityInfo:
    def test_the_per_call_flag_is_sent(self, sampler, relay):
        sampler.sample([1, 2, 3], record_stability_info=True)

        assert _last_sample_body(relay)["record_stability_info"] is True

    def test_the_session_level_flag_is_sent_at_creation(self, service, relay):
        service.create_sampler(
            base_model="Qwen/Qwen3.8-27B", record_stability_info=True
        )

        body = relay.requests_to(
            "/api/mindrsi/v1/service/sample/create_sampling_session"
        )[-1].json()
        assert body["record_stability_info"] is True


class TestSamplingPlane:
    def test_a_sampling_future_is_polled_on_the_sample_plane(self, sampler, relay):
        """The future belongs to the call that made it, not to /train."""
        sampler.sample([1, 2, 3]).result()

        assert relay.count("/api/mindrsi/v1/sample/retrieve_future") >= 1
        assert relay.count("/api/mindrsi/v1/train/retrieve_future") == 0


class TestStopSequences:
    """``stop`` is one untagged union upstream, so the SDK must not send two
    keys: MinT has no ``stop_tokens`` and would silently ignore it."""

    def test_text_stops_go_out_as_stop(self, sampler, relay):
        sampler.sample(
            [1, 2, 3],
            sampling_params=mind_recursive.SamplingParams(stop=["<end>", "</s>"]),
        )

        body = _last_sample_body(relay)
        assert body["sampling_params"]["stop"] == ["<end>", "</s>"]
        assert "stop_tokens" not in body["sampling_params"]

    def test_token_stops_are_folded_into_stop(self, sampler, relay):
        sampler.sample(
            [1, 2, 3],
            sampling_params=mind_recursive.SamplingParams(stop_tokens=[151643, 151645]),
        )

        params = _last_sample_body(relay)["sampling_params"]
        assert params["stop"] == [151643, 151645]
        assert "stop_tokens" not in params

    def test_mixing_text_and_token_stops_is_refused_locally(self):
        with pytest.raises(ValueError, match="not both"):
            mind_recursive.SamplingParams(stop=["<end>"], stop_tokens=[1]).to_wire()

    def test_the_request_declares_its_type(self, sampler, relay):
        sampler.sample([1, 2, 3])

        assert _last_sample_body(relay)["type"] == "sample"
