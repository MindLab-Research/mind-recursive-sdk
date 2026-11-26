"""The Tinker-shaped shells on top of the native API.

These tests pin the *call shapes* ported Tinker code uses, not a promise of
full compatibility: the uncovered surface (RestClient, AdminClient, custom
losses, external weights, tokenizers) is deliberately absent, so there is
nothing here to pin for it.
"""

from __future__ import annotations

import pytest

import mind_recursive
from mind_recursive._errors import ConfigurationError
from mind_recursive.types import AdamParams, ModelInput, SamplingParams
from tests.fake_relay import FakeRelay, build_service


@pytest.fixture
def service():
    client = build_service(FakeRelay())
    yield client
    client.close()


def test_creation_result_returns_the_created_object(service):
    """Upstream code calls ``create_...(...).result()``; creation is sync here."""
    trainer = service.create_lora_training_client(base_model="Qwen/Qwen3.8-27B")
    assert trainer.result() is trainer

    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
    assert sampler.result() is sampler


def test_create_training_client_from_state_shell(service):
    trainer = service.create_training_client_from_state(
        "mint://run-abc/checkpoints/step-1",
        base_model="Qwen/Qwen3.8-27B",
    )
    assert isinstance(trainer, mind_recursive.LoraTrainer)

    with pytest.raises(ConfigurationError, match="base_model"):
        service.create_training_client_from_state("mint://run-abc/checkpoints/step-1")


def test_create_training_client_from_state_with_optimizer_shell(service):
    trainer = service.create_training_client_from_state_with_optimizer(
        "mint://run-abc/checkpoints/step-1",
        base_model="Qwen/Qwen3.8-27B",
    )
    assert isinstance(trainer, mind_recursive.LoraTrainer)


def test_forward_delegates_to_forward_backward(service, monkeypatch):
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    seen: dict[str, object] = {}
    sentinel = object()

    def fake_forward_backward(
        data, loss_fn="cross_entropy", loss_fn_config=None, *, forward_only=False
    ):
        seen.update(
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            forward_only=forward_only,
        )
        return sentinel

    monkeypatch.setattr(trainer, "forward_backward", fake_forward_backward)
    result = trainer.forward(["datum"], "cross_entropy", {"k": 1})

    assert result is sentinel
    assert seen == {
        "data": ["datum"],
        "loss_fn": "cross_entropy",
        "loss_fn_config": {"k": 1},
        "forward_only": True,
    }


def test_optim_step_accepts_both_upstream_spellings(service):
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    trainer.optim_step(AdamParams(learning_rate=1e-5)).result()
    trainer.optim_step(adam_params=AdamParams(learning_rate=1e-5)).result()

    with pytest.raises(ValueError, match="not both"):
        trainer.optim_step(AdamParams(), adam_params=AdamParams())

    with pytest.raises(TypeError, match="AdamParams"):
        trainer.optim_step(object())


def test_sample_accepts_upstream_positional_form(service):
    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
    response = sampler.sample(
        ModelInput.from_ints([1, 2, 3]),
        2,
        SamplingParams(max_tokens=4),
        True,
    ).result()

    assert response.prompt_logprobs is not None


def test_sample_rejects_conflicting_prompt_logprob_spellings(service):
    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
    with pytest.raises(ValueError, match="conflicts"):
        sampler.sample(
            ModelInput.from_ints([1, 2, 3]),
            1,
            SamplingParams(),
            include_prompt_logprobs=False,
            prompt_logprobs=True,
        )


def test_compute_logprobs_is_a_shell_over_sample(service):
    sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
    logprobs = sampler.compute_logprobs(ModelInput.from_ints([1, 2, 3])).result()
    assert logprobs == pytest.approx([-0.2, -0.4, -0.6])


def test_save_state_accepts_positional_arguments_and_path_property(service):
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    path = trainer.save_state("step-1", 3600, False).result()

    assert path == "mint://run-abc/state/step-1"
    assert path.path == str(path)


SAVE_FOR_SAMPLER = "/api/mindrsi/v1/train/save_weights_for_sampler"
CREATE_SAMPLING_SESSION = "/api/mindrsi/v1/sample/create_sampling_session"
SERVICE_CREATE_SAMPLING_SESSION = (
    "/api/mindrsi/v1/service/sample/create_sampling_session"
)
REPORT_ARTIFACT_SUFFIX = "/artifact"


@pytest.fixture
def relay_service():
    relay = FakeRelay()
    client = build_service(relay)
    yield relay, client
    client.close()


def test_save_and_get_sampling_client_is_a_two_hop_export(relay_service):
    """Save on the training service under the reserved ``ephemeral`` name, then open a
    sampler from the returned path."""
    relay, service = relay_service
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    sampler = trainer.save_weights_and_get_sampling_client()

    assert isinstance(sampler, mind_recursive.Sampler)
    assert sampler.result() is sampler
    # Hop 2 samples from the saved path, so the sampler is bound to a
    # model_path, not a base_model.
    assert sampler.model_path is not None

    body = relay.last(SAVE_FOR_SAMPLER).json()
    assert body["path"].startswith("ephemeral-")
    assert body["ttl_seconds"] == 86400
    # Hop 2 goes through the service plane (the inference service has to deploy the path),
    # never the relay-direct sample create.
    assert relay.count(CREATE_SAMPLING_SESSION) == 0
    assert relay.count(SERVICE_CREATE_SAMPLING_SESSION) == 1
    # The platform service deploys the path, so it needs the trainer's base
    # model and the training model it belongs to.
    create = relay.last(SERVICE_CREATE_SAMPLING_SESSION).json()
    assert create["model_path"] == sampler.model_path
    assert create["base_model"] == "Qwen/Qwen3.8-27B"
    assert create["training_model_id"] == trainer.model_id


def test_save_and_get_sampling_client_uses_a_distinct_path_per_step(relay_service):
    relay, service = relay_service
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    for _ in range(5):
        trainer.save_weights_and_get_sampling_client()

    saves = [r.json() for r in relay.requests_to(SAVE_FOR_SAMPLER)]
    assert len(saves) == 5
    # Each step saves a distinct ephemeral path: the inference service serves the previous
    # step's deployment until the new one is ready, so they must not collide.
    paths = [b["path"] for b in saves]
    assert all(p.startswith("ephemeral-") for p in paths)
    assert len(set(paths)) == 5
    # The throwaway export is never reported as a user artefact.
    assert not [r for r in relay.requests if r.path.endswith(REPORT_ARTIFACT_SUFFIX)]


def test_ephemeral_names_count_per_trainer(relay_service):
    """Exports are keyed by training model, so two trainers may both start at
    ``ephemeral-1``; the sequence only has to be distinct within one trainer."""
    relay, service = relay_service
    first = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    second = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")

    first.save_weights_and_get_sampling_client()
    second.save_weights_and_get_sampling_client()

    paths = [r.json()["path"] for r in relay.requests_to(SAVE_FOR_SAMPLER)]
    assert paths == ["ephemeral-1", "ephemeral-1"]


def test_save_and_get_sampling_client_takes_no_name(service):
    """A name that silently did nothing was worse than no name at all.

    Ported code that passes one now fails on the call rather than saving a
    throwaway the caller believes it named.
    """
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    with pytest.raises(TypeError):
        trainer.save_weights_and_get_sampling_client("step-1")


def test_the_tinker_spelling_is_a_real_alias():
    """Same function object, so the two names cannot drift apart."""
    from mind_recursive.trainer import LoraTrainer

    assert (
        LoraTrainer.save_weights_and_get_sampling_client
        is LoraTrainer.save_weights_and_get_sampler
    )
    assert (
        LoraTrainer.save_weights_and_get_sampling_client_async
        is LoraTrainer.save_weights_and_get_sampler_async
    )


def test_save_and_get_sampling_client_carries_record_stability_info(relay_service):
    relay, service = relay_service
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    sampler = trainer.save_weights_and_get_sampling_client(record_stability_info=True)
    sampler.sample(
        ModelInput.from_ints([1, 2, 3]), 1, SamplingParams(max_tokens=4)
    ).result()
    assert relay.last("/api/mindrsi/v1/sample/asample").json()["record_stability_info"]


def test_save_and_get_sampling_client_inflight_skips_the_save(relay_service):
    relay, service = relay_service
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    sampler = trainer.save_weights_and_get_sampling_client(kind="inflight")
    assert sampler.kind == "inflight"
    assert relay.count(SAVE_FOR_SAMPLER) == 0


def test_save_and_get_sampling_client_rejects_an_unknown_kind(service):
    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    with pytest.raises(ValueError, match="kind must be"):
        trainer.save_weights_and_get_sampling_client(kind="bogus")


def test_retry_config_is_rejected_rather_than_ignored(service):
    with pytest.raises(TypeError, match="retry_config"):
        service.create_sampling_client(base_model="Qwen/Qwen3.8-27B", retry_config=object())

    trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    with pytest.raises(TypeError, match="retry_config"):
        trainer.create_sampling_client("mint://run-abc/adapters/step-1", object())


def test_close_accepts_upstream_shape(service):
    assert service.close("success", detail="done").result() is None


def test_tinker_namespace_and_capabilities():
    """`from mind_recursive import tinker` is the compatibility namespace."""
    from mind_recursive import tinker

    assert tinker.TrainingClient is mind_recursive.LoraTrainer
    assert tinker.SamplingClient is mind_recursive.Sampler

    relay = FakeRelay()
    service = build_service(relay)
    try:
        caps = tinker.ServiceClient.get_server_capabilities(service)
        assert caps.names() == ["Qwen/Qwen3.8-27B"]
    finally:
        service.close()


@pytest.mark.asyncio
async def test_tinker_capabilities_async():
    from mind_recursive import tinker
    from tests.fake_relay import build_async_service

    relay = FakeRelay()
    service = build_async_service(relay)
    try:
        caps = await tinker.ServiceClient.get_server_capabilities_async(service)
        assert caps.names() == ["Qwen/Qwen3.8-27B"]
    finally:
        await service.aclose()
