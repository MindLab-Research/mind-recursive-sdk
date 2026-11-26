"""Native async surface: same wire, awaited instead of blocked."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import mind_recursive
from tests.fake_relay import (
    FakeRelay,
    build_async_service,
    patch_user_async,
)

TRAIN = "/api/mindrsi/v1/train"
SERVICE = "/api/mindrsi/v1/service"


def _batch(n: int = 2, length: int = 4) -> list[mind_recursive.Datum]:
    return [
        mind_recursive.Datum.from_tokens(list(range(i * length, i * length + length)))
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_user_capabilities_and_health_async():
    relay = FakeRelay()
    user = mind_recursive.UserClient(api_key="sk-test-key", base_url="https://relay.test")
    patch_user_async(user, relay)

    caps = await user.get_server_capabilities_async()
    assert caps.names() == ["Qwen/Qwen3.8-27B"]
    assert await user.health_check_async() is True

    relay.fail_paths[f"{SERVICE}/healthz"] = (503, {"error": "down"})
    assert await user.health_check_async() is False
    await user.aclose()


@pytest.mark.asyncio
async def test_async_training_step_and_save():
    relay = FakeRelay()
    service = build_async_service(relay)

    trainer = await service.create_lora_trainer_async(
        base_model="Qwen/Qwen3.8-27B", rank=16
    )
    out = await trainer.forward_backward_async(_batch())
    assert out.loss == pytest.approx(0.5)
    assert out.num_tokens == 128

    step = await trainer.optim_step_async()
    assert step["grad_norm"] == pytest.approx(1.25)

    path = await trainer.save_weights_only_async("async-save")
    assert path == "mint://run-abc/sampler/async-save"
    assert relay.count(f"{TRAIN}/save_weights_for_sampler") == 1

    info = await trainer.get_info_async()
    assert info.model_data["base_model"] == "Qwen/Qwen3.8-27B"
    await service.aclose()


@pytest.mark.asyncio
async def test_async_forward_backward_splits_chunks_concurrently():
    relay = FakeRelay()
    relay.client_config["fwdbwd_max_chunk_len"] = 4
    service = build_async_service(relay)

    trainer = await service.create_lora_trainer_async(base_model="Qwen/Qwen3.8-27B")
    # Two 4-token datums exceed a 4-token chunk budget -> two submissions.
    out = await trainer.forward_backward_async(_batch(n=2, length=4))
    assert relay.count(f"{TRAIN}/forward_backward") == 2
    assert out.num_tokens == 256
    await service.aclose()


@pytest.mark.asyncio
async def test_async_sampling_and_logprobs():
    relay = FakeRelay()
    relay.sample_token_groups = [[5, 6, 7]]
    service = build_async_service(relay)

    sampler = await service.create_sampler_async(base_model="Qwen/Qwen3.8-27B")
    response = await sampler.sample_async([1, 2, 3], num_samples=1)
    assert response.sequences[0].tokens == [5, 6, 7]
    np.testing.assert_allclose(
        response.sequences[0].logprobs, [-0.1, -0.1, -0.1], rtol=1e-6
    )

    logprobs = await sampler.compute_logprobs_async([1, 2, 3])
    assert logprobs is not None and len(logprobs) == 3
    assert logprobs[0] == pytest.approx(-0.2, abs=1e-6)
    assert await sampler.get_base_model_async() == "Qwen/Qwen3.8-27B"
    await service.aclose()


@pytest.mark.asyncio
async def test_async_first_token_creation_and_max_context():
    relay = FakeRelay()
    service = build_async_service(relay)
    caps = await service.get_server_capabilities_async()
    assert caps.find("Qwen/Qwen3.8-27B").max_context_length == 131072
    await service.aclose()


@pytest.mark.asyncio
async def test_close_async_finishes_session_and_closes_transport():
    relay = FakeRelay()
    service = build_async_service(relay)
    session_id = service.session_id
    await service.aclose()
    assert relay.count(f"/api/mindrsi/v1/service/session/{session_id}/finish") == 1
    with pytest.raises(mind_recursive.MindRecursiveError):
        await service.get_server_capabilities_async()


@pytest.mark.asyncio
async def test_async_state_round_trip():
    relay = FakeRelay()
    service = build_async_service(relay)
    trainer = await service.create_lora_trainer_from_state_async(
        "mint://run-abc/checkpoints/step-500",
        base_model="Qwen/Qwen3.8-27B",
    )
    assert trainer.model_id == "sess-0001_0"

    saved = await trainer.save_state_async("state-async")
    assert saved
    await trainer.load_state_async(saved)
    await trainer.load_state_with_optimizer_async(saved)
    await service.aclose()


@pytest.mark.asyncio
async def test_async_heartbeat_runs_on_the_event_loop():
    relay = FakeRelay()
    service = build_async_service(relay, heartbeat_seconds=0.05)
    session_id = service.session_id
    await service.get_server_capabilities_async()
    await asyncio.sleep(0.16)
    assert (
        relay.count(f"/api/mindrsi/v1/service/session/{session_id}/heartbeat") >= 1
    )
    await service.aclose()
