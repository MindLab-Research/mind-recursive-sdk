"""``create_lora_trainer_from_state``: one submission instead of two.

The point of the method is that no second step exists: the server creates the
model and loads the checkpoint together. These tests assert on the request the
SDK actually sends, because "one call instead of create + load" is a wire
claim, not an API-shape claim.
"""

from __future__ import annotations

import pytest

import mind_recursive
from mind_recursive._errors import ConfigurationError
from tests.fake_relay import FakeRelay, build_service

TRAIN = "/api/mindrsi/v1/train"
FROM_STATE = "/api/mindrsi/v1/service/train/create_model_from_state"
CHECKPOINT = "mint://run-abc/checkpoints/step-500"
BASE = "Qwen/Qwen3.8-27B"


@pytest.fixture
def relay() -> FakeRelay:
    return FakeRelay()


@pytest.fixture
def service(relay: FakeRelay):
    client = build_service(relay, client_id="client-xyz")
    yield client
    client.close()


def _body(relay: FakeRelay) -> dict:
    return relay.last(FROM_STATE).json()


class TestWire:
    def test_one_request_replaces_create_and_load(self, service, relay):
        trainer = service.create_lora_trainer_from_state(CHECKPOINT, base_model=BASE)

        assert trainer.model_id == f"{service.session_id}_0"
        assert trainer.base_model == BASE
        assert relay.count(FROM_STATE) == 1
        assert relay.count("/api/mindrsi/v1/service/train/create_model") == 0
        assert relay.count(f"{TRAIN}/load_weights") == 0

    def test_inherited_structure_is_not_invented_locally(self, service, relay):
        """"Use what the checkpoint has" must mean absent keys, not defaults."""
        trainer = service.create_lora_trainer_from_state(CHECKPOINT, base_model=BASE)

        assert _body(relay) == {
            "session_id": service.session_id,
            "model_seq_id": 0,
            "state_path": CHECKPOINT,
            "base_model": BASE,
            "type": "create_model_from_state",
        }
        # The effective structure lives on the server; a locally filled
        # LoraConfig would be a guess.
        assert trainer.lora_config is None

    def test_explicit_structure_is_forwarded_verbatim(self, service, relay):
        trainer = service.create_lora_trainer_from_state(
            CHECKPOINT,
            base_model=BASE,
            rank=8,
            train_mlp=False,
            train_attn=True,
            load_optimizer=False,
            user_metadata={"run": "sft-1"},
        )

        body = _body(relay)
        assert body["lora_config"] == {
            "rank": 8,
            "train_mlp": False,
            "train_attn": True,
        }
        assert body["load_optimizer"] is False
        assert body["user_metadata"] == {"run": "sft-1"}
        assert trainer.lora_config == mind_recursive.LoraConfig(
            rank=8, train_mlp=False, train_attn=True
        )

    def test_partial_structure_stays_partial(self, service, relay):
        """Only what the caller named goes out; the rest is the server's."""
        trainer = service.create_lora_trainer_from_state(
            CHECKPOINT, base_model=BASE, rank=8
        )

        assert _body(relay)["lora_config"] == {"rank": 8}
        assert "load_optimizer" not in _body(relay)
        # Not every field was named, so the trainer cannot claim a full config.
        assert trainer.lora_config is None


class TestLocalValidation:
    """Bad arguments fail at the call site, before a request is sent."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"path": ""},
            {"path": 123},
            {"base_model": ""},
            {"rank": 0},
            {"rank": -1},
            {"rank": True},
            {"rank": 1.5},
            {"train_mlp": 1},
            {"train_attn": "yes"},
            {"load_optimizer": 0},
        ],
    )
    def test_invalid_arguments_send_nothing(self, service, relay, kwargs):
        args = {"path": CHECKPOINT, "base_model": BASE, **kwargs}
        with pytest.raises(ConfigurationError):
            service.create_lora_trainer_from_state(**args)
        assert relay.count(FROM_STATE) == 0

    def test_sdk_does_not_keep_a_model_registry(self, service, relay):
        """Serving-ability is the backend's call; the SDK forwards the name."""
        trainer = service.create_lora_trainer_from_state(
            CHECKPOINT, base_model="not/a-real-model"
        )
        assert trainer.base_model == "not/a-real-model"
        assert relay.count(FROM_STATE) == 1
