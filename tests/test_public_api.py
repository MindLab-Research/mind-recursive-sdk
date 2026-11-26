"""The shape of the public API: one client, three things created from it.

These tests exist because the shape is a promise, not an implementation
detail. A trainer, sampler or reporter that grew its own transport, or a
second top-level client, would break the single-cleanup guarantee the README
makes without breaking any behavioural test.
"""

from __future__ import annotations

import inspect
from typing import get_args

import pytest

import mind_recursive
from mind_recursive import tinker
from tests.fake_relay import FakeRelay, build_service


@pytest.fixture
def service():
    client = build_service(FakeRelay())
    yield client
    client.close()


class TestOneClient:
    def test_the_only_exported_clients_are_the_two_halves(self):
        """Exactly two clients, split by whether the work needs a session.

        ``ServiceClient`` creates and holds one; ``UserClient`` creates
        nothing. Any third client name on the surface means the split grew a
        case neither half covers, which is worth noticing.
        """
        clients = [n for n in mind_recursive.__all__ if n.endswith("Client")]
        assert sorted(clients) == ["ServiceClient", "UserClient"]

    def test_session_bound_work_is_not_on_the_user_client(self):
        """The split is about what the call operates on, not convenience."""
        for name in ("create_lora_trainer", "create_sampler", "log", "metric"):
            assert not hasattr(mind_recursive.UserClient, name)

    def test_session_less_queries_are_not_on_the_service_client(self):
        """Asking what the backend serves must not require a billable session."""
        for name in ("get_server_capabilities", "health_check"):
            assert callable(getattr(mind_recursive.UserClient, name))
            assert not hasattr(mind_recursive.ServiceClient, name)

    def test_message_level_inference_is_not_part_of_the_sdk(self):
        """Removed deliberately: generic LLM calls are out of scope."""
        assert not hasattr(mind_recursive, "InferenceClient")
        with pytest.raises(ImportError):
            from mind_recursive import inference  # noqa: F401

    def test_rendering_is_not_an_sdk_concern(self):
        """Messages -> tokens depends on chat templates and dataset policy.

        The SDK deals in Datum. Rendering lives with the recipes/cookbook, so
        no renderer may reappear on the package surface.
        """
        assert not hasattr(mind_recursive, "renderers")
        with pytest.raises(ImportError):
            from mind_recursive import renderers  # noqa: F401
        assert "transformers" not in str(mind_recursive.__all__)

    def test_the_two_creators_exist_on_the_service_client(self):
        for name in ("create_lora_trainer", "create_sampler"):
            assert callable(getattr(mind_recursive.ServiceClient, name))

    def test_reporting_is_on_the_client_not_a_handle(self):
        """One writer per session, so there is nothing to create."""
        for name in ("log", "event", "metric", "artifact", "summary"):
            assert callable(getattr(mind_recursive.ServiceClient, name))
        assert not hasattr(mind_recursive.ServiceClient, "create_reporter")

    def test_created_objects_share_the_service_transport(self, service):
        """The point of the abstraction: no handle opens its own connection."""
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        sampler = service.create_sampler(base_model="Qwen/Qwen3.8-27B")

        for handle in (trainer, sampler):
            assert handle._service is service
            assert not hasattr(handle, "_transport")

    def test_closing_the_service_client_is_the_only_cleanup(self, service):
        """No handle may carry its own close(); there is one lifecycle."""
        trainer = service.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
        assert not hasattr(trainer, "close")


class TestTinkerAliases:
    """``mind_recursive.tinker`` is where Tinker-name parity lives, and only there."""

    def test_class_aliases_are_the_native_classes(self):
        assert tinker.TrainingClient is mind_recursive.LoraTrainer
        assert tinker.SamplingClient is mind_recursive.Sampler

    def test_method_shells_delegate_to_the_native_methods(self):
        sc = mind_recursive.ServiceClient
        assert sc.create_lora_training_client is sc.create_lora_trainer
        # create_sampling_client is a thin wrapper, not an alias: its Tinker
        # signature has a positional retry_config slot that has to be rejected
        # explicitly. What matters is that it builds a native Sampler.

    def test_tinker_names_do_not_leak_into_the_native_namespace(self):
        assert not hasattr(mind_recursive, "TrainingClient")
        assert not hasattr(mind_recursive, "SamplingClient")

    def test_an_alias_really_builds_a_native_object(self, service):
        trainer = service.create_lora_training_client(base_model="Qwen/Qwen3.8-27B")
        assert isinstance(trainer, mind_recursive.LoraTrainer)

    def test_sampling_shell_builds_a_native_sampler(self, service):
        sampler = service.create_sampling_client(base_model="Qwen/Qwen3.8-27B")
        assert isinstance(sampler, mind_recursive.Sampler)
        assert sampler.result() is sampler

class TestTypedParameters:
    """The Literals that make a typo a type error instead of a bad train.

    ``loss_fn`` is the sharp one: a wrong spelling is not an exception, it is
    one silently mistrained step, because the name goes straight to the server
    and the GRPO recipe depends on the exact ``importance_sampling``.
    """

    def test_literals_are_reachable_from_the_package(self):
        """A caller annotating their own helper needs these names."""
        for name in ("LossFnType", "SamplerKind", "TensorDtype"):
            assert name in mind_recursive.__all__
            assert hasattr(mind_recursive, name)

    def test_loss_fn_is_typed_not_bare_str(self):
        annotation = inspect.signature(
            mind_recursive.LoraTrainer.forward_backward
        ).parameters["loss_fn"].annotation
        assert annotation != "str"
        assert "LossFnType" in str(annotation)

    def test_sampler_kind_is_typed_on_both_entry_points(self):
        for owner in (mind_recursive.ServiceClient, mind_recursive.LoraTrainer):
            annotation = inspect.signature(
                owner.create_sampler
            ).parameters["kind"].annotation
            assert "SamplerKind" in str(annotation), owner

    def test_sampler_kind_literal_matches_the_runtime_tuple(self):
        """The annotation and the ValueError must not drift apart."""
        import mind_recursive._protocol as protocol

        assert get_args(mind_recursive.SamplerKind) == protocol.SAMPLER_KINDS


class TestUpstreamSpellings:
    """``mind_recursive.tinker`` promises ported code compiles; absences break that.

    Both Tinker and Fireworks name these endpoints ``save_state`` /
    ``load_state`` and have no ``save_weights`` at all, so ported code calls
    the names we lacked. An alias table cannot show a missing name, which is
    why this is a test.
    """

    def test_the_four_upstream_names_exist(self):
        """The save/load surface both Tinker and Fireworks expose."""
        for name in (
            "save_state",
            "save_weights_for_sampler",
            "load_state",
            "load_state_with_optimizer",
        ):
            assert hasattr(mind_recursive.LoraTrainer, name), name

    def test_the_upstream_save_spelling_is_an_alias(self):
        """Same function object, so the two names cannot drift."""
        assert mind_recursive.LoraTrainer.save_weights_for_sampler is (
            mind_recursive.LoraTrainer.save_weights_only
        )

    def test_the_boolean_switch_is_not_public(self):
        """Two named saves, not one method with ``train_state=``.

        A boolean at the call site says nothing about which artefact comes
        back, and the two differ in where the server writes them and in what
        the result can be used for.
        """
        assert not hasattr(mind_recursive.LoraTrainer, "save_weights")
        assert not hasattr(mind_recursive.LoraTrainer, "save_training_state")

    def test_load_state_is_not_an_alias_because_the_default_differs(self):
        """Upstream ``load_state`` means weights only; ours defaults to both.

        Aliasing them would make ported code silently restore optimizer state
        it asked not to -- the kind of mistake that shows up as a training
        curve that does not resume cleanly, not as an error.
        """
        assert mind_recursive.LoraTrainer.load_state is not mind_recursive.LoraTrainer.load_weights


    def test_base_model_accepts_the_upstream_positional_call(self):
        """``create_lora_training_client("Qwen/...")`` is the common form."""
        parameter = inspect.signature(
            mind_recursive.ServiceClient.create_lora_trainer
        ).parameters["base_model"]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


class TestModelInfo:
    """``get_info`` returns a type, so callers stop guessing key names."""

    def test_names_the_fields_the_server_sends(self):
        info = mind_recursive.ModelInfo.from_wire(
            {
                "model_id": "run_0",
                "model_name": "Qwen/Qwen3.8-27B",
                "is_lora": True,
                "lora_rank": 32,
                "type": "get_info",
            }
        )
        assert info.model_id == "run_0"
        assert info.lora_rank == 32
        assert info.is_lora is True

    def test_keeps_unmodelled_keys_instead_of_dropping_them(self):
        """A server that grows a field must not lose it or raise."""
        info = mind_recursive.ModelInfo.from_wire({"model_id": "x", "future_field": 7})
        assert info.extra == {"future_field": 7}

    def test_drops_the_envelope_discriminator(self):
        """``type`` describes the response, not the model."""
        info = mind_recursive.ModelInfo.from_wire({"model_id": "x", "type": "get_info"})
        assert "type" not in info.extra

    def test_absent_fields_read_as_unsaid_not_as_error(self):
        info = mind_recursive.ModelInfo.from_wire({})
        assert info.lora_rank is None
        assert info.model_id is None

    def test_rank_is_coerced_to_int(self):
        """The wire may spell a number as a string; the attribute is an int."""
        assert mind_recursive.ModelInfo.from_wire({"lora_rank": "16"}).lora_rank == 16
