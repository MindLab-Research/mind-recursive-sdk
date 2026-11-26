"""Tinker-named symbols, for porting existing Tinker or cookbook code.

Change the import line and the rest of the file usually compiles::

    # before
    import tinker
    # after
    from mind_recursive import tinker

The native names are ``LoraTrainer`` / ``Sampler`` and
``ServiceClient.create_lora_trainer`` / ``create_sampler``; this module exposes
the Tinker spellings as aliases of exactly those objects, so the two namespaces
never drift.

What carries over: method names and call shapes for the covered training and
sampling surface. The shells are deliberately thin:

* creation calls return the native object directly, and also expose
  ``result()`` (returning itself) so upstream's ``create_...(...).result()``
  style works;
* ``close(status, detail)`` accepts the upstream call shape and returns a
  completed future-shaped object -- the relay's finish endpoint carries no
  fields, so the reason is not transmitted;
* ``TrainingClient.forward``, ``optim_step(optim_params=...)``,
  ``Sampler.compute_logprobs`` and positional ``save_state`` /
  ``save_weights_for_sampler`` / ``create_sampling_client`` are supported as
  shells over the native implementation;
* ``save_weights_and_get_sampling_client()`` is an ephemeral export, as
  upstream: no ``path`` is sent, the server overwrites one fixed
  ``ephemeral`` copy per training model (24h TTL), and
  ``name`` is ignored with a ``DeprecationWarning``. ``kind='inflight'`` skips
  the export.

The save surface matches upstream's shape -- two named saves rather than one
with a boolean: :meth:`~mind_recursive.LoraTrainer.save_weights_only` (weights alone,
upstream's ``save_weights_for_sampler``, kept as an alias) and
:meth:`~mind_recursive.LoraTrainer.save_state` (weights plus optimizer state, the
upstream name as-is). Save futures resolve to the path string; it is also a
``str`` subclass with a ``.path`` property, so upstream's response-object
access still reads correctly.

Loading differs on purpose. Upstream's ``load_state`` means *weights only*,
while our :meth:`~mind_recursive.LoraTrainer.load_weights` defaults to restoring the
optimizer too, so ``load_state`` and ``load_state_with_optimizer`` are real
methods here rather than aliases: aliasing them would make ported code
silently resume optimizer state it asked not to. Both are deprecated upstream
in favour of ``create_training_client_from_state``.

What does not carry over, and is not faked:

* type identity: ``isinstance(d, tinker.Datum)`` against the real package is
  false, and ``except tinker.TinkerError`` must become
  ``except mind_recursive.MindRecursiveError``;
* async twins are provided with the same names as upstream (``forward_backward_async``,
  ``sample_async``, ...), backed by a native ``httpx.AsyncClient``;
* ``RestClient`` / ``AdminClient``, custom losses (``create_custom_loss`` /
  ``forward_backward_custom``), external weights export, console URLs,
  telemetry, custom-loss handles, and ``get_tokenizer`` (this SDK keeps tokenisation
  in the recipe layer);
* non-Adam optimizers -- ``optim_params`` accepts our ``AdamParams`` only;
* ``retry_config`` arguments -- retry policy lives on ``ServiceClient`` and is
  rejected rather than silently ignored.

This is the accepted cost of owning the client.

Tinker's ``RestClient`` has a counterpart in spirit --
:class:`mind_recursive.UserClient`, which likewise needs no session -- but not in
name, because the method sets differ: ``list_training_runs``,
``copy_weights`` and ``get_audit_log`` have no equivalent here, and a shared
name with different contents is worse for porting than a new one. It is not
aliased below for the same reason.

Deliberately absent: ``RestClient`` and ``AdminClient``. Through the relay
their list endpoints return *global* resources rather than the caller's --
a shared upstream key makes that unavoidable server-side (see
plan-b-training-via-relay.md). Not implementing them keeps that surface out of
users' hands; Fireworks omits them for the same reason.
"""

from __future__ import annotations

from ._errors import MindRecursiveError as TinkerError
from ._futures import APIFuture
from .service import ServiceClient as _ServiceClient, _json_object
from .trainer import LoraTrainer as TrainingClient
from .sampler import Sampler as SamplingClient
from .types import ServerCapabilities, SupportedModel


from . import types
from .types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ImageChunk,
    LoraConfig,
    ModelInput,
    SampleResponse,
    SamplingParams,
    TopkLogprobs,
    TensorData,
)


def _parse_capabilities(payload: dict) -> ServerCapabilities:
    return ServerCapabilities(
        supported_models=[
            SupportedModel(
                model_name=item.get("model_name"),
                max_context_length=item.get("max_context_length"),
                trainable=item.get("trainable"),
                sampleable=item.get("sampleable"),
            )
            for item in payload.get("supported_models") or []
        ]
    )
class ServiceClient(_ServiceClient):
    """Native client plus the Tinker-only capability reads.

    The native client deliberately keeps its own shape; methods that exist only
    in Tinker's client live here so ported code keeps compiling.
    """

    def get_server_capabilities(self):
        """Tinker's ``ServiceClient.get_server_capabilities`` (sync)."""
        from . import _protocol as P
        from ._transport import JSON_CONTENT_TYPE

        response = self.transport.request(
            "GET",
            P.PATH_CAPABILITIES,
            base_url=self.config.service_base_url,
            extra_headers=self._headers(),
            accept=JSON_CONTENT_TYPE,
        )
        return _parse_capabilities(_json_object(response))

    async def get_server_capabilities_async(self):
        """Tinker's ``ServiceClient.get_server_capabilities_async``."""
        from . import _protocol as P
        from ._transport import JSON_CONTENT_TYPE

        response = await self.async_transport.request(
            "GET",
            P.PATH_CAPABILITIES,
            base_url=self.config.service_base_url,
            extra_headers=self._headers(),
            accept=JSON_CONTENT_TYPE,
        )
        return _parse_capabilities(_json_object(response))


__all__ = [
    "ServiceClient",
    "TrainingClient",
    "SamplingClient",
    "APIFuture",
    "TinkerError",
    "types",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ImageChunk",
    "LoraConfig",
    "ModelInput",
    "SampleResponse",
    "SamplingParams",
    "TopkLogprobs",
    "TensorData",
]
