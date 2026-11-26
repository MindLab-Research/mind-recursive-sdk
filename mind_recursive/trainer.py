"""``LoraTrainer``: the training loop primitives.

The step is ``forward_backward`` then ``optim_step``, each returning a future.
Three details are not optional:

**Protobuf.** ``forward_backward`` has no JSON path upstream; the body is
protobuf and may be zstd-compressed when the server's ``client/config`` says so.

**Chunking.** ``client/config`` returns ``fwdbwd_max_chunk_len`` and
``fwdbwd_max_chunk_bytes_count``. Exceeding either earns a 413, so a batch
larger than the budget is split and the per-chunk metrics are recombined.

**Adam parameters.** The backend accepts exactly one set and rejects anything
else with ``UNSUPPORTED: FSDP2 requires beta1=0.9, beta2=0.999, eps=1e-8 and
lora_plus_ratio=1``. Our defaults are those values, so a caller who only sets
a learning rate cannot trip it.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any, Callable, Sequence

from . import _protocol as P
from ._errors import ConfigurationError
from ._futures import APIFuture, FutureResolver
from ._proto import encode_forward_backward_request
from .types import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    LoraConfig,
    LossFnType,
    ModelInfo,
    SamplerKind,
)

__all__ = ["LoraTrainer"]

log = logging.getLogger("mind_recursive.trainer")


class _ArtifactFuture(APIFuture[str]):
    """A save future that reports the finished path as an artifact.

    An artifact is a fact about a save, not a separate user action: the
    trainer already knows the path and the kind, so making the caller repeat
    them is an invitation to forget. The hook fires once, on the first
    successful ``result()``; a failed save reports nothing.
    """

    __slots__ = ("_report", "_reported")

    def __init__(
        self,
        request_id: str,
        resolver: FutureResolver,
        parse: Callable[[dict[str, Any] | bytes], str],
        report: Callable[[str], None],
    ) -> None:
        super().__init__(request_id, resolver, parse)
        self._report = report
        self._reported = False

    def result(self, timeout: float | None = None) -> str:
        value = super().result(timeout)
        with self._lock:
            if self._reported:
                return value
            self._reported = True
        try:
            self._report(value)
        except Exception as exc:  # noqa: BLE001 - reporting must not break a save
            log.warning("reporting the saved artifact failed (ignored): %s", exc)
        return value


#: Prefix for a training model's ephemeral sampler exports. Each
#: save_weights_and_get_sampling_client() saves ``ephemeral-<seq>``; a named
#: weights-only save must not use this prefix.
EPHEMERAL_SAMPLER_NAME = "ephemeral"

#: TTL of the ephemeral export. The server expires the throwaway weights after
#: this, so a forgotten sampler does not pin storage forever.
EPHEMERAL_SAMPLER_TTL_SECONDS = 86400


def _checkpoint_name(name: str, method: str) -> str:
    """Validate a checkpoint name the way the server does.

    The server rejects sampler saves whose ``path`` is not a non-empty
    URL-safe name (``valid_checkpoint_name`` in the server's enqueue path).
    Validating here turns a rejected enqueue into a local error at the call
    site, before any request is sent.
    """
    if not isinstance(name, str) or not name:
        raise ConfigurationError(f"{method} requires a non-empty checkpoint name")
    if name in (".", "..") or not all(
        ch.isascii() and (ch.isalnum() or ch in "-_.") for ch in name
    ):
        raise ConfigurationError(
            f"{method} checkpoint name must be URL-safe "
            "(ASCII letters, digits, '-', '_' or '.'), got {name!r}"
        )
    return name


def _maybe_compress(body: bytes) -> tuple[bytes, str | None]:
    """zstd-compress when the extra is installed.

    Compression is requested by the server, but the codec is an optional
    dependency. Rather than fail a training run over a missing extra, we warn
    once and send the body uncompressed -- the server accepts both; only the
    bandwidth differs.
    """
    try:
        import zstandard as zstd
    except ImportError:
        log.warning(
            "server asked for zstd-compressed forward_backward bodies but "
            "'zstandard' is not installed; sending uncompressed "
            "(pip install 'mind_recursive[zstd]')"
        )
        return body, None
    return zstd.ZstdCompressor().compress(body), "zstd"


class LoraTrainer:
    """A training model inside a session.

    Obtained from :meth:`mind_recursive.ServiceClient.create_lora_trainer` or
    :meth:`mind_recursive.ServiceClient.create_lora_trainer_from_state`, not
    constructed directly.
    """

    def __init__(
        self,
        *,
        service: Any,
        model_id: str,
        base_model: str,
        lora_config: LoraConfig | None,
    ) -> None:
        self._service = service
        self._model_id = model_id
        self._base_model = base_model
        self._lora_config = lora_config
        # seq_id must be >= 1: a 0 trips the server's range check.
        self._seq = 0

    # -- introspection --------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_model(self) -> str:
        return self._base_model

    @property
    def lora_config(self) -> LoraConfig | None:
        """The requested structure, or ``None`` when the server resolves it.

        ``create_lora_trainer_from_state`` leaves the structure to the
        checkpoint unless the caller passes every field, so this is ``None``
        for a partially specified or fully inherited trainer. Use
        ``get_info()`` for the effective rank.
        """
        return self._lora_config

    @property
    def session_id(self) -> str | None:
        return self._service.session_id

    def result(self, timeout: float | None = None) -> LoraTrainer:
        """Tinker-style future compatibility: return this trainer.

        ``ServiceClient.create_lora_training_client`` returns a trainer
        directly; upstream returns a future of one. Model creation already
        blocked by the time this object exists, so ``.result()`` is the
        identity here.
        """
        return self

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # -- the step -------------------------------------------------------------
    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: dict[str, float | str] | None = None,
        *,
        forward_only: bool = False,
    ) -> APIFuture[ForwardBackwardOutput]:
        """Run a forward and backward pass, accumulating gradients.

        A batch exceeding the negotiated chunk budget is split transparently;
        the returned future then resolves to the combined result.

        Args:
            data: the batch.
            loss_fn: server-side loss name, e.g. ``cross_entropy``,
                ``importance_sampling``, ``ppo``.
            loss_fn_config: optional loss parameters.
            forward_only: forward pass only, no gradient accumulation.

        Returns:
            A future resolving to :class:`~mind_recursive.types.ForwardBackwardOutput`.
        """
        batch = list(data)
        if not batch:
            raise ValueError("forward_backward needs at least one datum")

        chunks = self._split_batch(batch)
        if len(chunks) == 1:
            return self._submit_forward_backward(
                chunks[0], loss_fn, loss_fn_config, forward_only
            )

        log.debug(
            "splitting %d datums into %d chunks for the negotiated budget",
            len(batch),
            len(chunks),
        )
        futures = [
            self._submit_forward_backward(chunk, loss_fn, loss_fn_config, forward_only)
            for chunk in chunks
        ]
        return _CombinedForwardBackward(futures)  # type: ignore[return-value]

    def forward(
        self,
        data: Sequence[Datum],
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: dict[str, float | str] | None = None,
    ) -> APIFuture[ForwardBackwardOutput]:
        """A forward pass without gradients, Tinker-style.

        Same request and same return type as :meth:`forward_backward`; the
        only difference is ``forward_only=True``. Kept as a named method
        because ported Tinker code calls it directly and the intent reads
        better than the flag.
        """
        return self.forward_backward(
            data,
            loss_fn,
            loss_fn_config=loss_fn_config,
            forward_only=True,
        )

    def _split_batch(self, batch: list[Datum]) -> list[list[Datum]]:
        """Split a batch to respect ``fwdbwd_max_chunk_{len,bytes_count}``.

        Datum size is measured by encoding it, which is exact rather than
        estimated. A single datum over the byte budget is passed through
        whole: splitting one example would change the training semantics, so
        the server's 413 is the honest outcome.
        """
        max_len, max_bytes = self._service.client_config.chunk_budget()
        chunks: list[list[Datum]] = []
        current: list[Datum] = []
        current_len = 0
        current_bytes = 0

        for datum in batch:
            size = len(self._encode_datum_for_sizing(datum))
            tokens = max(datum.model_input.length, 1)
            too_many = max_len > 0 and current_len + tokens > max_len
            too_big = max_bytes > 0 and current_bytes + size > max_bytes
            if current and (too_many or too_big):
                chunks.append(current)
                current, current_len, current_bytes = [], 0, 0
            current.append(datum)
            current_len += tokens
            current_bytes += size

        if current:
            chunks.append(current)
        return chunks or [batch]

    @staticmethod
    def _encode_datum_for_sizing(datum: Datum) -> bytes:
        from ._proto.encode import encode_datum

        return encode_datum(datum)

    def _submit_forward_backward(
        self,
        batch: list[Datum],
        loss_fn: LossFnType,
        loss_fn_config: dict[str, float | str] | None,
        forward_only: bool,
    ) -> APIFuture[ForwardBackwardOutput]:
        body = encode_forward_backward_request(
            model_id=self._model_id,
            seq_id=self._next_seq(),
            data=batch,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            forward_only=forward_only,
        )
        encoding: str | None = None
        if self._service.client_config.proto_compress_fwdbwd:
            body, encoding = _maybe_compress(body)

        payload = self._service.post_proto(
            P.PATH_FORWARD_BACKWARD, body, content_encoding=encoding
        )
        request_id = payload.get("request_id")
        if not request_id:
            raise ConfigurationError(
                f"forward_backward returned no request_id: {payload!r}"
            )
        return APIFuture(
            str(request_id),
            self._service.resolver(),
            P.parse_forward_backward_output,
        )

    def optim_step(
        self,
        optim_params: AdamParams | None = None,
        *,
        adam_params: AdamParams | None = None,
    ) -> APIFuture[dict[str, Any]]:
        """Apply the accumulated gradients.

        Args:
            optim_params: Tinker's spelling. This SDK supports Adam only;
                non-Adam optimizer params have no wire representation here and
                are rejected locally.
            adam_params: legacy keyword alias for the same value. Passing both
                raises, matching upstream.

        Defaults satisfy the backend's FSDP2 constraint, so ``optim_step()``
        always passes validation. A violating override is sent as given, with a
        warning -- we do not silently rewrite a caller's hyperparameters.

        Returns:
            A future resolving to the server's response (carries ``grad_norm``
            when the server reports it).
        """
        if optim_params is not None and adam_params is not None:
            raise ValueError("pass optim_params or adam_params, not both")
        params = adam_params if adam_params is not None else optim_params
        if params is None:
            params = AdamParams()
        if not isinstance(params, AdamParams):
            raise TypeError(
                "mind_recursive supports AdamParams only; a non-Adam optimizer has no "
                "wire representation in this SDK"
            )
        violations = params.violations()
        if violations:
            detail = ", ".join(
                f"{name}={given!r} (backend requires {required!r})"
                for name, (given, required) in violations.items()
            )
            log.warning(
                "optim_step Adam parameters will be rejected by the backend: %s", detail
            )

        request_id = self._service._submit(
            P.PATH_OPTIM_STEP,
            {
                "adam_params": params.to_wire(),
                "model_id": self._model_id,
                "seq_id": self._next_seq(),
                "type": "optim_step",
            },
        )
        return APIFuture(request_id, self._service.resolver(), P.parse_optim_step)

    # -- weights --------------------------------------------------------------
    def _save(
        self,
        name: str,
        *,
        train_state: bool,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
    ) -> APIFuture[str]:
        """Shared body of the two public saves; ``train_state`` picks the endpoint.

        Private because the public surface is the upstream one -- two named
        methods, :meth:`save_weights_only` and :meth:`save_state` --
        rather than one method with a boolean. A boolean at the call site says
        nothing about which artefact comes back, and the two differ in where
        the server writes them and in what you can do with the result.

        ``name`` identifies the artefact within this training run and must be
        URL-safe: ASCII letters, digits, ``-``, ``_`` or ``.``. A name that
        already exists is a conflict unless ``overwrite`` is set. A
        weights-only save may not be named ``ephemeral``: the server keeps each
        training model's ephemeral export under that name and overwrites it on
        every :meth:`save_weights_and_get_sampling_client`.

        When the returned future resolves, its path is reported as an artifact
        (``sampler_weights`` or ``checkpoint``), carrying ``user_metadata`` if
        one was given. Reporting is best-effort and never turns a successful
        save into a failure.
        """
        if user_metadata is not None and not isinstance(user_metadata, dict):
            # save_weights_only's third positional is user_metadata, as
            # upstream; save_state's is overwrite. Copying one call shape onto
            # the other would otherwise send a bool as metadata.
            raise ConfigurationError(
                "user_metadata must be a dict of strings, got "
                f"{type(user_metadata).__name__}; pass overwrite= by keyword"
            )
        if not train_state and (
            name == EPHEMERAL_SAMPLER_NAME
            or name.startswith(f"{EPHEMERAL_SAMPLER_NAME}-")
        ):
            raise ConfigurationError(
                f"the {EPHEMERAL_SAMPLER_NAME!r} prefix is reserved for the "
                "ephemeral export of save_weights_and_get_sampling_client(); "
                "pick another name"
            )

        if train_state:
            body: dict[str, Any] = {
                "model_id": self._model_id,
                "seq_id": self._next_seq(),
                "overwrite": overwrite,
                "path": _checkpoint_name(name, "save_weights"),
                "type": "save_weights",
            }
            endpoint = P.PATH_SAVE_WEIGHTS
        else:
            body = {
                "model_id": self._model_id,
                "seq_id": self._next_seq(),
                "kind": "snapshot",
                "overwrite": overwrite,
                "path": _checkpoint_name(name, "save_weights"),
                "type": "save_weights_for_sampler",
            }
            endpoint = P.PATH_SAVE_WEIGHTS_FOR_SAMPLER

        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if user_metadata:
            body["user_metadata"] = user_metadata
        request_id = self._service._submit(endpoint, body)
        artifact_type = "checkpoint" if train_state else "sampler_weights"
        return _ArtifactFuture(
            request_id,
            self._service.resolver(),
            P.parse_weights_path,
            lambda path: self._service.artifact(
                path,
                type=artifact_type,
                name=name,
                metadata=dict(user_metadata) if user_metadata else None,
            ),
        )

    def save_weights_only(
        self,
        name: str,
        ttl_seconds: int | None = None,
        user_metadata: dict[str, str] | None = None,
        *,
        overwrite: bool = False,
    ) -> APIFuture[str]:
        """Save the weights alone, for sampling or deployment.

        The server writes them under ``mint://.../adapters/...`` and the
        result can be handed straight to
        :meth:`~mind_recursive.ServiceClient.create_sampler` or deployed. **This is
        the artefact most runs want.**

        One of the two saves. Upstream spells this one
        ``save_weights_for_sampler``, kept here as an alias; ``_only`` says
        what you get rather than what you might do with it, which is the part
        that matters when the other save also produces usable weights.

        This is the persistent save: pass a name to keep the weights. Reusing a
        name is a conflict unless ``overwrite=True``. ``ephemeral`` is reserved
        and raises locally.
        """
        return self._save(
            name,
            train_state=False,
            ttl_seconds=ttl_seconds,
            overwrite=overwrite,
            user_metadata=user_metadata,
        )

    def save_state(
        self,
        name: str,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
    ) -> APIFuture[str]:
        """Save weights *and* optimizer state, for resuming training.

        Written under ``mint://.../checkpoints/...``. Only this product can
        start a training resume -- hand it to
        :meth:`~mind_recursive.ServiceClient.create_lora_trainer_from_state`, or to
        :meth:`load_state_with_optimizer` to load it into a live model.
        A weights-only path cannot: the server rejects it as a training
        source.

        The other of the two saves, named as both upstreams name it.
        """
        return self._save(
            name,
            train_state=True,
            ttl_seconds=ttl_seconds,
            overwrite=overwrite,
            user_metadata=user_metadata,
        )

    def load_weights(
        self, path: str, *, optimizer: bool = True
    ) -> APIFuture[dict[str, Any]]:
        """Load weights from a ``mint://`` path into this model.

        Args:
            path: a checkpoint path returned by :meth:`save_weights`.
            optimizer: also restore optimizer state. Defaults to ``True``,
                matching the server's own absent-field default, because
                resuming training wants it.

                Note this is *not* upstream's default: Tinker splits the two
                cases into ``load_state`` (weights only) and
                ``load_state_with_optimizer``, and both of those are
                deprecated there. Those two names exist here as thin methods
                so ported code keeps its meaning -- aliasing ``load_state``
                to this one would silently restore the optimizer.
        """
        request_id = self._service._submit(
            P.PATH_LOAD_WEIGHTS,
            {
                "model_id": self._model_id,
                "path": path,
                "seq_id": self._next_seq(),
                "optimizer": bool(optimizer),
                "type": "load_weights",
            },
        )
        return APIFuture(request_id, self._service.resolver(), P.parse_load_weights)

    def load_state(self, path: str) -> APIFuture[dict[str, Any]]:
        """Load the weights alone, leaving the optimizer as it is.

        Upstream's ``load_state`` means *weights only*, so this cannot be an
        alias of :meth:`load_weights`, which defaults to restoring the
        optimizer too: aliasing them would make ported code silently resume
        optimizer state it asked not to.

        Both this and :meth:`load_state_with_optimizer` are deprecated
        upstream in favour of
        :meth:`~mind_recursive.ServiceClient.create_lora_trainer_from_state`, which
        starts a fresh model from a checkpoint instead of mutating a live one.
        They exist here so ported code keeps its meaning.
        """
        return self.load_weights(path, optimizer=False)

    def load_state_with_optimizer(self, path: str) -> APIFuture[dict[str, Any]]:
        """Load weights and optimizer state -- the resume case.

        Upstream's explicit spelling for what :meth:`load_weights` does by
        default. See :meth:`load_state` for why these two are methods rather
        than aliases.
        """
        return self.load_weights(path, optimizer=True)

    def get_info(self) -> ModelInfo:
        """The server's record of this model.

        Synchronous -- this is the one training endpoint that does not go
        through a future. Unmodelled keys stay in ``ModelInfo.extra``, so a
        server that grows a field does not break the call.
        """
        payload = self._service._post_json(
            P.PATH_GET_INFO, {"model_id": self._model_id, "type": "get_info"}
        )
        return ModelInfo.from_wire(payload)

    # -- sampling from this model --------------------------------------------
    def create_sampler(
        self,
        *,
        model_path: str | None = None,
        kind: SamplerKind = "snapshot",
        record_stability_info: bool = False,
    ) -> Any:
        """Create a sampler for this training model's weights.

        ``kind="snapshot"`` needs a ``model_path`` from :meth:`save_weights`:
        fixed weights, so samples stay comparable across steps.
        ``kind="inflight"`` binds to this live model instead and needs no path
        -- samples follow the weights as training steps, which is what an
        on-policy loop wants and what saving a checkpoint per step is a
        workaround for.
        """
        if kind == "snapshot" and not model_path:
            raise ValueError(
                "kind='snapshot' samples fixed weights, so it needs a "
                "model_path from save_weights_only() -- or pass kind='inflight' "
                "to "
                "sample this model's current weights"
            )
        return self._service.create_sampler(
            model_path=model_path,
            model_id=self._model_id if kind == "inflight" else None,
            kind=kind,
            record_stability_info=record_stability_info,
            # The platform service deploys these weights on the inference service, which
            # needs the base model to place them, and tracks the temporary
            # deployment per training model so it can release the previous one.
            deploy_base_model=self._base_model,
            training_model_id=self._model_id,
        )

    def save_weights_and_get_sampler(
        self,
        name: str,
        *,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
        record_stability_info: bool = False,
    ) -> Any:
        """Save weights, then open a sampler on them.

        Blocks on the save, because the path is the sampler's input. Returns
        the :class:`~mind_recursive.sampler.Sampler`, not a future -- there is nothing
        left to await by then. For weights that move with training, skip the
        save entirely and use ``create_sampler(kind="inflight")``.
        """
        path = self.save_weights_only(
            name,
            ttl_seconds=ttl_seconds,
            user_metadata=user_metadata,
            overwrite=overwrite,
        ).result()
        return self.create_sampler(
            model_path=path,
            record_stability_info=record_stability_info,
        )

    def create_sampling_client(
        self,
        model_path: str | None = None,
        retry_config: object | None = None,
        record_stability_info: bool = False,
        *,
        kind: SamplerKind = "snapshot",
    ) -> Any:
        """Tinker spelling for :meth:`create_sampler`.

        ``retry_config`` is rejected rather than ignored: retry policy lives on
        the ``ServiceClient`` in this SDK, and silently dropping a caller's
        retry settings would change failure behaviour.
        """
        if retry_config is not None:
            raise TypeError(
                "retry_config is not supported; set max_retries on ServiceClient"
            )
        return self.create_sampler(
            model_path=model_path,
            kind=kind,
            record_stability_info=record_stability_info,
        )

    def save_weights_and_get_sampling_client(
        self,
        name: str | None = None,
        retry_config: object | None = None,
        record_stability_info: bool = False,
        kind: SamplerKind = "snapshot",
    ) -> Any:
        """Snapshot the current weights into a throwaway sampler.

        Two hops, because training and sampling are different platforms now:

        1. Save the weights on the training service (``/train``) under a per-step name
           ``ephemeral-<seq>``, each with a 24h TTL. A distinct path per step
           is deliberate: the inference service can only sample a path once it is deployed,
           and the platform service keeps the previous step's deployment alive
           until the new one is ready, so the two must not share a path. This
           returns a ``mint://`` path.
        2. Open a snapshot sampler through ``/service/sample``: the platform
           service deploys that path on the inference service, releases the previous step's
           temporary deployment, and resolves the future to the sampler id.

        The ephemeral save is not reported as a user artefact: it is a
        throwaway, not a checkpoint the caller asked to keep.

        ``name`` is deprecated and has no effect; it is ignored with a
        ``DeprecationWarning``. For weights you want to keep, use
        :meth:`save_weights_and_get_sampler` with a name.

        ``kind='inflight'`` skips the save and binds a sampler to this live
        model instead; it cannot be combined with ``name``.
        """
        if retry_config is not None:
            raise TypeError(
                "retry_config is not supported; set max_retries on ServiceClient"
            )
        if kind == "inflight":
            if name is not None:
                raise ValueError(
                    "kind='inflight' saves nothing, so a name would not be used"
                )
            return self.create_sampler(
                kind="inflight",
                record_stability_info=record_stability_info,
            )
        if kind != "snapshot":
            raise ValueError(f"kind must be 'snapshot' or 'inflight', got {kind!r}")
        if name is not None:
            warnings.warn(
                "The 'name' parameter of save_weights_and_get_sampling_client() "
                "has no effect: the weights are an ephemeral export. For a "
                "persistent checkpoint use save_weights_and_get_sampler(name).",
                DeprecationWarning,
                stacklevel=2,
            )
        # Hop 1: weights-only save on the training service under a per-step name, 24h
        # TTL. Built directly rather than via save_weights_only() because that
        # method reports a user artefact and forbids the ``ephemeral`` prefix
        # -- both wrong for a throwaway export.
        seq = self._next_seq()
        body: dict[str, Any] = {
            "model_id": self._model_id,
            "seq_id": seq,
            "kind": "snapshot",
            "path": f"{EPHEMERAL_SAMPLER_NAME}-{seq}",
            "ttl_seconds": EPHEMERAL_SAMPLER_TTL_SECONDS,
            "type": "save_weights_for_sampler",
        }
        request_id = self._service._submit(P.PATH_SAVE_WEIGHTS_FOR_SAMPLER, body)
        model_path = APIFuture(
            request_id,
            self._service.resolver(),
            P.parse_weights_path,
        ).result()
        # Hop 2: deploy the saved path on the inference service and open the sampler. This
        # goes through ``/service/sample`` and waits on the deployment.
        return self.create_sampler(
            model_path=model_path,
            record_stability_info=record_stability_info,
        )

    # -- async twins ----------------------------------------------------------
    async def forward_backward_async(
        self,
        data: Sequence[Datum],
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: dict[str, float | str] | None = None,
        *,
        forward_only: bool = False,
    ) -> ForwardBackwardOutput:
        """Async twin of :meth:`forward_backward`.

        Chunks are submitted concurrently; each future is polled on its own
        async resolver, then the chunk metrics are recombined exactly as in
        the sync path.
        """
        batch = list(data)
        if not batch:
            raise ValueError("forward_backward needs at least one datum")
        chunks = self._split_batch(batch)
        prepared: list[tuple[bytes, str | None]] = []
        for chunk in chunks:
            body = encode_forward_backward_request(
                model_id=self._model_id,
                seq_id=self._next_seq(),
                data=chunk,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                forward_only=forward_only,
            )
            encoding: str | None = None
            if self._service.client_config.proto_compress_fwdbwd:
                body, encoding = _maybe_compress(body)
            prepared.append((body, encoding))

        async def submit(item: tuple[bytes, str | None]) -> str:
            body, encoding = item
            payload = await self._service.post_proto_async(
                P.PATH_FORWARD_BACKWARD, body, content_encoding=encoding
            )
            request_id = payload.get("request_id")
            if not request_id:
                raise ConfigurationError(
                    f"forward_backward returned no request_id: {payload!r}"
                )
            return str(request_id)

        request_ids = await asyncio.gather(*(submit(item) for item in prepared))
        resolver = self._service.async_resolver()
        parts = await asyncio.gather(
            *(
                resolver.resolve(
                    request_id, timeout=None, parse=P.parse_forward_backward_output
                )
                for request_id in request_ids
            )
        )
        return _merge_outputs(list(parts))

    async def forward_async(
        self,
        data: Sequence[Datum],
        loss_fn: LossFnType = "cross_entropy",
        loss_fn_config: dict[str, float | str] | None = None,
    ) -> ForwardBackwardOutput:
        """Async twin of :meth:`forward`."""
        return await self.forward_backward_async(
            data, loss_fn, loss_fn_config, forward_only=True
        )

    async def optim_step_async(
        self,
        optim_params: AdamParams | None = None,
        *,
        adam_params: AdamParams | None = None,
    ) -> dict[str, Any]:
        """Async twin of :meth:`optim_step`."""
        if optim_params is not None and adam_params is not None:
            raise ValueError("pass optim_params or adam_params, not both")
        params = adam_params if adam_params is not None else optim_params
        if params is None:
            params = AdamParams()
        if not isinstance(params, AdamParams):
            raise TypeError(
                "mind_recursive supports AdamParams only; a non-Adam optimizer has no "
                "wire representation in this SDK"
            )
        request_id = await self._service._asubmit(
            P.PATH_OPTIM_STEP,
            {
                "adam_params": params.to_wire(),
                "model_id": self._model_id,
                "seq_id": self._next_seq(),
                "type": "optim_step",
            },
        )
        return await self._service.async_resolver().resolve(
            request_id, timeout=None, parse=P.parse_optim_step
        )

    async def _save_async(
        self,
        name: str,
        *,
        train_state: bool,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
    ) -> str:
        if user_metadata is not None and not isinstance(user_metadata, dict):
            raise ConfigurationError(
                "user_metadata must be a dict of strings, got "
                f"{type(user_metadata).__name__}; pass overwrite= by keyword"
            )
        if not train_state and (
            name == EPHEMERAL_SAMPLER_NAME
            or name.startswith(f"{EPHEMERAL_SAMPLER_NAME}-")
        ):
            raise ConfigurationError(
                f"the {EPHEMERAL_SAMPLER_NAME!r} prefix is reserved for the "
                "ephemeral export of save_weights_and_get_sampling_client(); "
                "pick another name"
            )
        if train_state:
            body: dict[str, Any] = {
                "model_id": self._model_id,
                "seq_id": self._next_seq(),
                "overwrite": overwrite,
                "path": _checkpoint_name(name, "save_weights"),
                "type": "save_weights",
            }
            endpoint = P.PATH_SAVE_WEIGHTS
        else:
            body = {
                "model_id": self._model_id,
                "seq_id": self._next_seq(),
                "kind": "snapshot",
                "overwrite": overwrite,
                "path": _checkpoint_name(name, "save_weights"),
                "type": "save_weights_for_sampler",
            }
            endpoint = P.PATH_SAVE_WEIGHTS_FOR_SAMPLER
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if user_metadata:
            body["user_metadata"] = user_metadata
        request_id = await self._service._asubmit(endpoint, body)
        path = await self._service.async_resolver().resolve(
            request_id, timeout=None, parse=P.parse_weights_path
        )
        artifact_type = "checkpoint" if train_state else "sampler_weights"
        try:
            await asyncio.to_thread(
                self._service.artifact,
                path,
                type=artifact_type,
                name=name,
                metadata=dict(user_metadata) if user_metadata else None,
            )
        except Exception:  # noqa: BLE001 - reporting must not fail a save
            log.warning("artifact report failed for %s (ignored)", path)
        return path

    async def save_weights_only_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
        user_metadata: dict[str, str] | None = None,
        *,
        overwrite: bool = False,
    ) -> str:
        """Async twin of :meth:`save_weights_only`."""
        return await self._save_async(
            name,
            train_state=False,
            ttl_seconds=ttl_seconds,
            overwrite=overwrite,
            user_metadata=user_metadata,
        )

    save_weights_for_sampler_async = save_weights_only_async

    async def save_state_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
    ) -> str:
        """Async twin of :meth:`save_state`."""
        return await self._save_async(
            name,
            train_state=True,
            ttl_seconds=ttl_seconds,
            overwrite=overwrite,
            user_metadata=user_metadata,
        )

    async def load_weights_async(
        self, path: str, *, optimizer: bool = True
    ) -> dict[str, Any]:
        """Async twin of :meth:`load_weights`."""
        request_id = await self._service._asubmit(
            P.PATH_LOAD_WEIGHTS,
            {
                "model_id": self._model_id,
                "path": path,
                "seq_id": self._next_seq(),
                "optimizer": bool(optimizer),
                "type": "load_weights",
            },
        )
        return await self._service.async_resolver().resolve(
            request_id, timeout=None, parse=P.parse_load_weights
        )

    async def load_state_async(self, path: str) -> dict[str, Any]:
        """Async twin of :meth:`load_state` (weights only)."""
        return await self.load_weights_async(path, optimizer=False)

    async def load_state_with_optimizer_async(self, path: str) -> dict[str, Any]:
        """Async twin of :meth:`load_state_with_optimizer`."""
        return await self.load_weights_async(path, optimizer=True)

    async def get_info_async(self) -> ModelInfo:
        """Async twin of :meth:`get_info`."""
        payload = await self._service._apost_json(
            P.PATH_GET_INFO, {"model_id": self._model_id, "type": "get_info"}
        )
        return ModelInfo.from_wire(payload)

    async def create_sampler_async(
        self,
        *,
        model_path: str | None = None,
        kind: SamplerKind = "snapshot",
        record_stability_info: bool = False,
    ) -> Any:
        """Async twin of :meth:`create_sampler`."""
        if kind == "snapshot" and not model_path:
            raise ValueError(
                "kind='snapshot' samples fixed weights, so it needs a "
                "model_path from save_weights_only() -- or pass kind='inflight' "
                "to sample this model's current weights"
            )
        return await self._service.create_sampler_async(
            model_path=model_path,
            model_id=self._model_id if kind == "inflight" else None,
            kind=kind,
            record_stability_info=record_stability_info,
            deploy_base_model=self._base_model,
            training_model_id=self._model_id,
        )

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        retry_config: object | None = None,
        record_stability_info: bool = False,
        *,
        kind: SamplerKind = "snapshot",
    ) -> Any:
        """Async twin of :meth:`create_sampling_client`."""
        if retry_config is not None:
            raise TypeError(
                "retry_config is not supported; set max_retries on ServiceClient"
            )
        return await self.create_sampler_async(
            model_path=model_path,
            kind=kind,
            record_stability_info=record_stability_info,
        )

    async def save_weights_and_get_sampler_async(
        self,
        name: str,
        *,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
        user_metadata: dict[str, str] | None = None,
        record_stability_info: bool = False,
    ) -> Any:
        """Async twin of :meth:`save_weights_and_get_sampler`."""
        path = await self.save_weights_only_async(
            name,
            ttl_seconds=ttl_seconds,
            user_metadata=user_metadata,
            overwrite=overwrite,
        )
        return await self.create_sampler_async(
            model_path=path,
            record_stability_info=record_stability_info,
        )

    async def save_weights_and_get_sampling_client_async(
        self,
        name: str | None = None,
        retry_config: object | None = None,
        record_stability_info: bool = False,
        kind: SamplerKind = "snapshot",
    ) -> Any:
        """Async twin of :meth:`save_weights_and_get_sampling_client`."""
        if retry_config is not None:
            raise TypeError(
                "retry_config is not supported; set max_retries on ServiceClient"
            )
        if kind == "inflight":
            if name is not None:
                raise ValueError(
                    "kind='inflight' saves nothing, so a name would not be used"
                )
            return await self.create_sampler_async(
                kind="inflight", record_stability_info=record_stability_info
            )
        if kind != "snapshot":
            raise ValueError(f"kind must be 'snapshot' or 'inflight', got {kind!r}")
        if name is not None:
            warnings.warn(
                "The 'name' parameter of save_weights_and_get_sampling_client() "
                "has no effect: the weights are an ephemeral export. For a "
                "persistent checkpoint use save_weights_and_get_sampler(name).",
                DeprecationWarning,
                stacklevel=2,
            )
        seq = self._next_seq()
        body: dict[str, Any] = {
            "model_id": self._model_id,
            "seq_id": seq,
            "kind": "snapshot",
            "path": f"{EPHEMERAL_SAMPLER_NAME}-{seq}",
            "ttl_seconds": EPHEMERAL_SAMPLER_TTL_SECONDS,
            "type": "save_weights_for_sampler",
        }
        request_id = await self._service._asubmit(
            P.PATH_SAVE_WEIGHTS_FOR_SAMPLER, body
        )
        model_path = await self._service.async_resolver().resolve(
            request_id, timeout=None, parse=P.parse_weights_path
        )
        return await self.create_sampler_async(
            model_path=model_path,
            record_stability_info=record_stability_info,
        )

    #: Upstream's spelling of the weights-only save. Same function object, so
    #: the two names cannot drift. ``save_state`` needs no alias: it is the
    #: upstream name already.
    save_weights_for_sampler = save_weights_only

    def __repr__(self) -> str:
        return f"<LoraTrainer {self._model_id} ({self._base_model})>"


class _CombinedForwardBackward:
    """Presents several chunk futures as one.

    Metrics are recombined the way the server would have: sums stay sums, and
    means are re-weighted by each chunk's token count rather than averaged
    naively, which would bias toward small chunks.
    """

    __slots__ = ("_futures", "_value")

    def __init__(self, futures: list[APIFuture[ForwardBackwardOutput]]) -> None:
        self._futures = futures
        self._value: ForwardBackwardOutput | None = None

    @property
    def request_id(self) -> str:
        return ",".join(f.request_id for f in self._futures)

    def done(self) -> bool:
        return all(f.done() for f in self._futures)

    def result(self, timeout: float | None = None) -> ForwardBackwardOutput:
        if self._value is not None:
            return self._value
        parts = [f.result(timeout) for f in self._futures]
        self._value = _merge_outputs(parts)
        return self._value

    def __repr__(self) -> str:
        return f"<APIFuture (combined, {len(self._futures)} chunks)>"


def _merge_outputs(parts: list[ForwardBackwardOutput]) -> ForwardBackwardOutput:
    """Recombine chunked ``forward_backward`` results."""
    if len(parts) == 1:
        return parts[0]

    merged: dict[str, float] = {}
    weights = [float(p.metrics.get("num_tokens:sum") or 0.0) for p in parts]
    total_weight = sum(weights)

    keys = {key for part in parts for key in part.metrics}
    for key in keys:
        values = [(p.metrics.get(key), w) for p, w in zip(parts, weights)]
        present = [(v, w) for v, w in values if v is not None]
        if not present:
            continue
        if key.endswith(":sum"):
            merged[key] = sum(v for v, _ in present)
        elif key.endswith(":mean") and total_weight > 0:
            # Token-weighted, so a 1-datum chunk does not count as much as a
            # 1000-datum chunk.
            weighted = sum(v * w for v, w in present)
            denominator = sum(w for _, w in present) or 1.0
            merged[key] = weighted / denominator
        elif key.endswith(":max"):
            merged[key] = max(v for v, _ in present)
        elif key.endswith(":min"):
            merged[key] = min(v for v, _ in present)
        else:
            merged[key] = sum(v for v, _ in present) / len(present)

    outputs: list[dict] = []
    for part in parts:
        outputs.extend(part.loss_fn_outputs)
    return ForwardBackwardOutput(
        loss_fn_output_type=parts[0].loss_fn_output_type,
        loss_fn_outputs=outputs,
        metrics=merged,
    )
