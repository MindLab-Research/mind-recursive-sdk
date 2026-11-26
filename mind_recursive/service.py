"""``ServiceClient``: one training session, created and owned.

Constructing one opens the session -- ``POST client/config`` followed by
``POST create_session`` -- and that is the class's whole definition: it exists
to create a session and hold it. There is no read-only mode. The trainer and
sampler are handles onto resources inside the session, not clients; a client
may create several trainers, each adding a model under the same session.
Reporting is not a handle either: a session has exactly one writer, so
``log`` / ``event`` / ``metric`` / ``artifact`` / ``summary`` live on the
client itself.

Anything that does not need a session is not here. Capabilities, health and
(when the platform serves them) ``whoami`` / ``list_sessions`` / billing live
on :class:`~mind_recursive.user.UserClient`, which creates nothing. Splitting them
means a connectivity check no longer leaves a session record behind, and this
class's arguments -- ``heartbeat_seconds``, ``tags`` -- are all about the
session it is opening, with no mode in which they mean nothing.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Mapping

from . import _protocol as P
from ._config import Config, build_config, web_session_url
from ._console import emit
from ._errors import ConfigurationError, GoneError, SessionClosedError
from ._futures import APIFuture, AsyncFutureResolver, FutureResolver
from ._session import DEFAULT_HEARTBEAT_SECONDS, ClientConfig, Heartbeat
from ._transport import AsyncTransport, JSON_CONTENT_TYPE, Transport
from .reporter import Reporter
from .types import LoraConfig, SamplerKind, ServerCapabilities, SupportedModel

__all__ = ["ServiceClient", "SDK_VERSION"]

from mind_recursive._version import SDK_VERSION as _SDK_VERSION

log = logging.getLogger("mind_recursive.service")

#: Re-exported from ``mind_recursive._version`` so the literal lives in exactly one
#: place; the server reads it to resolve version-gated flags.
SDK_VERSION = _SDK_VERSION


class ServiceClient:
    """One training session on the Mind Recursive platform, opened on construction.

    Trainers, samplers and the reporter are all created from here and none
    hold a connection of their own. This client owns the transport, the
    session and the billing identity, so closing it is the only cleanup a
    caller has to perform.

    Constructing one creates a server-side session, which is a billable
    record. For questions that need no session -- which models are served, is
    the backend up -- use :class:`~mind_recursive.user.UserClient` instead.

    Args:
        api_key: user key (``sk-...``). Falls back to ``MIND_RECURSIVE_API_KEY``.
        base_url: platform origin, with or without the fixed ``/api/mindrsi/v1``
            suffix. Falls back to ``MIND_RECURSIVE_BASE_URL``, then the
            production default.
        timeout: per-request timeout in seconds for the data plane.
        max_retries: retries for retryable failures (transport, 429, 5xx).
        heartbeat_seconds: session keep-alive interval.
        name: human-readable name for this session, shown as the title of
            the run in the web console. Optional and free-form; it
            is a label, not an id -- several sessions may share one.
        client_id: opaque id correlating every session this process opens, so
            the platform can group them. Recorded as a field on the session,
            sent in ``create_session``. It is neither a credential nor a
            routing key, so it is not a request header; free-form business
            labels belong in ``tags`` or ``user_metadata`` on the same call.
        tags: session tags, sent in ``create_session``.
        user_metadata: session metadata, sent in ``create_session``.

    Example::

        import mind_recursive

        sc = mind_recursive.ServiceClient(name="GRPO on GSM8K", client_id="grpo-exp-20261118")
        trainer = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=32)
        sc.log("training starts")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        connect_timeout: float = 30.0,
        max_retries: int = 3,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        name: str | None = None,
        client_id: str | None = None,
        trust_env: bool = False,
        config: Config | None = None,
        tags: list[str] | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or build_config(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            connect_timeout=connect_timeout,
            max_retries=max_retries,
            trust_env=trust_env,
        )
        self._transport = Transport(self._config)
        self._heartbeat_seconds = heartbeat_seconds
        self._name = name
        self._async_transport: AsyncTransport | None = None
        self._async_lock = threading.Lock()
        self._async_heartbeat_task: asyncio.Task[None] | None = None
        self._client_id = client_id

        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._client_config: ClientConfig | None = None
        self._heartbeat: Heartbeat | None = None
        self._closed = False
        # Staged here, then copied into the create_session body below.
        self._pending_tags = list(tags or [])
        self._pending_metadata: dict[str, Any] = dict(user_metadata or {})
        self._model_seq = 0
        self._sampling_seq = 0
        #: One reporter per client, and it lives on the client: a session has
        #: exactly one writer, so there is nothing to create or disambiguate.
        self._reporter = Reporter(service=self)

        # Constructing this class *is* creating the session; there is no mode
        # in which it holds none. A failure here leaves nothing behind.
        try:
            self._open_session()
        except Exception:
            self._transport.close()
            raise

    # -- introspection --------------------------------------------------------
    @property
    def config(self) -> Config:
        return self._config

    @property
    def session_id(self) -> str | None:
        """The upstream session id.

        Set from construction onwards -- there is no window in which a live
        client has no session -- and cleared by :meth:`close`.
        """
        return self._session_id

    @property
    def name(self) -> str | None:
        """The session's display name, as passed to the constructor."""
        return self._name

    @property
    def client_id(self) -> str | None:
        """The process-correlation id recorded on this session, if any."""
        return self._client_id

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """Per-request headers on top of the transport's own.

        The SDK sends no custom headers, so this is empty unless a caller
        passes something. ``client_id`` is not one: it is neither a credential
        nor a routing key -- just a label saying which sessions came from one
        process -- so it belongs on the session resource, and
        ``create_session`` records it. Neither Tinker nor Fireworks declares
        an attribution header either.
        """
        return dict(extra or {})

    # -- session --------------------------------------------------------------
    def require_session(self) -> str:
        """The session id, for the trainer / sampler / reporter to address.

        The session exists from construction, so this only has to rule out a
        client that has since been closed.
        """
        with self._lock:
            if self._closed or self._session_id is None:
                raise SessionClosedError("this ServiceClient has been closed")
            return self._session_id

    @property
    def client_config(self) -> ClientConfig:
        """Server-negotiated flags, fetched on first access."""
        with self._lock:
            if self._client_config is None:
                self._client_config = self._negotiate_config()
            return self._client_config

    def _negotiate_config(self) -> ClientConfig:
        """``POST client/config``.

        A failure here is not fatal: we fall back to documented defaults rather
        than refusing to train, but we say so, because the defaults are a guess
        at someone else's policy.
        """
        try:
            payload = self._post_json(
                P.PATH_CLIENT_CONFIG,
                {"sdk_version": SDK_VERSION},
                base_url=self._config.service_base_url,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "client/config negotiation failed (%s); falling back to defaults", exc
            )
            return ClientConfig()
        return ClientConfig.from_wire(payload)

    def _open_session(self) -> None:
        """``POST create_session`` and start the heartbeat.

        Caller must hold ``self._lock``. Config is negotiated first because the
        response governs how subsequent requests must behave.
        """
        if self._client_config is None:
            self._client_config = self._negotiate_config()

        body: dict[str, Any] = {
            "tags": list(self._pending_tags),
            "sdk_version": SDK_VERSION,
        }
        # A field on the session, not a header: it identifies the process that
        # opened the session, which is a property of the session.
        if self._name:
            body["name"] = self._name
        if self._client_id:
            body["client_id"] = self._client_id
        if self._pending_metadata:
            body["user_metadata"] = dict(self._pending_metadata)

        payload = self._post_json(
            P.PATH_CREATE_SESSION,
            body,
            base_url=self._config.session_base_url,
        )
        session_id = payload.get("session_id")
        if not session_id:
            raise ConfigurationError(
                f"create_session returned no session_id: {payload!r}"
            )
        self._session_id = str(session_id)

        for level, key in (("warning", "warning_message"), ("info", "info_message")):
            message = payload.get(key)
            if message:
                getattr(log, level)("create_session: %s", message)

        # The session is the thing the caller wants to look at in the browser,
        # so hand them the link instead of making them assemble it from an id.
        # Terminal only: a billing/console URL is not a training report.
        emit(f"Session created: {self._session_id}")
        emit(f"Web console: {web_session_url(self._config.web_url, self._session_id)}")

        self._heartbeat = Heartbeat(
            self._send_heartbeat,
            interval=self._heartbeat_seconds,
            # 410 means the server reclaimed the session. Retrying a beat for a
            # resource that is gone only adds noise; the contract says stop.
            stop_on=(GoneError,),
        )
        self._heartbeat.start()
        log.debug("opened session %s", self._session_id)

    def _send_heartbeat(self) -> None:
        session_id = self._session_id
        if session_id is None:
            return
        self._post_json(
            P.PATH_HEARTBEAT.format(session_id=session_id),
            {"session_id": session_id},
            # A heartbeat that outlives its own interval is worthless.
            timeout=min(self._heartbeat_seconds, 30.0),
            max_retries=1,
            base_url=self._config.session_base_url,
        )

    def next_model_seq(self) -> int:
        """Allocate the next model sequence number within this session."""
        with self._lock:
            seq = self._model_seq
            self._model_seq += 1
            return seq

    def next_sampling_seq(self) -> int:
        with self._lock:
            seq = self._sampling_seq
            self._sampling_seq += 1
            return seq

    # -- trainer --------------------------------------------------------------
    def create_lora_trainer(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = False,
        user_metadata: dict[str, Any] | None = None,
        optimizer: Any | None = None,
    ) -> Any:
        """Create a LoRA training client.

        The parameter order matches upstream, so ported positional calls land
        on the right fields. ``train_unembed`` defaults to ``False``, unlike
        the stock SDK's ``True``: training the LM head yields weights that fail
        to deploy, so the safe value is the default here.

        ``optimizer`` accepts the AdamW configuration only; a non-Adam family
        cannot be expressed on this wire and is rejected locally.

        Returns:
            A :class:`~mind_recursive.trainer.LoraTrainer`.
        """
        from .trainer import LoraTrainer

        if optimizer is not None:
            opt_type = (
                optimizer.get("type")
                if isinstance(optimizer, Mapping)
                else getattr(optimizer, "type", None)
            )
            if opt_type not in (None, "adamw"):
                raise ConfigurationError(
                    f"only the AdamW optimizer is supported, got {opt_type!r}"
                )

        lora = LoraConfig(
            rank=rank,
            seed=seed,
            train_unembed=train_unembed,
            train_mlp=train_mlp,
            train_attn=train_attn,
        )
        session_id = self.require_session()
        model_seq = self.next_model_seq()
        body: dict[str, Any] = {
            "session_id": session_id,
            "model_seq_id": model_seq,
            "base_model": base_model,
            "lora_config": lora.to_wire(),
            "optimizer_config": {"type": "adamw"},
            "type": "create_model",
        }
        if user_metadata:
            body["user_metadata"] = user_metadata

        model_id = self._create_trainer(P.PATH_CREATE_MODEL, body)
        return LoraTrainer(
            service=self, model_id=model_id, base_model=base_model, lora_config=lora
        )

    def create_lora_trainer_from_state(
        self,
        path: str,
        *,
        base_model: str,
        rank: int | None = None,
        train_mlp: bool | None = None,
        train_attn: bool | None = None,
        load_optimizer: bool | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a LoRA training client from a saved training state.

        One submission does what ``create_lora_trainer`` + ``load_weights``
        does in two: the server creates the model and loads the checkpoint
        together, so a freshly initialized adapter is never produced only to
        be overwritten.

        Args:
            path: a training-state checkpoint path returned by
                ``save_state`` (``mint://.../checkpoints/...``).
                Weights-only sampler exports (``adapters/...``) are not
                training states; the server rejects them.
            base_model: which served model to train. Required in this release:
                the server does not infer it from the artifact yet.
            rank / train_mlp / train_attn: leave ``None`` to use what the
                checkpoint was saved with. Only the values you pass are sent;
                the rest are the server's to resolve.
            load_optimizer: leave ``None`` to follow the checkpoint (restore
                the optimizer state when it has one). ``False`` forces a
                weights-only resume; ``True`` on a weights-only checkpoint is
                rejected by the server.
            user_metadata: free-form labels recorded on the resource.

        Returns:
            A :class:`~mind_recursive.LoraTrainer`. ``trainer.lora_config`` is
            ``None`` unless ``rank``, ``train_mlp`` and ``train_attn`` were all
            passed: the effective structure lives on the server and this
            release has no read-back for it (``get_info()`` returns the rank).
        """
        from .trainer import LoraTrainer

        if not isinstance(path, str) or not path:
            raise ConfigurationError(
                "create_lora_trainer_from_state requires a non-empty checkpoint path"
            )
        if not isinstance(base_model, str) or not base_model:
            raise ConfigurationError(
                "create_lora_trainer_from_state requires a non-empty base_model"
            )
        if rank is not None and (
            isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
        ):
            raise ConfigurationError(
                f"rank must be a positive integer or None, got {rank!r}"
            )
        for name, value in (
            ("train_mlp", train_mlp),
            ("train_attn", train_attn),
            ("load_optimizer", load_optimizer),
        ):
            if value is not None and not isinstance(value, bool):
                raise ConfigurationError(f"{name} must be a bool or None, got {value!r}")

        # Only what the caller actually asked for goes on the wire; an absent
        # key is how this API says "let the artifact decide".
        lora: dict[str, Any] = {}
        if rank is not None:
            lora["rank"] = rank
        if train_mlp is not None:
            lora["train_mlp"] = train_mlp
        if train_attn is not None:
            lora["train_attn"] = train_attn

        body: dict[str, Any] = {
            "session_id": self.require_session(),
            "model_seq_id": self.next_model_seq(),
            "state_path": path,
            "base_model": base_model,
            "type": "create_model_from_state",
        }
        if lora:
            body["lora_config"] = lora
        if load_optimizer is not None:
            body["load_optimizer"] = load_optimizer
        if user_metadata:
            body["user_metadata"] = dict(user_metadata)

        model_id = self._create_trainer(P.PATH_CREATE_MODEL_FROM_STATE, body)
        config = (
            LoraConfig(rank=rank, train_mlp=train_mlp, train_attn=train_attn)
            if rank is not None and train_mlp is not None and train_attn is not None
            else None
        )
        return LoraTrainer(
            service=self, model_id=model_id, base_model=base_model, lora_config=config
        )

    def create_training_client_from_state(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
        optimizer: Any | None = None,
    ) -> Any:
        """Tinker-named shell for :meth:`create_lora_trainer_from_state`.

        Loads weights only, matching upstream's ``create_training_client_from_state``.
        This SDK's server does not infer ``base_model`` from the checkpoint, so
        it is required here even though upstream treats it as optional.
        ``weights_access_token`` and non-Adam optimizers are rejected rather
        than silently ignored.
        """
        return self._create_from_state_compat(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=optimizer,
            load_optimizer=False,
        )

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> Any:
        """Tinker-named shell for a resume that restores optimizer state."""
        return self._create_from_state_compat(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=None,
            load_optimizer=True,
        )

    def _create_from_state_compat(
        self,
        path: str,
        *,
        base_model: str | None,
        user_metadata: dict[str, str] | None,
        weights_access_token: str | None,
        optimizer: Any | None,
        load_optimizer: bool,
    ) -> Any:
        """Shared body of the two Tinker-named state creation shells."""
        if base_model is None:
            raise ConfigurationError(
                "base_model is required in this release: the server does not "
                "infer it from the checkpoint yet; pass base_model=..."
            )
        if weights_access_token is not None:
            raise ConfigurationError(
                "weights_access_token is not supported by this relay"
            )
        if optimizer is not None:
            opt_type = (
                optimizer.get("type")
                if isinstance(optimizer, Mapping)
                else getattr(optimizer, "type", None)
            )
            if opt_type not in (None, "adamw"):
                raise ConfigurationError(
                    f"only the AdamW optimizer is supported, got {opt_type!r}"
                )
        return self.create_lora_trainer_from_state(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            load_optimizer=load_optimizer,
        )

    def create_sampler(
        self,
        model_path: str | None = None,
        *,
        base_model: str | None = None,
        model_id: str | None = None,
        kind: SamplerKind = "snapshot",
        record_stability_info: bool = False,
        deploy_base_model: str | None = None,
        training_model_id: str | None = None,
    ) -> Any:
        """Create a token-level sampling client.

        Args:
            base_model: sample the base model.
            model_path: a ``mint://`` path to sample trained weights.
            model_id: a live training model to sample from, required by
                ``kind="inflight"`` and ignored otherwise.
            kind: ``"snapshot"`` samples fixed weights -- a saved path or a base
                model -- and is what evaluation wants. ``"inflight"`` samples
                the training model's *current* weights, which move as training
                steps, so an on-policy loop does not have to save a checkpoint
                per step. Prefer :meth:`mind_recursive.LoraTrainer.create_sampler`,
                which fills ``model_id`` in for you.
            record_stability_info: record per-sequence stability information
                for every sample from this client, referenced from training
                requests by ``sequence_id`` (``stable_reinforce``). The
                per-call parameter on :meth:`mind_recursive.Sampler.sample` sets it
                for one request instead.
            deploy_base_model: base model the trained weights were trained
                from. Weights have to be deployed before they can be sampled,
                and the deployment needs it. :meth:`mind_recursive.LoraTrainer.create_sampler`
                fills it in; pass it yourself when sampling a ``model_path``
                from a ``ServiceClient``.
            training_model_id: the training model the weights belong to. Each
                new sampler on it releases the previous temporary deployment.
                Filled in by the trainer.

        Returns:
            A :class:`~mind_recursive.sampler.Sampler`.
        """
        from .sampler import Sampler

        if kind not in P.SAMPLER_KINDS:
            raise ValueError(f"kind must be one of {P.SAMPLER_KINDS}")
        if kind == "inflight":
            if not model_id:
                raise ValueError(
                    "kind='inflight' samples a live training model, so it needs "
                    "model_id -- use LoraTrainer.create_sampler() to pass it"
                )
            if base_model or model_path:
                raise ValueError(
                    "kind='inflight' takes its weights from model_id; "
                    "base_model and model_path describe fixed weights"
                )
        elif not base_model and not model_path:
            raise ValueError("pass base_model or model_path")

        session_id = self.require_session()
        body: dict[str, Any] = {
            "session_id": session_id,
            "sampling_session_seq_id": self.next_sampling_seq(),
            "type": "create_sampling_session",
        }
        if kind != "snapshot":
            body["kind"] = kind
        if model_id:
            body["model_id"] = model_id
        if record_stability_info:
            body["record_stability_info"] = True
        if base_model:
            body["base_model"] = base_model
        if model_path:
            body["model_path"] = model_path

        # Every sampler is created through the platform service
        # (``/service/sample``). The inference service only opens a sampler under an
        # session of its own; the platform service creates that session for
        # this Service Session, keeps it alive and finishes it when the
        # session ends. A public ``base_model`` names an already-deployed
        # model and resolves at once; trained weights (``model_path``) are
        # deployed first, so that future stays pending while the deployment
        # comes up. Sampling itself (``asample``) stays relay-direct.
        if model_path or kind == "inflight":
            # Deploying trained weights needs the base they were trained from.
            if deploy_base_model and "base_model" not in body:
                body["base_model"] = deploy_base_model
            if training_model_id:
                body["training_model_id"] = training_model_id
        request_id = self._submit_on(
            P.PATH_SERVICE_CREATE_SAMPLING_SESSION,
            body,
            base_url=self._config.service_sample_base_url,
        )
        sampling_session_id = APIFuture(
            request_id,
            self.resolver(base_url=self._config.service_base_url),
            P.parse_sampling_session_id,
        ).result()
        return Sampler(
            service=self,
            sampling_session_id=str(sampling_session_id),
            base_model=base_model,
            model_path=model_path,
            kind=kind,
            record_stability_info=record_stability_info,
        )

    #: Tinker's spelling of the LoRA creation call. Kept as an alias so the
    #: safe ``train_unembed=False`` default cannot drift away from it.
    create_lora_training_client = create_lora_trainer

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: object | None = None,
        record_stability_info: bool = False,
        *,
        model_id: str | None = None,
        kind: SamplerKind = "snapshot",
    ) -> Any:
        """Tinker spelling for :meth:`create_sampler`.

        The first four parameters are positional, matching upstream.
        ``retry_config`` is rejected rather than ignored: retry policy lives on
        this client's ``max_retries`` argument, and silently dropping it would
        change failure behaviour.
        """
        if retry_config is not None:
            raise TypeError(
                "retry_config is not supported; set max_retries on ServiceClient"
            )
        return self.create_sampler(
            model_path=model_path,
            base_model=base_model,
            model_id=model_id,
            kind=kind,
            record_stability_info=record_stability_info,
        )

    # -- async creation -------------------------------------------------------
    async def create_lora_trainer_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = False,
        user_metadata: dict[str, Any] | None = None,
        optimizer: Any | None = None,
    ) -> Any:
        """Async twin of :meth:`create_lora_trainer`."""
        from .trainer import LoraTrainer

        if optimizer is not None:
            opt_type = (
                optimizer.get("type")
                if isinstance(optimizer, Mapping)
                else getattr(optimizer, "type", None)
            )
            if opt_type not in (None, "adamw"):
                raise ConfigurationError(
                    f"only the AdamW optimizer is supported, got {opt_type!r}"
                )

        lora = LoraConfig(
            rank=rank,
            seed=seed,
            train_unembed=train_unembed,
            train_mlp=train_mlp,
            train_attn=train_attn,
        )
        body: dict[str, Any] = {
            "session_id": self.require_session(),
            "model_seq_id": self.next_model_seq(),
            "base_model": base_model,
            "lora_config": lora.to_wire(),
            "optimizer_config": {"type": "adamw"},
            "type": "create_model",
        }
        if user_metadata:
            body["user_metadata"] = user_metadata
        request_id = await self._asubmit_on(
            P.PATH_CREATE_MODEL, body, base_url=self._config.service_train_base_url
        )
        model_id = await self.async_resolver(
            base_url=self._config.service_base_url
        ).resolve(request_id, timeout=None, parse=P.parse_model_id)
        return LoraTrainer(
            service=self, model_id=model_id, base_model=base_model, lora_config=lora
        )

    create_lora_training_client_async = create_lora_trainer_async

    async def create_lora_trainer_from_state_async(
        self,
        path: str,
        *,
        base_model: str,
        rank: int | None = None,
        train_mlp: bool | None = None,
        train_attn: bool | None = None,
        load_optimizer: bool | None = None,
        user_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Async twin of :meth:`create_lora_trainer_from_state`."""
        from .trainer import LoraTrainer

        if not isinstance(path, str) or not path:
            raise ConfigurationError(
                "create_lora_trainer_from_state requires a non-empty checkpoint path"
            )
        if not isinstance(base_model, str) or not base_model:
            raise ConfigurationError(
                "create_lora_trainer_from_state requires a non-empty base_model"
            )
        if rank is not None and (
            isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
        ):
            raise ConfigurationError(
                f"rank must be a positive integer or None, got {rank!r}"
            )
        for name, value in (
            ("train_mlp", train_mlp),
            ("train_attn", train_attn),
            ("load_optimizer", load_optimizer),
        ):
            if value is not None and not isinstance(value, bool):
                raise ConfigurationError(f"{name} must be a bool or None, got {value!r}")

        lora: dict[str, Any] = {}
        if rank is not None:
            lora["rank"] = rank
        if train_mlp is not None:
            lora["train_mlp"] = train_mlp
        if train_attn is not None:
            lora["train_attn"] = train_attn
        body: dict[str, Any] = {
            "session_id": self.require_session(),
            "model_seq_id": self.next_model_seq(),
            "state_path": path,
            "base_model": base_model,
            "type": "create_model_from_state",
        }
        if lora:
            body["lora_config"] = lora
        if load_optimizer is not None:
            body["load_optimizer"] = load_optimizer
        if user_metadata:
            body["user_metadata"] = dict(user_metadata)
        request_id = await self._asubmit_on(
            P.PATH_CREATE_MODEL_FROM_STATE,
            body,
            base_url=self._config.service_train_base_url,
        )
        model_id = await self.async_resolver(
            base_url=self._config.service_base_url
        ).resolve(request_id, timeout=None, parse=P.parse_model_id)
        config = (
            LoraConfig(rank=rank, train_mlp=train_mlp, train_attn=train_attn)
            if rank is not None and train_mlp is not None and train_attn is not None
            else None
        )
        return LoraTrainer(
            service=self, model_id=model_id, base_model=base_model, lora_config=config
        )

    async def create_training_client_from_state_async(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
        optimizer: Any | None = None,
    ) -> Any:
        """Async twin of :meth:`create_training_client_from_state`."""
        if base_model is None:
            raise ConfigurationError(
                "base_model is required in this release: the server does not "
                "infer it from the checkpoint yet; pass base_model=..."
            )
        if weights_access_token is not None:
            raise ConfigurationError("weights_access_token is not supported by this relay")
        if optimizer is not None:
            opt_type = (
                optimizer.get("type")
                if isinstance(optimizer, Mapping)
                else getattr(optimizer, "type", None)
            )
            if opt_type not in (None, "adamw"):
                raise ConfigurationError(
                    f"only the AdamW optimizer is supported, got {opt_type!r}"
                )
        return await self.create_lora_trainer_from_state_async(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            load_optimizer=False,
        )

    async def create_training_client_from_state_with_optimizer_async(
        self,
        path: str,
        base_model: str | None = None,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> Any:
        """Async twin of :meth:`create_training_client_from_state_with_optimizer`."""
        if base_model is None:
            raise ConfigurationError(
                "base_model is required in this release: the server does not "
                "infer it from the checkpoint yet; pass base_model=..."
            )
        if weights_access_token is not None:
            raise ConfigurationError("weights_access_token is not supported by this relay")
        return await self.create_lora_trainer_from_state_async(
            path,
            base_model=base_model,
            user_metadata=user_metadata,
            load_optimizer=True,
        )

    async def create_sampler_async(
        self,
        model_path: str | None = None,
        *,
        base_model: str | None = None,
        model_id: str | None = None,
        kind: SamplerKind = "snapshot",
        record_stability_info: bool = False,
        deploy_base_model: str | None = None,
        training_model_id: str | None = None,
    ) -> Any:
        """Async twin of :meth:`create_sampler`."""
        from .sampler import Sampler

        if kind not in P.SAMPLER_KINDS:
            raise ValueError(f"kind must be one of {P.SAMPLER_KINDS}")
        if kind == "inflight":
            if not model_id:
                raise ValueError(
                    "kind='inflight' samples a live training model, so it needs "
                    "model_id -- use LoraTrainer.create_sampler() to pass it"
                )
            if base_model or model_path:
                raise ValueError(
                    "kind='inflight' takes its weights from model_id; "
                    "base_model and model_path describe fixed weights"
                )
        elif not base_model and not model_path:
            raise ValueError("pass base_model or model_path")

        body: dict[str, Any] = {
            "session_id": self.require_session(),
            "sampling_session_seq_id": self.next_sampling_seq(),
            "type": "create_sampling_session",
        }
        if kind != "snapshot":
            body["kind"] = kind
        if model_id:
            body["model_id"] = model_id
        if record_stability_info:
            body["record_stability_info"] = True
        if base_model:
            body["base_model"] = base_model
        if model_path:
            body["model_path"] = model_path
        if model_path or kind == "inflight":
            if deploy_base_model and "base_model" not in body:
                body["base_model"] = deploy_base_model
            if training_model_id:
                body["training_model_id"] = training_model_id
        request_id = await self._asubmit_on(
            P.PATH_SERVICE_CREATE_SAMPLING_SESSION,
            body,
            base_url=self._config.service_sample_base_url,
        )
        sampling_session_id = await self.async_resolver(
            base_url=self._config.service_base_url
        ).resolve(request_id, timeout=None, parse=P.parse_sampling_session_id)
        return Sampler(
            service=self,
            sampling_session_id=str(sampling_session_id),
            base_model=base_model,
            model_path=model_path,
            kind=kind,
            record_stability_info=record_stability_info,
        )

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: object | None = None,
        record_stability_info: bool = False,
        *,
        model_id: str | None = None,
        kind: SamplerKind = "snapshot",
    ) -> Any:
        """Async twin of :meth:`create_sampling_client`."""
        if retry_config is not None:
            raise TypeError(
                "retry_config is not supported; set max_retries on ServiceClient"
            )
        return await self.create_sampler_async(
            model_path=model_path,
            base_model=base_model,
            model_id=model_id,
            kind=kind,
            record_stability_info=record_stability_info,
        )

    # -- reporting ------------------------------------------------------------
    # -- reporting ------------------------------------------------------------
    # One reporter per client, and it lives here: a session has exactly one
    # writer, and every metric, log and artifact belongs to the client that
    # owns the session. The five methods POST through the client's own
    # transport (:mod:`mind_recursive.reporter`), are best-effort, and never raise --
    # reporting must not break training.

    def log(
        self,
        message: str,
        *,
        level: str = "info",
        step: int | None = None,
    ) -> None:
        """Record one runtime log line. Sugar over :meth:`event` with ``type="log"``.

        Args:
            message: human-readable text.
            level: ``debug`` | ``info`` | ``warning`` | ``error``.
            step: optional step number, to line the log up with metrics.
        """
        self._reporter.log(message, level=level, step=step)

    def event(
        self,
        type: str,  # noqa: A002 - the contract field is named "type"
        *,
        level: str = "info",
        code: str | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        """Record a structured timeline event.

        ``log`` is a special case: ``log(text)`` is ``event("log", message=text)``.

        Args:
            type: dotted open-set name, e.g. ``upstream.retrying``. Upstream
                requires ``^[a-z0-9_.]+$``, at most 128 bytes.
            level: ``debug`` | ``info`` | ``warning`` | ``error``.
            code: machine-readable reason code, e.g. ``429``.
            message: human-readable text.
            data: structured extras (must be a JSON object, not a list).
            step: optional step number, to line the event up with metrics.
        """
        self._reporter.event(
            type, level=level, code=code, message=message, data=data, step=step
        )

    def metric(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        stage: str | None = None,
        total: int | None = None,
        timings: dict[str, float] | None = None,
        counters: dict[str, int] | None = None,
    ) -> None:
        """Record one step's metrics. Values follow the server's metric types.

        Args:
            metrics: this step's values. ``None`` values are dropped rather
                than sent, since a null would occupy a key without locking it.
            step: the step number. Points are upserted by step.
            stage: optional label such as ``training`` or ``evaling``.
            total: revised total step count, for progress display.
            timings: named durations for this step, emitted as
                ``sys.step_seconds``.
            counters: named counters for this step, emitted as ``sys.<name>``.
        """
        self._reporter.metric(
            metrics, step=step, stage=stage, total=total,
            timings=timings, counters=counters,
        )

    def artifact(
        self,
        uri: str,
        *,
        type: str,  # noqa: A002 - the contract field is named "type"
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a produced artefact by reference. Append-only.

        Only the pointer is kept; the bytes are never fetched. ``uri`` must
        carry a scheme -- ``mint://``, ``oss://``, ``hf://``, ``https://``.
        """
        self._reporter.artifact(
            uri, type=type, name=name, version=version,
            description=description, metadata=metadata,
        )

    def summary(self, mapping: dict[str, Any] | None = None, /, **kwargs: Any) -> None:
        """Record terminal scalars. Merged by key, last write wins.

        ``sc.summary(final_loss=0.4)`` and ``sc.summary({"final_loss": 0.4})``
        are equivalent and may be mixed.
        """
        self._reporter.summary(mapping, **kwargs)

    # -- plumbing used by the client classes ----------------------------------
    def resolver(self, *, base_url: str | None = None) -> FutureResolver:
        """A resolver bound to one plane's ``retrieve_future``.

        Training calls default to the train plane. A sampling call passes the
        sample base so its future is polled on the plane that produced it; the
        wire path is the same, the base is the distinction.
        """
        base = base_url or self._config.train_base_url
        if base == self._config.sample_base_url:
            path = P.PATH_SAMPLE_RETRIEVE_FUTURE
        elif base == self._config.service_base_url:
            path = P.PATH_SERVICE_RETRIEVE_FUTURE
        else:
            path = P.PATH_RETRIEVE_FUTURE
        return FutureResolver(
            lambda request_id: self._poll_future(request_id, base_url=base, path=path)
        )

    def _poll_future(
        self, request_id: str, *, base_url: str, path: str
    ) -> dict[str, Any] | bytes:
        """One ``retrieve_future`` round trip.

        ``Accept`` offers protobuf as well as JSON because a
        ``forward_backward`` result arrives binary. The response's own
        ``Content-Type`` decides how it is read; we never assume.
        """
        response = self._transport.request(
            "POST",
            path,
            base_url=base_url,
            pool="train",
            json_body={"request_id": request_id, "allow_metadata_only": True},
            accept=P.ACCEPT_PROTO_OR_JSON,
            extra_headers=self._headers(),
            # 408 is this endpoint's normal "still pending" signal, not a
            # retryable failure; the resolver consumes the try_again body.
            allow_statuses=frozenset({408}),
        )
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "protobuf" in content_type or "octet-stream" in content_type:
            return response.content
        return response.json()

    # -- async transport -------------------------------------------------------
    @property
    def async_transport(self) -> AsyncTransport:
        """The lazy async twin of :attr:`transport`.

        Created on first async call inside the running event loop; the session
        itself was opened by the constructor over the sync transport.
        """
        if self._closed:
            raise SessionClosedError("this ServiceClient has been closed")
        with self._async_lock:
            if self._async_transport is None:
                self._async_transport = AsyncTransport(self._config)
            if self._async_heartbeat_task is None and self._heartbeat is not None:
                # From the first async call on, the heartbeat belongs to the
                # event loop instead of a background thread.
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:  # pragma: no cover - async methods run in a loop
                    pass
                else:
                    self._heartbeat.stop()
                    self._heartbeat = None
                    self._async_heartbeat_task = loop.create_task(
                        self._async_heartbeat_loop()
                    )
            return self._async_transport

    async def _async_heartbeat_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self._heartbeat_seconds)
                session_id = self._session_id
                if session_id is None:
                    return
                try:
                    await self.async_transport.request(
                        "POST",
                        P.PATH_HEARTBEAT.format(session_id=session_id),
                        base_url=self._config.session_base_url,
                        json_body={"session_id": session_id},
                        timeout=min(self._heartbeat_seconds, 30.0),
                        max_retries=1,
                        extra_headers=self._headers(),
                    )
                except GoneError:
                    return
                except Exception as exc:  # noqa: BLE001 - a heartbeat must not crash
                    log.warning("async session heartbeat failed: %s", exc)
        except asyncio.CancelledError:  # pragma: no cover - normal shutdown
            return

    def async_resolver(self, *, base_url: str | None = None) -> AsyncFutureResolver:
        """Async twin of :meth:`resolver`."""
        base = base_url or self._config.train_base_url
        if base == self._config.sample_base_url:
            path = P.PATH_SAMPLE_RETRIEVE_FUTURE
        elif base == self._config.service_base_url:
            path = P.PATH_SERVICE_RETRIEVE_FUTURE
        else:
            path = P.PATH_RETRIEVE_FUTURE
        return AsyncFutureResolver(
            lambda request_id: self._apoll_future(request_id, base_url=base, path=path)
        )

    async def _apoll_future(
        self, request_id: str, *, base_url: str, path: str
    ) -> dict[str, Any] | bytes:
        response = await self.async_transport.request(
            "POST",
            path,
            base_url=base_url,
            pool="train",
            json_body={"request_id": request_id, "allow_metadata_only": True},
            accept=P.ACCEPT_PROTO_OR_JSON,
            extra_headers=self._headers(),
            allow_statuses=frozenset({408}),
        )
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "protobuf" in content_type or "octet-stream" in content_type:
            return response.content
        return response.json()

    async def _apost_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        pool: str = "control",
        extra_headers: Mapping[str, str] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        response = await self.async_transport.request(
            "POST",
            path,
            base_url=base_url or self._config.train_base_url,
            pool=pool,
            json_body=body,
            accept=JSON_CONTENT_TYPE,
            extra_headers=self._headers(extra_headers),
            timeout=timeout,
            max_retries=max_retries,
        )
        return _json_object(response)

    async def _asubmit_on(
        self, path: str, body: dict[str, Any], *, base_url: str | None = None
    ) -> str:
        payload = await self._apost_json(path, body, base_url=base_url)
        request_id = payload.get("request_id")
        if not request_id:
            raise ConfigurationError(f"{path} returned no request_id: {payload!r}")
        return str(request_id)

    async def _asubmit(self, path: str, body: dict[str, Any]) -> str:
        return await self._asubmit_on(path, body)

    async def post_proto_async(
        self,
        path: str,
        body: bytes,
        *,
        content_encoding: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Async twin of :meth:`post_proto`."""
        headers = self._headers(extra_headers)
        if content_encoding:
            headers["Content-Encoding"] = content_encoding
        response = await self.async_transport.request(
            "POST",
            path,
            base_url=self._config.train_base_url,
            pool="train",
            content=body,
            content_type=P.PROTO_CONTENT_TYPE,
            accept=JSON_CONTENT_TYPE,
            extra_headers=headers,
        )
        return _json_object(response)

    async def get_server_capabilities_async(self) -> ServerCapabilities:
        """Async capabilities query; the sync equivalent lives on UserClient."""
        response = await self.async_transport.request(
            "GET",
            P.PATH_CAPABILITIES,
            base_url=self._config.service_base_url,
            extra_headers=self._headers(),
            accept=JSON_CONTENT_TYPE,
        )
        payload = _json_object(response)
        models = [
            SupportedModel(
                model_name=item.get("model_name"),
                max_context_length=item.get("max_context_length"),
                trainable=item.get("trainable"),
                sampleable=item.get("sampleable"),
            )
            for item in payload.get("supported_models") or []
        ]
        return ServerCapabilities(supported_models=models)

    async def close_async(self) -> None:
        """Async cleanup: finish the session and close both transports."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            heartbeat, self._heartbeat = self._heartbeat, None
            session_id, self._session_id = self._session_id, None
        transport = self._async_transport or AsyncTransport(self._config)
        if self._async_heartbeat_task is not None:
            task, self._async_heartbeat_task = self._async_heartbeat_task, None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if heartbeat is not None:
            heartbeat.stop()
        if session_id is not None:
            try:
                await transport.request(
                    "POST",
                    P.PATH_FINISH_SESSION.format(session_id=session_id),
                    base_url=self._config.session_base_url,
                    pool="control",
                    json_body={},
                    accept=JSON_CONTENT_TYPE,
                    extra_headers=self._headers(),
                    max_retries=0,
                )
            except Exception as exc:  # noqa: BLE001 - best effort, like close()
                log.debug("finishing session %s failed: %s", session_id, exc)
        await transport.aclose()
        self._async_transport = None
        self._transport.close()

    aclose = close_async

    def _create_trainer(self, path: str, body: dict[str, Any]) -> str:
        """Create a training model through the service plane.

        The platform service records the trainer under the session (so it can
        release it when the session ends) and forwards the create to the
        training engine. Its future is polled on the service base, and
        resolves to the Training Model ID that every ``/train`` step carries.
        """
        request_id = self._submit_on(
            path, body, base_url=self._config.service_train_base_url
        )
        return self.resolver(base_url=self._config.service_base_url).resolve(
            request_id, timeout=None, parse=P.parse_model_id
        )

    def _submit(self, path: str, body: dict[str, Any]) -> str:
        """Send a request that returns a bare ``request_id`` (train plane)."""
        return self._submit_on(path, body)

    def _submit_on(
        self, path: str, body: dict[str, Any], *, base_url: str | None = None
    ) -> str:
        """Send a request that returns a bare ``request_id`` on a given plane."""
        payload = self._post_json(path, body, base_url=base_url)
        request_id = payload.get("request_id")
        if not request_id:
            raise ConfigurationError(f"{path} returned no request_id: {payload!r}")
        return str(request_id)

    def _post_json(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        pool: str = "control",
        extra_headers: Mapping[str, str] | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        response = self._transport.request(
            "POST",
            path,
            base_url=base_url or self._config.train_base_url,
            pool=pool,
            json_body=body,
            accept=JSON_CONTENT_TYPE,
            extra_headers=self._headers(extra_headers),
            timeout=timeout,
            max_retries=max_retries,
        )
        return _json_object(response)

    def post_proto(
        self,
        path: str,
        body: bytes,
        *,
        content_encoding: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send a protobuf body on the training pool and read a JSON ack."""
        headers = self._headers(extra_headers)
        if content_encoding:
            headers["Content-Encoding"] = content_encoding
        response = self._transport.request(
            "POST",
            path,
            base_url=self._config.train_base_url,
            pool="train",
            content=body,
            content_type=P.PROTO_CONTENT_TYPE,
            accept=JSON_CONTENT_TYPE,
            extra_headers=headers,
        )
        return _json_object(response)

    @property
    def transport(self) -> Transport:
        return self._transport

    # -- lifecycle ------------------------------------------------------------
    def close(
        self,
        status: str | None = None,
        detail: str | None = None,
    ) -> Any:
        """Stop the heartbeat, finish the session, close connections.

        Ordering matters: the heartbeat stops *before* the session is finished,
        matching the constraint the existing reporter contract documents.
        Finishing is best-effort -- the server reclaims an idle session anyway,
        so a failure here must not mask whatever the caller was really doing.

        ``status`` and ``detail`` are accepted for Tinker call-shape
        compatibility (``close("success", detail=...)``). The relay's finish
        endpoint carries no request fields, so they are not transmitted. The
        return value is a completed future-shaped object, so ported code that
        calls ``close(...).result()`` also works.
        """
        with self._lock:
            if self._closed:
                return _CompletedFuture()
            self._closed = True
            heartbeat, self._heartbeat = self._heartbeat, None
            session_id, self._session_id = self._session_id, None

        if heartbeat is not None:
            heartbeat.stop()
        if session_id is not None:
            try:
                self._post_json(
                    P.PATH_FINISH_SESSION.format(session_id=session_id),
                    {},
                    max_retries=0,
                    base_url=self._config.session_base_url,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("finishing session %s failed: %s", session_id, exc)
        self._transport.close()
        return _CompletedFuture()

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        session = self._session_id or "no session"
        return f"<ServiceClient {self._config.origin} ({session})>"


class _CompletedFuture:
    """A completed future-shaped value for Tinker's ``close(...).result()``."""

    __slots__ = ()

    def result(self, timeout: float | None = None) -> None:
        return None

    def done(self) -> bool:
        return True


def _json_object(response: Any) -> dict[str, Any]:
    """Read a JSON object body, or ``{}`` for an empty one."""
    raw = response.content
    if not raw:
        return {}
    payload = response.json()
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"expected a JSON object from {response.request.url}, got {type(payload).__name__}"
        )
    return payload
