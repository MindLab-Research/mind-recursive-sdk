"""``UserClient``: user-level calls that need no session.

The SDK has two clients, split by what they operate on rather than by how they
authenticate -- both carry the same ``sk-`` key.

:class:`~mind_recursive.service.ServiceClient` creates and owns one training session:
constructing it opens the session, and everything it offers belongs inside
that session. :class:`UserClient` operates on the *user*: what the backend
serves, whether it answers at all, and -- once the platform exposes them --
which sessions and checkpoints the key owns and what they cost. None of that
needs a session, and creating one to ask would leave a billable record behind
for a question that never touched a GPU.

This is Tinker's ``RestClient`` split (``rest_client.py:68``, "no ServiceClient
or training session is needed"), with Tinker's connection behaviour too: a
session-less client's connections are cheap, so construction sends nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from . import _protocol as P
from ._config import Config, build_config
from ._errors import ConfigurationError
from ._transport import JSON_CONTENT_TYPE, AsyncTransport, Transport
from .platform import (
    Account,
    Chat,
    Datasets,
    Deployments,
    PlatformCapabilities,
    Training,
    Models,
    _InferenceHTTP,
    _Platform,
)
from .types import ServerCapabilities, SupportedModel

__all__ = ["UserClient"]

log = logging.getLogger("mind_recursive.user")

#: Queries answer in a second or two. ``ServiceClient``'s 600s default sizes
#: ``forward_backward``, which nothing here resembles.
DEFAULT_QUERY_TIMEOUT = 60.0


class UserClient:
    """User-level queries against the Mind Recursive platform. No session, nothing billable.

    Construction resolves credentials and URLs locally and sends no request, so
    a failure here is a configuration error, never a network one.

    Args:
        api_key: user key (``sk-...``). Falls back to ``MIND_RECURSIVE_API_KEY``.
        base_url: platform origin, with or without the fixed ``/api/mindrsi/v1``
            suffix. Falls back to ``MIND_RECURSIVE_BASE_URL``, then the
            production default.
        timeout: per-request timeout in seconds.
        max_retries: retries for retryable failures (transport, 429, 5xx).

    Example::

        import mind_recursive

        with mind_recursive.UserClient() as uc:
            if uc.health_check():
                for model in uc.get_server_capabilities().supported_models:
                    print(model.model_name)

    Platform resources hang off it as namespaces, all session-less, on the
    platform API (``/api/mint/cli/v1``)::

        uc.whoami()                       # GET /me
        uc.datasets.upload("train.jsonl", type="sft")
        run = uc.training.submit(dataset_version=..., base_model=..., config={...})
        uc.training.wait(run.execution_id)
        uc.models.get(run.version_id)
        dep = uc.deployments.wait(uc.deployments.create(version_id=...).deployment_id)
        uc.chat.completions.create(model=dep.public_model_id,
                                   messages=[{"role": "user", "content": "hi"}])

    See :mod:`mind_recursive.platform`. Not implemented yet: ``list_sessions``,
    ``list_checkpoints`` and billing usage for SDK sessions. Note that the upstream
    Tinker REST endpoints are *not* the answer: through the relay's shared
    upstream key their list calls return global resources rather than the
    caller's, which is why ``mind_recursive.tinker`` deliberately omits
    ``RestClient``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_QUERY_TIMEOUT,
        connect_timeout: float = 30.0,
        max_retries: int = 3,
        trust_env: bool = False,
        config: Config | None = None,
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
        self._async_transport: AsyncTransport | None = None
        self._closed = False
        self._platform = _Platform(self._config, self._transport, self)
        #: Datasets: upload, list, rows, validation report, download, delete.
        self.datasets = Datasets(self._platform)
        #: Managed training: submit / retry, and your training records
        #: (list, wait, cancel, progress, metrics, events).
        self.training = Training(self._platform)
        #: Your models: the versions managed training produced, and their lineage.
        self.models = Models(self._platform)
        #: Deployments of model versions: create, wait, stop, start, delete.
        self.deployments = Deployments(self._platform)
        _inference = _InferenceHTTP(self._platform)
        #: OpenAI format: ``uc.chat.completions.create(model=..., messages=...)``.
        self.chat = Chat(_inference)

    # -- introspection --------------------------------------------------------
    @property
    def config(self) -> Config:
        return self._config

    @property
    def transport(self) -> Transport:
        return self._transport

    # -- queries --------------------------------------------------------------
    def get_server_capabilities(self) -> ServerCapabilities:
        """What the backend serves right now.

        The cheapest end-to-end connectivity and credential check available:
        it reaches the backend, proves the key is accepted, and creates
        nothing.
        """
        payload = self._get_json(P.PATH_CAPABILITIES)
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

    def whoami(self) -> Account:
        """Who this key belongs to, and whether it may start training."""
        return Account.from_wire(self._platform.call("GET", "/me") or {})

    def get_platform_capabilities(self) -> PlatformCapabilities:
        """Which platform features (training, retry, deployment...) are on."""
        return PlatformCapabilities.from_wire(self._platform.call("GET", "/capabilities") or {})

    def health_check(self) -> bool:
        """Whether the training plane answers ``healthz``.

        Reports, never raises: a probe that throws is useless in the
        ``if not uc.health_check():`` it exists for. Use
        :meth:`get_server_capabilities` when you want the failure's reason.
        """
        try:
            self._get_json(P.PATH_HEALTHZ)
            return True
        except Exception:  # noqa: BLE001 - a health probe reports, never raises
            return False

    # -- plumbing -------------------------------------------------------------
    def _get_json(
        self,
        path: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ConfigurationError("this UserClient has been closed")
        response = self._transport.request(
            "GET",
            path,
            base_url=self._config.service_base_url,
            accept=JSON_CONTENT_TYPE,
            extra_headers=dict(extra_headers or {}),
        )
        raw = response.content
        if not raw:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise ConfigurationError(
                f"expected a JSON object from {response.request.url}, "
                f"got {type(payload).__name__}"
            )
        return payload

    # -- lifecycle ------------------------------------------------------------
    @property
    def async_transport(self) -> AsyncTransport:
        """Lazy async twin of :attr:`transport`."""
        if self._async_transport is None:
            self._async_transport = AsyncTransport(self._config)
        return self._async_transport

    async def get_server_capabilities_async(self) -> ServerCapabilities:
        """Async twin of :meth:`get_server_capabilities`."""
        payload = await self._aget_json(P.PATH_CAPABILITIES)
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

    async def health_check_async(self) -> bool:
        """Async twin of :meth:`health_check`."""
        try:
            await self._aget_json(P.PATH_HEALTHZ)
            return True
        except Exception:  # noqa: BLE001 - a health probe reports, never raises
            return False

    async def _aget_json(
        self,
        path: str,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise ConfigurationError("this UserClient has been closed")
        response = await self.async_transport.request(
            "GET",
            path,
            base_url=self._config.service_base_url,
            accept=JSON_CONTENT_TYPE,
            extra_headers=dict(extra_headers or {}),
        )
        raw = response.content
        if not raw:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise ConfigurationError(
                f"expected a JSON object from {response.request.url}, "
                f"got {type(payload).__name__}"
            )
        return payload

    async def close_async(self) -> None:
        """Async twin of :meth:`close`."""
        if self._closed:
            return
        self._closed = True
        if self._async_transport is not None:
            await self._async_transport.aclose()
            self._async_transport = None
        self._transport.close()

    aclose = close_async

    def close(self) -> None:
        """Close connections. There is no session to finish."""
        if self._closed:
            return
        self._closed = True
        self._transport.close()

    def __enter__(self) -> UserClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = " (closed)" if self._closed else ""
        return f"<UserClient {self._config.origin}{state}>"
