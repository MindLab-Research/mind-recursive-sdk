"""Credential and base-URL resolution.

Two rules:

1. **No global state.** Settings resolve when a client is constructed; nothing
   reads or rewrites ``os.environ`` at import time, so one process can talk to
   several backends.

2. **One fixed root.** The SDK takes the platform origin and appends the fixed
   ``/api/mindrsi/v1`` root itself; the planes differ by the path segment after
   it (``/train``, ``/sample``, ``/service``). ``MIND_RECURSIVE_BASE_URL`` is
   therefore the origin -- optionally with that suffix -- and nothing else. Any
   other path is rejected rather than reinterpreted.

**One built-in default: production.** With no ``base_url`` argument and no
``MIND_RECURSIVE_BASE_URL``, the SDK talks to :data:`DEFAULT_BASE_URL`, so a
user only has to set ``MIND_RECURSIVE_API_KEY``. Other environments set the
base URL explicitly; the default is a fallback, never an override.

Precedence for both key and URL: explicit argument > environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from ._errors import ConfigurationError

__all__ = [
    "Config",
    "resolve_base_urls",
    "resolve_api_key",
    "normalize_origin",
    "MIND_RECURSIVE_PREFIX",
    "SERVICE_SEGMENT",
    "SESSION_SEGMENT",
    "TRAIN_SEGMENT",
    "SAMPLE_SEGMENT",
    "SERVICE_SAMPLE_SEGMENT",
    "SERVICE_TRAIN_SEGMENT",
    "REPORT_SEGMENT",
    "PLATFORM_PREFIX",
    "ENV_API_KEYS",
    "ENV_BASE_URLS",
    "ENV_WEB_URLS",
    "DEFAULT_BASE_URL",
    "DEFAULT_WEB_URL",
    "resolve_web_url",
    "web_session_url",
]

#: Env names for each setting. Kept as tuples so lookups stay uniform.
ENV_API_KEYS = ("MIND_RECURSIVE_API_KEY",)
ENV_BASE_URLS = ("MIND_RECURSIVE_BASE_URL",)
ENV_WEB_URLS = ("MIND_RECURSIVE_WEB_URL",)

#: The production origin, used when neither ``base_url=`` nor
#: ``MIND_RECURSIVE_BASE_URL`` is set. Point other environments at their
#: origin explicitly.
DEFAULT_BASE_URL = "https://mintcn.macaron.xin"

#: Web console path for one session. The SDK only *prints* this link -- it is
#: not an API plane and never receives a request. ``MIND_RECURSIVE_WEB_URL``
#: overrides it for another deployment.
DEFAULT_WEB_URL = "https://mintrecursive.mindlab.im/v2train/records/sessions/"

ENV_API_KEY = ENV_API_KEYS[0]
ENV_BASE_URL = ENV_BASE_URLS[0]
ENV_WEB_URL = ENV_WEB_URLS[0]


def _first_env(names: tuple[str, ...], environ: dict[str, str]) -> str:
    """First non-blank value among ``names``, in preference order."""
    for name in names:
        value = environ.get(name, "").strip()
        if value:
            return value
    return ""

#: The one fixed root the SDK appends to the relay origin. There is no
#: per-plane base URL any more: the planes differ by the path segment after
#: this root, which names the destination platform (``/train``,
#: ``/sample``, ``/service``), and the relay routes on it. Keeping one root
#: means a caller configures an origin, not a plane.
MIND_RECURSIVE_PREFIX = "/api/mindrsi/v1"

#: Path segment for connectivity and negotiation: capability discovery,
#: health, and the client behaviour config. None of it is about training.
SERVICE_SEGMENT = "/service"

#: Path segment for session lifecycle: create, heartbeat, finish.
#: A session is the container both trainers and samplers live in, so its
#: lifecycle is not a training-plane concept and does not hang under ``/train``.
#: It is owned by the platform service, so it sits under ``/service``.
SESSION_SEGMENT = "/service/session"

#: Path segment for the training plane (session, model, steps, weights, and the
#: future that belongs to a training call).
TRAIN_SEGMENT = "/train"

#: Path segment for the sampling plane (sampling session, ``asample``, and the
#: future that belongs to a sampling call).
SAMPLE_SEGMENT = "/sample"

#: Path segment for creating a sampler whose weights come from training (a
#: saved ``model_path``). The inference service can only serve a path once it is deployed,
#: and that temporary deployment is owned by the platform service, so this
#: create goes through ``/service`` and resolves via the service plane's
#: future. A sampler keyed on a public ``base_model`` instead names an
#: already-deployed model and stays relay-direct on ``/sample``.
SERVICE_SAMPLE_SEGMENT = "/service/sample"

#: Path segment for creating a trainer. The platform service owns the
#: training control plane: it records each trainer under the session and
#: releases it when the session ends, so ``create_model`` and
#: ``create_model_from_state`` go through ``/service`` and resolve via the
#: service plane's future. Training steps stay relay-direct on ``/train``.
SERVICE_TRAIN_SEGMENT = "/service/train"

#: Path segment for the SDK's session reports: log / metric / artifact /
#: summary, keyed by the client's own ``session_id``.
REPORT_SEGMENT = "/service/report"

#: The platform API root: datasets, managed training, models and deployments
#: (``mind_recursive.platform``). A sibling of :data:`MIND_RECURSIVE_PREFIX`,
#: not a segment under it.
PLATFORM_PREFIX = "/api/mint/cli/v1"


def normalize_origin(url: str) -> str:
    """Reduce an accepted base-URL spelling to a bare relay origin.

    Accepts the origin with or without the fixed ``/api/mindrsi/v1`` suffix, and
    rejects any other path. There is deliberately no ``/train`` handling: the
    old spelling was exactly the confusion this design removes, so it fails
    loudly instead of being silently reinterpreted.

        >>> normalize_origin("https://relay.example.com")
        'https://relay.example.com'
        >>> normalize_origin("https://relay.example.com/api/mindrsi/v1/")
        'https://relay.example.com'

    Raises:
        ConfigurationError: not an absolute http(s) URL, carries a
            query/fragment, or carries a path other than the fixed prefix.
    """
    raw = (url or "").strip()
    if not raw:
        raise ConfigurationError("base_url is empty")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise ConfigurationError(
            f"base_url must be an absolute http(s) URL, got {raw!r}"
        )
    if not parts.netloc:
        raise ConfigurationError(f"base_url has no host: {raw!r}")
    if parts.query or parts.fragment:
        raise ConfigurationError(
            f"base_url must not carry a query or fragment, got {raw!r}"
        )
    path = parts.path.rstrip("/")
    if path.endswith(MIND_RECURSIVE_PREFIX):
        path = path[: -len(MIND_RECURSIVE_PREFIX)]
    path = path.rstrip("/")
    if path:
        raise ConfigurationError(
            f"base_url must be the relay origin (optionally with the fixed "
            f"{MIND_RECURSIVE_PREFIX} suffix), got path {parts.path!r}. The SDK "
            f"appends {MIND_RECURSIVE_PREFIX} itself; do not pass /train or a full "
            f"endpoint."
        )
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


@dataclass(frozen=True)
class Config:
    """Resolved connection settings. Immutable; built once per client."""

    api_key: str
    origin: str
    #: ``origin + MIND_RECURSIVE_PREFIX`` -- the one fixed root every call hangs off.
    api_base_url: str
    service_base_url: str
    session_base_url: str
    train_base_url: str
    sample_base_url: str
    service_sample_base_url: str
    service_train_base_url: str
    report_base_url: str
    #: ``origin + PLATFORM_PREFIX``. Defaulted so a hand-built ``Config``
    #: from before the platform calls existed still constructs.
    platform_base_url: str = ""
    #: Web console base for one session; a link is this plus the session id.
    #: Printed on session creation, never called.
    web_url: str = DEFAULT_WEB_URL
    timeout: float = 600.0
    connect_timeout: float = 30.0
    max_retries: int = 3
    #: ``trust_env=False`` on the httpx client. A local SOCKS proxy picked up
    #: from the ambient environment has previously hijacked traffic that was
    #: meant to stay on the relay path; the platform's billing and
    #: reporter clients set this for the same reason.
    trust_env: bool = False

    def redacted(self) -> dict[str, str]:
        """Loggable form -- never includes the key."""
        tail = self.api_key[-4:] if len(self.api_key) >= 4 else "?"
        return {"origin": self.origin, "api_key": f"sk-...{tail}"}


def resolve_api_key(api_key: str | None, env: dict[str, str] | None = None) -> str:
    """Resolve the API key: argument, then ``MIND_RECURSIVE_API_KEY``.

    The key's prefix is not validated locally; the server decides.
    """
    environ = os.environ if env is None else env
    if api_key is not None:
        key = api_key.strip()
    else:
        key = _first_env(ENV_API_KEYS, environ)
    if not key:
        raise ConfigurationError(
            f"no API key: pass api_key=... or set {ENV_API_KEYS[0]}; "
            "user keys start with 'sk-'"
        )
    return key


def resolve_web_url(env: dict[str, str] | None = None) -> str:
    """Resolve the web-console base: ``MIND_RECURSIVE_WEB_URL``, else default.

    This URL is only used to build the link printed when a session opens; it
    is not an API endpoint, so it is not validated as one.
    """
    environ = os.environ if env is None else env
    return _first_env(ENV_WEB_URLS, environ) or DEFAULT_WEB_URL


def web_session_url(web_url: str, session_id: str) -> str:
    """Join the console base and a session id with exactly one slash."""
    return f"{web_url.rstrip('/')}/{session_id}"


def resolve_base_urls(
    base_url: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve per-plane base URLs from one relay origin.

    Precedence: ``base_url`` argument > ``MIND_RECURSIVE_BASE_URL`` >
    :data:`DEFAULT_BASE_URL` (production).

    Returns:
        ``{"origin", "api", "service", "session", "train", "sample", "report",
        "platform"}``.

    Raises:
        ConfigurationError: an explicitly given base URL is empty or unusable.
    """
    environ = os.environ if env is None else env
    if base_url is not None:
        raw = base_url.strip()
        if not raw:
            raise ConfigurationError(
                "base_url is empty: pass the relay origin, or omit it to use "
                f"{ENV_BASE_URLS[0]} / the default {DEFAULT_BASE_URL}"
            )
    else:
        raw = _first_env(ENV_BASE_URLS, environ) or DEFAULT_BASE_URL
    origin = normalize_origin(raw)
    api = origin + MIND_RECURSIVE_PREFIX
    return {
        "origin": origin,
        "api": api,
        "service": api + SERVICE_SEGMENT,
        "session": api + SESSION_SEGMENT,
        "train": api + TRAIN_SEGMENT,
        "sample": api + SAMPLE_SEGMENT,
        "service_sample": api + SERVICE_SAMPLE_SEGMENT,
        "service_train": api + SERVICE_TRAIN_SEGMENT,
        "report": api + REPORT_SEGMENT,
        "platform": origin + PLATFORM_PREFIX,
    }


def build_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 600.0,
    connect_timeout: float = 30.0,
    max_retries: int = 3,
    trust_env: bool = False,
    env: dict[str, str] | None = None,
) -> Config:
    """Assemble a :class:`Config` from arguments and environment."""
    urls = resolve_base_urls(base_url, env)
    return Config(
        api_key=resolve_api_key(api_key, env),
        origin=urls["origin"],
        api_base_url=urls["api"],
        service_base_url=urls["service"],
        session_base_url=urls["session"],
        train_base_url=urls["train"],
        sample_base_url=urls["sample"],
        service_sample_base_url=urls["service_sample"],
        service_train_base_url=urls["service_train"],
        report_base_url=urls["report"],
        platform_base_url=urls["platform"],
        web_url=resolve_web_url(env),
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        trust_env=trust_env,
    )
