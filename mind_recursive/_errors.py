"""Exception hierarchy.

Our own classes, not Tinker's. Users who previously wrote
``except tinker.XxxError`` have to change those lines; that cost is accepted.

Two axes matter to callers:

* **retryable** -- transport faults and 429/5xx. The transport layer retries
  these itself; the exception only escapes once the budget is spent.
* **category** -- ``"user"`` or ``"server"``, carried through from the
  upstream error envelope (``{"error": ..., "category": ...}``) so callers can
  tell "I sent something wrong" from "the backend is unwell" without parsing
  status codes.
"""

from __future__ import annotations

__all__ = [
    "MindRecursiveError",
    "ConfigurationError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "ConflictError",
    "GoneError",
    "PayloadTooLargeError",
    "RateLimitError",
    "ServerError",
    "RequestFailedError",
    "FutureTimeoutError",
    "SessionClosedError",
    "status_to_error",
]


class MindRecursiveError(Exception):
    """Base class for everything this SDK raises."""


class ConfigurationError(MindRecursiveError):
    """Bad or missing local configuration -- no request was attempted."""


class APIError(MindRecursiveError):
    """An HTTP request reached the relay and came back unusable."""

    #: Whether retrying the identical request could plausibly succeed.
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        category: str | None = None,
        path: str | None = None,
        body: bytes | None = None,
        code: int | str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.category = category
        self.path = path
        self.body = body
        #: Platform calls only: the server business code (``int``, e.g. 140058)
        #: or the relay entry code (``str``, e.g. ``"credential_rejected"``).
        self.code = code
        #: Relay request id from an entry rejection, for support tickets.
        self.request_id = request_id

    def __str__(self) -> str:
        bits = [self.message]
        if self.code is not None:
            bits.append(f"code={self.code}")
        if self.status is not None:
            bits.append(f"status={self.status}")
        if self.path:
            bits.append(f"path={self.path}")
        if self.category:
            bits.append(f"category={self.category}")
        return " ".join(bits)


class APIConnectionError(APIError):
    """The request never produced a response (DNS, TCP, TLS, reset)."""

    retryable = True


class APITimeoutError(APIConnectionError):
    """The request exceeded its timeout budget."""

    retryable = True


class AuthenticationError(APIError):
    """401. The API key is missing, malformed, or rejected."""


class PermissionDeniedError(APIError):
    """403. Authenticated, but not allowed to touch this resource."""


class NotFoundError(APIError):
    """404.

    On the training plane this is most often the ``/train`` suffix trap rather
    than a genuinely missing resource -- see ``_config.resolve_base_urls``.
    """


class BadRequestError(APIError):
    """400. The server rejected the request shape."""


class ConflictError(APIError):
    """409. State conflict, e.g. reporting to a sealed session."""


class GoneError(APIError):
    """410. The resource existed and has been reclaimed."""


class PayloadTooLargeError(APIError):
    """413. Body exceeded a server limit.

    For ``forward_backward`` this means the chunking budget negotiated through
    ``client/config`` was not respected.
    """


class RateLimitError(APIError):
    """429. Retryable, and ``Retry-After`` is honoured when present."""

    retryable = True

    def __init__(self, *args, retry_after: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retry_after = retry_after


class ServerError(APIError):
    """5xx. Retryable."""

    retryable = True


class RequestFailedError(MindRecursiveError):
    """The future resolved to ``RequestFailedResponse``.

    The HTTP call succeeded; the *work* failed. Carries the upstream
    ``category`` verbatim -- this is the channel through which backend
    constraints surface, e.g.::

        UNSUPPORTED: FSDP2 requires beta1=0.9, beta2=0.999, eps=1e-8
        and lora_plus_ratio=1
    """

    def __init__(self, error: str, *, category: str | None = None, request_id: str | None = None):
        super().__init__(error)
        self.error = error
        self.category = category
        self.request_id = request_id

    def __str__(self) -> str:
        prefix = f"[{self.category}] " if self.category else ""
        suffix = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{prefix}{self.error}{suffix}"


class FutureTimeoutError(MindRecursiveError):
    """``result(timeout=...)`` expired while the request was still pending."""

    def __init__(self, request_id: str, timeout: float, queue_state: str | None = None):
        state = f", last queue_state={queue_state}" if queue_state else ""
        super().__init__(
            f"future {request_id} did not resolve within {timeout}s{state}"
        )
        self.request_id = request_id
        self.timeout = timeout
        self.queue_state = queue_state


class SessionClosedError(MindRecursiveError):
    """The session was closed locally; its clients are no longer usable."""


#: status -> exception class. 404 and 429 get bespoke handling in the caller.
_BY_STATUS: dict[int, type[APIError]] = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    410: GoneError,
    413: PayloadTooLargeError,
    429: RateLimitError,
}


def status_to_error(status: int) -> type[APIError]:
    """Map an HTTP status onto the exception class to raise."""
    if status in _BY_STATUS:
        return _BY_STATUS[status]
    if status >= 500:
        return ServerError
    return APIError
