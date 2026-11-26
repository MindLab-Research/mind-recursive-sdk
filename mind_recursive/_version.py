"""Package version, read from installed metadata.

Single source of truth is ``pyproject.toml``. Nothing here is hand-written:
a second hard-coded literal silently drifts, and ``SDK_VERSION`` is sent to
the server to resolve version-gated flags, so a stale value changes behaviour
rather than just printing wrong.

Lives in its own module because ``mind_recursive/__init__`` imports from
``mind_recursive.service``, so ``service`` cannot import back from ``mind_recursive``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _metadata_version

try:
    __version__ = _metadata_version("mind-recursive")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0.0.0"

#: Sent in ``create_session`` and ``client/config``. Honest about which client
#: this is: pretending to be the stock SDK would make server-side debugging
#: harder for everyone.
SDK_VERSION = f"mind-recursive-python/{__version__}"

__all__ = ["SDK_VERSION", "__version__"]
