"""Terminal echo for user-facing SDK events.

The reporting surface has two audiences. ``sc.log`` / ``sc.metric`` /
``sc.artifact`` write to the web console, but the person running the script is
looking at a terminal, and a training script that has to add its own ``print``
next to every report is a worse experience than one that does not. These
methods therefore echo the same information locally.

**Why logging, not print.** A library that prints cannot be silenced, pollutes
stdout, and gives no way to route the text into the caller's own log file.
``logging`` keeps stdout clean for program output and lets a caller raise the
level to hide the echo. To stay zero-config for a plain script, a ``stderr``
handler is installed on the ``mind_recursive`` logger the first time anything
is emitted -- but only when the application has not configured logging itself
(no handler on either the package logger or the root logger). An application
that configures logging keeps full control.

This module is deliberately tiny and import-free apart from the standard
library; importing the package must not configure handlers as a side effect.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["emit"]

#: One logger name for everything the SDK echoes, so a caller can silence the
#: SDK with ``logging.getLogger("mind_recursive").setLevel(logging.WARNING)``.
LOGGER_NAME = "mind_recursive"

_handler_checked = False


def _ensure_handler() -> None:
    """Install the zero-config stderr handler once, if the app has not.

    Runs at first emit rather than at import: importing a library must not
    configure logging behind the application's back. When the application has
    already attached a handler (root or package), it owns formatting and
    filtering, so nothing is added and no duplicate lines appear.
    """
    global _handler_checked
    if _handler_checked:
        return
    _handler_checked = True

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers or logging.getLogger().handlers:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    # Respect an explicit level the caller set before the first report; only
    # pick INFO when nobody has expressed a preference.
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


def emit(message: str, *, level: int = logging.INFO) -> None:
    """Echo one line to the terminal.

    Best-effort by construction: a logging failure must never turn a training
    call into an exception, for the same reason reporting is best-effort.
    """
    try:
        _ensure_handler()
        logging.getLogger(LOGGER_NAME).log(level, message)
    except Exception:  # noqa: BLE001 - echoing must not break training
        pass
