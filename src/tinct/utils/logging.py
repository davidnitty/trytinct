"""Structured logging helpers.

Uses Python's stdlib ``logging`` for machine-readable records and ``rich`` for
pleasant console output. Importing this module does not pull in any heavy ML
dependency.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


_console: Optional["Console"] = None


def get_console() -> "Console":
    """Return a shared rich console, creating a plain stdout fallback if rich
    is unavailable."""
    global _console
    if _console is None:
        if _HAS_RICH:
            from rich.console import Console
            _console = Console()
        else:  # pragma: no cover
            class _Plain:
                def print(self, *args, **kwargs):  # noqa: D102
                    print(*args)
            _console = _Plain()  # type: ignore[assignment]
    return _console


def setup_logging(level: int = logging.INFO, verbose: bool = False) -> None:
    """Configure the root logger (once). ``verbose`` enables debug output."""
    level = logging.DEBUG if verbose else level
    handlers: list[logging.Handler]
    if _HAS_RICH:
        handlers = [RichHandler(rich_tracebacks=True, markup=True)]
    else:  # pragma: no cover
        handlers = [logging.StreamHandler(sys.stderr)]

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
