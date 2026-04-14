"""Lightweight runtime environment checks.

This module only detects whether Python is running inside an isolated
environment and emits a soft warning when it is not.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys


def running_in_virtualenv(*, prefix: str | Path | None = None, base_prefix: str | Path | None = None) -> bool:
    """Return whether the current interpreter is running in an isolated environment.

    The check only compares the interpreter prefix with the base prefix and does
    not depend on a specific environment name.
    """
    if prefix is None:
        return False
    if base_prefix is None:
        return True
    return Path(prefix).resolve() != Path(base_prefix).resolve()


def warn_if_not_running_in_virtualenv(*, logger: logging.Logger, project_root: Path, script_name: str) -> None:
    """Warn when no virtual environment is active without blocking execution.

    Isolated environments are still recommended for reproducibility, but this
    helper only logs a warning.
    """
    prefix = getattr(sys, "prefix", None)
    base_prefix = getattr(sys, "base_prefix", prefix)
    if running_in_virtualenv(prefix=prefix, base_prefix=base_prefix):
        logger.info("Python environment: virtual environment detected at %s", Path(prefix).resolve())
        return

    logger.warning(
        "%s is running without an isolated virtual environment. This is allowed, but it is safer to install "
        "dependencies inside a dedicated environment such as %s.",
        script_name,
        (project_root / ".venv").resolve(),
    )
