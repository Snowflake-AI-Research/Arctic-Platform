"""Minimal stdlib-logging shim.

Replaces prime-rl's loguru-based logger with a plain ``logging.Logger`` so the
carved-out Qwen3.5 loading path has no ``loguru`` dependency. Only the
``.info``/``.debug``/``.warning``/``.error`` methods are used by this package.
"""

from __future__ import annotations

import logging

_LOGGER: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        logger = logging.getLogger("dss.qwen35")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)7s %(message)s", "%H:%M:%S"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        _LOGGER = logger
    return _LOGGER
