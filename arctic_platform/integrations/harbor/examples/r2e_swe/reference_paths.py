"""Locate the prime-rl checkout this example reads its harness from.

The agent program, its grading policy and the chat template are read from a
prime-rl checkout rather than vendored here, so the rollouts stay
byte-identical to the run being reproduced. That checkout is not part of this
repository and its location is site-specific, so it is supplied through the
environment instead of being hard-coded::

    export PRIME_RL_ROOT=/path/to/prime-rl

Resolution is deliberately lazy. Several modules here are imported for things
that need no checkout at all -- the grading policy constants, the
terminal-stop-condition logic -- and those imports must not fail just because
the variable is unset.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "PRIME_RL_ROOT"
R2E_DATASET_ENV = "R2E_DATASET"
K3S_DIR_ENV = "K3S_DIR"

_VERIFIERS_REL = "deps/verifiers/verifiers/v1"
_TEMPLATE_REL = "prime_snowrl/configs/chat_templates/qwen35_preserve_all_thinking.jinja"


def root() -> Path | None:
    """The configured checkout, or ``None`` when unset."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def require_root() -> Path:
    value = root()
    if value is None:
        raise RuntimeError(
            f"{ENV_VAR} is not set. Point it at a prime-rl checkout: "
            f"export {ENV_VAR}=/path/to/prime-rl"
        )
    if not value.is_dir():
        raise RuntimeError(f"{ENV_VAR}={value} is not a directory")
    return value


def verifiers_v1() -> Path:
    return require_root() / _VERIFIERS_REL


def harness_dir() -> Path:
    return verifiers_v1() / "harnesses" / "mini_swe_agent_plus"


def chat_template() -> Path:
    return require_root() / _TEMPLATE_REL


def r2e_dataset() -> Path:
    """The R2E-Gym instance table the driver samples tasks from.

    A site-local export of the public dataset rather than something this repo
    ships, so it is configured the same way as the checkout::

        export R2E_DATASET=/path/to/train.jsonl
    """
    value = os.environ.get(R2E_DATASET_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{R2E_DATASET_ENV} is not set. Point it at the R2E-Gym instance "
            f"table: export {R2E_DATASET_ENV}=/path/to/train.jsonl"
        )
    return Path(value)


def k3s_dir() -> Path:
    """Where the in-pod k3s install put its binary and kubeconfig."""
    return Path(os.environ.get(K3S_DIR_ENV, "").strip() or "/data-fast/k3s")


def default_chat_template() -> str:
    """Template path for an argparse default; empty when unconfigured.

    The driver treats an empty ``--chat-template`` as "use the stock one", so
    an unset checkout degrades to stock rather than crashing at parse time.
    """
    value = root()
    return str(value / _TEMPLATE_REL) if value else ""
