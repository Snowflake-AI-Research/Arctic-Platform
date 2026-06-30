"""Arctic RL integration for SkyRL — vendored into Arctic-Platform.

Lives at ``recipes/rl/skyrl/_lib/arctic_rl/`` in the Arctic-Platform repo and
is added to ``PYTHONPATH`` by the recipe launchers. Invoked via SkyRL's core
dispatch::

    python -m skyrl.train.entrypoints.main_base \\
        trainer.override_entrypoint=arctic_rl.entrypoint <flags>

Provides ``ArcticPPOTrainer`` and ``ArcticGenerator`` that route all GPU work
to an Arctic RL server; depends on ``arctic_platform.rl`` for the client.

Upstream source: ``integrations/arctic_rl/`` in NovaSky-AI/SkyRL — see
``VENDOR.md`` for the pinned SHA and re-sync instructions.
"""

from . import envs as _envs  # noqa: F401  side-effect: register `bird` env
from .generator import ArcticGenerator
from .trainer import ArcticPPOTrainer

__all__ = ["ArcticPPOTrainer", "ArcticGenerator"]
