"""FSDP-native SkyRL entrypoint that registers ``long_context_qa`` on the
driver + arranges for Ray workers to do the same.

Mirrors upstream's ``integrations/arctic_rl/examples/fsdp_bird_entry.py``: stock
``main_base`` doesn't know about our env id, so we side-effect-import the
recipe-local ``arctic_rl.envs`` package here (registers ``long_context_qa`` on
the driver) and monkey-patch ``prepare_runtime_environment`` to forward the
recipe dir onto worker ``runtime_env`` PYTHONPATH. We also replace
``main_base.skyrl_entrypoint`` with a version that re-imports the shim inside
Ray workers so registration fires there too.

This is the FSDP-native sibling of ``arctic_rl/entrypoint.py`` (Arctic RL
path). Both do the same three-interpreter-registration dance; the Arctic path
goes through ``ArcticRLExp``, this one goes through ``BasePPOExp``.
"""

import os
import sys
from pathlib import Path

import ray

_HERE = Path(__file__).resolve()
_RECIPE_DIR = _HERE.parent          # long_context_qa/
_RECIPE_DIR_STR = str(_RECIPE_DIR)

# Add the recipe dir to sys.path BEFORE importing so ``arctic_rl.envs`` is found.
# ``sitecustomize.py`` (also in this dir) is already imported by ``site.py`` at
# interpreter startup thanks to the launcher's PYTHONPATH, so
# ``long_context_qa`` env registration + the
# ``torch.nn.Module.named_non_persistent_buffers`` backport (needed by the
# SkyRL FSDP model wrapper on torch-2.10.0+cu128) are already in place by the
# time this file runs.
if _RECIPE_DIR_STR not in sys.path:
    sys.path.insert(0, _RECIPE_DIR_STR)

import arctic_rl.envs  # noqa: E402,F401  side-effect: register long_context_qa

import skyrl.train.entrypoints.main_base as _mb  # noqa: E402
import skyrl.train.utils.utils as _skyrl_utils  # noqa: E402

# Forward the recipe dir onto Ray workers' PYTHONPATH so worker-side imports of
# arctic_rl.envs resolve. Same idea as upstream's fsdp_bird_entry.py, minus the
# SKYRL_USE_LIGER passthrough (we don't need it here).
_original_prepare = _skyrl_utils.prepare_runtime_environment


def _patched_prepare(cfg):
    env_vars = _original_prepare(cfg)
    existing_pp = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if _RECIPE_DIR_STR not in existing_pp.split(os.pathsep):
        env_vars["PYTHONPATH"] = _RECIPE_DIR_STR + (
            os.pathsep + existing_pp if existing_pp else ""
        )
    if "SKYRL_USE_LIGER" in os.environ:
        env_vars["SKYRL_USE_LIGER"] = os.environ["SKYRL_USE_LIGER"]
    return env_vars


_skyrl_utils.prepare_runtime_environment = _patched_prepare


@ray.remote(num_cpus=1)
def _skyrl_entrypoint_with_lcq(cfg):
    """Ray worker task that re-registers ``long_context_qa`` before running."""
    import sys as _sys

    if _RECIPE_DIR_STR not in _sys.path:
        _sys.path.insert(0, _RECIPE_DIR_STR)
    import arctic_rl.envs  # noqa: F401  register long_context_qa in the worker

    exp = _mb.BasePPOExp(cfg)
    exp.run()


_mb.skyrl_entrypoint = _skyrl_entrypoint_with_lcq


if __name__ == "__main__":
    _mb.main()
