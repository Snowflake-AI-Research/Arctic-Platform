"""Recipe-side ``arctic_rl`` package shim.

This package exists for one reason: register the recipe-private
``long_context_qa`` env (and re-bind the upstream ``bird``/``bird_sql`` ids)
with ``skyrl_gym`` in a way that runs *both* on the SkyRL driver and inside
each Ray worker that ``skyrl_gym.make()``-s an env.

How the side-effect lands in workers:

1. The recipe launcher passes ``trainer.override_entrypoint=arctic_rl.entrypoint``.
2. SkyRL's ``main_base`` imports that module on the driver, which imports this
   package (here), which imports ``arctic_rl.envs`` for its ``register()``
   side-effects.
3. ``arctic_rl.entrypoint`` re-binds Arctic RL's ``@ray.remote`` worker task
   to our package, so the worker re-imports ``arctic_rl.entrypoint`` (and
   therefore this ``__init__.py``) when it deserializes the task — running the
   same registration in the worker's address space.

The actual Arctic RL × SkyRL machinery (config/trainer/generator) is *not*
vendored — it lives in the user's SkyRL checkout at
``$SKYRL_HOME/integrations/arctic_rl/``. The recipe launchers add
``$SKYRL_HOME`` and this directory to ``PYTHONPATH`` so both packages are
importable side-by-side.
"""

from . import envs as _envs  # noqa: F401  side-effect: register envs
