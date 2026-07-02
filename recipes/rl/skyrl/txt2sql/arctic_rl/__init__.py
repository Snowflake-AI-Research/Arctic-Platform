"""Recipe-side ``arctic_rl`` package shim.

This package exists for one reason: ensure ``bird`` / ``bird_sql`` are
registered with ``skyrl_gym`` on both the SkyRL driver and inside each Ray
worker that ``skyrl_gym.make()``-s an env.

Registration itself lives upstream — importing ``integrations.arctic_rl.envs``
runs it. This shim gives us a recipe-local module whose *import* triggers that
side-effect, without the recipe launcher ever having to name anything in
``$SKYRL_HOME/integrations/arctic_rl/examples/``.

How the side-effect lands in workers:

1. The recipe launcher passes ``trainer.override_entrypoint=arctic_rl.entrypoint``.
2. SkyRL's ``main_base`` imports that module on the driver, which imports this
   package (here), which imports upstream ``integrations.arctic_rl.envs`` for
   its ``register()`` side-effects.
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

import integrations.arctic_rl.envs as _envs  # noqa: F401  register bird / bird_sql
