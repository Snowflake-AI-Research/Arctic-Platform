"""Auto-import hook for SkyRL × Arctic RL recipe env registration.

Python's ``site.py`` imports a module named ``sitecustomize`` (if found on
``sys.path``) at every interpreter startup. The launchers put this directory
on ``PYTHONPATH``, so this file runs in:

  - the SkyRL driver process,
  - the Ray actor that runs the ``@ray.remote skyrl_entrypoint`` task,
  - every ``multiprocessing.spawn`` child that
    ``integrations.arctic_rl.generator``'s reward-scoring
    ``ProcessPoolExecutor`` spins up (these children re-run Python from
    scratch, so they don't inherit the parent's registered envs).

The PPE-child case is the load-bearing one: ``_score_one`` calls
``skyrl_gym.make(env_class)`` in the child and would otherwise raise
``RegistrationError: No registered env with id: bird`` because ``main_base``'s
imports never touched ``integrations.arctic_rl.envs`` in the child.
"""

try:
    import arctic_rl  # noqa: F401  side-effect: register bird / bird_sql
except Exception:
    # arctic_rl shim not importable (e.g. outside the recipe's PYTHONPATH);
    # let site.py continue without erroring the interpreter.
    pass
