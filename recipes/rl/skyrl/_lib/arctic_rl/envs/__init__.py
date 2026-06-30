"""Register the env classes added by this recipe directory.

Only ``long_context_qa`` lives here — there's no upstream home for it yet
(will be PR'd into SkyRL alongside its recipe).

``bird`` / ``bird_sql`` are *not* re-registered here: importing
``integrations.arctic_rl`` from the user's SkyRL clone already runs the
upstream registration as a side-effect (the shim's ``entrypoint.py`` imports
``integrations.arctic_rl.config``, which triggers it). Doing it again here
would raise ``RegistrationError: name already registered``.

Uses ``__name__`` for the entry_point so registration works regardless of the
path this package gets checked out at.
"""

from skyrl_gym.envs.registration import register

register(id="long_context_qa", entry_point=f"{__name__}.long_context_qa:LongContextQAEnv")
