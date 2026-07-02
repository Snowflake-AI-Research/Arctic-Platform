# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recipe-side entrypoint shim around ``integrations.arctic_rl.entrypoint``.

Purpose: keep the BIRD launchers dispatching to a recipe-local Ray
``skyrl_entrypoint`` task (defined here) rather than upstream's, so this
recipe never has to name anything under ``$SKYRL_HOME/integrations/arctic_rl/examples/``.
Registration of ``bird`` / ``bird_sql`` still happens via upstream's
``integrations.arctic_rl.envs`` — see ``__init__.py`` for why the driver +
worker both fire that import as a side-effect.

Everything else (``ArcticRLExp``, ``main()`` body, config schemas) is reused
verbatim from upstream — ``$SKYRL_HOME`` must be on ``PYTHONPATH`` (the
launchers do this).
"""

import os
import sys
from typing import Any
from typing import Optional

import ray
from integrations.arctic_rl.config import ArcticRLTrainerConfig
from integrations.arctic_rl.config import ArcticSkyRLConfig
from integrations.arctic_rl.config import build_rl_config
from integrations.arctic_rl.entrypoint import ArcticRLExp
from loguru import logger
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg

from arctic_platform.rl import ArcticRLClientConfig
from arctic_platform.rl import create_arctic_rl_client


@ray.remote(num_cpus=1)
def skyrl_entrypoint(
    cfg: SkyRLTrainConfig,
    reconnect_config: Optional[ArcticRLClientConfig] = None,
    server_state: Optional[Any] = None,
):
    """Ray task that runs the Arctic RL training loop.

    Defined here (not upstream) so workers import THIS module — see module
    docstring for why.
    """
    exp = ArcticRLExp(cfg, reconnect_config=reconnect_config, server_state=server_state)
    exp.run()


def main() -> None:
    """Driver entry. Behavior matches ``integrations.arctic_rl.entrypoint.main``
    exactly, with two diffs:

    - imports ``ArcticSkyRLConfig`` / ``ArcticRLTrainerConfig`` /
      ``build_rl_config`` from the upstream package (we don't vendor them).
    - dispatches to the local ``skyrl_entrypoint`` ray task above (instead of
      upstream's), so workers re-import this package.

    Also forwards the dir holding this ``arctic_rl/`` package onto worker
    ``PYTHONPATH``, just like upstream forwards the SkyRL repo root, so the
    relative import chain (``arctic_rl`` -> upstream envs) resolves in workers too.
    """
    argv = [a for a in sys.argv[1:] if not a.startswith("trainer.override_entrypoint=")]
    cfg = ArcticSkyRLConfig.from_cli_overrides(argv)
    if cfg.trainer.arctic_rl is None:
        cfg.trainer.arctic_rl = ArcticRLTrainerConfig()
    validate_cfg(cfg)

    rl_config = build_rl_config(cfg)
    logger.info("Pre-initializing ArcticRL jobs (before ray.init)…")
    pre_client = create_arctic_rl_client(rl_config)
    reconnect_cfg = pre_client.reconnect_config()
    server_state = pre_client.get_server_state() if rl_config.comm_protocol == "ray" else None
    logger.info(
        f"ArcticRL jobs ready — training={pre_client.training_job_id}, "
        f"sample={pre_client.sampling_job_id}, log_prob={pre_client.log_prob_job_id}"
    )

    from skyrl.train.utils.utils import prepare_runtime_environment

    env_vars = prepare_runtime_environment(cfg)
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("ARCTIC_")})
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("WANDB_")})

    # Forward $SKYRL_HOME (so workers can ``import integrations.arctic_rl.*``)
    # and the parent dir of this ``arctic_rl/`` shim (so workers can ``import
    # arctic_rl.entrypoint``) onto Ray workers' PYTHONPATH.
    skyrl_home = os.environ.get("SKYRL_HOME")
    if not skyrl_home:
        raise RuntimeError(
            "SKYRL_HOME is not set. Clone SkyRL at the pinned commit "
            "(see this recipe's README) and ``export SKYRL_HOME=<clone>``."
        )
    _pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _existing_pp = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = os.pathsep.join(p for p in (skyrl_home, _pkg_parent, _existing_pp) if p)

    runtime_env = {"env_vars": env_vars}
    ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
    ray.get(
        skyrl_entrypoint.options(runtime_env=runtime_env).remote(
            cfg,
            reconnect_config=reconnect_cfg,
            server_state=server_state,
        )
    )


if __name__ == "__main__":
    main()
