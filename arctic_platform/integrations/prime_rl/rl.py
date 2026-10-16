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

"""Launch an RL run whose trainer and policy inference run on Snowflake Cortex.

The orchestrator and env servers run locally exactly as under ``rl``; the Cortex worker
(``arctic_platform.integrations.prime_rl.worker``) stands in for the trainer and the
inference server. The
launcher owns the Cortex job: it creates it, waits for it to run, and cancels it on exit.
"""

import json
import os
import signal
import sys
import uuid
from pathlib import Path

from huggingface_hub import snapshot_download

from prime_rl.entrypoints.dashboard import ensure_dashboard, log_dashboard_url
from prime_rl.entrypoints.rl import (
    ENVS_DIR,
    ORCHESTRATOR_CONFIG,
    env_server_names,
    env_servers,
    rl_config_components,
    write_subconfigs,
)
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import (
    clean_future_steps,
    format_config_message,
    format_log_message,
    prepare_attempt_dirs,
    validate_run_dir,
    write_launch_artifacts,
)
from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, ProcessGroup, set_proc_title

from .config import CortexRLConfig, CortexWorkerConfig
from .jobs import CortexJobs, job_body

WORKER_MODULE = "arctic_platform.integrations.prime_rl.worker"
CORTEX_WORKER_CONFIG = "cortex.json"
CORTEX_JOB_BODY = "cortex_job_body.json"
CORTEX_JOB = "cortex_job.json"


def write_worker_config(config: CortexRLConfig, job_id: str, config_dir: Path) -> Path:
    worker = CortexWorkerConfig(
        cortex=config.cortex,
        job_id=job_id,
        trainer=config.trainer,
        max_model_len=config.orchestrator.seq_len,
        renderer=config.orchestrator.renderer,
    )
    path = config_dir / CORTEX_WORKER_CONFIG
    path.write_text(json.dumps(dump_resolved_config(worker), indent=2))
    return path


def cortex_rl(config: CortexRLConfig) -> None:
    os.environ.setdefault("PRL_RUN_ID", uuid.uuid4().hex)
    assert config.run.name is not None  # resolved at construction
    os.environ["PRL_RUN_NAME"] = config.run.name
    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    clean = config.clean and not os.environ.get("NEVER_CLEAN")
    validate_run_dir(config.run_dir, output_dir=config.output_dir, resuming=False, clean=clean)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    clean_future_steps(config.run_dir, -1)

    config_dir, log_dir = prepare_attempt_dirs(config.run_dir)
    write_launch_artifacts(config_dir, "cortex-rl")
    write_subconfigs(config, config_dir)
    body = job_body(config)
    (config_dir / CORTEX_JOB_BODY).write_text(json.dumps(body, indent=2))
    components = [*rl_config_components(config, config_dir), ("Cortex job", config_dir / CORTEX_JOB_BODY)]
    logger.info(f"Configs:\n{format_config_message(config_dir, 'cortex-rl', components)}")
    if config.dry_run:
        logger.success("Dry run complete. To start the run, remove --dry-run from your command.")
        return

    # Config and tokenizer only: the weights live on Cortex.
    snapshot_download(
        repo_id=config.trainer.model.name,
        repo_type="model",
        allow_patterns=["*.json", "*.txt", "tokenizer*", "*.jinja"],
    )

    # A plain SIGTERM would skip the cleanup below and leave the job holding its GPUs.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))
    jobs = CortexJobs(config.cortex.connection, config.cortex.request_timeout)
    created = config.cortex.job_id is None
    job_id = jobs.create(body) if created else config.cortex.job_id
    (config.run_dir / CORTEX_JOB).write_text(json.dumps({"job_id": job_id, "created": created}, indent=2))
    logger.info(f"{'Created' if created else 'Attached to'} Cortex job {job_id}")
    try:
        jobs.wait_running(job_id, config.cortex.job_ready_timeout)
        worker_config = write_worker_config(config, job_id, config_dir)
        logger.info(
            format_log_message(
                log_dir=log_dir,
                trainer=False,
                orchestrator=True,
                inference=False,
                env_names={split: env_server_names(config, split) for split in ("train", "eval")},
            )
            + f"\n  {'Cortex worker:':<18}tail -F {log_dir / 'cortex.log'}"
        )
        dashboard_url = ensure_dashboard(config.output_dir, logger) if config.dashboard else None

        # Shared W&B mode, as under `rl`: the worker logs the trainer side of one run.
        wandb_shared_env = {"WANDB_SHARED_MODE": "1", "WANDB_RUN_ID": os.environ["PRL_RUN_ID"]}
        with ProcessGroup() as processes:
            for split, source in env_servers(config):
                name = source.resolved_name
                logger.info(f"Starting {name} server")
                processes.start(
                    f"{split} env server {name}",
                    ["env-server", "@", (config_dir / ENVS_DIR / split / f"{name}.json").as_posix()],
                    env={**os.environ, **DEFAULT_COMMON_ENV_VARS, **config.env_vars, **config.orchestrator.env_vars},
                    log_path=log_dir / ENVS_DIR / split / f"{name}.log",
                )

            logger.info(f"Starting the Cortex worker at {config.cortex.base_url}")
            processes.start(
                "cortex",
                [sys.executable, "-m", WORKER_MODULE, "@", worker_config.as_posix()],
                env={
                    **os.environ,
                    **DEFAULT_COMMON_ENV_VARS,
                    "LOGURU_FORCE_COLORS": "1",
                    **config.env_vars,
                    **config.trainer.env_vars,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "trainer",
                },
                log_path=log_dir / "cortex.log",
            )

            logger.info("Starting orchestrator")
            processes.start(
                "orchestrator",
                ["orchestrator", "@", (config_dir / ORCHESTRATOR_CONFIG).as_posix()],
                env={
                    **os.environ,
                    **DEFAULT_COMMON_ENV_VARS,
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": f"python -m {__package__}.rl",
                    "WANDB_ARGS": json.dumps(sys.argv),
                    **config.env_vars,
                    **config.orchestrator.env_vars,
                    **wandb_shared_env,
                    "WANDB_SHARED_LABEL": "orchestrator",
                },
                log_path=log_dir / "orchestrator.log",
            )

            logger.success("Launcher complete")
            log_dashboard_url(logger, dashboard_url)
            # The worker serves the policy until the orchestrator is done; its failure still ends the run.
            processes.wait("orchestrator")
            logger.success("Training finished!")
    finally:
        if created and not config.cortex.keep_job:
            logger.info(f"Cancelling Cortex job {job_id}")
            jobs.cancel(job_id)
        jobs.close()


def main():
    set_proc_title("Launcher")
    cortex_rl(cli(CortexRLConfig))


if __name__ == "__main__":
    main()
