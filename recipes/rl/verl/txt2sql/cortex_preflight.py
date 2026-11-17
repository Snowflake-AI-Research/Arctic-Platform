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
"""Driver checks run_qwen3_8b_bird_grpo_cortex.sh makes before any Cortex job exists.

Reads AP_ROOT and VERL_HOME from the environment. EXPECTED_IMAGE_TAG is optional;
when set, ARCTIC_CORTEX_IMAGE_TAG must equal it. Never prints the PAT.
"""

import os
import sys


def main() -> None:
    ap_root = os.path.realpath(os.environ["AP_ROOT"])
    verl_home = os.path.realpath(os.environ["VERL_HOME"])
    expected_tag = (os.environ.get("EXPECTED_IMAGE_TAG") or "").strip()

    import arctic_platform
    import verl
    import vllm

    for name, module, root in (("arctic_platform", arctic_platform, ap_root), ("verl", verl, verl_home)):
        path = os.path.realpath(module.__file__)
        print(name, path)
        if not path.startswith(root + os.sep):
            sys.exit(f"{name} resolves outside {root}")
    print("vllm", vllm.__version__)

    from arctic_platform.client.config import CortexConfig
    from arctic_platform.client.config import cortex_job_packs
    from arctic_platform.client.transports.cortex import job_create_payload
    from arctic_platform.integrations.verl.adapter import ArcticRLClientWrapper

    env_tag = (os.environ.get("ARCTIC_CORTEX_IMAGE_TAG") or "").strip()
    config = CortexConfig()
    # Behavioral check: the env tag must reach the job body as debug.job.image_tag.
    if (config.image_tag or "").strip() != env_tag:
        sys.exit("CortexConfig does not read ARCTIC_CORTEX_IMAGE_TAG")
    sent_tag = job_create_payload([], config).get("debug", {}).get("job", {}).get("image_tag")
    if env_tag and sent_tag != env_tag:
        sys.exit(f"job_create_payload does not send ARCTIC_CORTEX_IMAGE_TAG={env_tag}")
    if not env_tag and sent_tag is not None:
        sys.exit("job_create_payload sends an image tag although ARCTIC_CORTEX_IMAGE_TAG is unset")
    if expected_tag and env_tag != expected_tag:
        sys.exit(f"ARCTIC_CORTEX_IMAGE_TAG is not {expected_tag}")
    if not env_tag:
        print("ARCTIC_CORTEX_IMAGE_TAG unset: the job uses the account's default image")
    if cortex_job_packs({"pack": False}) is not False:
        sys.exit("pack=false still packs")
    if cortex_job_packs({"zorro_train_enable": True, "pack": True}) is not False:
        sys.exit("ZoRRo still packs")
    if not hasattr(ArcticRLClientWrapper, "_send_cortex_update_actor"):
        sys.exit("the verl adapter has no Cortex backend")
    print(
        f"cortex host={config.host} database={config.database} schema={config.schema_}"
        f" pat_set={config.pat is not None}"
    )
    print("preflight ok: image hook, pack rules, Cortex adapter")


if __name__ == "__main__":
    main()
