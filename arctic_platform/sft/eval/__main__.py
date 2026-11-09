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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run a packed SFT wall probe for one model and arm.

Example::

    CUDA_VISIBLE_DEVICES=0 python -m arctic_platform.sft.eval \
        --model Qwen/Qwen3-8B --arm gc --start-length 262144 \
        --out /data-fast/tmp/ap-sft-eval/qwen3-8b-gc.json
"""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import torch
from transformers import AutoTokenizer

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.sft.eval.arms import PROBE_STEPS
from arctic_platform.sft.eval.arms import WARMUP_STEPS
from arctic_platform.sft.eval.arms import job_config
from arctic_platform.sft.eval.packing import load_ultrachat_texts
from arctic_platform.sft.eval.packing import pack_exact_length
from arctic_platform.sft.eval.wall import ProbeResult
from arctic_platform.sft.eval.wall import at_wall
from arctic_platform.sft.eval.wall import climb_then_bisect
from arctic_platform.sft.eval.wall import host_rss_gib
from arctic_platform.sft.eval.wall import peak_gpu_mib
from arctic_platform.sft.eval.wall import timed_steps
from arctic_platform.sft.eval.wall import write_history


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _build_worker(model_name: str, arm: str) -> DeepSpeedWorker:
    worker = DeepSpeedWorker(rank=0, world_size=1, master_port=_free_port())
    worker.initialize(socket.gethostname(), job_config(model_name, arm, world_size=1))
    return worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", required=True, choices=["gc", "tiling", "offload", "tiling_offload"])
    parser.add_argument("--start-length", type=int, default=None, help="defaults to the AT wall for this arm")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/data/hf_cache"))
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.hf_home) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.hf_home) / "hub"))

    start = args.start_length if args.start_length is not None else at_wall(args.model, args.arm)
    print(f"eval_start model={args.model} arm={args.arm} start_length={start}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = load_ultrachat_texts(seed=42)
    worker = _build_worker(args.model, args.arm)

    def probe(length: int) -> ProbeResult:
        print(f"probe_begin length={length}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        batch = pack_exact_length(tokenizer, texts, length, pad_token_id=int(tokenizer.pad_token_id))

        def one_step() -> None:
            worker.forward_backward(batch)
            worker.step()

        times, failure = timed_steps(one_step, warmup=WARMUP_STEPS, steps=PROBE_STEPS)
        result = ProbeResult(
            length=length,
            passed=failure is None,
            failure=failure,
            step_s=times,
            peak_gpu_mib=peak_gpu_mib(),
            host_rss_gib=host_rss_gib(),
        )
        print(
            f"probe_end length={length} passed={result.passed} failure={result.failure} "
            f"peak_gpu_mib={result.peak_gpu_mib} host_rss_gib={result.host_rss_gib}"
        )
        return result

    wall, history = climb_then_bisect(start, probe)
    write_history(args.out, args.model, args.arm, wall, history)
    print(f"eval_done wall={wall} probes={len(history)} out={args.out}")
    worker.destroy()


if __name__ == "__main__":
    main()
