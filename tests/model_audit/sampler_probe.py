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
"""Sampler-side probe of the model audit: vLLM with the platform's weight-sync extension. Not collected by pytest.

Prompt log-probs of the audit rows, aligned with the trainer's (index ``t`` is the log-prob of token ``t + 1``):
each row alone, the same again after a prefix-cache reset, and all rows in one batch. With ``--sync`` the engine
then loads a ``[(name, tensor)]`` file and the rows are scored again. ``--sync-transport file`` loads it through
``WeightSyncExtension.load_weights_from_shm_path``, the production CPU-file sync; ``stream`` feeds it to the
placement ``sync_weights_broadcast`` runs on what NCCL receives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import audit_lib  # noqa: E402

WORKER_EXTENSION = "arctic_platform.inference.server.weight_sync.WeightSyncExtension"


def _score(llm, rows: list[list[int]]) -> list[torch.Tensor]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=0, detokenize=False)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=row) for row in rows], params, use_tqdm=False)
    scored = []
    for row, output in zip(rows, outputs):
        lp = torch.full((len(row),), float("nan"))
        for t in range(1, len(row)):
            entry = output.prompt_logprobs[t]
            lp[t - 1] = entry[row[t]].logprob
        scored.append(lp)
    return scored


def _stream_sync(worker, path: str, weight_format: str) -> dict:
    """``sync_weights_broadcast``'s receive-side placement, with the file standing in for the NCCL stream."""
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

    from arctic_platform.inference.server.weight_sync.receiver import _refresh_cached_weight_copies

    if weight_format == "hf":
        load = worker._load_batched
    elif worker.model_config.quantization:
        return {"unsupported": "the stream transport does not cover sync_weights_broadcast's fp8 path"}
    elif get_tensor_model_parallel_world_size() == 1:
        load = worker._load_direct
    else:
        load = worker._load_batched
    weights = torch.load(path, map_location="cpu", weights_only=True)

    class _FileStream:
        def receive_weights(self):
            for name, tensor in weights:
                yield name, tensor.to(worker.device)

    model = worker.model_runner.model
    loaded, _, _, _, skipped = load(model, _FileStream())
    refreshed = _refresh_cached_weight_copies(model)
    torch.cuda.synchronize()
    return {"loaded": loaded, "skipped": skipped, "cached_weight_copies_refreshed": refreshed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sync", action="append", default=[], help="fmt=path of a weight file to sync, repeatable")
    parser.add_argument("--sync-transport", choices=["file", "stream"], default="file")
    parser.add_argument("--engine-args", default="{}", help="JSON overrides of the profile's tiny sampler args")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result: dict = {}
    try:
        import vllm
        from vllm import LLM

        profile = audit_lib.load_profile(args.profile)
        _, rows, _ = audit_lib.probe_inputs(profile, args.ckpt)
        engine_args = {**profile.sampler_args(), **json.loads(args.engine_args)}
        llm = LLM(model=args.ckpt, dtype=profile.dtype, seed=0, worker_extension_cls=WORKER_EXTENSION, **engine_args)
        result["engine"] = {"vllm": vllm.__version__, "args": engine_args}
        result["alone"] = [_score(llm, [row])[0] for row in rows]
        llm.reset_prefix_cache()
        result["alone_repeat"] = [_score(llm, [row])[0] for row in rows]
        llm.reset_prefix_cache()
        result["batched"] = _score(llm, rows)
        for item in args.sync:
            fmt, path = item.split("=", 1)
            try:
                if args.sync_transport == "stream":
                    result[f"sync_{fmt}_rpc"] = llm.collective_rpc(_stream_sync, args=(path, fmt))
                    gaps = {r["unsupported"] for r in result[f"sync_{fmt}_rpc"] if "unsupported" in r}
                    if gaps:
                        result[f"sync_{fmt}_unsupported"] = sorted(gaps)
                        continue
                else:
                    result[f"sync_{fmt}_rpc"] = llm.collective_rpc("load_weights_from_shm_path", args=(path,))
            except Exception as err:  # noqa: BLE001
                # The unsynced scores above stay valid; the sync check reports this.
                result[f"sync_{fmt}_error"] = f"{type(err).__name__}: {str(err).strip()[:2000]}"
                break
            llm.reset_prefix_cache()
            result[f"after_sync_{fmt}"] = [_score(llm, [row])[0] for row in rows]
        result["status"] = {"ok": True}
    except Exception as err:
        result["status"] = {"ok": False, "error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()}
        print(result["status"]["traceback"], file=sys.stderr, flush=True)
    torch.save(result, args.out)


if __name__ == "__main__":
    main()
