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

"""End-to-end smoke test for the Arctic Tinker HTTP layer.

Boot the Arctic server (native flags), then provision + bind Tinker via
``recipes/rl/tinker/serve.sh``. Everything below is pure upstream
``tinker`` SDK.

Usage::

    # terminal 1
    python -m arctic_platform.rl.http_server \
        --host 0.0.0.0 --port 7000 \
        --training-gpus 1 --sampling-gpus 1 --colocate

    # terminal 2
    MODEL=Qwen/Qwen3-0.6B MAX_PROMPT=512 MAX_RESPONSE=128 \
        recipes/rl/tinker/serve.sh
    python -m arctic_platform.rl.examples.tinker_smoke \
        --url http://localhost:7000 --base-model Qwen/Qwen3-0.6B --steps 5
"""

from __future__ import annotations

import argparse
import os
import time

import tinker
import tinker.types as t
from transformers import AutoTokenizer


def _build_datum(prompt_ids: list[int], seq, max_response_length: int) -> t.Datum:
    """Toy reward: length-normalized response length. Prompt positions are
    zero-masked on every loss-input array (SkyRL-tx cookbook convention);
    non-zero entries in ``weights`` / ``advantages`` mark the response
    tail and drive the prompt/response split on the server."""
    resp_ids = list(seq.tokens)
    all_ids = prompt_ids + resp_ids
    n_prompt, n_resp = len(prompt_ids), len(resp_ids)
    adv = float(n_resp) / max(1, max_response_length)
    return t.Datum(
        model_input=t.ModelInput.from_ints(all_ids),
        loss_fn_inputs={
            "target_tokens": [0] * n_prompt + resp_ids,
            "advantages": [0.0] * n_prompt + [adv] * n_resp,
            "logprobs": [0.0] * n_prompt + list(seq.logprobs or [0.0] * n_resp),
            "weights": [0.0] * n_prompt + [1.0] * n_resp,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:7000")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--api-key", default="tml-dummy")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default="Q: what is 2+2?\nA:")
    p.add_argument("--max-response-length", type=int, default=128,
                   help="Must match MAX_RESPONSE used at /tinker/bind.")
    args = p.parse_args()

    os.environ["TINKER_BASE_URL"] = args.url
    os.environ["TINKER_API_KEY"] = args.api_key

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt = t.ModelInput.from_ints(prompt_ids)

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=0)
    print(f"[bootstrap] training client ready for {args.base_model} (FFT via rank=0)")

    for step in range(args.steps):
        t0 = time.time()
        sampler = tc.save_weights_and_get_sampling_client()
        keep = []
        while len(keep) < args.num_samples:
            need = args.num_samples - len(keep)
            samples = sampler.sample(
                prompt=prompt, num_samples=need,
                sampling_params=t.SamplingParams(max_tokens=args.max_tokens, temperature=0.7),
            ).result()
            keep.extend(s for s in samples.sequences if len(s.tokens) > 0)
        batch = [_build_datum(prompt_ids, s, args.max_response_length) for s in keep]

        fbwd = tc.forward_backward(
            data=batch, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
        ).result()
        tc.optim_step(t.AdamParams(learning_rate=1e-6)).result()

        loss = fbwd.metrics.get("loss:mean", float("nan"))
        stop_rate = sum(1 for s in keep if s.stop_reason == "stop") / len(keep)
        preview = tokenizer.decode(keep[0].tokens[:16])
        print(f"[step {step}] loss:mean={loss:.4f} stop_rate={stop_rate:.2f} "
              f"first_preview={preview!r} dt={time.time() - t0:.2f}s")

    print("[done] smoke passed")


if __name__ == "__main__":
    main()
