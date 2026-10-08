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

"""Run the single-GPU reference in its own process and write the result as JSON.

The reference runs in a subprocess so its CUDA context and allocator caches are fully reclaimed before the
gateway claims the node's GPUs. Freeing tensors in-process leaves the context resident, which shrinks what
the Arctic Platform workers can allocate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="arctic_platform.correctness.reference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--token-budget", type=int, default=65536)
    parser.add_argument("--ce-chunk", type=int, default=2048)
    parser.add_argument("--lm-head-token-chunk", type=int)
    parser.add_argument("--lm-head-vocab-chunk", type=int, default=8192)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument(
        "--matmul-precision",
        default="highest",
        choices=("highest", "high", "medium"),
        help="float32 matmul precision; the engine's matmul_precision setting",
    )
    parser.add_argument(
        "--mixer-packing",
        dest="mixer_packing",
        action="store_true",
        help=(
            "hand the gated delta net the sequence boundaries the engine hands it, so "
            "both run the varlen convolution and delta-rule kernels"
        ),
    )
    parser.add_argument(
        "--fp32-lm-head", dest="fp32_lm_head", action="store_true", help="apply the engine's fp32_lm_head behavior"
    )
    parser.add_argument(
        "--fused-cross-entropy",
        choices=("liger",),
        help="use the config's fused output projection and cross-entropy kernel",
    )
    parser.add_argument("--peft-config", help="JSON PEFT config to apply before forward-backward")
    parser.add_argument("--peft-adapter", help="Arctic Platform-initialized PEFT adapter directory")
    parser.add_argument("--optimizer-config", help="JSON AdamW config for an optimizer-step reference")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-clipping", type=float)
    parser.add_argument("--optimizer-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--optimizer-output-dir")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "pin reduction order so the comparison repeats; the caller also sets "
            "CUBLAS_WORKSPACE_CONFIG and FLASH_ATTENTION_DETERMINISTIC, which have "
            "to be in the environment before CUDA initializes"
        ),
    )
    args = parser.parse_args()

    if args.deterministic:
        from .process_determinism import pin_reduction_order

        pin_reduction_order(args.model, args.attn)

    from ..harness.batches import load
    from .hf_single_gpu import run

    batch = load(Path(args.batch))
    result = run(
        args.model,
        batch,
        token_budget=args.token_budget,
        ce_chunk=args.ce_chunk,
        dtype=args.dtype,
        attn_implementation=args.attn,
        fp32_lm_head=args.fp32_lm_head,
        fused_cross_entropy=args.fused_cross_entropy or False,
        mixer_packing=args.mixer_packing,
        matmul_precision=args.matmul_precision,
        peft_config=json.loads(args.peft_config) if args.peft_config else None,
        peft_adapter_path=args.peft_adapter,
        seed=args.seed,
        lm_head_token_chunk_size=args.lm_head_token_chunk,
        lm_head_vocab_chunk_size=args.lm_head_vocab_chunk,
        optimizer_config=json.loads(args.optimizer_config) if args.optimizer_config else None,
        learning_rate=args.learning_rate,
        gradient_clipping=args.gradient_clipping,
        optimizer_dtype=args.optimizer_dtype,
        optimizer_output_dir=args.optimizer_output_dir,
    )
    Path(args.out).write_text(
        json.dumps(
            {
                "loss": result.loss,
                "grad_norms": result.grad_norms,
                "active_tokens": result.active_tokens,
                "microbatches": result.microbatches,
                "peak_gib": result.peak_gib,
                "skipped_no_grad": result.skipped_no_grad,
                "optimizer_state_manifest": result.optimizer_state_manifest,
                "optimizer_gradient_norm": result.optimizer_gradient_norm,
            }
        )
    )


if __name__ == "__main__":
    main()
