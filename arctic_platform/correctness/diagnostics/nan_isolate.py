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

"""Locate the non-finite loss: is it the model's depth or the sequence-parallel row length?

Onboarding raised ``FloatingPointError`` on both configs after two settings moved together -- rows went from
65536 tokens to 2048, and the layer count the single-GPU budget allows went from 16 to 28. Either could
produce a non-finite loss, and the traceback names neither: it reports a rank, and the rank differs between
runs.

Four arms, each printing a loss or the exception it raised. The reference arm says whether the weights
themselves are the problem, since it runs on one GPU with no sequence parallelism at all. The two Arctic Platform arms
on the deep model differ only in row length, and the shallow arm differs from them only in depth.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize  # noqa: E402

SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"


def model_for(layers: int) -> tuple[str, int]:
    from transformers import AutoConfig

    dest = f"{CACHE_ROOT}/synthetic-Qwen3.8-27B-{layers}L-seed{SEED}"
    spec = materialize(SOURCE, dest, layers)
    cfg = AutoConfig.from_pretrained(spec.cache_path, trust_remote_code=True)
    return spec.cache_path, getattr(cfg, "text_config", cfg).vocab_size


def dss_loss(training: dict, model_path: str, attn: str, n_gpus: int, body_bytes: bytes) -> str:
    payload = build_payload(training, model_path, SEED, attn_implementation=attn)
    work = Path(tempfile.mkdtemp(prefix="nanprobe-"))
    try:
        with gateway(work, n_gpus) as url:
            with running_job(url, payload) as job_id:
                step = fwd_bwd_step(url, job_id, body_bytes)
        return f"loss {step.avg_loss:.6f}"
    except Exception as exc:  # the server raises FloatingPointError through a 500
        text = str(exc)
        marker = "FloatingPointError:"
        return "FloatingPointError " + text.split(marker)[-1].strip()[:110] if marker in text else text[:160]


def reference_loss(model_path: str, batch) -> str:
    tmp = Path(tempfile.mkdtemp(prefix="nanprobe-ref-"))
    save(batch, tmp / "batch.pt")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "arctic_platform.correctness.reference",
            "--model",
            model_path,
            "--batch",
            str(tmp / "batch.pt"),
            "--out",
            str(tmp / "ref.json"),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return "failed: " + (out.stderr.strip().splitlines() or ["(no stderr)"])[-1][:140]
    return f"loss {json.loads((tmp / 'ref.json').read_text())['loss']:.6f}"


def main(config_path: str) -> int:
    cfg = load_config(Path(config_path))
    attn = cfg.training.get("attn_implementation", "flash_attention_3")

    deep_path, vocab = model_for(28)
    shallow_path, vocab_shallow = model_for(16)

    short = build_batch("short", 1, 2048, vocab, seed=SEED)
    long_ = build_batch("long", 1, 65536, vocab, seed=SEED)
    short_shallow = build_batch("short", 1, 2048, vocab_shallow, seed=SEED)

    print(f"config {cfg.config_id}, attn {attn}, n_gpus {cfg.n_gpus}")
    print(
        f"short row 2,048 tokens {short.pad_fraction:.1%} padding; "
        f"long row 65,536 tokens {long_.pad_fraction:.1%} padding\n",
        flush=True,
    )

    results = []
    for label, fn in [
        ("reference 28L, 2048-token row (1 GPU, no SP)", lambda: reference_loss(deep_path, short)),
        (
            "Arctic Platform 28L, 2048-token row",
            lambda: dss_loss(cfg.training, deep_path, attn, cfg.n_gpus, pack(short)),
        ),
        (
            "Arctic Platform 28L, 65536-token row",
            lambda: dss_loss(cfg.training, deep_path, attn, cfg.n_gpus, pack(long_)),
        ),
        (
            "Arctic Platform 16L, 2048-token row",
            lambda: dss_loss(cfg.training, shallow_path, attn, cfg.n_gpus, pack(short_shallow)),
        ),
    ]:
        print(f"[probe] {label} ...", flush=True)
        outcome = fn()
        results.append((label, outcome))
        print(f"[probe] {label:<46} {outcome}\n", flush=True)

    print(f"{'arm':<46} outcome")
    for label, outcome in results:
        print(f"{label:<46} {outcome}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(main(args[0] if args else "qwen3.8-27b-8gpu-plain.config"))
