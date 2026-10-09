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

"""Ablation: does the residual disagreement live in the full-attention layers?

With the reduced model carrying the checkpoint's own weights, Arctic Platform and the single-GPU reference no longer
show a systematic offset, but individual gradient norms still disagree by up to 1% -- enough to cross a
1e-3 absolute gate. Self-attention layers are 7 of the 28 under the period-4 layer pattern and supply
roughly half of the tensors that cross it.

This builds a second model from the same weights with the full-attention layers removed: the linear
attention layers are kept in order and renumbered, so every tensor still holds a checkpoint value and only
the layer mix changes. It is an ablation and not a proposed configuration -- a stack with no full attention
is not the model anyone runs.

A disagreement that disappears here is produced by the full-attention path. One that survives is produced
by something every layer shares.
"""

from __future__ import annotations

import copy
import glob
import os
import re
import statistics
import sys
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
from arctic_platform.correctness.harness.names import align  # noqa: E402
from arctic_platform.correctness.harness.runner import run_reference  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

# noqa: E402
ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"


def layer_type_field(text_cfg):
    types = list(getattr(text_cfg, "layer_types", []) or [])
    if not types:
        raise SystemExit("this architecture has no layer_types list; the ablation cannot select layers")
    return types


def build_ablated(source_slice: str, dest: str, drop_types) -> tuple:
    """Write a model holding only the layers whose type is not in ``drop_types``, renumbered from zero."""
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(source_slice, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    types = layer_type_field(text)
    keep = [i for i, t in enumerate(types) if t not in drop_types]
    dropped = [i for i, t in enumerate(types) if t in drop_types]
    remap = {old: new for new, old in enumerate(keep)}

    text.num_hidden_layers = len(keep)
    text.layer_types = [types[i] for i in keep]

    dest_path = Path(dest)
    if not (dest_path / "config.json").exists():
        cls = getattr(transformers, cfg.architectures[0])
        model = cls(cfg).to(torch.bfloat16)
        state = {}
        for shard in sorted(glob.glob(f"{source_slice}/*.safetensors")):
            for key, value in load_file(shard).items():
                match = re.search(r"(?:language_model\.)?layers\.(\d+)\.", key)
                if match:
                    old = int(match.group(1))
                    if old not in remap:
                        continue
                    key = key[: match.start(1)] + str(remap[old]) + key[match.end(1) :]
                state[key] = value
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
        print(f"[ablate] kept {len(keep)} layers, dropped {len(dropped)} at {dropped}", flush=True)
        print(f"[ablate] tensors without a checkpoint value: {len(missing)}", flush=True)
        if missing:
            print(f"[ablate] first missing: {missing[:5]}", flush=True)
        dest_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(dest_path, safe_serialization=True, max_shard_size="5GB")
        for extra in glob.glob(f"{source_slice}/*token*") + glob.glob(f"{source_slice}/*.txt"):
            Path(dest_path / Path(extra).name).write_bytes(Path(extra).read_bytes())
        del model
    return str(dest_path), keep, dropped


def statistics_for(dss, reference):
    pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
    ratios = sorted(a / b for _, a, b in pairs if b)
    diffs = [(abs(a - b), name) for name, a, b in pairs]
    worst, worst_name = max(diffs)
    return (
        ratios[len(ratios) // 2],
        sum(1 for r in ratios if r > 1.0),
        len(ratios),
        sum(1 for d, _ in diffs if d > STATED_CRITERION_ABS),
        worst,
        worst_name,
        statistics.median(b for _, _, b in pairs),
        len(only_dss),
        len(only_ref),
    )


def main(config_path: str, spec_path: str) -> int:
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    layers = int(os.environ.get("PROBE_LAYERS", 0)) or spec.model.num_hidden_layers
    fp32_lm_head = bool(cfg.training.get("fp32_lm_head", False))
    # Flash attention's backward is not reproducible, so a difference measured under it cannot be
    # attributed to a layer type; sdpa repeats exactly.
    attn = os.environ.get("PROBE_ATTN") or cfg.training.get("attn_implementation", "flash_attention_3")
    n_gpus = int(cfg.training.get("n_gpus", 8))

    intact = materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{layers}L", layers).cache_path

    source_cfg = AutoConfig.from_pretrained(intact, trust_remote_code=True)
    text = getattr(source_cfg, "text_config", source_cfg)
    types = layer_type_field(text)
    full_types = {t for t in types if "full" in t or "self" in t or t == "attention"}
    # Which half to remove. ``full`` leaves a stack of linear-attention layers; ``linear`` leaves a stack
    # of full-attention layers, which is what isolates the gated delta net's own contribution.
    which = os.environ.get("PROBE_DROP", "full")
    drop_types = full_types if which == "full" else (set(types) - full_types)
    print(
        f"layer types present: {sorted(set(types))}; treating {sorted(full_types)} as full attention; "
        f"dropping {sorted(drop_types)}",
        flush=True,
    )

    ablated, keep, dropped = build_ablated(intact, f"{CACHE_ROOT}/Qwen3.8-27B-{layers}L-no-{which}-attn", drop_types)

    model_cfg = AutoConfig.from_pretrained(intact, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = correctness_workdir("attn-ablation-")
    batch_rows = int(os.environ.get("PROBE_ROWS", n_gpus))
    batch = build_batch("gas1", batch_rows, ROW_TOKENS, vocab, seed=SEED)
    batch_path = work / "batch.pt"
    save(batch, batch_path)
    body = pack(batch)
    print(
        f"{batch.rows} x {ROW_TOKENS:,}-token sequences, {batch.active_tokens:,} real tokens, attn {attn}, "
        f"fp32_lm_head {fp32_lm_head}, sp {cfg.training.get('sp_size')}\n",
        flush=True,
    )

    arms = ((f"intact ({len(types)} layers)", intact), (f"no {which} attention ({len(keep)} layers)", ablated))

    rows = []
    for label, model_path in arms:
        print(f"[ref] {label} ...", flush=True)
        reference = run_reference(
            model_path,
            batch_path,
            work / f"ref-{len(rows)}.json",
            token_budget=65536,
            ce_chunk=2048,
            attn=attn,
            fp32_lm_head=fp32_lm_head,
        )
        print(f"[ref] {label}: loss {reference['loss']:.6f}", flush=True)

        training = copy.deepcopy(cfg.training)
        payload = build_payload(training, model_path, SEED, attn_implementation=attn)
        print(f"[dss] {label} ...", flush=True)
        with gateway(work, n_gpus) as url:
            with running_job(url, payload) as job_id:
                dss = fwd_bwd_step(url, job_id, body)

        median_ratio, above, total, over, worst, worst_name, median_norm, ud, ur = statistics_for(dss, reference)
        rows.append(
            (
                label,
                reference["loss"],
                median_norm,
                median_ratio,
                above,
                total,
                over,
                worst,
                worst_name,
                abs(dss.avg_loss - reference["loss"]),
            )
        )
        print(
            f"[dss] {label}: median ratio {median_ratio:.6f}  above 1.0 {above}/{total}  "
            f"over 1e-3 {over}  worst {worst:.3e} {worst_name}"
            f"{f'  UNMATCHED {ud}/{ur}' if ud or ur else ''}\n",
            flush=True,
        )

    print(
        f"{'arm':30} {'ref loss':>10} {'median norm':>12} {'median ratio':>13} {'above 1.0':>11} "
        f"{'over 1e-3':>10} {'worst abs':>11} {'loss delta':>11}"
    )
    for label, loss, norm, ratio, above, total, over, worst, _, loss_delta in rows:
        print(
            f"{label:30} {loss:>10.4f} {norm:>12.4f} {ratio:>13.6f} {above:>7}/{total:<3} "
            f"{over:>10} {worst:>11.3e} {loss_delta:>11.3e}"
        )
    print("")
    print(
        "Tensor counts differ between the arms, so compare the rate rather than the count: "
        f"{rows[0][6]}/{rows[0][5]} against {rows[1][6]}/{rows[1][5]}."
    )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    raise SystemExit(
        main(
            args[0] if args else "arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config",
            args[1] if len(args) > 1 else SPEC,
        )
    )
