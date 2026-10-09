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

"""Measure whether Qwen3.6 gradient disagreement grows with DP width.

The reviewed Qwen3.6 config uses sequence parallelism and expert parallelism. This diagnostic keeps the
input batch fixed, runs one single-GPU reference for the selected arm, and then asks Arctic Platform to run
the same batch at SP1/EP1 on several data-parallel widths. It is diagnostic-only: no product config files
or correctness report tables are changed.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.arms import correctness_microbatch_tokens  # noqa: E402
from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import load as load_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import LoadedConfig  # noqa: E402
from arctic_platform.correctness.harness.config import deepspeed_train_batch_size  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.names import align  # noqa: E402
from arctic_platform.correctness.harness.runner import run_reference  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import ArmSpec  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.reference.model_features import uses_mixer_packing  # noqa: E402

DEFAULT_CONFIG = "arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-8gpus-64k.config"
DEFAULT_SPEC = "arctic_platform/correctness/specs/qwen3.6-35b-a3b-h200-train-sft-8gpus-64k.json"
DEFAULT_WIDTHS = (1, 2, 4, 8)
REL_ABS_GATE = 1e-3


@dataclass(frozen=True)
class WidthSummary:
    width: int
    median_ratio: float
    above_one: int
    over_gate: int
    compared: int
    worst_delta: float
    worst_name: str
    loss_delta: float
    model_calls: int | None
    packed_rows: int | None


def parse_widths(value: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not widths:
        raise ValueError("at least one width is required")
    if any(width < 1 for width in widths):
        raise ValueError(f"widths must be positive, got {widths}")
    return widths


def arm_by_name(spec: TestSpec, name: str) -> ArmSpec:
    for arm in spec.arms:
        if arm.name == name:
            return arm
    raise ValueError(f"spec has no arm {name!r}; available arms are {', '.join(arm.name for arm in spec.arms)}")


def loss_settings(cfg: LoadedConfig) -> tuple[bool, bool | str, int | None, int]:
    training = cfg.effective_training
    return (
        bool(training.get("fp32_lm_head", False)),
        cfg.fused_cross_entropy,
        cfg.lm_head_token_chunk_size,
        int(training.get("fused_lm_head_vocab_chunk_size", 8192)),
    )


def training_for_width(cfg: LoadedConfig, width: int) -> dict:
    """Return a local diagnostic payload config narrowed to SP1/EP1 at ``width`` GPUs."""
    training = copy.deepcopy(cfg.effective_training)
    training["n_gpus"] = width
    training["sp_size"] = 1
    training["ep_size"] = 1
    training.setdefault("ds_worker_config", {})["ep_size"] = 1

    required_batch = deepspeed_train_batch_size(training, width)
    if required_batch is not None:
        training["train_batch_size"] = required_batch
        if isinstance(training.get("ds_config"), dict):
            training["ds_config"]["train_batch_size"] = required_batch
    return training


def summarize_width(width: int, dss, reference: dict, gate: float = REL_ABS_GATE) -> WidthSummary:
    pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
    if only_dss or only_ref:
        print(f"[width {width}] unmatched gradient names: dss={len(only_dss)} reference={len(only_ref)}", flush=True)
    diffs = []
    ratios = []
    over_gate = 0
    for name, ap_value, ref_value in pairs:
        delta = abs(ap_value - ref_value)
        relative = delta / max(abs(ref_value), gate)
        diffs.append((delta, name))
        if ref_value:
            ratios.append(ap_value / ref_value)
        if delta > gate or relative > gate:
            over_gate += 1
    ratios.sort()
    worst_delta, worst_name = max(diffs) if diffs else (0.0, "none")
    return WidthSummary(
        width=width,
        median_ratio=ratios[len(ratios) // 2] if ratios else 0.0,
        above_one=sum(1 for ratio in ratios if ratio > 1.0),
        over_gate=over_gate,
        compared=len(pairs),
        worst_delta=worst_delta,
        worst_name=worst_name,
        loss_delta=abs(dss.avg_loss - float(reference["loss"])),
        model_calls=dss.model_calls,
        packed_rows=dss.packed_rows,
    )


def print_summary(summary: WidthSummary) -> None:
    print(
        f"width {summary.width:<2} median {summary.median_ratio:.6f}  "
        f"above1 {summary.above_one}/{summary.compared}  "
        f"over1e-3-rel/abs {summary.over_gate}/{summary.compared}  "
        f"worst {summary.worst_delta:.3e} {summary.worst_name}  "
        f"loss_delta {summary.loss_delta:.3e}  "
        f"model_calls {summary.model_calls}  packed_rows {summary.packed_rows}",
        flush=True,
    )


def _workdir(value: str | None) -> Path:
    if value:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return correctness_workdir("qwen36-dp-width-")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen36_dp_width")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("spec", nargs="?", default=DEFAULT_SPEC)
    parser.add_argument("--widths", default=",".join(str(width) for width in DEFAULT_WIDTHS))
    parser.add_argument("--arm", default="gas1", choices=("gas1", "gas4"))
    parser.add_argument("--work-dir")
    parser.add_argument("--reference-attn")
    args = parser.parse_args(argv)

    widths = parse_widths(args.widths)
    cfg = load_config(Path(args.config))
    spec = TestSpec.read(Path(args.spec))
    arm = arm_by_name(spec, args.arm)
    model_path = spec.model.cache_path
    attn = cfg.attention_implementation
    reference_attn = args.reference_attn or attn

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = _workdir(args.work_dir)
    batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, vocab, seed=SEED)
    batch_path = work / f"batch-{arm.name}.pt"
    save(batch, batch_path)

    fp32_lm_head, fused_cross_entropy, token_chunk_size, vocab_chunk_size = loss_settings(cfg)
    reference_training = cfg.effective_training
    reference_token_budget = correctness_microbatch_tokens(cfg.max_tokens_per_mb)

    print(f"workdir {work}", flush=True)
    print(
        f"model {model_path}; arm {arm.name}: rows={arm.global_batch_size}, row_tokens={arm.max_seq_len}, "
        f"active_tokens={batch.active_tokens}",
        flush=True,
    )
    print(f"reference attn={reference_attn}; AP attn={attn}; widths={','.join(str(width) for width in widths)}")

    print("[reference] running once ...", flush=True)
    reference = run_reference(
        model_path,
        batch_path,
        work / f"ref-{arm.name}.json",
        token_budget=reference_token_budget,
        ce_chunk=2048,
        attn=reference_attn,
        fp32_lm_head=fp32_lm_head,
        seed=SEED,
        fused_cross_entropy=fused_cross_entropy,
        mixer_packing=uses_mixer_packing(model_path),
        matmul_precision=reference_training.get("matmul_precision", "highest"),
        peft_config=reference_training.get("peft_config"),
        lm_head_token_chunk_size=token_chunk_size,
        lm_head_vocab_chunk_size=vocab_chunk_size,
        gradient_clipping=cfg.training.get("gradient_clipping"),
        optimizer_dtype=cfg.optimizer_dtype,
    )
    print(f"[reference] loss {reference['loss']:.6f}; gradients {len(reference['grad_norms'])}", flush=True)

    body = pack(load_batch(batch_path), model_provider=str(cfg.training.get("model_provider", "huggingface")))
    rows = []
    for width in widths:
        training = training_for_width(cfg, width)
        payload = build_payload(training, model_path, SEED, attn_implementation=attn, gradient_norms_per_param=True)
        print(f"[dss width {width}] running AP with sp_size=1 ep_size=1 ...", flush=True)
        with gateway(work / f"gateway-width-{width}", width) as session:
            with running_job(session, payload) as job:
                dss = fwd_bwd_step(session, job, body)
        summary = summarize_width(width, dss, reference)
        rows.append(summary)
        print_summary(summary)

    print("")
    print("width  median_ratio  above1       over1e-3-rel/abs  worst_abs    loss_delta   worst_tensor")
    for row in rows:
        print(
            f"{row.width:<5}  {row.median_ratio:>12.6f}  {row.above_one:>4}/{row.compared:<4}  "
            f"{row.over_gate:>8}/{row.compared:<8}  {row.worst_delta:>9.3e}  "
            f"{row.loss_delta:>10.3e}  {row.worst_name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
