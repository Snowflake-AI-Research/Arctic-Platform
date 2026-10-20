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

"""Compare Qwen3.6 DP/non-SP against one-row and grouped reference mixer calls.

This is a narrow diagnostic for the call-shape hypothesis behind the Qwen3.6 single-step failures. The
product path is unchanged: Arctic Platform runs the config normally, while the independent reference runs
twice from the same materialized batch. The first reference arm preserves the correctness suite's current
one-row-per-call mixer packing. The second arm passes AP-style row boundaries while grouping multiple rows
into one decoder call.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
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
from arctic_platform.correctness.harness.optimizer_capture import optimizer_capture_worker  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import ArmSpec  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.reference.model_features import uses_mixer_packing  # noqa: E402

DEFAULT_CONFIG = "arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-8gpus-64k.config"
DEFAULT_SPEC = "arctic_platform/correctness/specs/qwen3.6-35b-a3b-h200-train-sft-8gpus-64k.json"
OPTIMIZER_LEARNING_RATE = 1e-2
CUBLAS_WORKSPACE = ":16:8"


@dataclass(frozen=True)
class GradSummary:
    label: str
    over: int
    compared: int
    worst: float
    worst_name: str
    loss_delta: float
    median_ratio: float


@dataclass(frozen=True)
class OptimizerSummary:
    label: str
    over: int
    compared: int
    worst: float
    worst_name: str


def loss_settings(cfg: LoadedConfig) -> tuple[bool, bool | str, int | None, int]:
    """Resolve the same reference loss knobs used by the correctness runner."""
    training = cfg.effective_training
    return (
        bool(training.get("fp32_lm_head", False)),
        cfg.fused_cross_entropy,
        cfg.lm_head_token_chunk_size,
        int(training.get("fused_lm_head_vocab_chunk_size", 8192)),
    )


def _workdir() -> Path:
    return correctness_workdir("qwen36-grouping-")


def _arm(spec: TestSpec, name: str) -> ArmSpec:
    for arm in spec.arms:
        if arm.name == name:
            return arm
    raise ValueError(f"spec has no arm {name!r}; available arms are {', '.join(arm.name for arm in spec.arms)}")


def _reference_token_budget(cfg: LoadedConfig, spec: TestSpec) -> int:
    from arctic_platform.correctness.harness.arms import correctness_microbatch_tokens

    budget = correctness_microbatch_tokens(cfg.max_tokens_per_mb)
    optimizer_settings = spec.test_settings.get("single-step-optimizer") or {}
    if "reference_token_budget" in optimizer_settings:
        budget = min(budget, int(optimizer_settings["reference_token_budget"]))
    return budget


def _reference_model_path(spec: TestSpec) -> str:
    optimizer_settings = spec.test_settings.get("single-step-optimizer") or {}
    return str(optimizer_settings.get("model_cache_path") or spec.model.cache_path)


def _run_reference(
    model_path: str,
    batch_path: Path,
    out_path: Path,
    *,
    token_budget: int,
    ce_chunk: int,
    attn: str,
    fp32_lm_head: bool,
    fused_cross_entropy: bool | str,
    mixer_packing_group_rows: int,
    matmul_precision: str,
    peft_config: dict | None,
    lm_head_token_chunk_size: int | None,
    lm_head_vocab_chunk_size: int,
    optimizer_config: dict,
    gradient_clipping: float | None,
    optimizer_dtype: str,
    optimizer_output_dir: Path,
) -> dict:
    """Run the single-GPU reference in a subprocess without importing the full harness runner."""
    cmd = [
        sys.executable,
        "-m",
        "arctic_platform.correctness.reference",
        "--model",
        model_path,
        "--batch",
        str(batch_path),
        "--out",
        str(out_path),
        "--token-budget",
        str(token_budget),
        "--ce-chunk",
        str(ce_chunk),
        "--attn",
        attn,
        "--seed",
        str(SEED),
        "--mixer-packing",
        "--mixer-packing-group-rows",
        str(mixer_packing_group_rows),
        "--matmul-precision",
        matmul_precision,
        "--optimizer-config",
        json.dumps(optimizer_config, sort_keys=True),
        "--learning-rate",
        str(OPTIMIZER_LEARNING_RATE),
        "--optimizer-dtype",
        optimizer_dtype,
        "--optimizer-output-dir",
        str(optimizer_output_dir),
        "--deterministic",
    ]
    if fp32_lm_head:
        cmd.append("--fp32-lm-head")
    if fused_cross_entropy:
        cmd += ["--fused-cross-entropy", "liger" if fused_cross_entropy is True else str(fused_cross_entropy)]
    if peft_config:
        cmd += ["--peft-config", json.dumps(peft_config, sort_keys=True)]
    if lm_head_token_chunk_size is not None:
        cmd += [
            "--lm-head-token-chunk",
            str(lm_head_token_chunk_size),
            "--lm-head-vocab-chunk",
            str(lm_head_vocab_chunk_size),
        ]
    if gradient_clipping is not None:
        cmd += ["--gradient-clipping", str(gradient_clipping)]
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE,
    }
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"reference failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    return json.loads(out_path.read_text())


def _compare_optimizer_artifacts(*args, **kwargs):
    """Load the optimizer comparator without importing all correctness checks at CLI startup."""
    module_path = Path(__file__).resolve().parents[1] / "checks" / "optimizer_state.py"
    module_name = "arctic_platform.correctness.checks.optimizer_state_grouping_control"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load optimizer comparator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "arctic_platform.correctness.checks"
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module.compare_optimizer_artifacts(*args, **kwargs)


def _grad_summary(label: str, dss, reference: dict, gate: float) -> GradSummary:
    pairs, only_dss, only_ref = align(dss.grad_norms, reference["grad_norms"])
    if only_dss or only_ref:
        print(f"[{label}] unmatched gradient names: dss={len(only_dss)} reference={len(only_ref)}", flush=True)
    diffs = sorted(((abs(a - b), name, a, b) for name, a, b in pairs), reverse=True)
    ratios = sorted(a / b for _name, a, b in pairs if b)
    worst, worst_name, _a, _b = diffs[0] if diffs else (0.0, "none", 0.0, 0.0)
    summary = GradSummary(
        label=label,
        over=sum(1 for diff, *_rest in diffs if diff > gate),
        compared=len(pairs),
        worst=worst,
        worst_name=worst_name,
        loss_delta=abs(dss.avg_loss - float(reference["loss"])),
        median_ratio=ratios[len(ratios) // 2] if ratios else 0.0,
    )
    print(
        f"[{label}] grads over {gate:.3e}: {summary.over}/{summary.compared}; "
        f"worst {summary.worst:.3e} {summary.worst_name}; "
        f"median ratio {summary.median_ratio:.6f}; loss delta {summary.loss_delta:.3e}",
        flush=True,
    )
    print(f"[{label}] worst gradient tensors", flush=True)
    for diff, name, dss_value, ref_value in diffs[:8]:
        print(f"  {name:56} dss {dss_value:>12.6f} ref {ref_value:>12.6f} diff {diff:>10.3e}", flush=True)
    return summary


def _optimizer_summary(
    label: str,
    dss_manifest: str,
    reference_manifest: str,
    *,
    optimizer_config: dict,
    gate: float,
) -> OptimizerSummary:
    comparison = _compare_optimizer_artifacts(
        Path(dss_manifest),
        Path(reference_manifest),
        optimizer_config=optimizer_config,
        learning_rate=OPTIMIZER_LEARNING_RATE,
    )
    if comparison.only_dss or comparison.only_reference:
        print(
            f"[{label}] unmatched optimizer names: dss={len(comparison.only_dss)} "
            f"reference={len(comparison.only_reference)}",
            flush=True,
        )
    deltas = sorted(comparison.deltas, key=lambda item: item.delta_norm, reverse=True)
    worst = deltas[0] if deltas else None
    summary = OptimizerSummary(
        label=label,
        over=sum(1 for item in deltas if item.delta_norm > gate),
        compared=len(deltas),
        worst=worst.delta_norm if worst else 0.0,
        worst_name=f"{worst.name}::{worst.state}" if worst else "none",
    )
    print(
        f"[{label}] optimizer over {gate:.3e}: {summary.over}/{summary.compared}; "
        f"worst {summary.worst:.3e} {summary.worst_name}",
        flush=True,
    )
    print(f"[{label}] worst optimizer tensors", flush=True)
    for item in deltas[:8]:
        print(f"  {item.name + '::' + item.state:72} delta {item.delta_norm:>10.3e}", flush=True)
    return summary


def _build_dss_payload(
    training: dict,
    model_path: str,
    attn: str,
    optimizer_state_output_dir: Path,
    *,
    include_gradient_telemetry: bool,
) -> dict:
    """Build the AP request for the optimizer diagnostic.

    Optimizer artifacts already carry the gradients used for the step comparison. Per-parameter gradient
    telemetry is a separate debug path, so keep it off unless explicitly requested for a gradient-only read.
    """
    return build_payload(
        training,
        model_path,
        SEED,
        attn_implementation=attn,
        optimizer_state_output_dir=optimizer_state_output_dir,
        gradient_norms_per_param=include_gradient_telemetry,
    )


def _ap_dp_non_sp_training(cfg: LoadedConfig) -> dict:
    """Return the diagnostic-only AP config with sequence parallelism disabled."""
    training = copy.deepcopy(cfg.training)
    training["sp_size"] = 1
    training["n_gpus"] = cfg.n_gpus

    required_batch = deepspeed_train_batch_size(training, cfg.n_gpus)
    if required_batch is not None:
        training["train_batch_size"] = required_batch
        if isinstance(training.get("ds_config"), dict):
            training["ds_config"]["train_batch_size"] = required_batch
    return training


def _print_interpretation(one_row: OptimizerSummary | GradSummary, grouped: OptimizerSummary | GradSummary) -> None:
    print("", flush=True)
    if grouped.over < one_row.over and grouped.worst < one_row.worst:
        print(
            "INTERPRETATION: grouped reference is closer to Arctic Platform. That supports the call-shape/"
            "grouped-backward hypothesis and points the next patch at reference grouping or the grouped GDN path.",
            flush=True,
        )
    elif grouped.over == one_row.over and grouped.worst >= one_row.worst * 0.8:
        print(
            "INTERPRETATION: grouped reference did not materially close the gap. That disfavors row grouping as "
            "the main cause and points back to AP-vs-reference kernel/model numerics.",
            flush=True,
        )
    else:
        print(
            "INTERPRETATION: grouping changed the comparison but did not cleanly close it. Inspect the top tensors "
            "above; the hypothesis is only partially supported.",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen36_grouping_control")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("spec", nargs="?", default=DEFAULT_SPEC)
    parser.add_argument("--arm", default="gas1", choices=("gas1", "gas4"))
    parser.add_argument("--group-rows", type=int, default=8)
    parser.add_argument("--reference-attn")
    parser.add_argument(
        "--include-gradient-telemetry",
        action="store_true",
        help=(
            "also request AP per-parameter gradient norms; by default the optimizer diagnostic skips this "
            "separate telemetry path"
        ),
    )
    args = parser.parse_args(argv)

    loaded_cfg = load_config(Path(args.config))
    cfg = loaded_cfg.at_gpu_width(loaded_cfg.n_gpus)
    spec = TestSpec.read(Path(args.spec))
    arm = _arm(spec, args.arm)
    model_path = _reference_model_path(spec)
    if not uses_mixer_packing(model_path):
        raise RuntimeError(
            f"{model_path} does not declare linear_attention layers; grouping control is not applicable"
        )

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = _workdir()
    batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, vocab, seed=SEED)
    batch_path = work / f"batch-{arm.name}.pt"
    save(batch, batch_path)

    fp32_lm_head, fused_cross_entropy, token_chunk_size, vocab_chunk_size = loss_settings(cfg)
    reference_training = cfg.effective_training
    reference_budget = _reference_token_budget(cfg, spec)
    attn = cfg.attention_implementation
    reference_attn = args.reference_attn or attn
    optimizer_config = cfg.training.get("optimizer")
    if not isinstance(optimizer_config, dict):
        raise RuntimeError("Qwen3.6 grouping diagnostic requires an explicit optimizer config")

    print(f"workdir {work}", flush=True)
    print(
        f"model {model_path}; arm {arm.name}: rows={arm.global_batch_size}, row_tokens={arm.max_seq_len}, "
        f"active_tokens={batch.active_tokens}; grouped reference rows/call={args.group_rows}",
        flush=True,
    )
    print(
        f"AP DP/non-SP on {cfg.n_gpus} GPU(s), attn={attn}; reference attn={reference_attn}; "
        f"reference token budget={reference_budget}; gradient telemetry={args.include_gradient_telemetry}",
        flush=True,
    )
    print(
        f"loss fp32_lm_head={fp32_lm_head}, fused_cross_entropy={fused_cross_entropy!r}, "
        f"lm_head_token_chunk={token_chunk_size}, lm_head_vocab_chunk={vocab_chunk_size}",
        flush=True,
    )

    references = {}
    for label, group_rows in (("one-row", 1), (f"grouped-{args.group_rows}", args.group_rows)):
        print(f"[reference {label}] running ...", flush=True)
        out_path = work / f"ref-{label}.json"
        optimizer_output_dir = work / "optimizer" / "reference" / label / "states"
        optimizer_output_dir.parent.mkdir(parents=True, exist_ok=True)
        references[label] = _run_reference(
            model_path,
            batch_path,
            out_path,
            token_budget=reference_budget,
            ce_chunk=2048,
            attn=reference_attn,
            fp32_lm_head=fp32_lm_head,
            fused_cross_entropy=fused_cross_entropy,
            mixer_packing_group_rows=group_rows,
            matmul_precision=reference_training.get("matmul_precision", "highest"),
            peft_config=reference_training.get("peft_config"),
            lm_head_token_chunk_size=token_chunk_size,
            lm_head_vocab_chunk_size=vocab_chunk_size,
            optimizer_config=optimizer_config,
            gradient_clipping=cfg.training.get("gradient_clipping"),
            optimizer_dtype=cfg.optimizer_dtype,
            optimizer_output_dir=optimizer_output_dir,
        )
        print(
            f"[reference {label}] loss {references[label]['loss']:.6f}; "
            f"microbatches {references[label]['microbatches']}; peak {references[label]['peak_gib']:.2f} GiB",
            flush=True,
        )

    training = _ap_dp_non_sp_training(cfg)
    dss_optimizer_dir = work / "optimizer" / "dss" / arm.name
    dss_optimizer_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_dss_payload(
        training,
        model_path,
        attn,
        dss_optimizer_dir,
        include_gradient_telemetry=args.include_gradient_telemetry,
    )

    print("[dss] running AP DP/non-SP ...", flush=True)
    with optimizer_capture_worker(), gateway(work / "gateway", cfg.n_gpus) as session:
        with running_job(session, payload) as job:
            dss = fwd_bwd_step(
                session,
                job,
                pack(batch, model_provider=str(cfg.training.get("model_provider", "huggingface"))),
                learning_rate=OPTIMIZER_LEARNING_RATE,
            )
    print(
        f"[dss] loss {dss.avg_loss:.6f}; model_calls={dss.model_calls}; "
        f"packed_rows={dss.packed_rows}; optimizer_manifest={dss.optimizer_state_manifest}",
        flush=True,
    )

    grad_gate = (
        spec.tolerance_for("single-step-grads")
        if "single-step-grads" in spec.test_tolerances
        else STATED_CRITERION_ABS
    )
    optimizer_gate = (
        spec.tolerance_for("single-step-optimizer")
        if "single-step-optimizer" in spec.test_tolerances
        else STATED_CRITERION_ABS
    )
    print("", flush=True)
    grouped_label = f"grouped-{args.group_rows}"
    if args.include_gradient_telemetry:
        one_row_grads = _grad_summary("one-row reference", dss, references["one-row"], grad_gate)
        grouped_grads = _grad_summary(f"{grouped_label} reference", dss, references[grouped_label], grad_gate)
    else:
        print(
            "[dss] gradient telemetry disabled; comparing optimizer artifacts only. "
            "Rerun with --include-gradient-telemetry for grad-norm summaries.",
            flush=True,
        )

    if dss.optimizer_state_manifest:
        print("", flush=True)
        one_row_opt = _optimizer_summary(
            "one-row reference",
            dss.optimizer_state_manifest,
            references["one-row"]["optimizer_state_manifest"],
            optimizer_config=optimizer_config,
            gate=optimizer_gate,
        )
        grouped_opt = _optimizer_summary(
            f"{grouped_label} reference",
            dss.optimizer_state_manifest,
            references[grouped_label]["optimizer_state_manifest"],
            optimizer_config=optimizer_config,
            gate=optimizer_gate,
        )
        _print_interpretation(one_row_opt, grouped_opt)
    elif args.include_gradient_telemetry:
        _print_interpretation(one_row_grads, grouped_grads)
    else:
        raise RuntimeError("AP returned no optimizer artifact manifest; no diagnostic signal was available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
