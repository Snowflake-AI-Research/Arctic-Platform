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

"""Classify Qwen3.6 routed expert gradient scaling against optimizer artifacts.

This is the bounded follow-up to ``qwen36_expert_grad_telemetry``. It runs the
same DP/non-SP AP arm and compares routed expert norms at three points:

* independent HF reference gradients;
* AP full gradients reported by ``step()`` before ``engine.step()``;
* AP optimizer artifact gradients snapshotted immediately before ``engine.step()``.

If AP pre-step and artifact norms agree, artifact capture is reading the same
gradient the optimizer receives. If both are high by a group-size-like factor,
the issue is missing averaging before optimizer input. If only artifact/pre-step
is high, the issue is in artifact capture/globalization.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import DEFAULT_CONFIG  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import DEFAULT_SPEC  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _arm  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _expert_compare_name  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _format_optional  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _is_routed_expert_name  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _load_manifest_entries  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _loss_settings  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _reference_model_path  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _reference_token_budget  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import (  # noqa: E402
    _run_dss_with_raw_metrics,
)
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _safe_workdir  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _tensor_norm  # noqa: E402
from arctic_platform.correctness.diagnostics.qwen36_expert_grad_telemetry import _training_for_dp_non_sp  # noqa: E402
from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import load as load_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.names import canonical_dss_tensor_groups  # noqa: E402
from arctic_platform.correctness.harness.optimizer_capture import optimizer_capture_worker  # noqa: E402
from arctic_platform.correctness.harness.runner import run_reference  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.reference.model_features import uses_mixer_packing  # noqa: E402

SCALE_REL_TOL = 0.20
HIGH_RATIO = 1.25


@dataclass(frozen=True)
class ScaleCandidate:
    name: str
    factor: float


@dataclass(frozen=True)
class ScalingRow:
    name: str
    reference: float
    pre_step: float | None
    artifact_gradient: float | None
    exp_avg_effective: float | None
    pre_to_ref: float | None
    artifact_to_ref: float | None
    artifact_to_pre: float | None
    exp_avg_effective_to_ref: float | None
    expert_count: int | None
    classification: str

    @property
    def sort_key(self) -> float:
        return abs((self.artifact_to_ref or self.pre_to_ref or 1.0) - 1.0)


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return value / reference


def _close_to(value: float | None, target: float, *, rel_tol: float = SCALE_REL_TOL) -> bool:
    if value is None or target == 0.0:
        return False
    return abs(value - target) / abs(target) <= rel_tol


def scale_candidates(n_gpus: int, ep_size: int, router_top_k: int | None) -> list[ScaleCandidate]:
    """Return plausible missing-average factors for this topology."""
    candidates = [ScaleCandidate("no scaling", 1.0)]
    if ep_size > 1:
        candidates.append(ScaleCandidate("expert_parallel", float(ep_size)))
    expert_dp = n_gpus // max(ep_size, 1)
    if expert_dp > 1:
        candidates.append(ScaleCandidate("expert_data_parallel", float(expert_dp)))
    if n_gpus > 1:
        candidates.append(ScaleCandidate("world_size", float(n_gpus)))
    if router_top_k is not None and router_top_k > 1:
        candidates.append(ScaleCandidate("router_top_k", float(router_top_k)))

    unique: dict[float, ScaleCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.factor, candidate)
    return list(unique.values())


def closest_scale(value: float | None, candidates: Sequence[ScaleCandidate]) -> ScaleCandidate | None:
    if value is None:
        return None
    matches = [candidate for candidate in candidates if _close_to(value, candidate.factor)]
    if not matches:
        return None
    return min(matches, key=lambda candidate: abs(candidate.factor - value))


def _normalized_routed_norms(values: Mapping[str, float]) -> dict[str, float]:
    return {_expert_compare_name(name): float(value) for name, value in values.items() if _is_routed_expert_name(name)}


def _expert_counts(per_expert: Mapping[str, Sequence[float]]) -> dict[str, int]:
    return {
        _expert_compare_name(name): len(values) for name, values in per_expert.items() if _is_routed_expert_name(name)
    }


def artifact_tensor_norms(manifest_path: str | Path, tensor_key: str) -> dict[str, float]:
    """Return routed expert norms for one optimizer artifact tensor key."""
    import torch

    entries = _load_manifest_entries(manifest_path)
    groups = canonical_dss_tensor_groups(entries)
    result: dict[str, float] = {}
    for name, group in groups.items():
        if not _is_routed_expert_name(name):
            continue
        values = []
        for original in group.originals:
            _norm, tensor = _tensor_norm(entries[original], tensor_key)
            values.append(tensor)
        combined = values[0] if group.concatenate_dim is None else torch.cat(values, dim=group.concatenate_dim)
        result[name] = float(torch.linalg.vector_norm(combined).cpu())
    return result


def effective_exp_avg_norms(manifest_path: str | Path, beta1: float) -> dict[str, float]:
    """Convert Adam's first-moment norm to the gradient scale used by the optimizer."""
    if beta1 >= 1.0:
        raise ValueError(f"Adam beta1 must be less than 1.0, got {beta1}")
    scale = 1.0 / (1.0 - beta1)
    return {name: value * scale for name, value in artifact_tensor_norms(manifest_path, "exp_avg").items()}


def classify_scaling(
    *,
    pre_to_ref: float | None,
    artifact_to_ref: float | None,
    artifact_to_pre: float | None,
    exp_avg_effective_to_ref: float | None,
    candidates: Sequence[ScaleCandidate],
) -> str:
    """Describe where the scaling first appears."""
    artifact_matches_pre = _close_to(artifact_to_pre, 1.0, rel_tol=0.05)
    pre_scale = closest_scale(pre_to_ref, candidates)
    artifact_pre_scale = closest_scale(artifact_to_pre, candidates)
    moment_matches_reference = _close_to(exp_avg_effective_to_ref, 1.0, rel_tol=0.10)
    moment_matches_pre = _close_to(exp_avg_effective_to_ref, pre_to_ref or 0.0, rel_tol=0.10)

    if pre_to_ref is not None and pre_to_ref > HIGH_RATIO and artifact_matches_pre:
        if moment_matches_reference:
            return "pre-step high; optimizer appears to rescale before moments"
        if moment_matches_pre:
            suffix = f" near {pre_scale.name}" if pre_scale and pre_scale.factor != 1.0 else ""
            return f"pre-step high{suffix}; optimizer moments consume same scale"
        suffix = f" near {pre_scale.name}" if pre_scale and pre_scale.factor != 1.0 else ""
        return f"pre-step high{suffix}; artifact matches pre-step"

    if artifact_to_ref is not None and artifact_to_ref > HIGH_RATIO and artifact_pre_scale:
        return f"artifact/globalization adds scale near {artifact_pre_scale.name}"

    if artifact_to_ref is not None and artifact_to_ref > HIGH_RATIO:
        return "artifact high, but not by a configured group-size factor"

    if (
        pre_to_ref is not None
        and pre_to_ref <= HIGH_RATIO
        and artifact_to_ref is not None
        and artifact_to_ref <= HIGH_RATIO
    ):
        return "no routed expert inflation at these capture points"

    return "insufficient shared routed expert data"


def summarize_scaling(
    reference: Mapping[str, float],
    pre_step: Mapping[str, float],
    artifact_gradient: Mapping[str, float],
    exp_avg_effective: Mapping[str, float],
    per_expert: Mapping[str, Sequence[float]],
    *,
    candidates: Sequence[ScaleCandidate],
) -> list[ScalingRow]:
    reference_norms = _normalized_routed_norms(reference)
    pre_step_norms = _normalized_routed_norms(pre_step)
    artifact_norms = _normalized_routed_norms(artifact_gradient)
    exp_avg_norms = _normalized_routed_norms(exp_avg_effective)
    counts = _expert_counts(per_expert)
    rows = []
    for name in sorted(set(reference_norms) & (set(pre_step_norms) | set(artifact_norms) | set(exp_avg_norms))):
        ref = reference_norms[name]
        pre = pre_step_norms.get(name)
        artifact = artifact_norms.get(name)
        exp_avg = exp_avg_norms.get(name)
        pre_to_ref = _ratio(pre, ref)
        artifact_to_ref = _ratio(artifact, ref)
        artifact_to_pre = _ratio(artifact, pre)
        exp_avg_to_ref = _ratio(exp_avg, ref)
        rows.append(
            ScalingRow(
                name=name,
                reference=ref,
                pre_step=pre,
                artifact_gradient=artifact,
                exp_avg_effective=exp_avg,
                pre_to_ref=pre_to_ref,
                artifact_to_ref=artifact_to_ref,
                artifact_to_pre=artifact_to_pre,
                exp_avg_effective_to_ref=exp_avg_to_ref,
                expert_count=counts.get(name),
                classification=classify_scaling(
                    pre_to_ref=pre_to_ref,
                    artifact_to_ref=artifact_to_ref,
                    artifact_to_pre=artifact_to_pre,
                    exp_avg_effective_to_ref=exp_avg_to_ref,
                    candidates=candidates,
                ),
            )
        )
    rows.sort(key=lambda row: row.sort_key, reverse=True)
    return rows


def _optimizer_beta1(training: Mapping[str, Any]) -> float:
    optimizer = training.get("optimizer") or {}
    if not isinstance(optimizer, Mapping):
        return 0.9
    params = optimizer.get("params") if isinstance(optimizer.get("params"), Mapping) else optimizer
    betas = params.get("betas") if isinstance(params, Mapping) else None
    if isinstance(betas, Sequence) and not isinstance(betas, (str, bytes)) and betas:
        return float(betas[0])
    return 0.9


def print_rows(rows: Sequence[ScalingRow], limit: int) -> None:
    print("")
    print(
        "name                                                       ref_norm      pre/ref   artifact/ref  "
        "artifact/pre  moment/ref  experts  classification",
        flush=True,
    )
    for row in rows[:limit]:
        print(
            f"{row.name[:58]:58}  "
            f"{row.reference:>10.6f}  "
            f"{_format_optional(row.pre_to_ref):>9}  "
            f"{_format_optional(row.artifact_to_ref):>12}  "
            f"{_format_optional(row.artifact_to_pre):>12}  "
            f"{_format_optional(row.exp_avg_effective_to_ref):>10}  "
            f"{_format_optional(row.expert_count):>7}  "
            f"{row.classification}",
            flush=True,
        )


def print_interpretation(rows: Sequence[ScalingRow]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    print("")
    if not counts:
        print("INTERPRETATION: no shared routed expert rows were available for scaling classification.", flush=True)
        return
    dominant, total = max(counts.items(), key=lambda item: item[1])
    print(f"INTERPRETATION: dominant routed expert pattern ({total}/{len(rows)}): {dominant}.", flush=True)


def _load_router_top_k(model_path: str) -> int | None:
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    for attr in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k"):
        value = getattr(text_cfg, attr, None)
        if value is not None:
            return int(value)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen36_expert_grad_scaling")
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("spec", nargs="?", default=DEFAULT_SPEC)
    parser.add_argument("--arm", default="gas1", choices=("gas1", "gas4"))
    parser.add_argument("--work-dir")
    parser.add_argument("--reference-attn")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args(argv)

    loaded_cfg = load_config(Path(args.config))
    cfg = loaded_cfg.at_gpu_width(loaded_cfg.n_gpus)
    spec = TestSpec.read(Path(args.spec))
    arm = _arm(spec, args.arm)
    model_path = _reference_model_path(spec)
    attn = cfg.attention_implementation
    reference_attn = args.reference_attn or attn
    router_top_k = _load_router_top_k(model_path)
    candidates = scale_candidates(cfg.n_gpus, int(cfg.training.get("ep_size", 1)), router_top_k)

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(getattr(model_cfg, "text_config", model_cfg), "vocab_size")

    work = _safe_workdir(args.work_dir)
    batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, vocab, seed=SEED)
    batch_path = work / f"batch-{arm.name}.pt"
    save(batch, batch_path)

    fp32_lm_head, fused_cross_entropy, token_chunk_size, vocab_chunk_size = _loss_settings(cfg)
    beta1 = _optimizer_beta1(cfg.training)

    print(f"workdir {work}", flush=True)
    print(
        f"model {model_path}; arm {arm.name}: rows={arm.global_batch_size}, row_tokens={arm.max_seq_len}, "
        f"active_tokens={batch.active_tokens}",
        flush=True,
    )
    print(
        "scale candidates: " + ", ".join(f"{candidate.name}={candidate.factor:g}" for candidate in candidates),
        flush=True,
    )

    print("[reference] running for gradient norms ...", flush=True)
    reference = run_reference(
        model_path,
        batch_path,
        work / f"ref-{arm.name}.json",
        token_budget=_reference_token_budget(cfg, spec),
        ce_chunk=2048,
        attn=reference_attn,
        fp32_lm_head=fp32_lm_head,
        seed=SEED,
        fused_cross_entropy=fused_cross_entropy,
        mixer_packing=uses_mixer_packing(model_path),
        matmul_precision=cfg.effective_training.get("matmul_precision", "highest"),
        peft_config=cfg.effective_training.get("peft_config"),
        lm_head_token_chunk_size=token_chunk_size,
        lm_head_vocab_chunk_size=vocab_chunk_size,
        gradient_clipping=cfg.training.get("gradient_clipping"),
        optimizer_dtype=cfg.optimizer_dtype,
    )
    print(f"[reference] loss {reference['loss']:.6f}; gradients {len(reference['grad_norms'])}", flush=True)

    training = _training_for_dp_non_sp(cfg)
    dss_optimizer_dir = work / "optimizer" / "dss" / arm.name
    dss_optimizer_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(
        training,
        model_path,
        SEED,
        attn_implementation=attn,
        optimizer_state_output_dir=dss_optimizer_dir,
        gradient_norms_per_param=True,
    )
    body = pack(load_batch(batch_path), model_provider=str(cfg.training.get("model_provider", "huggingface")))

    print("[dss] running AP DP/non-SP with pre-step telemetry and optimizer artifacts ...", flush=True)
    with optimizer_capture_worker(), gateway(work / "gateway", cfg.n_gpus) as session:
        dss = _run_dss_with_raw_metrics(session, payload, body)
    if dss.optimizer_state_manifest is None:
        raise RuntimeError("AP run did not return optimizer_state_manifest")
    print(
        f"[dss] loss {dss.avg_loss:.6f}; pre_step_norms={len(dss.global_norms)}; "
        f"per_expert={len(dss.per_expert)}; optimizer_manifest={dss.optimizer_state_manifest}",
        flush=True,
    )

    artifact_gradients = artifact_tensor_norms(dss.optimizer_state_manifest, "gradient")
    exp_avg_effective = effective_exp_avg_norms(dss.optimizer_state_manifest, beta1)
    rows = summarize_scaling(
        reference["grad_norms"],
        dss.global_norms,
        artifact_gradients,
        exp_avg_effective,
        dss.per_expert,
        candidates=candidates,
    )
    print(
        f"routed expert rows compared={len(rows)}; artifact_gradients={len(artifact_gradients)}; "
        f"effective_exp_avg={len(exp_avg_effective)}; beta1={beta1}",
        flush=True,
    )
    print_rows(rows, args.top)
    print_interpretation(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
