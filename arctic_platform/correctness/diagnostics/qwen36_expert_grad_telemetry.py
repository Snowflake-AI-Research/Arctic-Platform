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

"""Compare Qwen3.6 expert grad telemetry with optimizer artifact gradients.

This is a bounded diagnostic for the Qwen3.6 DP/non-SP gas1 failure. It runs one
reference and one Arctic Platform job from the same batch, then compares routed
expert gradient norms from three sources:

* reference ``grad_norms`` from the independent backward pass;
* Arctic Platform ``gradient_norms_per_param`` / ``gradient_norms_per_expert`` telemetry;
* Arctic Platform optimizer artifact pre-step ``gradient`` tensors.

If AP artifact gradients match reference while telemetry is high, the next patch
target is telemetry reconstruction. If AP artifact gradients are high while
Adam moments are small, the next target is optimizer-step scaling.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from arctic_platform.correctness.harness.arms import correctness_microbatch_tokens  # noqa: E402
from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import load as load_batch  # noqa: E402
from arctic_platform.correctness.harness.batches import save  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import _globalize_expert_norms  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import _rank_owned_mapping  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import _rank_owned_path  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import _scalar  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import build_payload  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import gateway  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import pack  # noqa: E402
from arctic_platform.correctness.harness.dss_driver import running_job  # noqa: E402
from arctic_platform.correctness.harness.names import canonical_dss_tensor_groups  # noqa: E402
from arctic_platform.correctness.harness.names import normalize  # noqa: E402
from arctic_platform.correctness.harness.optimizer_capture import optimizer_capture_worker  # noqa: E402
from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE  # noqa: E402
from arctic_platform.correctness.harness.runner import run_reference  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import ArmSpec  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.harness.workdir import correctness_workdir  # noqa: E402
from arctic_platform.correctness.reference.model_features import uses_mixer_packing  # noqa: E402

DEFAULT_CONFIG = "arctic_platform/correctness/configs/qwen3.6-35b-a3b/h200/train-sft-8gpus-64k.config"
DEFAULT_SPEC = "arctic_platform/correctness/specs/qwen3.6-35b-a3b-h200-train-sft-8gpus-64k.json"
RATIO_GATE = 1.25
MATCH_RATIO = 1.05


@dataclass(frozen=True)
class RawDssResult:
    avg_loss: float
    per_param: dict[str, float]
    per_expert: dict[str, list[float]]
    global_norms: dict[str, float]
    model_calls: int | None
    packed_rows: int | None
    optimizer_state_manifest: str | None


@dataclass(frozen=True)
class ExpertRow:
    name: str
    reference: float
    telemetry: float | None
    artifact: float | None
    telemetry_ratio: float | None
    artifact_ratio: float | None
    telemetry_delta: float | None
    artifact_delta: float | None
    expert_count: int | None

    @property
    def hottest_delta(self) -> float:
        return max(abs(self.telemetry_delta or 0.0), abs(self.artifact_delta or 0.0))


def _arm(spec: TestSpec, name: str) -> ArmSpec:
    for arm in spec.arms:
        if arm.name == name:
            return arm
    raise ValueError(f"spec has no arm {name!r}; available arms are {', '.join(arm.name for arm in spec.arms)}")


def _loss_settings(cfg) -> tuple[bool, bool | str, int | None, int]:
    training = cfg.effective_training
    return (
        bool(training.get("fp32_lm_head", False)),
        cfg.fused_cross_entropy,
        cfg.lm_head_token_chunk_size,
        int(training.get("fused_lm_head_vocab_chunk_size", 8192)),
    )


def _safe_workdir(value: str | None) -> Path:
    if value:
        root = Path(value)
        root.mkdir(parents=True, exist_ok=True)
        return root
    return correctness_workdir("qwen36-expert-grad-telemetry-")


def _reference_token_budget(cfg, spec: TestSpec) -> int:
    budget = correctness_microbatch_tokens(cfg.max_tokens_per_mb)
    optimizer_settings = spec.test_settings.get("single-step-optimizer") or {}
    if "reference_token_budget" in optimizer_settings:
        budget = min(budget, int(optimizer_settings["reference_token_budget"]))
    return budget


def _reference_model_path(spec: TestSpec) -> str:
    optimizer_settings = spec.test_settings.get("single-step-optimizer") or {}
    return str(optimizer_settings.get("model_cache_path") or spec.model.cache_path)


def _training_for_dp_non_sp(cfg) -> dict[str, Any]:
    training = copy.deepcopy(cfg.training)
    training["n_gpus"] = cfg.n_gpus
    training["sp_size"] = 1
    return training


def _as_float_mapping(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def _as_float_list_mapping(value: Mapping[str, Any]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key, item in value.items():
        if isinstance(item, (list, tuple)):
            result[str(key)] = [float(part) for part in item]
    return result


def _metric_mapping(source: Mapping[str, Any], metrics: Mapping[str, Any], key: str) -> dict[str, Any]:
    return _rank_owned_mapping(source.get(key) or metrics.get(key) or {})


def _run_dss_with_raw_metrics(session, payload: dict, body: dict) -> RawDssResult:
    with running_job(session, payload) as job:
        response = job.client.fwd_bwd(body)
        stepped = job.client.step(OPTIMIZER_LEARNING_RATE)
    step_metrics = stepped.get("metrics") or {}
    response_metrics = response.get("metrics") or {}
    per_param = _as_float_mapping(_metric_mapping(stepped, step_metrics, "gradient_norms_per_param"))
    per_expert = _as_float_list_mapping(_metric_mapping(stepped, step_metrics, "gradient_norms_per_expert"))
    loss = response.get("avg_loss", response_metrics.get("loss"))
    if loss is None:
        raise RuntimeError(f"forward-backward response carries no loss: {sorted(response)}")
    calls = response_metrics.get("sft_model_calls")
    rows = response_metrics.get("sft_packed_rows")
    return RawDssResult(
        avg_loss=_scalar(loss),
        per_param=per_param,
        per_expert=per_expert,
        global_norms=_globalize_expert_norms(per_param, per_expert),
        model_calls=int(calls) if calls is not None else None,
        packed_rows=int(rows) if rows is not None else None,
        optimizer_state_manifest=_rank_owned_path(
            stepped.get("optimizer_state_manifest") or step_metrics.get("optimizer_state_manifest")
        ),
    )


def _load_manifest_entries(manifest_path: str | Path) -> dict[str, Path]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    entries: dict[str, Path] = {}
    for item in manifest["parameters"]:
        name = normalize(str(item["name"]))
        if name in entries:
            raise ValueError(f"optimizer artifact has duplicate normalized parameter name: {name}")
        entries[name] = manifest_path.parent / str(item["file"])
    return entries


def _tensor_norm(path: Path, key: str):
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if key not in artifact:
        raise ValueError(f"{path} is missing tensor {key!r}")
    value = artifact[key].to(dtype=torch.float32)
    return float(torch.linalg.vector_norm(value).cpu()), value


def _dss_artifact_gradient_norms(manifest_path: str | Path) -> dict[str, float]:
    import torch

    entries = _load_manifest_entries(manifest_path)
    groups = canonical_dss_tensor_groups(entries)
    result: dict[str, float] = {}
    for name, group in groups.items():
        if not _is_routed_expert_name(name):
            continue
        values = []
        for original in group.originals:
            _norm, tensor = _tensor_norm(entries[original], "gradient")
            values.append(tensor)
        combined = values[0] if group.concatenate_dim is None else torch.cat(values, dim=group.concatenate_dim)
        result[name] = float(torch.linalg.vector_norm(combined).cpu())
    return result


def _is_routed_expert_name(name: str) -> bool:
    normalized = normalize(name)
    if normalized.endswith(".weight"):
        normalized = normalized.removesuffix(".weight")
    return ".mlp.experts." in normalized and (
        normalized.endswith(".gate_up_proj")
        or normalized.endswith(".down_proj")
        or normalized.endswith(".w1")
        or normalized.endswith(".w2")
        or normalized.endswith(".w3")
    )


def _expert_compare_name(name: str) -> str:
    normalized = normalize(name)
    if normalized.endswith(".weight") and _is_routed_expert_name(normalized):
        return normalized.removesuffix(".weight")
    return normalized


def _ratio(value: float | None, reference: float) -> float | None:
    if value is None or reference == 0.0:
        return None
    return value / reference


def _expert_counts(per_expert: Mapping[str, Sequence[float]]) -> dict[str, int]:
    return {
        _expert_compare_name(name): len(values) for name, values in per_expert.items() if _is_routed_expert_name(name)
    }


def summarize_experts(
    reference_artifacts: Mapping[str, float],
    telemetry: Mapping[str, float],
    artifact: Mapping[str, float],
    per_expert: Mapping[str, Sequence[float]],
) -> list[ExpertRow]:
    telemetry_norms = {
        _expert_compare_name(key): float(value) for key, value in telemetry.items() if _is_routed_expert_name(key)
    }
    artifact_norms = {
        _expert_compare_name(key): float(value) for key, value in artifact.items() if _is_routed_expert_name(key)
    }
    reference_norms = {
        _expert_compare_name(key): float(value)
        for key, value in reference_artifacts.items()
        if _is_routed_expert_name(key)
    }
    counts = _expert_counts(per_expert)
    rows = []
    for name in sorted(set(reference_norms) & (set(telemetry_norms) | set(artifact_norms))):
        reference = reference_norms[name]
        telem = telemetry_norms.get(name)
        art = artifact_norms.get(name)
        rows.append(
            ExpertRow(
                name=name,
                reference=reference,
                telemetry=telem,
                artifact=art,
                telemetry_ratio=_ratio(telem, reference),
                artifact_ratio=_ratio(art, reference),
                telemetry_delta=None if telem is None else telem - reference,
                artifact_delta=None if art is None else art - reference,
                expert_count=counts.get(name),
            )
        )
    rows.sort(key=lambda row: row.hottest_delta, reverse=True)
    return rows


def _format_optional(value: float | int | None, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if math.isfinite(value):
        return f"{value:.{precision}f}"
    return str(value)


def print_rows(rows: Sequence[ExpertRow], limit: int) -> None:
    print("")
    print(
        "name                                                       ref_norm      telemetry    telem/ref  "
        "artifact     artifact/ref  experts",
        flush=True,
    )
    for row in rows[:limit]:
        print(
            f"{row.name[:58]:58}  "
            f"{row.reference:>10.6f}  "
            f"{_format_optional(row.telemetry):>10}  "
            f"{_format_optional(row.telemetry_ratio):>9}  "
            f"{_format_optional(row.artifact):>10}  "
            f"{_format_optional(row.artifact_ratio):>12}  "
            f"{_format_optional(row.expert_count):>7}",
            flush=True,
        )


def print_interpretation(rows: Sequence[ExpertRow]) -> None:
    compared = [row for row in rows if row.telemetry_ratio is not None and row.artifact_ratio is not None]
    telemetry_high_artifact_matches = [
        row
        for row in compared
        if row.telemetry_ratio is not None
        and row.artifact_ratio is not None
        and row.telemetry_ratio > RATIO_GATE
        and abs(row.artifact_ratio - 1.0) <= MATCH_RATIO - 1.0
    ]
    artifact_high = [row for row in compared if row.artifact_ratio is not None and row.artifact_ratio > RATIO_GATE]
    print("")
    if telemetry_high_artifact_matches and not artifact_high:
        print(
            "INTERPRETATION: AP artifact gradients match reference while routed expert telemetry is high. "
            "Telemetry reconstruction is the next patch target.",
            flush=True,
        )
    elif artifact_high:
        print(
            "INTERPRETATION: AP optimizer-artifact gradients are high for routed experts. If optimizer moments remain "
            "small, inspect DeepSpeed expert gradient scaling during optimizer step next.",
            flush=True,
        )
    elif compared:
        print(
            "INTERPRETATION: neither AP routed expert telemetry nor optimizer-artifact gradients show a large routed "
            "expert norm inflation against reference in the printed rows.",
            flush=True,
        )
    else:
        print("INTERPRETATION: no shared routed expert rows were available for comparison.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen36_expert_grad_telemetry")
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

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    work = _safe_workdir(args.work_dir)
    batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, vocab, seed=SEED)
    batch_path = work / f"batch-{arm.name}.pt"
    save(batch, batch_path)

    fp32_lm_head, fused_cross_entropy, token_chunk_size, vocab_chunk_size = _loss_settings(cfg)
    optimizer_config = cfg.training.get("optimizer")
    if not isinstance(optimizer_config, dict):
        raise RuntimeError("Qwen3.6 expert telemetry diagnostic requires an explicit optimizer config")

    print(f"workdir {work}", flush=True)
    print(
        f"model {model_path}; arm {arm.name}: rows={arm.global_batch_size}, row_tokens={arm.max_seq_len}, "
        f"active_tokens={batch.active_tokens}",
        flush=True,
    )
    print(
        f"AP DP/non-SP on {cfg.n_gpus} GPU(s), sp_size=1, ep_size={cfg.training.get('ep_size')}; "
        f"AP attn={attn}; reference attn={reference_attn}",
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
    print(
        f"[reference] loss {reference['loss']:.6f}; gradients {len(reference['grad_norms'])}",
        flush=True,
    )

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

    print("[dss] running AP DP/non-SP with raw gradient telemetry and optimizer artifacts ...", flush=True)
    with optimizer_capture_worker(), gateway(work / "gateway", cfg.n_gpus) as session:
        dss = _run_dss_with_raw_metrics(session, payload, body)
    if dss.optimizer_state_manifest is None:
        raise RuntimeError("AP run did not return optimizer_state_manifest")
    print(
        f"[dss] loss {dss.avg_loss:.6f}; per_param={len(dss.per_param)}; per_expert={len(dss.per_expert)}; "
        f"model_calls={dss.model_calls}; packed_rows={dss.packed_rows}; "
        f"optimizer_manifest={dss.optimizer_state_manifest}",
        flush=True,
    )

    dss_artifact_norms = _dss_artifact_gradient_norms(dss.optimizer_state_manifest)
    rows = summarize_experts(reference["grad_norms"], dss.global_norms, dss_artifact_norms, dss.per_expert)
    print(
        f"routed expert rows compared={len(rows)}; reference_grad_norms={len(reference['grad_norms'])}; "
        f"dss_artifacts={len(dss_artifact_norms)}",
        flush=True,
    )
    print_rows(rows, args.top)
    print_interpretation(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
