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

# !/usr/bin/env python3
"Render the fixed Hugging Face repeatability driver for one onboarded config."

from __future__ import annotations

import argparse
from pathlib import Path


def render(args: argparse.Namespace) -> str:
    from arctic_platform.correctness.harness.config import DEFAULT_REFERENCE_CALIBRATION_RUNS
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import load_config

    config = load_config(args.config)
    training = config.training
    lm_head_config = config.effective_training
    frozen_config_checksum = config_checksum(config)
    attention = config.reference_attention_implementation
    tokens = args.tokens if args.tokens is not None else config.max_seq_len
    fp32_lm_head = bool(lm_head_config.get("fp32_lm_head", False))
    fused_cross_entropy = lm_head_config.get("fused_cross_entropy", False)
    matmul_precision = training.get("matmul_precision", "highest")
    token_chunk = config.lm_head_token_chunk_size
    vocab_chunk = lm_head_config.get("fused_lm_head_vocab_chunk_size", 8192)
    peft_config = training.get("peft_config")
    peft_adapter_path = getattr(args, "peft_adapter_path", None)
    template = (
        "'Measure gradient-norm repeatability of the fixed reference.'\n\nfrom __future__ import"
        " annotations\n\nimport hashlib\nimport importlib.metadata\nimport json\nimport math\nimport os\nimport"
        " platform\nimport time\nfrom decimal import Decimal, ROUND_CEILING\nimport sys\nfrom concurrent.futures"
        " import ThreadPoolExecutor\nfrom pathlib import Path\n\nimport torch\nfrom transformers import"
        " AutoConfig\nfrom arctic_platform.correctness.harness.report import reference_progress\n\nfrom"
        " arctic_platform.correctness.harness.batches import build_batch, save as save_batch\nfrom"
        " arctic_platform.correctness.harness.runner import run_reference\nfrom"
        " arctic_platform.correctness.harness.batches import SEED\nfrom arctic_platform.correctness.harness.config"
        " import hash_directory\nfrom arctic_platform.correctness.onboarding.synth_model import"
        " materialize_pretrained\nfrom arctic_platform.correctness.reference.model_features import"
        " uses_mixer_packing\nfrom arctic_platform.correctness.harness.storage import correctness_workdir,"
        " require_data_fast_path\n\nSOURCE_PATH = __SOURCE__\nLAYERS = __LAYERS__\nMODEL_PATH = __MODEL__\nOUTPUT ="
        " Path(__OUTPUT__)\nWORK_DIR = __WORK_DIR__\nSPEC_PATH = Path(__SPEC_PATH__)\nTEST_ID ="
        " __TEST_ID__\nCONFIG_CHECKSUM = __CONFIG_CHECKSUM__\nRUNS = __RUNS__\nTOKENS ="
        ' __TOKENS__\nREFERENCE_TOKEN_BUDGET = __REFERENCE_TOKEN_BUDGET__\nGATE_QUANTUM = Decimal("0.001")\n# How far'
        " above the run-to-run variation the calibrated gate sits. Sixteen calibration runs measure the\n# tail"
        " imperfectly: on the CPU-offloaded Qwen3.8-27B config a factor of two produced a 2e-03 gate that the\n#"
        " validation run then exceeded at 2.929e-03, so a factor that only just covers the observed spread does not\n#"
        " survive the next run. Four keeps that measurement inside the gate with room, and stays far below the\n#"
        " disagreement a real defect produces -- a dropped microbatch gradient is a fraction of the correct norm,\n#"
        " not a per-mille difference.\nGATE_SAFETY_FACTOR = 4.0\n\n\ndef _select_gate(raw_variation: float) ->"
        " tuple[float, float]:\n    computed_gate = GATE_SAFETY_FACTOR * raw_variation\n    rounded_gate = (\n       "
        " Decimal(str(computed_gate)) / GATE_QUANTUM\n    ).to_integral_value(rounding=ROUND_CEILING) * GATE_QUANTUM\n"
        "    return computed_gate, max(float(GATE_QUANTUM), float(rounded_gate))\n\n\ndef _batch_hash(batch) -> str:\n"
        '    digest = hashlib.sha256()\n    for name in ("input_ids", "position_ids", "labels"):\n        tensor ='
        " getattr(batch, name).detach().cpu().contiguous()\n        digest.update(name.encode())\n       "
        " digest.update(str(tensor.dtype).encode())\n        digest.update(str(tuple(tensor.shape)).encode())\n       "
        " digest.update(tensor.numpy().tobytes())\n    return digest.hexdigest()\n\n\ndef _environment() -> dict:\n   "
        ' packages = {}\n    for distribution in ("torch", "transformers", "flash-attn", "flash-linear-attention"):\n '
        "       try:\n            packages[distribution] = importlib.metadata.version(distribution)\n        except"
        " importlib.metadata.PackageNotFoundError:\n            packages[distribution] = None\n    return {\n       "
        ' "python": sys.version.split()[0],\n        "platform": platform.platform(),\n        "cuda":'
        ' torch.version.cuda,\n        "packages": packages,\n    }\n\n\ndef main() -> int:\n    print(f"interpreter'
        ' {sys.prefix}", flush=True)\n    print("Preparing the HF calibration fixture", flush=True)\n   '
        " materialize_pretrained(SOURCE_PATH, MODEL_PATH, LAYERS)\n    model_cfg ="
        " AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)\n    inner = getattr(model_cfg,"
        ' "text_config", model_cfg)\n    layer_types = list(getattr(inner, "layer_types", ["full_attention"] *'
        ' int(inner.num_hidden_layers)))\n    spec = json.loads(SPEC_PATH.read_text())\n    arms = spec.get("arms",'
        ' [])\n    if not arms or len({arm["name"] for arm in arms}) != len(arms):\n        raise ValueError("gradient'
        ' calibration requires nonempty, uniquely named specification arms")\n    mixer_packing ='
        ' uses_mixer_packing(MODEL_PATH)\n    print(f"determinism: off, {RUNS} independent processes per arm",'
        ' flush=True)\n    tensor_dir = (\n        require_data_fast_path(WORK_DIR, label="gradient calibration'
        ' tensors")\n        if WORK_DIR is not None else correctness_workdir("reference-calibration-")\n    )\n   '
        " from arctic_platform.correctness.harness.config import reference_calibration_worker_count\n    lanes ="
        " reference_calibration_worker_count(__WORKERS__, runs=RUNS)\n\n    def one_run(index: int, batch_path: Path,"
        ' run_dir: Path) -> tuple[int, float, dict]:\n        started = time.monotonic()\n        print(f"HF'
        ' calibration start: arm={run_dir.name}, run={index}, device={index % lanes}", flush=True)\n        result ='
        " run_reference(\n            MODEL_PATH,\n            batch_path,\n            run_dir /"
        ' f"run-{index:02d}.json",\n            token_budget=REFERENCE_TOKEN_BUDGET,\n            ce_chunk=2048,\n    '
        "        attn=__ATTENTION__,\n            fp32_lm_head=__FP32_LM_HEAD__,\n           "
        " fused_cross_entropy=__FUSED_CROSS_ENTROPY__,\n            mixer_packing=mixer_packing,\n           "
        " matmul_precision=__MATMUL_PRECISION__,\n            peft_config=__PEFT_CONFIG__,\n           "
        " peft_adapter_path=__PEFT_ADAPTER_PATH__,\n            seed=SEED,\n           "
        " lm_head_token_chunk_size=__TOKEN_CHUNK__,\n            lm_head_vocab_chunk_size=__VOCAB_CHUNK__,\n          "
        "  deterministic=False,\n            cuda_device=index % lanes,\n        )\n        return index,"
        " time.monotonic() - started, result\n\n    ranges = []\n    arm_measurements = {}\n    for arm in arms:\n    "
        '    arm_name = arm["name"]\n        batch = build_batch(arm_name, arm["global_batch_size"],'
        ' arm["max_seq_len"], inner.vocab_size, seed=SEED)\n        batch_path = tensor_dir / f"batch-{arm_name}.pt"\n'
        '        save_batch(batch, batch_path)\n        run_dir = OUTPUT.parent / "calibration-runs" / arm_name\n     '
        "   run_dir.mkdir(parents=True, exist_ok=True)\n        completed = {}\n        for start in range(0, RUNS,"
        " lanes):\n            with ThreadPoolExecutor(max_workers=lanes) as pool:\n                futures ="
        " {pool.submit(one_run, index, batch_path, run_dir): index\n                           for index in"
        " range(start, min(start + lanes, RUNS))}\n                for _, (index, elapsed, result) in"
        ' reference_progress(\n                    futures, label=f"Finding tolerance {arm_name}", total=RUNS,'
        " completed=len(completed)\n                ):\n                    completed[index] = result\n               "
        '     print(f"HF calibration done: arm={arm_name}, run={index}, "\n                         '
        " f\"loss={result['loss']:.6f}, peak={result['peak_gib']:.1f}GiB, \"\n                         "
        ' f"elapsed={elapsed:.1f}s", flush=True)\n        print(f"Summarizing gradient variation for {arm_name}",'
        " flush=True)\n        results = [completed[index] for index in range(RUNS)]\n        names ="
        ' sorted(results[0]["grad_norms"])\n        if not names:\n            raise RuntimeError(f"{arm_name}:'
        ' reference runs returned no gradients")\n        for result in results:\n            if'
        ' sorted(result["grad_norms"]) != names:\n                raise RuntimeError(f"{arm_name}: reference runs'
        ' returned different gradient parameter sets")\n            if sorted(result["skipped_no_grad"]) !='
        ' sorted(results[0]["skipped_no_grad"]):\n                raise RuntimeError(f"{arm_name}: reference runs'
        ' skipped different parameters")\n            if result["active_tokens"] != batch.active_tokens:\n            '
        '    raise RuntimeError(f"{arm_name}: reference active-token count differs from the calibration batch")\n     '
        '       if not all(math.isfinite(value) and value >= 0 for value in result["grad_norms"].values()):\n         '
        '       raise RuntimeError(f"{arm_name}: reference gradient norms must be finite and nonnegative")\n          '
        '  if not math.isfinite(result["loss"]):\n                raise RuntimeError(f"{arm_name}: reference loss must'
        ' be finite")\n        for name in names:\n            values = [result["grad_norms"][name] for result in'
        ' results]\n            ranges.append({"arm": arm_name, "name": name, "minimum": min(values),\n               '
        '            "maximum": max(values), "range": max(values) - min(values), "values": values})\n        losses ='
        ' [result["loss"] for result in results]\n        arm_measurements[arm_name] = {\n           '
        ' "global_batch_size": batch.rows, "max_seq_len": batch.seq_len,\n            "active_tokens":'
        ' batch.active_tokens, "batch_hash": _batch_hash(batch),\n            "microbatches": [result["microbatches"]'
        ' for result in results],\n            "skipped_no_grad": results[0]["skipped_no_grad"],\n           '
        ' "losses": losses, "loss_range": max(losses) - min(losses),\n            "peak_gib": [result["peak_gib"] for'
        ' result in results],\n        }\n    ranges.sort(key=lambda row: row["range"], reverse=True)\n   '
        ' raw_variation = ranges[0]["range"]\n    computed_gate, selected_gate = _select_gate(raw_variation)\n   '
        ' print(f"Hashing calibration model files: {MODEL_PATH}", flush=True)\n    payload = {\n       '
        ' "configuration": {\n            "model": f"{Path(SOURCE_PATH).name} {LAYERS}-layer hybrid slice",\n         '
        '   "active_gdn_layers": layer_types.count("linear_attention"),\n            "active_full_attention_layers":'
        ' layer_types.count("full_attention"),\n            "tokens": TOKENS,\n            "attention_implementation":'
        ' __ATTENTION__,\n            "mixer_packing": mixer_packing,\n            "reference_token_budget":'
        ' REFERENCE_TOKEN_BUDGET,\n            "runs": RUNS,\n            "seed_reset_each_run": SEED,\n           '
        ' "deterministic_algorithms": False,\n            "cudnn_deterministic": False,\n           '
        ' "cublas_workspace_config": None,\n            "flash_attention_deterministic": False,\n           '
        ' "config_checksum": CONFIG_CHECKSUM,\n            "model_hash": hash_directory(Path(MODEL_PATH)),\n       '
        ' },\n        "environment": _environment(),\n        "arms": arm_measurements,\n        "loss_range":'
        ' max(arm["loss_range"] for arm in arm_measurements.values()),\n        "gradient_ranges": ranges,\n       '
        ' "tolerance": {\n            "worst_arm": ranges[0]["arm"],\n            "raw_max_same_tensor_range":'
        ' raw_variation,\n            "multiplier": GATE_SAFETY_FACTOR,\n            "computed_gate": computed_gate,\n'
        '            "rounding_quantum": float(GATE_QUANTUM),\n            "minimum_gate": float(GATE_QUANTUM),\n     '
        '       "selected_gate": selected_gate,\n        },\n    }\n    OUTPUT.parent.mkdir(parents=True,'
        ' exist_ok=True)\n    print(f"Writing calibration results: {OUTPUT}", flush=True)\n   '
        ' OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")\n    print(f"loss range:'
        ' {payload[\'loss_range\']:.9e}")\n    print(f"raw variation: {raw_variation:.9e}")\n    print(f"computed'
        ' {GATE_SAFETY_FACTOR:g}x gate: {computed_gate:.9e}")\n    print(f"selected gate: {selected_gate:.9e}")\n   '
        " print(f\"{'weight':64} {'minimum':>13} {'maximum':>13} {'range':>13}\")\n    for row in ranges[:10]:\n"
        "        print(f\"{row['name']:64} {row['minimum']:>13.9f} {row['maximum']:>13.9f}"
        " {row['range']:>13.6e}\")\n\n    spec = json.loads(SPEC_PATH.read_text())\n   "
        ' spec.setdefault("test_tolerances", {})[TEST_ID] = {\n        "absolute": selected_gate,\n       '
        ' "calibration_runs": RUNS,\n        "raw_max_same_tensor_range": raw_variation,\n        "multiplier":'
        ' GATE_SAFETY_FACTOR,\n        "computed_gate": computed_gate,\n        "rounding_quantum":'
        ' float(GATE_QUANTUM),\n        "minimum_gate": float(GATE_QUANTUM),\n        "worst_tensor":'
        ' ranges[0]["name"],\n        "status": f"calibrated from {RUNS} nondeterministic reference executions per'
        ' specification arm; {GATE_SAFETY_FACTOR:g}x rounded upward in 1e-3 units",\n    }\n   '
        ' SPEC_PATH.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\\n")\n    return 0\n\n\nif __name__ =='
        ' "__main__":\n    raise SystemExit(main())\n'
    )
    replacements = {
        "__SOURCE__": repr(str(args.source_checkpoint)),
        "__LAYERS__": str(args.layers),
        "__MODEL__": repr(str(args.model_path)),
        "__OUTPUT__": repr(str(args.output_json)),
        "__WORK_DIR__": repr(str(args.work_dir)) if getattr(args, "work_dir", None) is not None else "None",
        "__SPEC_PATH__": repr(str(args.test_spec)),
        "__TEST_ID__": repr(args.test_id),
        "__CONFIG_CHECKSUM__": repr(frozen_config_checksum),
        "__RUNS__": str(DEFAULT_REFERENCE_CALIBRATION_RUNS),
        "__WORKERS__": repr(getattr(args, "reference_calibration_workers", None)),
        "__TOKENS__": str(tokens),
        "__REFERENCE_TOKEN_BUDGET__": str(args.reference_token_budget),
        "__ATTENTION__": repr(attention),
        "__FP32_LM_HEAD__": repr(fp32_lm_head),
        "__FUSED_CROSS_ENTROPY__": repr(fused_cross_entropy),
        "__MATMUL_PRECISION__": repr(matmul_precision),
        "__PEFT_CONFIG__": repr(peft_config),
        "__PEFT_ADAPTER_PATH__": repr(str(peft_adapter_path)) if peft_adapter_path else "None",
        "__TOKEN_CHUNK__": repr(token_chunk),
        "__VOCAB_CHUNK__": str(vocab_chunk),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-script", type=Path, required=True)
    parser.add_argument(
        "--work-dir", type=Path, help="Tensor scratch directory under /data-fast (allocated automatically)"
    )
    parser.add_argument("--test-spec", type=Path, required=True)
    parser.add_argument("--test-id", default="single-step-grads")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--reference-token-budget", type=int, default=65536)
    args = parser.parse_args()
    args.output_script.parent.mkdir(parents=True, exist_ok=True)
    args.output_script.write_text(render(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
