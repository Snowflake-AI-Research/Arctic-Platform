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

"""Execute selected tests for one config and collect their results.

HF measurements are prepared explicitly during onboarding. Regression validates compact Git records
and reconstructed inputs before allocating any target GPUs. Normal-workload references stay live.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.model.implementations.debug.determinism import CUBLAS_WORKSPACE

from ..reference.model_features import uses_mixer_packing
from . import report as console
from .batches import SEED
from .batches import build_batch
from .batches import save
from .config import ArmDefinition
from .config import ArmSpec
from .config import LoadedConfig
from .config import TestSpec
from .config import correctness_microbatch_tokens
from .dss_driver import UnsafeCleanupError
from .dss_driver import build_payload
from .dss_driver import copy_checkpoint_peft_adapter
from .dss_driver import fwd_bwd_step
from .dss_driver import gateway
from .dss_driver import pack
from .dss_driver import running_job
from .registry import TestOutcome
from .registry import TestResult
from .registry import registered_tests
from .storage import correctness_workdir
from .storage import require_data_fast_path


@dataclass
class RunContext:
    config_id: str
    cfg: LoadedConfig
    spec: TestSpec
    arms: List[ArmSpec]
    attn_implementation: str
    workdir: Path
    batch_paths: Dict[str, Path]
    # The materialized model's vocabulary, so a check that builds its own batches draws the same token
    # range the runner's cases were drawn from.
    vocab_size: int
    reference_adapter_path: Path | None = None
    client_backend: OnPremConfig | CortexConfig = field(default_factory=OnPremConfig)
    # Populated only from a fixture binding validated against the reviewed local weights.
    target_model_name: str | None = None
    prepare_references: bool = False
    reference_records_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "reference_data")
    reference_record_paths: Dict[str, str] = field(default_factory=dict)
    reference_record_provenance: Dict[str, dict] = field(default_factory=dict)
    _reference: Dict[str, dict] = field(default_factory=dict)
    _reference_by_check: Dict[str, Dict[str, dict]] = field(default_factory=dict)
    _target: Dict[str, object] = field(default_factory=dict)
    # Seconds spent in each engine per case, so the report can say which case dominates the run.
    _timing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reference_attn_implementation: str | None = None

    @property
    def reference_attention_implementation(self) -> str:
        """Keep explicit HF overrides and legacy target overrides for other architectures."""
        if self.reference_attn_implementation is not None:
            return self.reference_attn_implementation
        reference = self.cfg.reference_attention_implementation
        return reference if reference != self.cfg.attention_implementation else self.attn_implementation

    def reference_for(self, arm: ArmSpec) -> dict:
        return self._reference[arm.name]

    def target_for(self, arm: ArmSpec):
        return self._target[arm.name]


def run_reference(
    model_path: str,
    batch_path: Path,
    out_path: Path,
    *,
    token_budget: int,
    ce_chunk: int,
    attn: str,
    fp32_lm_head: bool,
    seed: int = SEED,
    fused_cross_entropy: bool | str = False,
    mixer_packing: bool = False,
    matmul_precision: str = "highest",
    peft_config: dict | None = None,
    peft_adapter_path: Path | None = None,
    lm_head_token_chunk_size: int | None = None,
    lm_head_vocab_chunk_size: int = 8192,
    optimizer_config: dict | None = None,
    learning_rate: float | None = None,
    gradient_clipping: float | None = None,
    optimizer_dtype: str = "float32",
    optimizer_output_dir: Path | None = None,
    cuda_device: int | None = None,
    deterministic: bool = True,
) -> dict:
    """Invoke the reference as a subprocess so the node is clean before the gateway starts."""
    out_path = Path(out_path)
    if optimizer_output_dir is not None:
        optimizer_output_dir = require_data_fast_path(optimizer_output_dir, label="reference optimizer output")
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
        str(seed),
    ]
    if fp32_lm_head:
        cmd.append("--fp32-lm-head")
    if fused_cross_entropy:
        cmd += ["--fused-cross-entropy", "liger" if fused_cross_entropy is True else str(fused_cross_entropy)]
    if mixer_packing:
        cmd.append("--mixer-packing")
    if peft_config:
        cmd += ["--peft-config", json.dumps(peft_config, sort_keys=True)]
    if peft_adapter_path is not None:
        cmd += ["--peft-adapter", str(peft_adapter_path)]
    cmd += ["--matmul-precision", matmul_precision]
    if lm_head_token_chunk_size is not None:
        cmd += [
            "--lm-head-token-chunk",
            str(lm_head_token_chunk_size),
            "--lm-head-vocab-chunk",
            str(lm_head_vocab_chunk_size),
        ]
    if optimizer_config is not None:
        if learning_rate is None or optimizer_output_dir is None:
            raise ValueError("optimizer_config requires learning_rate and optimizer_output_dir")
        cmd += [
            "--optimizer-config",
            json.dumps(optimizer_config, sort_keys=True),
            "--learning-rate",
            str(learning_rate),
            "--optimizer-dtype",
            optimizer_dtype,
            "--optimizer-output-dir",
            str(optimizer_output_dir),
        ]
        if gradient_clipping is not None:
            cmd += ["--gradient-clipping", str(gradient_clipping)]
    # The reference holds a 13B-parameter model plus an fp32 gradient accumulator, so the margin left for
    # activations is thin and the default allocator spends it on fragmentation: an OOM at 28 layers had
    # 18.6 GiB reserved but unallocated. Expandable segments return that.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if deterministic:
        # A comparison has to be repeatable to mean anything, so both arms pin every reduction order that can
        # be pinned. The reference decides for itself whether to ask flash attention for a deterministic
        # backward, because only the process holding the kernel can ask it.
        env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
        cmd.append("--deterministic")
    else:
        # Only the onboarding calibration runs this way, to measure the spread the kernels actually have.
        env["FLASH_ATTENTION_DETERMINISTIC"] = "0"
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if cuda_device is not None:
        # One reference per GPU lets independent runs proceed concurrently; the run itself is single-GPU
        # either way, so pinning changes only which device it lands on.
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    try:
        log_dir = require_data_fast_path(out_path.parent, label="reference logs")
    except ValueError:
        log_dir = correctness_workdir("hf-reference-logs-")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = require_data_fast_path(log_dir / f"{out_path.stem}.stdout.log", label="reference stdout")
    stderr_log = require_data_fast_path(log_dir / f"{out_path.stem}.stderr.log", label="reference stderr")
    returncode_log = require_data_fast_path(log_dir / f"{out_path.stem}.returncode", label="reference return code")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout_log.write_text(proc.stdout)
    stderr_log.write_text(proc.stderr)
    returncode_log.write_text(f"{proc.returncode}\n")
    print(
        f"HF reference process: run={out_path.stem}, returncode={proc.returncode}, "
        f"stdout={stdout_log}, stderr={stderr_log}, returncode_file={returncode_log}",
        flush=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"reference failed ({proc.returncode}); full stdout: {stdout_log}; full stderr: {stderr_log}:\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
        )
    return json.loads(out_path.read_text())


def materialize_dss_peft_adapter(
    cfg: LoadedConfig,
    model_path: str,
    workdir: Path,
    *,
    attn_implementation: str,
) -> Path | None:
    """Export the product-initialized adapter so the independent reference starts from identical bytes."""
    if not cfg.training.get("peft_config"):
        return None
    workdir = require_data_fast_path(workdir, label="PEFT adapter workdir")
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    backend = None
    if cfg.model_name in {"zai-org/GLM-5.3-Flash", "zai-org/GLM-5.3-Flash-BF16"}:
        from .hosted import require_local_cortex_gateway

        if cfg.native is None or not isinstance(cfg.native.backend, CortexConfig):
            raise ValueError("GLM adapter initialization requires its configured local Cortex gateway")
        backend = cfg.native.backend
        require_local_cortex_gateway(backend)
        if cfg.training.get("model_provider") != "prime_rl" or cfg.sp_size != 8 or cfg.training.get("ep_size") != 8:
            raise ValueError("GLM adapter initialization requires the reviewed PrimeRL/SP8/EP8 route")
    workdir.mkdir(parents=True, exist_ok=True)
    gateway_dir = workdir / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = workdir / "adapter"
    payload = build_payload(
        cfg.training,
        model_path,
        SEED,
        attn_implementation=attn_implementation,
        weight_format=cfg.sub_job.get("weight_format"),
    )
    checkpoint_root = workdir / "checkpoint"
    if backend is not None:
        checkpoint_root = require_data_fast_path(checkpoint_root, label="GLM initial checkpoint")
        adapter_dir = require_data_fast_path(adapter_dir, label="GLM initial adapter")
        if adapter_dir.exists():
            raise FileExistsError(f"GLM initial adapter already exists: {adapter_dir}")
        checkpoint_root.mkdir()
        payload["hardware"] = cfg.gpu_type.upper() if cfg.gpu_type is not None else None
    with gateway(gateway_dir, cfg.n_gpus, **({"backend": backend} if backend is not None else {})) as session:
        with running_job(session, payload) as job:
            if backend is not None:
                from .hosted import _local_checkpoint_metadata

                # The native EP checkpointer writes the unmerged adapter under global_step0/default.
                saved = job.client.save_checkpoint(path=str(checkpoint_root), checkpoint_type="weights-only")
                tag = _local_checkpoint_metadata(saved, checkpoint_root, "initial adapter save")
                if tag != "global_step0":
                    raise RuntimeError("GLM initial adapter save must report step zero")
                exported_root = checkpoint_root
            else:
                saved = job.client.save_checkpoint(path=str(checkpoint_root), export_hf=True)
                exported_root = Path(saved.get("hf_path") or saved["path"]).parent
    return copy_checkpoint_peft_adapter(exported_root, adapter_dir)


# The optimizer comparison needs a step large enough to separate a real disagreement from rounding, so
# the shared step runs at this rate rather than the config's. Gradient norms are captured after reduction
# and before clipping and the optimizer, so the rate does not affect them.
OPTIMIZER_LEARNING_RATE = 1e-2


def assert_applicable_tests_registered(spec: TestSpec, config_id: str) -> None:
    """Refuse a spec that lists a check no module registers.

    The selection intersects the registry with ``applicable_tests``, so an id nothing registers drops out
    and the config runs no checks while still reporting success. Without this, a retired id left in a spec
    is indistinguishable from a config that legitimately has nothing to run.
    """
    registered = {entry.test_id for entry in registered_tests().values()}
    unknown = sorted(set(spec.applicable_tests) - registered)
    if unknown:
        raise ValueError(
            f"{config_id}: applicable_tests names {len(unknown)} check(s) that nothing registers "
            f"({', '.join(unknown)}); registered checks are {', '.join(sorted(registered))}. "
            "An unregistered id selects nothing, so this config would run no checks at all."
        )


def _at_width(results: List[TestResult], gpus: int) -> List[TestResult]:
    """Name the placement width on every result, so results from two widths stay distinguishable."""
    for result in results:
        result.gpus = gpus
    return results


def cortex_unavailable_reason(test, published_fixture: dict | None = None) -> str | None:
    """Reject checks whose artifacts or reviewed target identity are unavailable."""
    if test.test_id in {
        "single-step-grads",
        "single-step-optimizer",
    }:
        if published_fixture is not None:
            return None
        return (
            "Cortex comparison requires the reviewed reduced HF weights to be published; "
            "publish through a supported model-distribution route and pass the binding with --cortex-fixture"
        )
    if test.compares_to_reference:
        return "Cortex cannot load the reviewed node-local reduced reference checkpoint"
    return None


def _cortex_rejection(
    test,
    cfg,
    published_fixture: dict | None = None,
) -> TestResult | None:
    reason = cortex_unavailable_reason(test, published_fixture)
    if reason is None and test.compares_to_reference and cfg.training.get("peft_config"):
        reason = "Cortex reference comparisons cannot export the product-initialized LoRA adapter"
    if reason is None:
        return None
    missing_fixture = test.test_id in {"single-step-grads", "single-step-optimizer"} and published_fixture is None
    return TestResult(
        test_id=test.test_id,
        config_id=cfg.config_id,
        outcome=TestOutcome.FAIL,
        summary="published Cortex fixture required" if missing_fixture else "check not supported on Cortex",
        reason=reason,
    )


def execute(
    cfg: LoadedConfig,
    spec: TestSpec,
    arm_defs: List[ArmDefinition],
    *,
    vocab_size: int,
    attn: Optional[str] = None,
    select: Optional[List[str]] = None,
    reference_token_budget: Optional[int] = None,
    ce_chunk: int = 2048,
    reference_attn: Optional[str] = None,
    workdir: Optional[Path] = None,
    order: str = "batched",
    fail_fast: bool = False,
    client_backend: OnPremConfig | CortexConfig | None = None,
    cortex_fixture: Path | None = None,
    local_cortex_fixture: bool = False,
    local_gateway_lora_save0: bool = False,
    prepare_references: bool = False,
    peft_adapter: Path | None = None,
    reference_records_dir: Path | None = None,
    reference_calibration_runs: int = 16,
    reference_calibration_workers: int = 1,
    historical_gradient_references: dict | None = None,
) -> List[TestResult]:
    """Use the config's upstream reference attention unless explicitly overridden.

    GLM's upstream top-k mask uses SDPA while the AP target keeps sparse MLA. Other architectures retain
    their target attention default. Onboarding calibrates the selected reference backend; the HF side keeps
    samples in separate rows while Arctic Platform exercises packed/SP execution.
    """
    if workdir is not None:
        workdir = require_data_fast_path(workdir, label="correctness workdir")
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    if historical_gradient_references is not None:
        if (
            select != ["single-step-grads"]
            or prepare_references
            or cfg.is_rl
            or not isinstance(historical_gradient_references, dict)
            or not historical_gradient_references
            or set(historical_gradient_references) != {arm.name for arm in arm_defs}
        ):
            raise ValueError(
                "historical gradient references require only single-step-grads and exactly the selected SFT arms"
            )
        historical_gradient_references = deepcopy(historical_gradient_references)
    if (
        prepare_references
        and select is not None
        and (not select or not set(select) <= {"single-step-grads", "single-step-optimizer"})
    ):
        raise ValueError("reference preparation supports only explicit HF reference check selection")
    client_backend = client_backend if client_backend is not None else OnPremConfig()
    if local_gateway_lora_save0:
        raise ValueError("deterministic LoRA SAVE0 execution is retired; use --peft-adapter on-prem")
    shared_adapter_identity = None
    if peft_adapter is not None:
        from .shared_lora_adapter import adapter_identity

        if (
            cfg.is_rl
            or not cfg.training.get("peft_config")
            or select is not None
            and (not select or not set(select) <= {"single-step-grads", "single-step-optimizer"})
            or (not prepare_references and not isinstance(client_backend, OnPremConfig))
        ):
            raise ValueError(
                "supplied PEFT adapter requires on-prem first-two SFT checks or their reference preparation"
            )
        peft_adapter = require_data_fast_path(peft_adapter, label="shared PEFT adapter").resolve(strict=True)
        shared_adapter_identity = adapter_identity(peft_adapter, cfg.training["peft_config"])
    tests = registered_tests()
    assert_applicable_tests_registered(spec, cfg.config_id)
    chosen = [t for t in tests.values() if select is None or t.test_id in select]
    results: List[TestResult] = []
    published_fixture = None
    if local_cortex_fixture and not prepare_references:
        from .cortex import load_local_fixture

        if not isinstance(client_backend, CortexConfig) or cortex_fixture is not None:
            raise ValueError("local Cortex fixture requires CortexConfig without a publication receipt")
        if select and not set(select) <= {"single-step-grads", "single-step-optimizer"}:
            raise ValueError("local Cortex fixture supports only single-step-grads and single-step-optimizer")
        try:
            published_fixture = load_local_fixture(client_backend, spec)
        except ValueError as exc:
            return _at_width(
                [
                    TestResult(
                        test_id=test_id,
                        config_id=cfg.config_id,
                        outcome=TestOutcome.FAIL,
                        summary="local Cortex fixture identity unavailable",
                        reason=str(exc),
                    )
                    for test_id in select or tests
                ],
                cfg.n_gpus,
            )
    if cortex_fixture is not None:
        from .cortex import load_fixture

        if not isinstance(client_backend, CortexConfig):
            raise ValueError("--cortex-fixture requires --backend cortex")
        published_fixture = load_fixture(cortex_fixture, spec)
    if isinstance(client_backend, CortexConfig) and not prepare_references:
        rejected = []
        for test in chosen:
            if test.test_id not in spec.applicable_tests:
                continue
            rejection = _cortex_rejection(
                test,
                cfg,
                published_fixture,
            )
            if rejection is not None:
                results.append(rejection)
                rejected.append(test)
        chosen = [test for test in chosen if test not in rejected]

    for test in tests.values():
        if select is not None and test.test_id not in select:
            continue
        if test.test_id not in spec.applicable_tests:
            results.append(
                TestResult(
                    test_id=test.test_id,
                    config_id=cfg.config_id,
                    outcome=TestOutcome.INAPPLICABLE,
                    summary="not applicable to this config",
                    reason=spec.inapplicable.get(test.test_id, "not listed as applicable in the spec"),
                )
            )

    chosen = [t for t in chosen if t.test_id in spec.applicable_tests]
    if not chosen:
        return _at_width(results, cfg.n_gpus)

    if reference_token_budget is None:
        reference_token_budget = correctness_microbatch_tokens(cfg.max_tokens_per_mb)

    # Both checks read one execution. The optimizer comparison is the constrained side -- it holds an fp32
    # master and two Adam states per parameter -- so where the two disagree on a model slice or a token
    # budget, its choice is the one that fits both.
    stepping = next((t for t in chosen if t.test_id == "single-step-optimizer"), None)
    optimizer_settings = (spec.test_settings.get("single-step-optimizer") or {}) if stepping else {}
    optimizer_config = cfg.training.get("optimizer") if stepping else None
    if stepping:
        if not isinstance(optimizer_config, dict):
            raise ValueError("single-step-optimizer requires an explicit optimizer config")
        try:
            reference_token_budget = min(reference_token_budget, int(optimizer_settings["reference_token_budget"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("single-step-optimizer requires an onboarded reference_token_budget") from exc
    reference_model_path = str(optimizer_settings.get("model_cache_path", spec.model.cache_path))
    if local_cortex_fixture and stepping:
        if require_data_fast_path(reference_model_path, label="optimizer reference model") != require_data_fast_path(
            spec.model.cache_path, label="local Cortex reference model"
        ):
            raise ValueError("local optimizer fixture must match the frozen optimizer reference model")
    target_model = published_fixture["model_name"] if published_fixture is not None else reference_model_path
    tmp = workdir if workdir is not None else correctness_workdir("dss-correctness-")
    tmp.mkdir(parents=True, exist_ok=True)
    attn_impl = attn or cfg.attention_implementation

    batch_paths: Dict[str, Path] = {}
    arm_specs: List[ArmSpec] = []
    for definition in arm_defs:
        batch = build_batch(
            definition.name, definition.global_batch_size, definition.max_seq_len, vocab_size, seed=SEED
        )
        path = tmp / f"batch-{definition.name}.pt"
        save(batch, path)
        batch_paths[definition.name] = path
        arm_specs.append(
            ArmSpec(
                name=definition.name,
                global_batch_size=definition.global_batch_size,
                max_seq_len=definition.max_seq_len,
                total_tokens=batch.total_tokens,
                active_tokens=batch.active_tokens,
                dss_microbatches=definition.microbatches(
                    correctness_microbatch_tokens(cfg.max_tokens_per_mb), cfg.dp_size
                ),
                pad_fraction=batch.pad_fraction,
            )
        )

    context = RunContext(
        config_id=cfg.config_id,
        cfg=cfg,
        spec=spec,
        arms=arm_specs,
        attn_implementation=attn_impl,
        reference_attn_implementation=reference_attn,
        workdir=tmp,
        batch_paths=batch_paths,
        vocab_size=vocab_size,
        client_backend=client_backend,
        target_model_name=published_fixture["model_name"] if published_fixture is not None else None,
        prepare_references=prepare_references,
        **({"reference_records_dir": reference_records_dir} if reference_records_dir is not None else {}),
    )

    run_started = time.monotonic()
    per_arm_tests = [t for t in chosen if t.per_arm is not None]
    collect_gradient_norms = any(t.test_id == "single-step-grads" for t in chosen)

    def payload_for(arm: ArmSpec) -> dict:
        """One job per case, asking for the optimizer artifacts only when the step check reads them."""
        optimizer_state_output_dir = None
        if stepping and not isinstance(client_backend, CortexConfig):
            optimizer_state_output_dir = require_data_fast_path(
                tmp / "optimizer" / "dss" / arm.name, label="optimizer artifact output"
            )
            optimizer_state_output_dir.mkdir(parents=True, exist_ok=True)
        payload = build_payload(
            cfg.training,
            target_model,
            SEED,
            attn_implementation=attn_impl,
            optimizer_state_output_dir=optimizer_state_output_dir,
            gradient_norms_per_param=collect_gradient_norms,
            weight_format=cfg.sub_job.get("weight_format"),
        )
        if isinstance(client_backend, CortexConfig):
            payload["training_config"]["debug"].pop("optimizer_state_output_dir", None)
            if cfg.gpu_type is not None:
                payload["hardware"] = cfg.gpu_type.upper()
        if context.reference_adapter_path is not None:
            payload["training_config"]["debug"]["shared_peft_adapter"] = {
                "path": str(context.reference_adapter_path),
                "identity": deepcopy(shared_adapter_identity),
            }
        return payload

    def measure_target(session, job, arm: ArmSpec):
        body = body_for(arm)
        if stepping and isinstance(client_backend, CortexConfig):
            from .cortex_optimizer import capture_optimizer_step

            return capture_optimizer_step(
                session,
                job,
                body,
                learning_rate=OPTIMIZER_LEARNING_RATE,
                output_dir=tmp / "optimizer" / "dss" / arm.name,
                expected_dp_world_size=cfg.dp_size,
                optimizer_config=optimizer_config,
                **({"local_checkpoint": True} if local_cortex_fixture else {}),
            )
        return fwd_bwd_step(
            session,
            job,
            body,
            learning_rate=OPTIMIZER_LEARNING_RATE if stepping else 0.0,
        )

    def body_for(arm: ArmSpec) -> dict:
        batch = load_batch(batch_paths[arm.name])
        provider = str(cfg.training.get("model_provider", "huggingface"))
        return pack(batch, model_provider=provider)

    reference_training = cfg.effective_training
    if per_arm_tests and cfg.training.get("peft_config"):
        if peft_adapter is None:
            from .shared_lora_adapter import adapter_identity
            from .shared_lora_adapter import generate_adapter

            if not prepare_references and not isinstance(client_backend, OnPremConfig):
                raise ValueError("shared PEFT adapter initialization requires the on-prem correctness worker")
            peft_adapter = generate_adapter(
                reference_model_path, cfg.training["peft_config"], tmp / "initial-adapter", seed=SEED
            )
            shared_adapter_identity = adapter_identity(peft_adapter, cfg.training["peft_config"])
        reference_adapter_path = peft_adapter
    else:
        reference_adapter_path = None
    context.reference_adapter_path = reference_adapter_path
    if reference_adapter_path is not None and not prepare_references and not isinstance(client_backend, OnPremConfig):
        raise ValueError("supplied PEFT adapters require the on-prem correctness worker")

    from .batches import load as load_batch

    def measure_reference(
        arm: ArmSpec,
        *,
        output_name: str | None = None,
        cuda_device: int | None = None,
        deterministic: bool = True,
    ) -> dict:
        started = time.monotonic()
        output_name = output_name or arm.name
        device = cuda_device if cuda_device is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "default")
        print(
            f"HF reference start: arm={arm.name}, run={output_name}, device={device}, deterministic={deterministic}",
            flush=True,
        )
        optimizer_output_dir = None
        if stepping:
            optimizer_output_dir = require_data_fast_path(
                tmp / "optimizer" / "reference" / output_name / "states", label="reference optimizer output"
            )
            optimizer_output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            if reference_adapter_path is not None:
                from .shared_lora_adapter import adapter_identity

                if adapter_identity(reference_adapter_path) != shared_adapter_identity:
                    raise ValueError("shared adapter changed before HF reference execution")
            ref = run_reference(
                reference_model_path,
                batch_paths[arm.name],
                tmp / f"ref-{output_name}.json",
                token_budget=reference_token_budget,
                ce_chunk=ce_chunk,
                attn=reference_attn or cfg.reference_attention_implementation,
                fp32_lm_head=bool(reference_training.get("fp32_lm_head", False)),
                seed=SEED,
                fused_cross_entropy=cfg.fused_cross_entropy,
                mixer_packing=uses_mixer_packing(reference_model_path),
                matmul_precision=reference_training.get("matmul_precision", "highest"),
                peft_config=reference_training.get("peft_config"),
                peft_adapter_path=reference_adapter_path,
                lm_head_token_chunk_size=cfg.lm_head_token_chunk_size,
                lm_head_vocab_chunk_size=reference_training.get("fused_lm_head_vocab_chunk_size", 8192),
                optimizer_config=optimizer_config,
                learning_rate=OPTIMIZER_LEARNING_RATE if stepping else None,
                gradient_clipping=cfg.training.get("gradient_clipping"),
                optimizer_dtype=cfg.optimizer_dtype,
                optimizer_output_dir=optimizer_output_dir,
                cuda_device=cuda_device,
                deterministic=deterministic,
            )
            if (
                reference_adapter_path is not None
                and adapter_identity(reference_adapter_path) != shared_adapter_identity
            ):
                raise ValueError("shared adapter changed during HF reference execution")
        except Exception:
            print(
                f"HF reference failed: arm={arm.name}, run={output_name}, device={device}, "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
            raise
        elapsed = time.monotonic() - started
        print(
            f"HF reference done: arm={arm.name}, run={output_name}, device={device}, elapsed={elapsed:.1f}s",
            flush=True,
        )
        context._reference[arm.name] = ref
        context._timing.setdefault(arm.name, {})["reference"] = elapsed
        return ref

    from .reference_records import ReferencePrerequisiteError
    from .reference_records import batch_identity
    from .reference_records import load_record
    from .reference_records import model_identity
    from .reference_records import publish_record
    from .reference_records import workload_identity

    def reference_identity(test_id: str, arm: ArmSpec) -> dict:
        batch = load_batch(batch_paths[arm.name])
        initialization = "external-model"
        if reference_adapter_path is not None:
            initialization = deepcopy(shared_adapter_identity)
        recipe = {
            "model": model_identity(reference_model_path),
            "initialization": initialization,
            "synthetic_inputs": {
                "version": "private-cpu-generator-v1",
                "seed": SEED,
                "rows": arm.global_batch_size,
                "max_seq_len": arm.max_seq_len,
                "vocab_size": vocab_size,
                "batch": batch_identity(batch),
            },
            "attention": reference_attn or cfg.reference_attention_implementation,
            "token_budget": reference_token_budget,
            "ce_chunk": ce_chunk,
            "training": {
                "fp32_lm_head": bool(reference_training.get("fp32_lm_head", False)),
                "fused_cross_entropy": cfg.fused_cross_entropy,
                "mixer_packing": uses_mixer_packing(reference_model_path),
                "matmul_precision": reference_training.get("matmul_precision", "highest"),
                "peft_config": reference_training.get("peft_config"),
                "lm_head_token_chunk_size": cfg.lm_head_token_chunk_size,
                "lm_head_vocab_chunk_size": reference_training.get("fused_lm_head_vocab_chunk_size", 8192),
                "deterministic": True,
            },
            "optimizer": (
                {
                    "config": optimizer_config,
                    "learning_rate": OPTIMIZER_LEARNING_RATE,
                    "gradient_clipping": cfg.training.get("gradient_clipping"),
                    "dtype": cfg.optimizer_dtype,
                }
                if stepping
                else None
            ),
        }
        return workload_identity(context, test_id, recipe)

    def prepared_reference(arm: ArmSpec) -> None:
        identities = {test.test_id: reference_identity(test.test_id, arm) for test in per_arm_tests}
        records = {}
        for test_id, identity in identities.items():
            try:
                if historical_gradient_references is not None:
                    records[test_id] = load_record(
                        context, identity, historical_binding=historical_gradient_references[arm.name]
                    )
                    (tmp / "reference-record-provenance.json").write_text(
                        json.dumps(context.reference_record_provenance, indent=2, sort_keys=True, allow_nan=False)
                        + "\n"
                    )
                else:
                    records[test_id] = load_record(context, identity)
            except ReferencePrerequisiteError as exc:
                if not prepare_references or "prepared HF reference missing:" not in str(exc):
                    raise
        if len(records) != len(identities):
            if not prepare_references:
                raise ReferencePrerequisiteError("HF reference preparation required")
            reference = measure_reference(arm)
            compact = {key: value for key, value in reference.items() if key != "optimizer_state_manifest"}
            if stepping:
                from decimal import ROUND_CEILING
                from decimal import Decimal

                from ..checks.optimizer_state import max_pairwise_optimizer_norm_delta
                from ..checks.optimizer_state import summarize_optimizer_artifacts

                if reference_calibration_runs < 2 or reference_calibration_workers < 1:
                    raise ValueError("norm onboarding requires at least two calibration runs and one worker")
                compact["optimizer_norms"] = summarize_optimizer_artifacts(
                    Path(reference["optimizer_state_manifest"]),
                    optimizer_config=optimizer_config,
                    learning_rate=OPTIMIZER_LEARNING_RATE,
                )
                print(
                    f"Onboarding optimizer norm gate: {reference_calibration_runs} HF runs, "
                    f"{reference_calibration_workers} worker(s), arm={arm.name}",
                    flush=True,
                )
                manifests = []
                for start in range(0, reference_calibration_runs, reference_calibration_workers):
                    indices = range(start, min(start + reference_calibration_workers, reference_calibration_runs))
                    with ThreadPoolExecutor(max_workers=reference_calibration_workers) as pool:
                        futures = {
                            pool.submit(
                                measure_reference,
                                arm,
                                output_name=f"{arm.name}-calibration-{index}",
                                cuda_device=index % reference_calibration_workers,
                                deterministic=False,
                            ): index
                            for index in indices
                        }
                        completed = {}
                        for future in as_completed(futures):
                            completed[futures[future]] = Path(future.result()["optimizer_state_manifest"])
                        manifests.extend(completed[index] for index in indices)
                worst = max_pairwise_optimizer_norm_delta(
                    manifests, optimizer_config=optimizer_config, learning_rate=OPTIMIZER_LEARNING_RATE
                )
                spread = worst.delta_norm if worst is not None else 0.0
                quantum = Decimal("0.001")
                absolute = float(
                    max(
                        quantum,
                        (Decimal(str(4 * spread)) / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum,
                    )
                )
                compact["optimizer_norm_gate"] = {
                    "criterion": "optimizer-state-l2-norm-v1",
                    "absolute": absolute,
                    "status": "reviewed",
                    "provenance": {
                        "policy": "onboarding-norm-repeatability-4x-upward-1e-3-v1",
                        "calibration_runs": reference_calibration_runs,
                        "maximum_pairwise_norm_delta": spread,
                        "worst_quantity": f"{worst.name}::{worst.state}" if worst is not None else None,
                        "multiplier": 4,
                        "minimum_gate": 0.001,
                    },
                }
            for test_id, identity in identities.items():
                if test_id not in records:
                    publish_record(context, identity, compact)
                    records[test_id] = compact
        primary = "single-step-optimizer" if stepping else "single-step-grads"
        context._reference_by_check[arm.name] = records
        context._reference[arm.name] = records[primary]
        context._timing.setdefault(arm.name, {})["reference"] = 0.0

    # Validate every prerequisite before allocating any target job, including arm-at-a-time runs.
    for arm in arm_specs if per_arm_tests else []:
        prepared_reference(arm)
    if prepare_references:
        print(f"HF reference preparation complete: {len(context.reference_record_paths)} record(s)", flush=True)
        return []

    def judge(arm: ArmSpec) -> bool:
        """Print each per-arm verdict as it is reached. True when any verdict for this arm is not a pass."""
        bad = False
        for test in per_arm_tests:
            try:
                context._reference[arm.name] = context._reference_by_check[arm.name][test.test_id]
                result = test.per_arm(context, arm)
            except UnsafeCleanupError:
                raise
            except Exception as exc:  # noqa: BLE001 - one failing test must not hide the rest
                traceback.print_exc()
                result = TestResult(
                    test_id=test.test_id,
                    config_id=cfg.config_id,
                    arm=arm.name,
                    outcome=TestOutcome.FAIL,
                    summary=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
            results.append(result)
            print(console.arm_verdict(result, arm, context._timing.get(arm.name)), flush=True)
            bad = bad or result.outcome is not TestOutcome.PASS
        return bad

    execution_errors = set()

    def measure_target_arm(session, arm: ArmSpec) -> bool:
        try:
            with console.activity(f"Running Arctic Platform {arm.name}"):
                with running_job(session, payload_for(arm)) as job_id:
                    started = time.monotonic()
                    context._target[arm.name] = measure_target(session, job_id, arm)
                    context._timing.setdefault(arm.name, {})["dss"] = time.monotonic() - started
        except UnsafeCleanupError:
            raise
        except Exception as exc:  # noqa: BLE001 - startup failures must also reach the report
            traceback.print_exc()
            execution_errors.add(arm.name)
            for test in per_arm_tests:
                result = TestResult(
                    test_id=test.test_id,
                    config_id=cfg.config_id,
                    arm=arm.name,
                    outcome=TestOutcome.FAIL,
                    summary=f"execution error: {type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
                results.append(result)
                print(console.arm_verdict(result, arm, context._timing.get(arm.name)), flush=True)
            return False
        return True

    measured: List[ArmSpec] = []
    if per_arm_tests:
        if order == "batched":
            # One job per case, sharing the gateway. Reading the gradient norms costs an optimizer update,
            # because ``/step`` is the only endpoint that returns them, so a job that has served one case is
            # no longer holding the checkpoint's weights. Sharing it would make each case depend on the ones
            # before it and on the order they ran in.
            with ExitStack() as stack:
                with console.activity("Starting Arctic Platform gateway"):
                    url = stack.enter_context(gateway(tmp, cfg.n_gpus, backend=client_backend))
                for arm in arm_specs:
                    if not measure_target_arm(url, arm):
                        if fail_fast:
                            break
                        continue
                    measured.append(arm)
                    if judge(arm) and fail_fast:
                        break
        else:
            for arm in arm_specs:
                with ExitStack() as stack:
                    with console.activity("Starting Arctic Platform gateway"):
                        url = stack.enter_context(gateway(tmp, cfg.n_gpus, backend=client_backend))
                    if not measure_target_arm(url, arm):
                        if fail_fast:
                            break
                        continue
                measured.append(arm)
                if judge(arm) and fail_fast:
                    break

    skipped = (
        [a.name for a in arm_specs if a not in measured and a.name not in execution_errors] if per_arm_tests else []
    )
    if skipped or execution_errors:
        if skipped:
            print(console.stopped_early(skipped), flush=True)
        print(console.runtime_table(context._timing, time.monotonic() - run_started), flush=True)
        return _at_width(results, cfg.n_gpus)

    print(console.runtime_table(context._timing, time.monotonic() - run_started), flush=True)
    return _at_width(results, cfg.n_gpus)
