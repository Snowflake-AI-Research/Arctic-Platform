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

Ordering is forced by the hardware. The reference needs one GPU with nothing else on the node, and the
gateway claims every GPU it is given, so all reference arms run first, in a subprocess so their CUDA
context is reclaimed, and the gateway comes up afterwards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from arctic_platform.model.implementations.debug.determinism import CUBLAS_WORKSPACE

from ..reference.model_features import uses_mixer_packing
from . import console
from .arms import ArmDefinition
from .arms import correctness_microbatch_tokens
from .batches import build_batch
from .batches import save
from .config import LoadedConfig
from .dss_driver import build_payload
from .dss_driver import copy_checkpoint_peft_adapter
from .dss_driver import fwd_bwd_step
from .dss_driver import gateway
from .dss_driver import pack_microbatches
from .dss_driver import running_job
from .dss_driver import save_weights_only_checkpoint
from .optimizer_capture import optimizer_capture_worker
from .registry import TestOutcome
from .registry import TestResult
from .registry import registered_tests
from .rl_driver import rl_gpu_count
from .seeds import SEED
from .spec import ArmSpec
from .spec import TestSpec
from .workdir import correctness_workdir


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
    # Where a check's training jobs run. ``None`` starts a Ray gateway on this node and claims its GPUs;
    # the hosted command supplies a factory that creates jobs on the hosted control plane instead.
    transport_factory: Optional[Callable[[Path], object]] = None
    _reference: Dict[str, dict] = field(default_factory=dict)
    _target: Dict[str, object] = field(default_factory=dict)
    # Seconds spent in each engine per case, so the report can say which case dominates the run.
    _timing: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def transport_for(self, workdir: Path):
        """The transport a check creates its jobs through, entered by the caller and exited when it ends."""
        if self.transport_factory is not None:
            return self.transport_factory(workdir)
        from .dss_driver import GatewayTransport

        return GatewayTransport(
            workdir,
            self.cfg.training,
            self.spec.model.cache_path,
            SEED,
            slots=self.cfg.n_gpus,
            attn_implementation=self.attn_implementation,
        )

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
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"reference failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
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
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    workdir.mkdir(parents=True, exist_ok=True)
    gateway_dir = workdir / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = workdir / "adapter"
    payload = build_payload(
        cfg.training,
        model_path,
        SEED,
        attn_implementation=attn_implementation,
    )
    checkpoint_root = workdir / "checkpoint"
    with gateway(gateway_dir, cfg.n_gpus) as session:
        with running_job(session, payload) as job:
            exported = Path(save_weights_only_checkpoint(session, job, checkpoint_root))
    return copy_checkpoint_peft_adapter(exported.parent, adapter_dir)


# The optimizer comparison needs a step large enough to separate a real disagreement from rounding, so
# the shared step runs at this rate rather than the config's. Gradient norms are captured after reduction
# and before clipping and the optimizer, so the rate does not affect them.
OPTIMIZER_LEARNING_RATE = 1e-2

_QWEN3_8B_FULL_LONG_CONTEXT_CONFIG = "qwen3-8b-h200-train-sft-full-4gpus-64k"


def _ap_batch_processing_options(cfg: LoadedConfig, arm: ArmSpec) -> dict:
    """Select AP-only SFT loss packaging for cases that cannot materialize full vocab logits."""
    if cfg.config_id == _QWEN3_8B_FULL_LONG_CONTEXT_CONFIG and arm.name == "gas4":
        return {
            "loss_fn": "sft_ce",
            "processing_config": {
                "logits_optimization": "memory",
                "logits_optimization_peak_mem_size_in_gib": 4,
            },
        }
    return {}


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
) -> List[TestResult]:
    """Use the config's GPU-family FlashAttention kernel for the reference unless explicitly overridden.

    The onboarding calibration turns that kernel's nondeterministic backward into the frozen gate.
    Matching kernels avoids adding an SDPA-versus-FlashAttention difference and keeps the 64K reference
    memory-feasible. The HF side keeps samples in separate rows, while Arctic Platform exercises packed/SP execution.
    """
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    tests = registered_tests()
    assert_applicable_tests_registered(spec, cfg.config_id)
    chosen = [t for t in tests.values() if select is None or t.test_id in select]
    results: List[TestResult] = []

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
    tmp = Path(workdir) if workdir else correctness_workdir("dss-correctness-")
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
        workdir=tmp,
        batch_paths=batch_paths,
        vocab_size=vocab_size,
    )

    run_started = time.monotonic()
    per_arm_tests = [t for t in chosen if t.per_arm is not None]
    whole_run_tests = [t for t in chosen if t.per_arm is None]
    collect_gradient_norms = any(t.test_id == "single-step-grads" for t in chosen)

    def payload_for(arm: ArmSpec) -> dict:
        """One job per case, asking for the optimizer artifacts only when the step check reads them."""
        optimizer_state_output_dir = None
        if stepping:
            optimizer_state_output_dir = tmp / "optimizer" / "dss" / arm.name
            optimizer_state_output_dir.mkdir(parents=True, exist_ok=True)
        return build_payload(
            cfg.training,
            reference_model_path,
            SEED,
            attn_implementation=attn_impl,
            optimizer_state_output_dir=optimizer_state_output_dir,
            gradient_norms_per_param=collect_gradient_norms,
            gradient_accumulation_steps=arm.dss_microbatches,
        )

    reference_training = cfg.effective_training
    if cfg.training.get("peft_config"):
        with console.activity("Preparing LoRA adapter"):
            reference_adapter_path = materialize_dss_peft_adapter(
                cfg,
                reference_model_path,
                tmp / "reference-adapter-init",
                attn_implementation=attn_impl,
            )
    else:
        reference_adapter_path = None
    context.reference_adapter_path = reference_adapter_path

    from .batches import load as load_batch

    def measure_reference(arm: ArmSpec) -> None:
        started = time.monotonic()
        optimizer_output_dir = None
        if stepping:
            optimizer_output_dir = tmp / "optimizer" / "reference" / arm.name / "states"
            optimizer_output_dir.parent.mkdir(parents=True, exist_ok=True)
        ref = run_reference(
            reference_model_path,
            batch_paths[arm.name],
            tmp / f"ref-{arm.name}.json",
            token_budget=reference_token_budget,
            ce_chunk=ce_chunk,
            attn=reference_attn or cfg.attention_implementation,
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
        )
        elapsed = time.monotonic() - started
        context._reference[arm.name] = ref
        context._timing.setdefault(arm.name, {})["reference"] = elapsed

    def judge(arm: ArmSpec) -> bool:
        """Print each per-arm verdict as it is reached. True when any verdict for this arm is not a pass."""
        bad = False
        for test in per_arm_tests:
            try:
                result = test.per_arm(context, arm)
            except Exception as exc:  # noqa: BLE001 - one failing test must not hide the rest
                result = TestResult(
                    test_id=test.test_id,
                    config_id=cfg.config_id,
                    arm=arm.name,
                    outcome=TestOutcome.FAIL,
                    summary=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            print(console.arm_verdict(result, arm, context._timing.get(arm.name)), flush=True)
            bad = bad or result.outcome is not TestOutcome.PASS
        return bad

    measured: List[ArmSpec] = []
    if per_arm_tests:
        if order == "batched":
            for arm in arm_specs:
                with console.activity(f"Running reference {arm.name}"):
                    measure_reference(arm)
            # One job per case, sharing the gateway. Reading the gradient norms costs an optimizer update,
            # because ``/step`` is the only endpoint that returns them, so a job that has served one case is
            # no longer holding the checkpoint's weights. Sharing it would make each case depend on the ones
            # before it and on the order they ran in.
            with ExitStack() as stack:
                if stepping:
                    stack.enter_context(optimizer_capture_worker())
                with console.activity("Starting Arctic Platform gateway"):
                    url = stack.enter_context(gateway(tmp, cfg.n_gpus))
                for arm in arm_specs:
                    with console.activity(f"Running Arctic Platform {arm.name}"):
                        with running_job(url, payload_for(arm)) as job_id:
                            started = time.monotonic()
                            context._target[arm.name] = fwd_bwd_step(
                                url,
                                job_id,
                                pack_microbatches(
                                    load_batch(batch_paths[arm.name]),
                                    arm.dss_microbatches,
                                    model_provider=str(cfg.training.get("model_provider", "huggingface")),
                                    **_ap_batch_processing_options(cfg, arm),
                                ),
                                learning_rate=OPTIMIZER_LEARNING_RATE if stepping else 0.0,
                            )
                            context._timing.setdefault(arm.name, {})["dss"] = time.monotonic() - started
                    measured.append(arm)
                    if judge(arm) and fail_fast:
                        break
        else:
            for arm in arm_specs:
                with console.activity(f"Running reference {arm.name}"):
                    measure_reference(arm)
                with ExitStack() as stack:
                    if stepping:
                        stack.enter_context(optimizer_capture_worker())
                    with console.activity("Starting Arctic Platform gateway"):
                        url = stack.enter_context(gateway(tmp, cfg.n_gpus))
                    with console.activity(f"Running Arctic Platform {arm.name}"):
                        with running_job(url, payload_for(arm)) as job_id:
                            started = time.monotonic()
                            context._target[arm.name] = fwd_bwd_step(
                                url,
                                job_id,
                                pack_microbatches(
                                    load_batch(batch_paths[arm.name]),
                                    arm.dss_microbatches,
                                    model_provider=str(cfg.training.get("model_provider", "huggingface")),
                                    **_ap_batch_processing_options(cfg, arm),
                                ),
                                learning_rate=OPTIMIZER_LEARNING_RATE if stepping else 0.0,
                            )
                            context._timing.setdefault(arm.name, {})["dss"] = time.monotonic() - started
                measured.append(arm)
                if judge(arm) and fail_fast:
                    break

    skipped = [a.name for a in arm_specs if a not in measured] if per_arm_tests else []
    if skipped:
        print(console.stopped_early(skipped), flush=True)
        print(console.runtime_table(context._timing, time.monotonic() - run_started), flush=True)
        return results

    by_name = {arm.name: arm for arm in arm_specs}
    for test in whole_run_tests:
        try:
            produced = list(test.fn(context))
        except Exception as exc:  # noqa: BLE001 - one failing test must not hide the rest
            produced = [
                TestResult(
                    test_id=test.test_id,
                    config_id=cfg.config_id,
                    outcome=TestOutcome.FAIL,
                    summary=f"{type(exc).__name__}: {exc}",
                )
            ]
        # Printed here as well as in judge(): a result that is computed but never shown reads as a test
        # that silently did not run.
        for result in produced:
            print(
                console.arm_verdict(result, by_name.get(result.arm), context._timing.get(result.arm or "")), flush=True
            )
        results.extend(produced)
    print(console.runtime_table(context._timing, time.monotonic() - run_started), flush=True)
    return _at_width(results, cfg.n_gpus)


def execute_rl(
    cfg: LoadedConfig,
    spec: TestSpec,
    *,
    vocab_size: int,
    attn: Optional[str] = None,
    select: Optional[List[str]] = None,
    workdir: Optional[Path] = None,
) -> List[TestResult]:
    """Run the checks of a reinforcement-learning config that compare its two zones with each other.

    Tests against the single-GPU reference are the caller's to report as inapplicable. Every other applicable
    test runs once, and both sub-jobs run at the widths the config declares. The RL checks build their own
    requests from the training and sampling sub-jobs; a check that trains on generated batches, such as
    ``checkpoint-resume-loss``, runs one case per entry in the spec's ``arms``, as the hosted command does.
    """
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    assert_applicable_tests_registered(spec, cfg.config_id)
    tmp = Path(workdir) if workdir else correctness_workdir("dss-correctness-rl-")
    tmp.mkdir(parents=True, exist_ok=True)
    context = RunContext(
        config_id=cfg.config_id,
        cfg=cfg,
        spec=spec,
        arms=list(spec.arms),
        attn_implementation=attn or cfg.attention_implementation,
        workdir=tmp,
        batch_paths={},
        vocab_size=vocab_size,
    )
    results: List[TestResult] = []
    for test in registered_tests().values():
        if test.compares_to_reference or (select is not None and test.test_id not in select):
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
            continue
        with console.activity(f"Running {test.test_id}"):
            try:
                produced = list(test.fn(context))
            except Exception as exc:  # noqa: BLE001 - one failing check must not hide the rest
                produced = [
                    TestResult(
                        test_id=test.test_id,
                        config_id=cfg.config_id,
                        outcome=TestOutcome.FAIL,
                        summary=f"{type(exc).__name__}: {exc}",
                    )
                ]
        for result in produced:
            print(console.arm_verdict(result, None), flush=True)
        results.extend(produced)
    return _at_width(results, rl_gpu_count(cfg))
