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

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
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
from .dss_driver import pack
from .dss_driver import running_job
from .registry import TestOutcome
from .registry import TestResult
from .registry import registered_tests
from .rl_driver import rl_gpu_count
from .seeds import SEED
from .spec import ArmSpec
from .spec import TestSpec
from .workdir import correctness_workdir
from .workdir import require_data_fast_path


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
    # Optional test/onboarding override; normal regression chooses the transport from client_backend.
    transport_factory: Optional[Callable[[Path], object]] = None
    client_backend: OnPremConfig | CortexConfig = field(default_factory=OnPremConfig)
    # Populated only from a fixture binding validated against the reviewed local weights.
    target_model_name: str | None = None
    local_gateway_full_model: bool = False
    local_full_model_path: str | None = None
    _reference: Dict[str, dict] = field(default_factory=dict)
    _target: Dict[str, object] = field(default_factory=dict)
    # Seconds spent in each engine per case, so the report can say which case dominates the run.
    _timing: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def transport_for(self, workdir: Path, *, arm: ArmSpec | None = None):
        """The transport a check creates its jobs through, entered by the caller and exited when it ends."""
        if self.transport_factory is not None:
            return self.transport_factory(workdir)
        if isinstance(self.client_backend, CortexConfig):
            from .hosted import HostedTransport

            return HostedTransport(
                self.client_backend,
                self.cfg,
                attn_implementation=self.attn_implementation,
            )
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
    workdir = require_data_fast_path(workdir, label="PEFT adapter workdir")
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
            saved = job.client.save_checkpoint(path=str(checkpoint_root), export_hf=True)
            exported = Path(saved.get("hf_path") or saved["path"])
    return copy_checkpoint_peft_adapter(exported.parent, adapter_dir)


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
    unavailable = {
        "rl-weight-sync": "weight-sync verification requires worker-local environment access",
        "rl-router-replay": "router-replay verification requires worker-local trace instrumentation",
    }
    if test.test_id in unavailable:
        return unavailable[test.test_id]
    if test.test_id in {
        "single-step-grads",
        "single-step-optimizer",
        "e2e-sft-train-validate",
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
    test, cfg, published_fixture: dict | None = None, *, local_gateway_full_model: bool = False
) -> TestResult | None:
    reason = (
        None
        if local_gateway_full_model and test.test_id == "e2e-sft-train-validate"
        else cortex_unavailable_reason(test, published_fixture)
    )
    if reason is None and test.compares_to_reference and cfg.training.get("peft_config"):
        reason = "Cortex reference comparisons cannot export the product-initialized LoRA adapter"
    if reason is None:
        return None
    missing_fixture = (
        test.test_id in {"single-step-grads", "single-step-optimizer", "e2e-sft-train-validate"}
        and published_fixture is None
    )
    return TestResult(
        test_id=test.test_id,
        config_id=cfg.config_id,
        outcome=TestOutcome.FAIL,
        summary="published Cortex fixture required" if missing_fixture else "check not supported on Cortex",
        reason=reason,
    )


def validate_local_full_model(backend: CortexConfig, cfg: LoadedConfig, spec: TestSpec) -> Path:
    """Resolve the explicitly supported original model offline, without inventing a reduced binding."""
    from safetensors import safe_open
    from transformers.utils.hub import cached_file

    from .hosted import require_local_cortex_gateway

    require_local_cortex_gateway(backend)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("DSS_ALLOW_REMOTE_MODELS") != "1":
        raise ValueError("local full model requires offline HF cache and DSS_ALLOW_REMOTE_MODELS=1")
    # This explicit local mode is bounded to the pinned source of the requested Qwen3-8B config.
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    if cfg.model_name != "Qwen/Qwen3-8B" or spec.model.source_checkpoint != cfg.model_name:
        raise ValueError("local full model mode currently supports only the pinned Qwen/Qwen3-8B source")
    cache = require_data_fast_path(os.environ.get("HF_HUB_CACHE", ""), label="offline full model cache")
    config_file = cached_file(cfg.model_name, "config.json", local_files_only=True)
    if not isinstance(config_file, str):
        raise ValueError("original full model config is not present in the offline cache")
    root = require_data_fast_path(Path(config_file).parent, label="offline full model snapshot")
    if root != cache / "models--Qwen--Qwen3-8B" / "snapshots" / revision:
        raise ValueError("offline full model snapshot does not match the pinned original revision")

    def readable(relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("offline full model has an unsafe relative asset path")
        resolved = require_data_fast_path(root / path, label="offline full model asset")
        if not resolved.is_relative_to(cache) or not resolved.is_file() or resolved.stat().st_size == 0:
            raise ValueError("offline full model asset is missing, empty or outside its cache")
        return resolved

    config_bytes = readable("config.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30":
        raise ValueError("offline full model config does not match the pinned original Qwen3-8B config")
    config = json.loads(config_bytes)
    readable("tokenizer_config.json")
    readable("tokenizer.json")
    index = json.loads(readable("model.safetensors.index.json").read_text())
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not all(
            isinstance(name, str) and name and isinstance(shard, str) and shard.endswith(".safetensors")
            for name, shard in mapping.items()
        )
    ):
        raise ValueError("offline full model requires an indexed safetensors checkpoint")
    keys = set()
    for shard in sorted(set(mapping.values())):
        with safe_open(readable(shard), framework="pt", device="cpu") as stream:
            actual = set(stream.keys())
            if actual != {name for name, filename in mapping.items() if filename == shard} or keys & actual:
                raise ValueError("offline full model safetensors keys do not match the complete weight index")
            keys.update(actual)
    layers = {int(match.group(1)) for name in keys if (match := re.match(r"model\.layers\.(\d+)\.", name)) is not None}
    if layers != set(range(config["num_hidden_layers"])):
        raise ValueError("offline full model checkpoint is missing original transformer layers")
    return root


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
    local_gateway_full_model: bool = False,
) -> List[TestResult]:
    """Use the config's GPU-family FlashAttention kernel for the reference unless explicitly overridden.

    The onboarding calibration turns that kernel's nondeterministic backward into the frozen gate.
    Matching kernels avoids adding an SDPA-versus-FlashAttention difference and keeps the 64K reference
    memory-feasible. The HF side keeps samples in separate rows, while Arctic Platform exercises packed/SP execution.
    """
    if workdir is not None:
        workdir = require_data_fast_path(workdir, label="correctness workdir")
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    client_backend = client_backend if client_backend is not None else OnPremConfig()
    tests = registered_tests()
    assert_applicable_tests_registered(spec, cfg.config_id)
    chosen = [t for t in tests.values() if select is None or t.test_id in select]
    results: List[TestResult] = []
    published_fixture = None
    local_full_model_path = None
    if local_gateway_full_model:
        if (
            not isinstance(client_backend, CortexConfig)
            or cortex_fixture is not None
            or local_cortex_fixture
            or not select
            or not set(select) <= {"e2e-sft-train-validate", "inference-checkpoint-loss"}
        ):
            raise ValueError("local full model requires one loopback Cortex backend and only full-model checks")
        local_full_model_path = str(validate_local_full_model(client_backend, cfg, spec))
    if local_cortex_fixture:
        from .cortex_fixture import load_local_fixture

        if not isinstance(client_backend, CortexConfig) or cortex_fixture is not None:
            raise ValueError("local Cortex fixture requires CortexConfig without a publication receipt")
        if not select or not set(select) <= {"single-step-grads", "single-step-optimizer"}:
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
                    for test_id in select
                ],
                cfg.n_gpus,
            )
    if cortex_fixture is not None:
        from .cortex_fixture import load_fixture

        if not isinstance(client_backend, CortexConfig):
            raise ValueError("--cortex-fixture requires --backend cortex")
        published_fixture = load_fixture(cortex_fixture, spec)
    if isinstance(client_backend, CortexConfig):
        rejected = []
        for test in chosen:
            if test.test_id not in spec.applicable_tests:
                continue
            rejection = _cortex_rejection(
                test, cfg, published_fixture, local_gateway_full_model=local_gateway_full_model
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
        workdir=tmp,
        batch_paths=batch_paths,
        vocab_size=vocab_size,
        client_backend=client_backend,
        target_model_name=published_fixture["model_name"] if published_fixture is not None else None,
        local_gateway_full_model=local_gateway_full_model,
        local_full_model_path=local_full_model_path,
    )

    run_started = time.monotonic()
    per_arm_tests = [t for t in chosen if t.per_arm is not None]
    whole_run_tests = [t for t in chosen if t.per_arm is None]
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
        )
        if isinstance(client_backend, CortexConfig):
            payload["training_config"]["debug"].pop("optimizer_state_output_dir", None)
            if cfg.gpu_type is not None:
                payload["hardware"] = cfg.gpu_type.upper()
        return payload

    def measure_target(session, job, arm: ArmSpec):
        body = body_for(arm)
        if stepping and isinstance(client_backend, CortexConfig):
            from .cortex_optimizer_capture import capture_optimizer_step

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
            optimizer_output_dir = require_data_fast_path(
                tmp / "optimizer" / "reference" / arm.name / "states", label="reference optimizer output"
            )
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

    execution_errors = set()

    def measure_target_arm(session, arm: ArmSpec) -> bool:
        try:
            with console.activity(f"Running Arctic Platform {arm.name}"):
                with running_job(session, payload_for(arm)) as job_id:
                    started = time.monotonic()
                    context._target[arm.name] = measure_target(session, job_id, arm)
                    context._timing.setdefault(arm.name, {})["dss"] = time.monotonic() - started
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
                    reason=str(exc),
                )
                results.append(result)
                print(console.arm_verdict(result, arm, context._timing.get(arm.name)), flush=True)
            return False
        return True

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
                with console.activity(f"Running reference {arm.name}"):
                    measure_reference(arm)
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
    client_backend: OnPremConfig | CortexConfig | None = None,
) -> List[TestResult]:
    """Run the checks of a reinforcement-learning config that compare its two zones with each other.

    Tests against the single-GPU reference are the caller's to report as inapplicable. Every other applicable
    test runs once, and both sub-jobs run at the widths the config declares. The RL checks build their own
    requests from the training and sampling sub-jobs; a check that trains on generated batches, such as
    ``checkpoint-resume-loss``, runs one case per entry in the spec's ``arms``, as the hosted command does.
    """
    if workdir is not None:
        workdir = require_data_fast_path(workdir, label="RL correctness workdir")
    cfg = cfg.at_gpu_width(cfg.n_gpus)
    assert_applicable_tests_registered(spec, cfg.config_id)
    tmp = workdir if workdir is not None else correctness_workdir("dss-correctness-rl-")
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
        client_backend=client_backend if client_backend is not None else OnPremConfig(),
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
        if isinstance(context.client_backend, CortexConfig):
            rejection = _cortex_rejection(test, cfg)
            if rejection is not None:
                results.append(rejection)
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
