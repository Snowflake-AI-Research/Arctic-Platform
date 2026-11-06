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

"""DeepSpeed Ray worker actor used by the on-prem RL server."""

from __future__ import annotations

import asyncio
import logging
import numbers
import os
import socket
import time
from typing import Any

import deepspeed
import ray
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.common.deepspeed_engine import training_engine_spec
from arctic_platform.common.model_settings import ModelSettings
from arctic_platform.common.ray_cluster import primary_ip
from arctic_platform.common.training_settings import WorkerTrainingSettings
from arctic_platform.common.utils import combine_metric_microbatches
from arctic_platform.common.utils import dp_sp_world_size
from arctic_platform.common.utils import log_dp_shard_tokens
from arctic_platform.common.utils import merge_dict_shards
from arctic_platform.common.utils import sp_size_from_job_config
from arctic_platform.common.utils import split_dict
from arctic_platform.common.utils import unpack_batch
from arctic_platform.common.utils.bf16_zero_norm import is_bf16_zero_norm_assert
from arctic_platform.common.utils.debug import enable_full_determinism
from arctic_platform.common.utils.debug import pr0
from arctic_platform.common.utils.debug import see_memory_usage
from arctic_platform.common.weight_checksum import tensor_l2_sq
from arctic_platform.model import build_model
from arctic_platform.model import ds_worker_zorro_train
from arctic_platform.model import settings_dense_model_spec
from arctic_platform.model import settings_moe_model_spec
from arctic_platform.model.implementations.debug.gpu_memory import gpu_memory_debug_config
from arctic_platform.model.option_profile import check_model_options_profile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models (mirrors dss-platform sftp_server)
# ---------------------------------------------------------------------------

ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


def onprem_deepspeed_config(training: WorkerTrainingSettings, *, dtype: str, world_size: int, seed=None) -> dict:
    """The on-prem training engine's DeepSpeed config.

    No mpu is passed, so DeepSpeed counts every rank as data-parallel unless
    ``training.ds_config`` sets ``sequence_parallel_size``. Unless
    ``training.ds_config`` says otherwise, the engine autocasts to bf16 and
    accumulates gradients in FP32.
    """
    sequence_parallel = int((training.ds_config or {}).get("sequence_parallel_size") or 1)
    spec = training_engine_spec(
        training, dtype=dtype, world_size=world_size, sequence_parallel=sequence_parallel, seed=seed
    )
    ds_config = build_deepspeed_config(spec)
    ds_config.setdefault("torch_autocast", {"enabled": True, "dtype": "bfloat16"})
    # DeepSpeed accumulates in the model dtype when ``grad_accum_dtype`` is unset, so the FP32 default belongs to
    # the key, not the block: a supplied ``data_types`` without it still accumulates in FP32. ``None`` is how
    # DeepSpeed spells unset. A new block is built so a caller's shared ``data_types`` dict is never mutated.
    data_types = dict(ds_config.get("data_types") or {})
    if data_types.get("grad_accum_dtype") is None:
        data_types["grad_accum_dtype"] = "fp32"
    ds_config["data_types"] = data_types
    return ds_config


# ---------------------------------------------------------------------------
# DeepSpeed training actor
# ---------------------------------------------------------------------------


async def spawn_and_initialize_workers(gpus, master_port, config_dict, actor_options):
    """Create DeepSpeed ranks and initialize them. Destroy every spawned actor if any step fails.

    ``actor_options(rank)`` is the kwargs for ``DeepSpeedWorker.options``. Rank 0's host is the
    distributed rendezvous master so off-node ranks do not hang on their own localhost.
    """
    workers = []
    try:
        for rank in range(gpus):
            workers.append(DeepSpeedWorker.options(**actor_options(rank)).remote(rank, gpus, master_port))
        master_addr = await workers[0].get_ip.remote()
        await asyncio.gather(*[w.initialize.remote(master_addr, config_dict) for w in workers])
    except Exception:
        await asyncio.gather(*[w.destroy.remote() for w in workers], return_exceptions=True)
        raise
    return workers


@ray.remote
class DeepSpeedWorker:
    """Single-GPU worker for DeepSpeed training."""

    def __init__(self, rank: int, world_size: int, master_port: int):
        self.rank = rank
        self.world_size = world_size
        self.my_addr = socket.gethostname()
        self.master_addr = primary_ip()
        self.master_port = master_port
        self.engine = None
        self.optimizer = None
        self.num_micro_batches = 1
        self._warned_manual_lr_with_scheduler = False
        self._gpu_mem_preallocation = None
        self.sp_size = 1
        self._weight_sender = None
        self._on_gpu = True

    def get_ip(self) -> str:
        return self.my_addr

    def initialize(self, master_addr: str, job_config: dict) -> bool:
        self.master_addr = master_addr
        os.environ.update(
            {
                "RANK": str(self.rank),
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(self.world_size),
                "MASTER_ADDR": self.master_addr,
                "MASTER_PORT": str(self.master_port),
            }
        )

        if job_config.get("full_determinism", False):
            enable_full_determinism(seed=job_config.get("seed", 42))

        # aws-ofi-nccl generates a per-process topology and hands it to NCCL by setting NCCL_TOPO_FILE to a
        # /proc/self/fd/<N> path (an in-memory fd). That handle is only valid in the process that created it: once
        # inherited by a child, fd <N> resolves to an unrelated file, and because the plugin skips regenerating when
        # NCCL_TOPO_FILE is already set, NCCL loads a bogus topology and the multi-rank rendezvous deadlocks. This
        # worker has not run OFI yet, so any value present here is necessarily inherited and stale -- drop it.
        #
        # Dropping it does NOT cost performance: the plugin sets NCCL_TOPO_FILE only as an in-process mechanism to
        # pass its generated topology to NCCL, and it regenerates that topology during OFI init whenever the var is
        # unset. So popping the stale handle simply makes the plugin produce a fresh, correct, platform-optimal
        # topology for this process -- exactly what happens on a clean first init -- instead of reusing a poisoned
        # one. The narrow /proc/self/fd/ check also leaves an admin-provided static topology path untouched.
        topo_file = os.environ.get("NCCL_TOPO_FILE", "")
        if topo_file.startswith("/proc/self/fd/"):
            logger.warning(f"dropping inherited stale NCCL_TOPO_FILE={topo_file}; OFI will regenerate per-process")
            os.environ.pop("NCCL_TOPO_FILE", None)

        deepspeed.init_distributed()

        model_name = job_config["model_name"]
        self.job_type = job_config.get("job_type")
        pr0(f"{self.job_type=} {job_config=}")

        ds_worker_config = job_config.get("ds_worker_config") or {}
        ds_worker_config["world_size"] = self.world_size
        self.ds_worker_config = ds_worker_config
        self.sp_size = sp_size_from_job_config(job_config)
        dtype = str(job_config.get("dtype") or "bfloat16")

        # Training engines get an optimizer; the reference/log-prob engine is
        # forward-only and is configured from log_prob_config with no optimizer state.
        training = None
        if self.job_type == "log_prob":
            ds_config = self.ds_inference_config(job_config.get("log_prob_config") or {}, ds_worker_config)
            model_settings = ModelSettings.model_validate(job_config.get("training_config") or {})
        else:
            training = WorkerTrainingSettings.model_validate(job_config.get("training_config") or {})
            if training.optimizer is None:
                raise ValueError("on-prem training requires training.optimizer; a frozen model runs on Cortex only")
            mb_spec = training.mb_spec
            if mb_spec is not None and mb_spec.max_tokens_per_mb is not None:
                raise ValueError(
                    "on-prem training splits fwd-bwd batches by mb_spec.num_micro_batches; "
                    "mb_spec.max_tokens_per_mb is supported on Cortex only"
                )
            self.num_micro_batches = (mb_spec.num_micro_batches if mb_spec is not None else None) or 1
            ds_config = onprem_deepspeed_config(
                training, dtype=dtype, world_size=self.world_size, seed=job_config.get("seed")
            )
            model_settings = training
        self._has_optimizer = training is not None

        pr0(f"ds_worker[ds_config]: {self.job_type=} {ds_config=} {ds_worker_config=}")

        debug = gpu_memory_debug_config(
            {
                "debug": training.debug if training is not None else ds_worker_config.get("debug"),
                "prime_rl": model_settings.prime_rl,
            }
        )
        attn_implementation = model_settings.resolved_attn_implementation()
        if attn_implementation is None:
            raise ValueError("on-prem training requires training_config.attn_implementation")
        ep_size = model_settings.ep_size or 1
        check_model_options_profile(model_settings.model_dump(), model_path=model_name, sp_size=self.sp_size)
        # The MoE loader owns expert- and sequence-parallel layout, so only dense loading is gated here.
        if ep_size > 1:
            if ds_worker_config.get("zorro_train_enable"):
                raise ValueError("expert-parallel workers do not support ZoRRo Train")
            spec = settings_moe_model_spec(
                model_settings,
                model_path=model_name,
                sequence_parallel=self.sp_size,
                seq_len=job_config.get("max_seq_len"),
                weight_conversion_cache_dir=ds_worker_config.get("weight_conversion_cache_dir"),
                debug=debug,
            )
        else:
            if self.sp_size != 1:
                raise ValueError("on-prem sequence-parallel data dispatch is not wired; sp_size must be 1")
            if model_settings.model_provider != "huggingface":
                raise ValueError(
                    f"on-prem dense training does not support model_provider={model_settings.model_provider!r}"
                )
            spec = settings_dense_model_spec(
                model_settings,
                model_path=model_name,
                dtype=dtype,
                sequence_parallel=self.sp_size,
                zorro_train=ds_worker_zorro_train(ds_worker_config, attn_implementation),
                debug=debug,
            )
        loaded = build_model(spec)
        model = loaded.model
        self._gpu_mem_preallocation = getattr(model, "_arctic_gpu_mem_preallocation", None)

        zorro_train_enable = ds_worker_config.get("zorro_train_enable", False)
        self.dedup_actor_model_once_patcher = getattr(model, "_arctic_zorro_once_patcher", None)

        self.engine, self.optimizer, self.optimizer_choice = initialize_training_engine(
            model,
            ds_config,
            optimizer=training.optimizer if training is not None else None,
            ep_size=ep_size,
            mpu=None,
            model_path=model_name,
            rank=self.rank,
        )
        self._device = get_accelerator().device_name(self.engine.local_rank)

        gpu_id = torch.cuda.current_device()
        gpu_uuid = torch.cuda.get_device_properties(gpu_id).uuid
        logger.info("Rank %d initialized on GPU %d (uuid=%s, device=%s)", self.rank, gpu_id, gpu_uuid, self._device)
        self.cpu_device = torch.device("cpu")

        pr0(
            "ds_worker[after_initialize]:"
            f" {self.job_type=} {self.engine.global_steps=} {zorro_train_enable=} {model_name=}"
        )

        return True

    def ds_inference_config(self, log_prob_config: dict, ds_worker_config: dict) -> dict:
        """Build a forward-only DeepSpeed config (no optimizer state).

        Used for the reference / log-prob engine. Keeps ZeRO param sharding
        (stage + offload_param) and bf16, but omits the optimizer, gradient
        accumulation, gradient clipping, train_batch_size, and any
        offload_optimizer so DeepSpeed allocates no optimizer state.
        """
        src = dict(log_prob_config or {})
        zero = dict(src.get("zero_optimization", {}) or {})
        zero.pop("offload_optimizer", None)

        cfg: dict = {
            "train_micro_batch_size_per_gpu": src.get("train_micro_batch_size_per_gpu", 1),
        }
        if zero:
            cfg["zero_optimization"] = zero
        if "sequence_parallel_size" in src:
            cfg["sequence_parallel_size"] = src["sequence_parallel_size"]
        if ds_worker_config.get("use_autocast", False):
            cfg["torch_autocast"] = {"enabled": True, "dtype": "bfloat16"}
        cfg["bf16"] = {"enabled": True}
        return cfg

    # move batch to device
    def _move_batch_to_device(self, batch: Any, device: torch.device):
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(v, device) for v in batch]
        elif isinstance(batch, torch.Tensor):
            # Pin + non_blocking H2D: copy overlaps with subsequent CPU work on the
            # default stream; the next CUDA kernel that consumes the tensor waits.
            # ``device`` may be a str ("cuda:0") or a torch.device.
            dev = torch.device(device) if not isinstance(device, torch.device) else device
            if dev.type == "cuda" and batch.device.type == "cpu":
                if not batch.is_pinned():
                    try:
                        batch = batch.pin_memory()
                    except RuntimeError:
                        pass  # non-pageable / shared storage — fall through
                return batch.to(dev, non_blocking=True)
            return batch.to(dev)
        return batch

    def _inject_sft_global_token_meta(self, loss_fn: str, batch_data, meta_data: dict) -> None:
        """All-reduce valid-target count into ``meta["global_num_tokens"]`` + ``dp_size``.

        ``dp_size`` is ``world_size`` while shards stay disjoint. ``sp_size`` is
        still validated against ``world_size``.
        Opt-in via ``SFT_GLOBAL_TOKEN_LOSS_FNS``. No-op when labels are absent.
        """
        from arctic_platform.sft.processor import SFT_GLOBAL_TOKEN_LOSS_FNS
        from arctic_platform.sft.processor import count_valid_target_tokens

        if loss_fn not in SFT_GLOBAL_TOKEN_LOSS_FNS:
            return

        local_tokens = count_valid_target_tokens(batch_data)
        if local_tokens is None:
            return

        global_tokens = local_tokens
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.world_size > 1:
            tok = torch.tensor([local_tokens], device=self._device, dtype=torch.long)
            torch.distributed.all_reduce(tok, op=torch.distributed.ReduceOp.SUM)
            global_tokens = int(tok.item())
        meta_data["global_num_tokens"] = global_tokens
        dp_sp_world_size(self.world_size, self.sp_size)
        meta_data["dp_size"] = self.world_size

    def _forward_maybe_backward(self, batch: dict, backward: bool) -> dict:
        # torch.autograd.set_detect_anomaly(True)

        pr0(f"_forward_maybe_backward mode: {backward=}")
        PROFILE = False
        # if backward:
        #     PROFILE = True
        if PROFILE:
            torch.cuda.memory._record_memory_history(max_entries=int(1e12))
        see_memory_usage("_forward_maybe_backward start", force=True)

        from arctic_platform import sft_profile
        from arctic_platform.rl.processors.base_loss import _pop_loss_object

        loss_object = _pop_loss_object(batch)
        args, batch_data, meta_data, processing = unpack_batch(batch)
        with sft_profile.timed("h2d"):
            if isinstance(batch_data, list):
                batch_data = [self._move_batch_to_device(mb, self._device) for mb in batch_data]
            else:
                batch_data = self._move_batch_to_device(batch_data, self._device)
            if sft_profile.enabled() and torch.cuda.is_available():
                torch.cuda.synchronize()

        tag = "forward_only" if not backward else "forward_backward"

        if isinstance(batch_data, list):
            for i, mb in enumerate(batch_data):
                log_dp_shard_tokens(self.rank, f"{tag} shard mb{i}", mb, meta_data)
            pr0(f"[DeepSpeedWorker] {tag}: gas_list={len(batch_data)} {meta_data.keys()=} {processing.keys()=}")
        else:
            log_dp_shard_tokens(self.rank, f"{tag} shard", batch_data, meta_data)
            pr0(f"[DeepSpeedWorker] {tag}: {batch_data.keys()=} {meta_data.keys()=} {processing.keys()=}")
            for k, v in batch_data.items():
                pr0(f"[DeepSpeedWorker] {tag}: {k=}: shape={getattr(v, 'shape', type(v).__name__)}")

        # H3: list-of-microbatches from the client skips concat→split_dict.
        if isinstance(batch_data, list):
            micro_batch_data = batch_data
            if len(micro_batch_data) != self.num_micro_batches:
                raise ValueError(
                    f"Received {len(micro_batch_data)} microbatches but "
                    f"training.mb_spec.num_micro_batches={self.num_micro_batches}"
                )
        else:
            micro_batch_data = split_dict(batch_data, self.num_micro_batches)
        num_micro_batches = len(micro_batch_data)
        pipeline_micro_batch_outputs = []
        return_tensors = meta_data.get("worker_return_tensors", False)

        # Resolve before selecting a specialized pipeline so class registry
        # precedence applies equally to SFT and RL names. Do not fill a missing
        # loss_fn: a backward pass requires the name in processing.
        loss_fn = processing.get("loss_fn")
        from arctic_platform.common.registry import LOSS_FNS
        from arctic_platform.rl.processors import resolve_loss
        from arctic_platform.sft.processor import SFT_LOSS_FNS

        if loss_object is None and loss_fn is not None:
            loss_object = resolve_loss(loss_fn)
        if loss_object is not None:
            loss_object.model_call_count_callback(
                [num_micro_batches],
                processing.get("config") or {},
            )
        legacy_sft_loss = LOSS_FNS.get(loss_fn) if loss_fn in SFT_LOSS_FNS else None
        use_sft_pipeline = (
            legacy_sft_loss is not None
            and loss_object is not None
            and loss_object.is_legacy_adapter_for(legacy_sft_loss)
        )

        if use_sft_pipeline:
            self._inject_sft_global_token_meta(loss_fn, batch_data, meta_data)

        loss_reduction = None
        if not use_sft_pipeline:
            from arctic_platform.rl.processors import resolve_packed_loss_reduction

            loss_reduction = resolve_packed_loss_reduction(
                processing,
                [{**meta_data, **micro_batch} for micro_batch in micro_batch_data],
                require_declared=False,
                loss_object=loss_object,
            )

        pr0(f"mbs {len(micro_batch_data)=} {self.num_micro_batches=}")

        for i, micro_batch in enumerate(micro_batch_data):

            # time.sleep(1)
            # pr0(f"{i=}")
            # pr0(f"{micro_batch.keys()=}")

            log_dp_shard_tokens(
                self.rank,
                f"{tag} micro_batch {i}/{num_micro_batches}",
                micro_batch,
                meta_data,
            )

            DEBUG = False
            if DEBUG:
                from arctic_platform.rl.zorro_train import analyze_normal_batch_via_attention_mask

                analyze_normal_batch_via_attention_mask(
                    micro_batch["input_ids"], micro_batch["attention_mask"], response_len=meta_data["max_response_len"]
                )

            # die
            see_memory_usage(f"_forward_maybe_backward mb {i=}", force=True)
            if i == 0:
                pr0(f"[DeepSpeedWorker] {tag}: {i=}/{num_micro_batches=} {meta_data.keys()=} {processing.keys()=}")

            # DeepSpeed's accumulation count is 1, so only the last microbatch may be a boundary:
            # earlier ones must not all-reduce grads or step the optimizer.
            if backward and hasattr(self.engine, "set_gradient_accumulation_boundary"):
                self.engine.set_gradient_accumulation_boundary(i == num_micro_batches - 1)

            if use_sft_pipeline:
                from arctic_platform.sft.processor import run_sft_pipeline

                micro_batch_output = run_sft_pipeline(
                    self.engine,
                    micro_batch,
                    meta_data,
                    processing,
                    device=self._device,
                    backward=backward,
                )
            else:
                from arctic_platform.rl.processors import apply_packed_loss_reduction
                from arctic_platform.rl.processors import run_pipeline

                pipeline_backward = "loss_only" if backward and loss_reduction is not None else backward
                micro_batch_output = run_pipeline(
                    self.engine,
                    args,
                    micro_batch,
                    meta_data,
                    processing,
                    device=self._device,
                    backward=pipeline_backward,
                    pack=False,
                    return_tensors=return_tensors,
                    loss_object=loss_object,
                )
                if backward and loss_reduction is not None:
                    apply_packed_loss_reduction(
                        self.engine,
                        micro_batch_output.pop("loss_tensor"),
                        loss_reduction.loss_scales[i],
                        backward=True,
                    )

            if i == 0:
                pr0(f"[DeepSpeedWorker] {tag}: {i=}/{num_micro_batches=} {micro_batch_output.keys()=}")
            pipeline_micro_batch_outputs.append(micro_batch_output)

            # DS requires matching steps for backward pass
            if backward and i < num_micro_batches - 1:
                self._engine_step()

        pipeline_outputs = dict()
        for k, v in pipeline_micro_batch_outputs[0].items():
            if k == "metrics" and isinstance(v, dict):
                if loss_reduction is None:
                    # Legacy losses without packed metadata retain the
                    # historical GAS metric combiner.
                    pipeline_outputs[k] = combine_metric_microbatches([r[k] for r in pipeline_micro_batch_outputs])
                else:
                    from arctic_platform.rl.processors import combine_packed_metrics

                    pipeline_outputs[k] = combine_packed_metrics(
                        [r[k] for r in pipeline_micro_batch_outputs],
                        loss_reduction.reporting_weights,
                    )
            elif isinstance(v, dict):
                pipeline_outputs[k] = merge_dict_shards([r[k] for r in pipeline_micro_batch_outputs])
            elif isinstance(v, numbers.Number):
                values = [r[k] for r in pipeline_micro_batch_outputs]
                if k == "avg_loss" and loss_reduction is not None:
                    from arctic_platform.rl.processors import combine_packed_losses

                    pipeline_outputs[k] = combine_packed_losses(values, loss_reduction)
                else:
                    pipeline_outputs[k] = sum(values) / len(values)

        pipeline_outputs = self._move_batch_to_device(pipeline_outputs, self.cpu_device)

        see_memory_usage("_forward_maybe_backward end", force=True)
        if PROFILE:
            dir = "/tmp/mem-prof"
            rank = 0  # torch.distributed.get_rank()
            from pathlib import Path

            Path(dir).mkdir(exist_ok=True)
            torch.cuda.memory._dump_snapshot(f"{dir}/rank-{rank}.pickle")
            exit()

        pr0(f"[DeepSpeedWorker] {tag}: {pipeline_outputs.keys()=}")
        from arctic_platform import sft_profile

        if sft_profile.enabled():
            metrics = pipeline_outputs.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            metrics = sft_profile.merge_into_metrics(metrics)
            pipeline_outputs["metrics"] = metrics
            sft_profile.maybe_print(f"worker rank{self.rank} {tag}", metrics.get("_profile_ms"))
        return pipeline_outputs

    def forward_backward(self, batch: dict) -> dict:
        tname = timers.start("forward_backward")
        results = self._forward_maybe_backward(batch, backward=True)
        timers.stop_and_print_elapsed(tname)
        return results

    def forward_no_grad(self, batch: dict) -> dict:
        tname = timers.start("forward_no_grad")
        results = self._forward_maybe_backward(batch, backward=False)
        timers.stop_and_print_elapsed(tname)
        return results

    def _is_bf16_zero_norm_assert(self, exc: BaseException) -> bool:
        """True only for BF16_Optimizer's bare ``assert all_groups_norm > 0.``."""
        return is_bf16_zero_norm_assert(exc, getattr(self.engine, "optimizer", None))

    def _engine_step(self) -> None:
        """``engine.step()`` with a skip for DeepSpeed BF16_Optimizer's zero-norm assert.

        ZeRO-1/2 use ``BF16_Optimizer``, which asserts ``all_groups_norm > 0``.
        GRPO can produce an all-zero grad batch (identical group rewards). ZeRO-3
        does not assert; skip the optimizer update instead of crashing.
        Only that BF16 assert is skipped; every other ``AssertionError`` re-raises.
        """
        try:
            self.engine.step()
        except AssertionError as err:
            if self._is_bf16_zero_norm_assert(err):
                pr0("[DeepSpeedWorker] skip optimizer.step: global grad norm is 0")
                return
            raise

    def step(self, learning_rate: float | None = None) -> dict:
        from arctic_platform import sft_profile

        if learning_rate is not None:
            if getattr(self.engine, "lr_scheduler", None) is not None and not self._warned_manual_lr_with_scheduler:
                logger.info("Overriding the scheduled learning rate; the client's per-step learning rate wins.")
                self._warned_manual_lr_with_scheduler = True
            for group in self.optimizer.param_groups:
                group["lr"] = float(learning_rate)
        with sft_profile.timed("step"):
            self._engine_step()
            if sft_profile.enabled() and torch.cuda.is_available():
                torch.cuda.synchronize()
        # Pull grad_norm out of DeepSpeed so it can be logged by the trainer.
        # rename_dict in ray_trainer turns "grad_norm" -> "actor/grad_norm",
        # matching the FSDP baseline path in verl/workers/actor/dp_actor.py.
        grad_norm = self.engine.get_global_grad_norm()
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        metrics = dict(
            last_lr=self.engine.get_lr()[0],
        )
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm
        metrics = sft_profile.merge_into_metrics(metrics)
        if sft_profile.enabled():
            sft_profile.maybe_print(f"worker rank{self.rank} step", metrics.get("_profile_ms"))
        return dict(metrics=metrics, batch=dict())

    def save_checkpoint(self, path: str, export_hf: bool = False) -> dict:
        """Save DeepSpeed checkpoint; optionally export HF weights to ``{path}/hf/`` (rank 0)."""
        self.engine.save_checkpoint(path)
        hf_path = None
        if export_hf:
            hf_path = self.export_hf_checkpoint(path)
        return {
            "path": path,
            "hf_path": hf_path,
            "global_step": int(self.engine.global_steps),
        }

    def load_checkpoint(self, path: str) -> int:
        """Restore DeepSpeed state. Returns restored ``global_steps``, or 0 if none found."""
        load_path, _ = self.engine.load_checkpoint(path)
        if load_path is None:
            return 0
        return int(self.engine.global_steps)

    def export_hf_checkpoint(self, checkpoint_dir: str) -> str | None:
        """Convert a DeepSpeed checkpoint dir to HF ``save_pretrained`` under ``hf/``.

        Rank 0 only; other ranks return None after a barrier. Prefer DeepSpeed's
        ``get_fp32_state_dict_from_zero_checkpoint``; fall back to gathering live params.
        """
        import os

        hf_dir = os.path.join(checkpoint_dir, "hf")
        if self.rank == 0:
            os.makedirs(hf_dir, exist_ok=True)
            model = self.engine.module
            state_dict = None
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
            except Exception:
                logger.exception("zero_to_fp32 export failed; falling back to live gather")
            if state_dict is None:
                state_dict = {n: p.detach().cpu() for n, p in model.named_parameters()}
            try:
                model.save_pretrained(hf_dir, state_dict=state_dict, safe_serialization=True)
            except TypeError:
                # Older transformers: no state_dict kwarg — load then save.
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning("HF export load_state_dict missing=%s unexpected=%s", missing, unexpected)
                model.save_pretrained(hf_dir, safe_serialization=True)
            logger.info("Exported HF checkpoint to %s", hf_dir)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return hf_dir if self.rank == 0 else None

    @staticmethod
    def prune_checkpoint_dirs(parent_dir: str, keep: int) -> int:
        """Keep the newest ``keep`` ``checkpoint-*`` dirs under ``parent_dir``; remove older."""
        from arctic_platform.common.utils.checkpoint import prune_checkpoint_dirs

        return prune_checkpoint_dirs(parent_dir, keep)

    def compute_log_probs(self, batch: dict) -> torch.Tensor:
        """Full-sequence per-token log-probs for a ``{"batch","meta","processing"}`` DP shard.

        Takes a dict shard (like ``forward_no_grad``) -- ``unpack_batch`` pulls the encoded ``{input_ids,
        attention_mask}`` out of ``batch["batch"]`` -- runs a forward-only pass, and returns the shifted-label
        per-token log-probs ``[shard_B, S-1]`` on CPU. The DeepSpeed engine ignores ``top_k`` (unlike the vLLM path).
        """
        _, kwargs, _, _ = unpack_batch(batch)
        kwargs = {k: v.to(self._device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        with torch.no_grad():
            logits = self.engine(**kwargs).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        shifted_ids = kwargs["input_ids"][:, 1:]
        token_log_probs = log_probs[:, :-1].gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.cpu()

    def max_param_bytes(self) -> int:
        max_bytes = 0
        for p in self.engine.module.parameters():
            elem_size = p.data.element_size()
            numel = p.ds_numel if hasattr(p, "ds_id") else p.data.numel()
            max_bytes = max(max_bytes, numel * elem_size)
        return max_bytes

    def init_weight_sender(self, group, schedule, master_addr, base_port, bucket_size) -> bool:
        from arctic_platform.inference.server.weight_sync.sender import WeightSender

        self._weight_sender = WeightSender(
            group=group,
            schedule=schedule,
            master_addr=master_addr,
            base_port=base_port,
            device=torch.device(self._device),
            bucket_size=bucket_size,
        )
        return True

    def get_weights(self) -> list[tuple[str, torch.Tensor]]:
        weights = []
        for n, p in self.engine.module.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    weights.append((n, p.data))
            else:
                weights.append((n, p.data))
        return weights

    def weight_norm(self) -> dict:
        """Global L2 norm of the model's parameters (sum of squares + count).

        ZeRO-3 partitions each param across ranks, so ``GatheredParameters``
        materializes the full param on every rank; the resulting sum of
        squares is therefore the whole-model value (the server reads rank 0).
        Summing squares is layout-invariant, so it compares directly against
        the vLLM engine's norm despite vLLM fusing params differently. A debug
        check: a weight sync proves itself with its ``sent_l2_sq`` checksum.
        """
        sq_sum = torch.zeros((), dtype=torch.float64, device=self._device)
        num_params = 0
        for _, p in self.engine.module.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    sq_sum.add_(tensor_l2_sq(p.data).to(sq_sum.device))
            else:
                sq_sum.add_(tensor_l2_sq(p.data).to(sq_sum.device))
            num_params += 1
        sq_sum = sq_sum.item()
        return {"norm": sq_sum**0.5, "sq_sum": sq_sum, "num_params": num_params}

    def _weights_l2_sq(self, weights: list[tuple[str, torch.Tensor]]) -> float:
        total = torch.zeros((), dtype=torch.float64, device=self._device)
        for _, weight in weights:
            total.add_(tensor_l2_sq(weight).to(total.device))
        return total.item()

    def send_weights(self) -> dict:
        """Send the full model to this rank's NCCL group.

        ``sent_l2_sq`` is the sum of squares of the full tensors handed to the sender. Every sender rank gathers the
        whole model, so each one's value is authoritative for the replicas in its own group.
        """
        weights = self.get_weights()
        if self._weight_sender is not None:
            sent_l2_sq = self._weights_l2_sq(weights)
            result = self._weight_sender.send(weights)
            result["sent_l2_sq"] = sent_l2_sq
            return result
        return {"status": "not_initialized"}

    def get_cuda_ipc_handles(self) -> dict:
        """Create CUDA IPC handles for all model parameters.

        Returns a dict with names, dtypes, shapes and pickled IPC handles
        that can be opened by another process on the same GPU.

        For ZeRO-2 this reads p.data directly (full params on every rank).
        For ZeRO-3 this is incorrect — use gather_cuda_ipc_handles instead.
        """
        import base64
        import pickle

        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        names, dtype_names, shapes = [], [], []
        handles = []
        self._ipc_tensor_refs = []

        for name, p in self.engine.module.named_parameters():
            weight = p.data.detach().contiguous()
            self._ipc_tensor_refs.append(weight)
            handle = reduce_tensor(weight)
            handles.append({gpu_uuid: handle})
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))

        torch.cuda.synchronize()
        pickled = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": pickled,
            "num_params": len(names),
        }

    def gather_cuda_ipc_handles(self) -> dict:
        """Gather ZeRO-3 partitioned params and create CUDA IPC handles.

        All ranks must call this collectively (GatheredParameters is a
        collective op).  Inside the context manager the full param lives
        on every rank's GPU — each rank clones it and creates an IPC handle
        keyed by its own GPU UUID before the context manager frees the
        gathered tensor.  Every rank returns its payload; the server merges
        them so each colocated inference replica finds a handle for its GPU.

        ``sent_l2_sq`` is the sum of squares of the full tensors behind this
        rank's handles, so each rank's value is authoritative for the replicas
        on its own GPU.
        """
        import base64
        import pickle

        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        t0 = time.monotonic()
        names, dtype_names, shapes = [], [], []
        handles = []
        self._ipc_tensor_refs = []
        model = self.engine.module
        sent_l2_sq = torch.zeros((), dtype=torch.float64, device=self._device)

        # Every rank builds IPC handles for the full (gathered) weight on its
        # OWN physical GPU, keyed by that GPU's UUID. With colocated multi-GPU
        # inference, each vLLM replica lives on a distinct physical GPU (bundle
        # r == training rank r), so it needs a handle for *its* GPU. The server
        # merges these per-rank dicts so each replica finds its GPU's handle.
        # (Previously only rank 0 produced handles, which only worked when a
        # single sampling GPU was colocated with rank 0.)
        for name, p in model.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    weight = p.data.detach().clone().contiguous()
                    self._ipc_tensor_refs.append(weight)
                    handle = reduce_tensor(weight)
                    handles.append({gpu_uuid: handle})
                    names.append(name)
                    dtype_names.append(str(weight.dtype).split(".")[-1])
                    shapes.append(list(weight.shape))
            else:
                weight = p.data.detach().contiguous()
                self._ipc_tensor_refs.append(weight)
                handle = reduce_tensor(weight)
                handles.append({gpu_uuid: handle})
                names.append(name)
                dtype_names.append(str(weight.dtype).split(".")[-1])
                shapes.append(list(weight.shape))
            sent_l2_sq.add_(tensor_l2_sq(weight).to(sent_l2_sq.device))

        if hasattr(self.engine, "empty_partition_cache"):
            self.engine.empty_partition_cache()
        torch.cuda.synchronize()

        elapsed = time.monotonic() - t0
        pickled = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
        logger.info(
            "Rank %d gathered IPC handles in %.2fs (%d params, gpu=%s)", self.rank, elapsed, len(names), gpu_uuid
        )
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": pickled,
            "num_params": len(names),
            "gpu_uuid": gpu_uuid,
            "sent_l2_sq": sent_l2_sq.item(),
        }

    def release_ipc_handles(self) -> bool:
        """Release tensor references held for IPC handles."""
        self._ipc_tensor_refs = []
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        return True

    def get_parameter_names(self) -> list:
        """Return the model's parameter names in deterministic order.

        Used to drive the low-memory streaming weight sync one parameter at a
        time. Only names cross the Ray boundary; the live parameter is resolved
        on each rank inside ``get_cuda_ipc_handle`` so ZeRO-3 ``ds_id`` / live
        storage is preserved.
        """
        return [name for name, _ in self.engine.module.named_parameters()]

    def _param_by_name(self, name: str):
        """Resolve this rank's live module parameter for ``name`` (cached)."""
        cache = getattr(self, "_param_name_cache", None)
        if cache is None:
            cache = dict(self.engine.module.named_parameters())
            self._param_name_cache = cache
        return cache[name]

    def get_cuda_ipc_handle(self, name: str) -> dict:
        """Create a CUDA IPC handle payload for a single parameter on this
        rank's GPU.

        Memory-efficient counterpart to ``gather_cuda_ipc_handles``: only ONE
        full parameter is materialized at a time instead of the whole model.
        For ZeRO-3 params all ranks must call this collectively with the same
        ``name`` (``GatheredParameters`` is a collective op). The caller must
        invoke ``release_ipc_handles`` between params so peak extra GPU memory
        stays at one full parameter per GPU.

        ``sent_l2_sq`` covers this one parameter; the caller streaming the
        parameters adds them up per rank.
        """
        import base64
        import pickle

        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        p = self._param_by_name(name)

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        if hasattr(p, "ds_id"):
            with deepspeed.zero.GatheredParameters([p], enabled=True):
                weight = p.data.detach().clone().contiguous()
        else:
            weight = p.data.detach().contiguous()

        # Hold exactly one source tensor alive until release_ipc_handles().
        self._ipc_tensor_refs = [weight]
        handle = reduce_tensor(weight)
        torch.cuda.synchronize()

        return {
            "names": [name],
            "dtype_names": [str(weight.dtype).split(".")[-1]],
            "shapes": [list(weight.shape)],
            "ipc_handles_pickled": base64.b64encode(pickle.dumps([{gpu_uuid: handle}])).decode("utf-8"),
            "num_params": 1,
            "gpu_uuid": gpu_uuid,
            "sent_l2_sq": tensor_l2_sq(weight).item(),
        }

    def save_state_dict_to_path(self, path: str) -> dict:
        """Save model state dict to a file.

        Works for both ZeRO-2 (params are full) and ZeRO-3 (params are
        partitions — we just save whatever is local).  For ZeRO-3, the
        caller must ensure all ranks call this collectively and only
        rank 0's output is used.
        """
        t0 = time.monotonic()
        weights = [(n, p.data.cpu()) for n, p in self.engine.module.named_parameters()]
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(weights, path)
        num_params = len(weights)
        del weights
        import gc

        gc.collect()
        elapsed = time.monotonic() - t0
        logger.info("Rank %d saved state dict to %s in %.2fs (%d params)", self.rank, path, elapsed, num_params)
        return {"num_params": num_params, "elapsed": elapsed}

    def gather_and_save_state_dict(self, path: str) -> dict:
        """Gather ZeRO-3 partitioned params and save full state dict.

        All ranks must call this collectively.  Every parameter is wrapped
        in GatheredParameters so the all-gather runs on all ranks.  Only
        rank 0 copies the full tensor and writes to disk.

        Only rank 0 reports ``sent_l2_sq``, squared on the GPU over the tensors
        it writes: the file is the only copy the sampler reads.
        """
        import deepspeed

        t0 = time.monotonic()
        model = self.engine.module
        weights = []
        sent_l2_sq = torch.zeros((), dtype=torch.float64, device=self._device)
        for n, p in model.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    if self.rank == 0:
                        sent_l2_sq.add_(tensor_l2_sq(p.data).to(sent_l2_sq.device))
                        weights.append((n, p.data.cpu()))
            else:
                if self.rank == 0:
                    sent_l2_sq.add_(tensor_l2_sq(p.data).to(sent_l2_sq.device))
                    weights.append((n, p.data.cpu()))
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(weights, path)
        num_params = len(weights)
        del weights
        import gc

        gc.collect()
        if hasattr(self.engine, "empty_partition_cache"):
            self.engine.empty_partition_cache()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - t0
        logger.info("Rank %d gathered+saved state dict in %.2fs (%d params)", self.rank, elapsed, num_params)
        if self.rank != 0:
            return {"num_params": num_params, "elapsed": elapsed}
        return {"num_params": num_params, "elapsed": elapsed, "sent_l2_sq": sent_l2_sq.item()}

    def _log_mem(self, label):
        from deepspeed.runtime.utils import see_memory_usage

        see_memory_usage(f"[Rank {self.rank}] {label}", force=True)

    def _ds_offload(self, include):
        """Offload states using DeepSpeed native API.

        engine.offload_states works for ZeRO-3/ZeRO-2 without
        offload_optimizer.  When offload_optimizer is configured,
        DeepSpeed raises AssertionError ("Moving states across devices
        is not supported"); fall back to optimizer.offload_states which
        bypasses the engine assertion (see DeepSpeed issue #6596).
        """
        from deepspeed.runtime.zero.config import OffloadDeviceEnum

        try:
            self.engine.offload_states(include=include)
        except AssertionError as e:
            if "Moving states across devices" not in str(e):
                raise
            opt = getattr(self.engine, "optimizer", None)
            if opt is None:
                # Forward-only (inference) engine has no optimizer to fall back
                # to; the engine-level assertion only fires with offload_optimizer.
                raise
            opt.offload_states(
                include=include,
                device=OffloadDeviceEnum.cpu,
                pin_memory=True,
            )

    def _ds_reload(self):
        """Reload state to GPU.

        engine.reload_states works for most configs.  When
        offload_optimizer is configured, falls back to optimizer
        directly.  ZeRO-2 reload_states requires .grad on fp32 param
        partitions; create zero grads if missing.  Forward-only engines
        have no optimizer, so the optimizer-specific handling is skipped.
        """
        opt = getattr(self.engine, "optimizer", None)
        if opt is not None and hasattr(opt, "single_partition_of_fp32_groups"):
            for fp32_group in opt.single_partition_of_fp32_groups:
                if fp32_group.grad is None:
                    fp32_group.grad = torch.zeros_like(fp32_group)
        try:
            self.engine.reload_states()
        except AssertionError as e:
            if "Moving states across devices" not in str(e):
                raise
            if opt is None:
                raise
            opt.reload_states()

    def _move_params(self, device) -> None:
        """Move model parameters to ``device`` for a forward-only engine.

        DeepSpeed's offload_states/reload_states require a real optimizer, so a
        no-optimizer (inference) engine can't use them. Instead move each
        parameter's storage directly. Under ZeRO-3 the local shard lives in
        ``param.ds_tensor``; otherwise it is ``param.data``. This is only ever
        called while the engine is idle (the client wakes it before any forward
        and sleeps it after), so it never races a parameter all-gather.
        """
        for p in self.engine.module.parameters():
            shard = getattr(p, "ds_tensor", None)
            if shard is not None:
                shard.data = shard.data.to(device, non_blocking=True)
            else:
                p.data = p.data.to(device, non_blocking=True)

    def offload_to_cpu(self) -> dict:
        """Offload engine state to CPU using DeepSpeed native API.

        Trainable engines offload optimizer/grad states plus model params via
        the DeepSpeed offload_states API. A forward-only (no-optimizer) engine
        has no optimizer for that API, so it moves only its model params.
        """
        if not self._on_gpu:
            return {"status": "already_offloaded"}
        t0 = time.monotonic()
        self._log_mem("before offload_to_cpu")

        if getattr(self, "_has_optimizer", True):
            from deepspeed.runtime.zero.offload_states import OffloadStateTypeEnum

            self._ds_offload(
                include=[
                    OffloadStateTypeEnum.hp_params,
                    OffloadStateTypeEnum.lp_params,
                    OffloadStateTypeEnum.lp_grads,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                ]
            )
        else:
            self._move_params(self.cpu_device)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._on_gpu = False
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded to CPU in %.2fs (%.0f MB GPU remaining)", self.rank, elapsed, mem_mb)
        return {"status": "offloaded", "elapsed": elapsed, "gpu_mb": mem_mb}

    def backload_to_gpu(self) -> dict:
        """Reload engine state to GPU using DeepSpeed native API.

        Forward-only engines move only their model params back (no optimizer
        state to reload)."""
        if self._on_gpu:
            return {"status": "already_on_gpu"}
        t0 = time.monotonic()
        self._log_mem("before reload_states")

        if getattr(self, "_has_optimizer", True):
            self._ds_reload()
        else:
            self._move_params(torch.device(self._device))
        torch.cuda.empty_cache()
        self._log_mem("after reload_states + empty_cache")

        torch.cuda.synchronize()
        self._on_gpu = True
        elapsed = time.monotonic() - t0
        logger.info("Rank %d backloaded to GPU in %.2fs", self.rank, elapsed)
        return {"status": "on_gpu", "elapsed": elapsed}

    def offload_non_lp_states(self) -> dict:
        """Offload everything except bf16 params (for CUDA IPC sync)."""
        t0 = time.monotonic()
        self._log_mem("before offload_non_lp")

        from deepspeed.runtime.zero.offload_states import OffloadStateTypeEnum

        self._ds_offload(
            include=[
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.lp_grads,
                OffloadStateTypeEnum.contiguous_grad_buffer,
            ]
        )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded non-lp states in %.2fs (%.0f MB GPU)", self.rank, elapsed, mem_mb)
        return {"status": "offloaded_non_lp", "elapsed": elapsed, "gpu_mb": mem_mb}

    def offload_lp_params(self) -> dict:
        """Offload bf16 model params to CPU (after CUDA IPC sync)."""
        t0 = time.monotonic()

        from deepspeed.runtime.zero.offload_states import OffloadStateTypeEnum

        self._ds_offload(include=[OffloadStateTypeEnum.lp_params])

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._on_gpu = False
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded lp_params in %.2fs (%.0f MB GPU remaining)", self.rank, elapsed, mem_mb)
        return {"status": "offloaded_lp", "elapsed": elapsed, "gpu_mb": mem_mb}

    def empty_cache(self) -> dict:
        """Release ZeRO-3 partition cache and PyTorch cached memory."""
        if hasattr(self.engine, "empty_partition_cache"):
            self.engine.empty_partition_cache()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d empty_cache: %.0f MB GPU remaining", self.rank, mem_mb)
        return {"gpu_mb": mem_mb}

    def destroy(self) -> bool:
        if self._weight_sender is not None:
            self._weight_sender.destroy()
            self._weight_sender = None
        if dist.is_initialized():
            dist.destroy_process_group()
        self.engine = None
        self._gpu_mem_preallocation = None
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
