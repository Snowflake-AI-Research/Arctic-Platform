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

"""Local RL server matching the dss-platform sftp_server HTTP API.

Uses Ray to manage DeepSpeed workers and ArcticInference ReplicaPools.

Usage::

    python -m arctic_platform.rl.server \\
        --training-gpus 4 --sampling-gpus 2 --log-prob-gpus 2
"""

from __future__ import annotations

import io
import logging
import os
import time
from typing import Any

import deepspeed
import ray
import torch
import torch.distributed as dist
import uvicorn
from arctic_inference.server.weight_sync.sender import WeightSender
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM
import numbers
from arctic_platform.rl.processors import run_pipeline
from arctic_platform.rl.utils import (
    unpack_batch,
    merge_dict_shards,
    combine_metric_microbatches,
    split_dict,
    log_dp_shard_tokens,
)
from arctic_platform.rl.ray_cluster import primary_ip
from arctic_platform.rl.utils.debug import enable_full_determinism
from arctic_platform.rl.utils.debug import see_memory_usage, pr, pr0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models (mirrors dss-platform sftp_server)
# ---------------------------------------------------------------------------

ENABLE_TIMERS = True
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple
    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy
    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


def make_model_gradient_checkpointing_compatible(model):
    # Taken from arctic_platform/model/hf_factory.py
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model

# ---------------------------------------------------------------------------
# DeepSpeed training actor
# ---------------------------------------------------------------------------

import socket
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
        self._weight_sender: WeightSender | None = None
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

        deepspeed.init_distributed()

        model_name = job_config["model_name"]
        ds_config = job_config.get("ds_config") or {}
        self.job_type = job_config.get("job_type")
        pr0(f"{self.job_type=} {job_config=}")
        pr0(f"ds_worker[before_modify]: {self.job_type=} {ds_config=}")

        ds_worker_config = job_config.get("ds_worker_config") or {}
        ds_worker_config["world_size"] = self.world_size
        self.ds_worker_config = ds_worker_config

        # Build the DeepSpeed config per job type. Training engines get an
        # optimizer; the reference/log-prob engine is forward-only and is
        # configured from log_prob_config with no optimizer state.
        if self.job_type == "log_prob":
            log_prob_config = job_config.get("log_prob_config") or {}
            ds_config = self.ds_inference_config(log_prob_config, ds_worker_config)
            self._has_optimizer = False
        else:
            ds_config = self.ds_training_config(job_config, ds_config, ds_worker_config)
            self._has_optimizer = True

        pr0(f"ds_worker[after_modify]: {self.job_type=} {ds_config=} {ds_worker_config=}")

        attn_implementation = ds_worker_config.get("attn_implementation", "flash_attention_2")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )

        if ds_worker_config.get("use_liger", False):
            pr0(f"Using liger kernel w/ {attn_implementation=}")
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            # Apply Liger kernel to the model if use_liger is set to True
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(
                model=model,
                cross_entropy=False,
                fused_linear_cross_entropy=False,
                rope=True,
                rms_norm=True,
                swiglu=True,
            )
        # if ds_worker_config.get("use_liger", False):
        #     pr0(f"Using liger kernel w/ {attn_implementation=}")
        #     from liger_kernel.transformers import AutoLigerKernelForCausalLM
        #     model = AutoLigerKernelForCausalLM.from_pretrained(
        #         model_name,
        #         attn_implementation=attn_implementation,
        #         dtype=torch.bfloat16,
        #     )

        # else:
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         attn_implementation=attn_implementation,
        #         dtype=torch.bfloat16,
        #     )

        zorro_train_enable = ds_worker_config.get("zorro_train_enable", False)
        if zorro_train_enable:
            self.model_patch_in_zorro(model, ds_worker_config)

        init_kwargs = dict(model=model, config=ds_config)
        if self._has_optimizer:
            # Forward-only (log-prob) engines are initialized without an
            # optimizer so DeepSpeed allocates no optimizer state.
            init_kwargs["model_parameters"] = model.parameters()
        self.engine, _, _, _ = deepspeed.initialize(**init_kwargs)
        self._device = get_accelerator().device_name(self.engine.local_rank)

        gpu_id = torch.cuda.current_device()
        gpu_uuid = torch.cuda.get_device_properties(gpu_id).uuid
        logger.info("Rank %d initialized on GPU %d (uuid=%s, device=%s)",
                     self.rank, gpu_id, gpu_uuid, self._device)
        self.cpu_device = torch.device("cpu")

        enable_gradient_checkpointing = ds_worker_config.get("enable_gradient_checkpointing", True)
        if enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        pr0(f"ds_worker[after_initialize]: {self.job_type=} {self.engine.global_steps=} {zorro_train_enable=} {model_name=}")

        return True


    def ds_training_config(self, job_config: dict, ds_config: dict, ds_worker_config: dict) -> dict:
        """Build the DeepSpeed config for a trainable engine (with optimizer).

        Prefer the high-level training_config when provided (the framework sends
        training_config and the server owns the DeepSpeed details); otherwise
        fall back to a default AdamW optimizer.
        """
        training_config = job_config.get("training_config")
        if training_config is not None:
            opt_cfg = training_config.get("optimizer", {})
            ds_config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": opt_cfg.get("lr", 1e-5),
                    "betas": [opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)],
                    "eps": 1e-8,
                    "weight_decay": opt_cfg.get("weight_decay", 0.0),
                },
            }
            if "gradient_accumulation_steps" in training_config:
                ds_config["gradient_accumulation_steps"] = training_config["gradient_accumulation_steps"]
            if "gradient_clipping" in opt_cfg:
                ds_config["gradient_clipping"] = opt_cfg["gradient_clipping"]

            sched_cfg = training_config.get("lr_scheduler", None)
            horizon = training_config.get("training_horizon", 0)
            if sched_cfg is not None and horizon > 0:
                lr = opt_cfg.get("lr", 1e-5)
                warmup_steps = sched_cfg.get("warmup_ratio", 0.0) * horizon
                if sched_cfg.get("type", "constant") == "cosine":
                    ds_config["scheduler"] = {
                        "type": "WarmupCosineLR",
                        "params": {
                            "total_num_steps": horizon,
                            "warmup_num_steps": warmup_steps,
                            "warmup_min_ratio": 0.0,
                            "cos_min_ratio": sched_cfg.get("min_lr_ratio") or 0.0,
                            "warmup_type": "linear",

                        },
                    }
                elif warmup_steps > 0:
                    ds_config["scheduler"] = {
                        "type": "WarmupLR",
                        "params": {
                            "warmup_min_lr": 0.0,
                            "warmup_max_lr": lr,
                            "warmup_num_steps": warmup_steps,
                            "warmup_type": "linear",
                        },
                    }
                else:
                    # No LR scheduler, use constant LR
                    pass

        if ds_worker_config.get("use_autocast", False):
            ds_config["torch_autocast"] = {"enabled": True, "dtype": "bfloat16"}

        if ds_worker_config.get("fp32_gradients", False):
            ds_config["communication_data_type"] = "fp32"
            ds_config["data_types"] = {"grad_accum_dtype": "fp32"}

        # Set reasonable defaults as fallbacks
        ds_config.setdefault("train_micro_batch_size_per_gpu", 1)
        ds_config.setdefault("bf16", {"enabled": True})
        ds_config.setdefault(
            "optimizer",
            {
                "type": "AdamW",
                "params": {"lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8},
            },
        )
        return ds_config


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


    def model_patch_in_zorro(self, model, ds_worker_config):
        from arctic_platform.rl.zorro_train.qwen_model_patcher import Qwen3ModelOncePatcher

        #pr0(f"Patching ZoRRO")

        response_len = ds_worker_config.get("response_len")
        max_token_len = ds_worker_config.get("max_token_len")
        rollout_n = ds_worker_config.get("rollout_n")
        temperature = ds_worker_config.get("temperature")
        logits_optimization = ds_worker_config.get("logits_optimization", "none")
        logits_optimization_peak_mem_size_in_gib = ds_worker_config.get("logits_optimization_peak_mem_size_in_gib", 4)
        logits_compute_from_fp32_inputs = ds_worker_config.get("logits_compute_from_fp32_inputs", False)
        logits_compute_in_fp32 = ds_worker_config.get("logits_compute_in_fp32", False)
        use_unpad = ds_worker_config.get("use_unpad")
        world_size = ds_worker_config.get("world_size")

        self.dedup_actor_model_once_patcher = Qwen3ModelOncePatcher(model, response_len=response_len, max_token_len=max_token_len, rollout_n=rollout_n, temperature=temperature, logits_optimization=logits_optimization, logits_optimization_peak_mem_size_in_gib=logits_optimization_peak_mem_size_in_gib, logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs, logits_compute_in_fp32=logits_compute_in_fp32, use_unpad=use_unpad, world_size=world_size)
        self.dedup_actor_model_once_patcher.patch_forward()

    # move batch to device
    def _move_batch_to_device(self, batch: Any, device: torch.device):
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(v, device) for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    def _forward_maybe_backward(self, batch: dict, backward: bool) -> dict:
        #torch.autograd.set_detect_anomaly(True)

        pr0(f"_forward_maybe_backward mode: {backward=}")
        PROFILE = False
        # if backward:
        #     PROFILE = True
        if PROFILE:
            torch.cuda.memory._record_memory_history(max_entries=int(1e12))
        see_memory_usage("_forward_maybe_backward start", force=True)

        args, batch_data, meta_data, processing = unpack_batch(batch)
        batch_data = self._move_batch_to_device(batch_data, self._device)

        tag = "forward_only" if not backward else "forward_backward"

        log_dp_shard_tokens(self.rank, f"{tag} shard", batch_data, meta_data)

        pr0(f"[DeepSpeedWorker] {tag}: {batch_data.keys()=} {meta_data.keys()=} {processing.keys()=}")

        for k, v in batch_data.items():
            pr0(f"[DeepSpeedWorker] {tag}: {k=}: {v.shape=}")

        grad_accum_steps = self.engine.gradient_accumulation_steps()
        micro_batch_data = split_dict(batch_data, grad_accum_steps)
        num_micro_batches = len(micro_batch_data)
        pipeline_micro_batch_outputs = []
        return_tensors = meta_data.get("worker_return_tensors", False)

        pr0(f"mbs {len(micro_batch_data)=} {grad_accum_steps=}")

        for i, micro_batch in enumerate(micro_batch_data):
            import time
            #time.sleep(1)
            #pr0(f"{i=}")
            #pr0(f"{micro_batch.keys()=}")

            log_dp_shard_tokens(
                self.rank, f"{tag} micro_batch {i}/{num_micro_batches}", micro_batch, meta_data,
            )

            DEBUG = False
            if DEBUG:
                from arctic_platform.rl.zorro_train import analyze_normal_batch_via_attention_mask
                analyze_normal_batch_via_attention_mask(micro_batch["input_ids"], micro_batch["attention_mask"], response_len=meta_data["max_response_len"])

            #die
            see_memory_usage(f"_forward_maybe_backward mb {i=}", force=True)
            if i == 0:
                pr0(f"[DeepSpeedWorker] {tag}: {i=}/{num_micro_batches=} {meta_data.keys()=} {processing.keys()=}")

            micro_batch_output = run_pipeline(
                self.engine, args, micro_batch, meta_data, processing,
                device=self._device, backward=backward,
                pack=False,
                return_tensors=return_tensors,
            )

            if i == 0:
                pr0(f"[DeepSpeedWorker] {tag}: {i=}/{num_micro_batches=} {micro_batch_output.keys()=}")
            pipeline_micro_batch_outputs.append(micro_batch_output)

            # DS requires matching steps for backward pass
            if backward and i < num_micro_batches - 1:
                self.engine.step()

        pipeline_outputs = dict()
        for k, v in pipeline_micro_batch_outputs[0].items():
            if k == "metrics" and isinstance(v, dict):
                # Per-microbatch loss-fn metrics are emitted as paired
                # ``{name}.sum`` / ``{name}.tokens`` scalars (plus a few
                # passthrough numerics like ``kl_coef``). Sum them across
                # this rank's microbatches so each rank returns one scalar
                # per metric; ``ray_server.fwd_bwd`` / ``http_server.fwd_bwd``
                # then sums across DP ranks and collapses the paired keys
                # into a single global token-mean per metric per mini-batch.
                pipeline_outputs[k] = combine_metric_microbatches(
                    [r[k] for r in pipeline_micro_batch_outputs]
                )
            elif isinstance(v, dict):
                pipeline_outputs[k] = merge_dict_shards([r[k] for r in pipeline_micro_batch_outputs])
            elif isinstance(v, numbers.Number):
                # TODO: weight average needs to be implemented
                pipeline_outputs[k] = sum([r[k] for r in pipeline_micro_batch_outputs]) / len(pipeline_micro_batch_outputs)

        pipeline_outputs = self._move_batch_to_device(pipeline_outputs, self.cpu_device)

        see_memory_usage("_forward_maybe_backward end", force=True)
        if PROFILE:
            dir = "/tmp/mem-prof"
            rank = 0 # torch.distributed.get_rank()
            from pathlib import Path
            Path(dir).mkdir(exist_ok=True)
            torch.cuda.memory._dump_snapshot(f"{dir}/rank-{rank}.pickle")
            exit()

        pr0(f"[DeepSpeedWorker] {tag}: {pipeline_outputs.keys()=}")
        return pipeline_outputs

    def forward_backward(self, batch: dict) -> dict:
        tname = timers.start("forward_backward")
        results = self._forward_maybe_backward(batch, backward=True)
        timers.stop_and_print_elapsed(tname);
        return results

    def forward_no_grad(self, batch: dict) -> dict:
        tname = timers.start("forward_no_grad")
        results = self._forward_maybe_backward(batch, backward=False)
        timers.stop_and_print_elapsed(tname);
        return results

    def step(self) -> dict:
        self.engine.step()
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
        return dict(metrics=metrics, batch=dict())

    def save_checkpoint(self, path: str) -> bool:
        self.engine.save_checkpoint(path)
        return True

    def compute_log_probs(self, batch_bytes: bytes) -> bytes:
        batch = torch.load(io.BytesIO(batch_bytes), map_location=self._device)
        args, kwargs, _, _ = unpack_batch(batch)
        with torch.no_grad():
            logits = self.engine(*args, **kwargs).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        shifted_ids = kwargs["input_ids"][:, 1:]
        token_log_probs = log_probs[:, :-1].gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
        buf = io.BytesIO()
        torch.save(token_log_probs.cpu(), buf)
        return buf.getvalue()

    def max_param_bytes(self) -> int:
        max_bytes = 0
        for p in self.engine.module.parameters():
            elem_size = p.data.element_size()
            numel = p.ds_numel if hasattr(p, "ds_id") else p.data.numel()
            max_bytes = max(max_bytes, numel * elem_size)
        return max_bytes


    def init_weight_sender(self, group, schedule, master_addr, base_port, bucket_size) -> bool:
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

    def send_weights(self) -> dict:
        weights = self.get_weights()
        if self._weight_sender is not None:
            return self._weight_sender.send(weights)
        return {"status": "not_initialized"}


    def send_weights_ipc(self, group_id: int) -> dict:
        """Save weights to shared memory for colocated (same-GPU) transfer."""
        from arctic_inference.server.weight_sync.ipc_engine import save_weights_to_shm
        weights = [(n, p.data) for n, p in self.engine.module.named_parameters()]
        return save_weights_to_shm(weights, group_id)

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

        gpu_uuid = str(torch.cuda.get_device_properties(
            torch.cuda.current_device()).uuid)

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
        """
        import base64
        import pickle

        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(
            torch.cuda.current_device()).uuid)

        t0 = time.monotonic()
        names, dtype_names, shapes = [], [], []
        handles = []
        self._ipc_tensor_refs = []
        model = self.engine.module

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

        if hasattr(self.engine, 'empty_partition_cache'):
            self.engine.empty_partition_cache()
        torch.cuda.synchronize()

        elapsed = time.monotonic() - t0
        pickled = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
        logger.info("Rank %d gathered IPC handles in %.2fs (%d params, gpu=%s)",
                    self.rank, elapsed, len(names), gpu_uuid)
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles_pickled": pickled,
            "num_params": len(names),
            "gpu_uuid": gpu_uuid,
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
        """
        import base64
        import pickle

        import deepspeed
        from torch.multiprocessing.reductions import reduce_tensor

        p = self._param_by_name(name)

        gpu_uuid = str(torch.cuda.get_device_properties(
            torch.cuda.current_device()).uuid)

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
            "ipc_handles_pickled": base64.b64encode(
                pickle.dumps([{gpu_uuid: handle}])).decode("utf-8"),
            "num_params": 1,
            "gpu_uuid": gpu_uuid,
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
        logger.info("Rank %d saved state dict to %s in %.2fs (%d params)",
                     self.rank, path, elapsed, num_params)
        return {"num_params": num_params, "elapsed": elapsed}

    def gather_and_save_state_dict(self, path: str) -> dict:
        """Gather ZeRO-3 partitioned params and save full state dict.

        All ranks must call this collectively.  Every parameter is wrapped
        in GatheredParameters so the all-gather runs on all ranks.  Only
        rank 0 copies the full tensor and writes to disk.
        """
        import deepspeed
        t0 = time.monotonic()
        model = self.engine.module
        weights = []
        for n, p in model.named_parameters():
            if hasattr(p, "ds_id"):
                with deepspeed.zero.GatheredParameters([p], enabled=True):
                    if self.rank == 0:
                        weights.append((n, p.data.cpu()))
            else:
                if self.rank == 0:
                    weights.append((n, p.data.cpu()))
        if self.rank == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(weights, path)
        num_params = len(weights)
        del weights
        import gc
        gc.collect()
        if hasattr(self.engine, 'empty_partition_cache'):
            self.engine.empty_partition_cache()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - t0
        logger.info("Rank %d gathered+saved state dict in %.2fs (%d params)",
                     self.rank, elapsed, num_params)
        return {"num_params": num_params, "elapsed": elapsed}

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
        if opt is not None and hasattr(opt, 'single_partition_of_fp32_groups'):
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
            self._ds_offload(include=[
                OffloadStateTypeEnum.hp_params,
                OffloadStateTypeEnum.lp_params,
                OffloadStateTypeEnum.lp_grads,
                OffloadStateTypeEnum.contiguous_grad_buffer,
            ])
        else:
            self._move_params(self.cpu_device)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self._on_gpu = False
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded to CPU in %.2fs (%.0f MB GPU remaining)",
                     self.rank, elapsed, mem_mb)
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
        self._ds_offload(include=[
            OffloadStateTypeEnum.hp_params,
            OffloadStateTypeEnum.lp_grads,
            OffloadStateTypeEnum.contiguous_grad_buffer,
        ])

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        elapsed = time.monotonic() - t0
        mem_mb = torch.cuda.memory_allocated() / 1e6
        logger.info("Rank %d offloaded non-lp states in %.2fs (%.0f MB GPU)",
                     self.rank, elapsed, mem_mb)
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
        logger.info("Rank %d offloaded lp_params in %.2fs (%.0f MB GPU remaining)",
                     self.rank, elapsed, mem_mb)
        return {"status": "offloaded_lp", "elapsed": elapsed, "gpu_mb": mem_mb}

    def empty_cache(self) -> dict:
        """Release ZeRO-3 partition cache and PyTorch cached memory."""
        if hasattr(self.engine, 'empty_partition_cache'):
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
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
