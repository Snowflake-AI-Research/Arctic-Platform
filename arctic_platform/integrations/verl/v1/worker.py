# Copyright 2026 Snowflake Inc.
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
"""V1 forwarder worker for the Arctic backend.

Subclasses :class:`verl.workers.engine_workers.ActorRolloutRefWorker` and
overrides only the surfaces Arctic implements itself; Ray dispatch,
profiler wiring, and worker-group plumbing come from the base class
unchanged. Reuses the V0 worker's payload helpers so the wire format to
the Arctic RL server is byte-identical to V0.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from typing import Awaitable
from typing import Optional
from typing import TypeVar

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.remote_backend.base import RemoteBackend
from verl.remote_backend.base import RemoteBackendRegistry
from verl.single_controller.base.decorator import Dispatch
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn
from verl.single_controller.base.decorator import register
from verl.utils import hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.flops_counter import FlopsCounter
from verl.utils.profiler import DistProfiler
from verl.utils.profiler import DistProfilerExtension
from verl.workers.config import ActorConfig
from verl.workers.config import HFModelConfig
from verl.workers.engine_workers import ActorRolloutRefWorker

# Eager: populates RemoteBackendRegistry in every Ray child process; the
# V0 worker supplies payload marshaling helpers reused below.
from arctic_platform.integrations.verl import adapter as _adapter  # noqa: F401
from arctic_platform.integrations.verl.worker import ArcticRLActorRolloutRefWorker as _ArcticV0Worker

T = TypeVar("T")


class _AsyncRunner:
    """Runs coroutines on a dedicated background event loop.

    V1's ``@register`` decorator stack drops coroutine-function status, so
    the ``single_controller`` machinery binds worker methods with the sync
    fast-path (``ray.get(handle.method.remote(...))``). We expose
    ``compute_log_prob`` / ``update_actor`` / ``update_weights`` as sync
    methods and drive the async backend RPCs on this persistent loop
    (avoids per-call ``asyncio.run`` teardown of the Ray client's loop state).
    """

    _instance: Optional["_AsyncRunner"] = None

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="ArcticV1WorkerAsyncLoop",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def get(cls) -> "_AsyncRunner":
        if cls._instance is None:
            cls._instance = _AsyncRunner()
        return cls._instance

    def run(self, coro: Awaitable[T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


def _njt_to_left_padded(njt: torch.Tensor, pad_value, max_len: int) -> torch.Tensor:
    """Densify a nested-jagged tensor into ``[B, max_len]`` left-padded with ``pad_value``."""
    offsets = njt.offsets().tolist()
    values = njt.values()
    batch = len(offsets) - 1
    dense = values.new_full((batch, max_len), fill_value=pad_value)
    for b in range(batch):
        seg = values[offsets[b] : offsets[b + 1]]
        length = seg.shape[0]
        if length == 0:
            continue
        dense[b, max_len - length :] = seg
    return dense


def _njt_to_right_padded(njt: torch.Tensor, pad_value, max_len: int) -> torch.Tensor:
    """Densify a nested-jagged tensor into ``[B, max_len]`` right-padded with ``pad_value``."""
    offsets = njt.offsets().tolist()
    values = njt.values()
    batch = len(offsets) - 1
    dense = values.new_full((batch, max_len), fill_value=pad_value)
    for b in range(batch):
        seg = values[offsets[b] : offsets[b + 1]]
        length = seg.shape[0]
        if length == 0:
            continue
        dense[b, :length] = seg
    return dense


# Response-aligned fields that get left-padded (or right-padded onto a
# ``response_length`` tensor). Values are ``(pad_value, dtype_default)``;
# ``dtype_default=None`` means keep the tensor's own dtype.
_RESPONSE_FIELDS: dict[str, tuple[float, Optional[torch.dtype]]] = {
    "response_mask": (0, None),
    "loss_mask": (0, None),
    "old_log_probs": (0.0, None),
    "advantages": (0.0, None),
    "ref_log_prob": (0.0, None),
    "rollout_is_weights": (1.0, None),
}


def _to_v0_padded_batch(
    data: TensorDict,
    *,
    pad_token_id: int,
    max_prompt_len: int,
    max_response_len: int,
) -> TensorDict:
    """Convert a V1 nested-jagged batch into V0's dense-padded shape.

    V0's Arctic adapter and the Arctic-side load balancer index the wire
    payload with Python ``list[int]`` slices (unsupported on NJT), so we
    densify here to the V0 layout:

    * ``prompts``: ``[B, max_prompt_len]`` left-padded with ``pad_token_id``.
    * ``responses`` / response-aligned fields: ``[B, max_response_len]``
      right-padded.
    * ``attention_mask``: ``[B, max_prompt_len + max_response_len]`` dense.
    * ``input_ids`` / ``position_ids`` are left to the V0 helper
      ``_no_padding_2_padding_prompt_response``.

    ``max_prompt_len`` / ``max_response_len`` MUST be the CONFIG maxes
    (``config.data.{max_prompt_length,max_response_length}``), not batch-
    local maxes: ZoRRo's :class:`Qwen3ModelOncePatcher` derives
    ``prompt_len = input_ids.shape[1] - config.max_response_length`` on
    every forward, so padding to a batch-local response max sends its
    response slice into the prompt column and
    ``extract_unpadded_responses_from_deduped_packed_ids`` returns the
    wrong span. No-op for fields that are already dense.
    """
    prompts = data["prompts"]
    responses = data["responses"]

    if not prompts.is_nested and not responses.is_nested:
        # Already V0-shaped (dense batch); trust the caller.
        return data

    prompt_lens = prompts.offsets().diff()
    response_lens = responses.offsets().diff()

    out = data.clone(recurse=False)

    out["prompts"] = _njt_to_left_padded(prompts, pad_token_id, max_prompt_len)
    out["responses"] = _njt_to_right_padded(responses, pad_token_id, max_response_len)

    # Rebuild a dense left-pad-prompt / right-pad-response mask sized to
    # the config maxes so ZoRRo's post-fwd unpad reads the right slice.
    batch = prompt_lens.shape[0]
    seq_len = max_prompt_len + max_response_len
    device = prompts.values().device
    mask = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
    prompt_lens_cpu = prompt_lens.tolist()
    response_lens_cpu = response_lens.tolist()
    for b in range(batch):
        pl = int(prompt_lens_cpu[b])
        rl = int(response_lens_cpu[b])
        mask[b, max_prompt_len - pl : max_prompt_len] = 1
        mask[b, max_prompt_len : max_prompt_len + rl] = 1
    out["attention_mask"] = mask

    for key, (pad_val, _dtype) in _RESPONSE_FIELDS.items():
        if key not in out.keys():
            continue
        val = out[key]
        if getattr(val, "is_nested", False):
            out[key] = _njt_to_right_padded(val, pad_val, max_response_len)

    return out


class ArcticV1ActorRolloutRefWorker(ActorRolloutRefWorker):
    """CPU-only V1 forwarder that drives an Arctic :class:`RemoteBackend`.

    Single-instance worker group (``n_gpus_per_node * nnodes == 1``); the
    backend owns its own training / sampling / log-prob GPUs. Overrides
    only ``init_model`` (rank-0 mesh registration; Arctic already owns
    engines/servers), the compute/update methods (forward to the backend,
    reusing V0 payload marshaling), and the weight-sync + checkpoint
    hooks. ``to`` / ``set_loss_fn`` / ``reset`` / ``load_checkpoint`` are
    no-ops (backend owns state).
    """

    def __init__(
        self,
        config: DictConfig,
        role: str,
        distillation_config: Optional[Any] = None,
        *,
        main_config: DictConfig,
        backend_handle: dict,
        **kwargs,
    ):
        # Skip ActorRolloutRefWorker.__init__: it reads megatron/veomni
        # router-replay + profiler fields Arctic doesn't populate. Set the
        # base state we need explicitly.
        from verl.single_controller.base import Worker

        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.role = role
        self.actor = None
        self.ref = None
        self.rollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        # Routing decisions happen inside the Arctic inference engine.
        self.enable_routing_replay = False

        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=None, tool_config=None))

        # --- Arctic-specific setup -------------------------------------- #

        self.main_config: DictConfig = main_config

        # CONFIG maxes for the V1->V0 densifier; see ``_to_v0_padded_batch``.
        self._config_max_prompt_len: int = int(self.main_config.data.max_prompt_length)
        self._config_max_response_len: int = int(self.main_config.data.max_response_length)

        backend_name = self.main_config.trainer.get("remote_backend")
        if backend_name is None:
            raise ValueError(
                "ArcticV1ActorRolloutRefWorker requires main_config.trainer.remote_backend to be set. "
                "Use the Hydra `remote_backend=arctic` group choice or set the field explicitly."
            )

        # Zorro fast path emits response-aligned model output; the forwarder
        # shifts it back to "predict-next" before writing to TransferQueue so
        # the pad->njt helper uses one uniform slice.
        self.zorro_train_enable = bool(
            OmegaConf.select(
                self.main_config,
                "remote_backend.train.zorro_train.enable",
                default=False,
            )
        )

        backend_cls = RemoteBackendRegistry.get(backend_name)
        self.backend: RemoteBackend = backend_cls.from_config(self.main_config, handle=backend_handle)

        if self._is_actor:
            self.model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
            self.actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            self.actor_config.model_config = self.model_config

            trust_remote_code = self.config.model.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(self.config.model.path, trust_remote_code=trust_remote_code)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

            self.flops_counter = FlopsCounter(self.model_config.hf_config)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Register mesh dispatch metadata for actor / ref / rollout. With the
        # single-forwarder invariant (n_gpus_per_node * nnodes == 1) rank 0
        # is the only rank; dp_rank=0 and is_collect=True mirror the V0 setup.
        self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("ref", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        return  # backend owns device residency

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        return  # loss is owned by backend (or sent in-band per call)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        return  # backend owns engine state

    # ------------------------------------------------------------------ #
    # Core dispatched ops
    # ------------------------------------------------------------------ #
    # Reuse the V0 worker's `_run_log_prob` / `_run_update_actor` payload
    # marshaling to keep the wire between forwarder and Arctic server
    # byte-identical to V0. Only the dispatch decorators + method
    # signatures change to satisfy the V1 base-class contract.

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        data = _to_v0_padded_batch(
            data,
            pad_token_id=self.pad_token_id,
            max_prompt_len=self._config_max_prompt_len,
            max_response_len=self._config_max_response_len,
        )
        return _AsyncRunner.get().run(_ArcticV0Worker._run_log_prob(self, data, ref=True, calculate_entropy=False))

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        data = _to_v0_padded_batch(
            data,
            pad_token_id=self.pad_token_id,
            max_prompt_len=self._config_max_prompt_len,
            max_response_len=self._config_max_response_len,
        )
        return _AsyncRunner.get().run(_ArcticV0Worker._run_log_prob(self, data, ref=False, calculate_entropy=True))

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: TensorDict) -> TensorDict:
        data = _to_v0_padded_batch(
            data,
            pad_token_id=self.pad_token_id,
            max_prompt_len=self._config_max_prompt_len,
            max_response_len=self._config_max_response_len,
        )
        return _AsyncRunner.get().run(_ArcticV0Worker._run_update_actor(self, data))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def update_weights(self, global_steps: int = None, mode: str = "auto"):
        # mode is ignored: CheckpointEngineManager only dispatches this on the
        # 'remote_backend' short-circuit path, where the backend owns transfer.
        # blocking=False mirrors the built-in ActorRolloutRefWorker.update_weights
        # so CheckpointEngineManager can wrap the resulting ObjectRef in ray.get.
        _AsyncRunner.get().run(self.backend.update_weights())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert self._is_actor, "save_checkpoint only supported on actor role"
        _AsyncRunner.get().run(self.backend.save_checkpoint())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        # Arctic reloads at driver startup via `create_arctic_rl_client`; the
        # per-worker load path is a no-op. The trainer only calls this on
        # `trainer.resume_mode != "disable"`, and only when a `global_step_*`
        # dir exists; ignoring the call there is safe.
        return

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def destroy(self):
        if self.backend is not None:
            _AsyncRunner.get().run(self.backend.destroy())
            self.backend = None
