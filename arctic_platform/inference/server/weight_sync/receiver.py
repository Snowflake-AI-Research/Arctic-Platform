"""WeightSyncExtension — vLLM worker extension for receiving weights.

Registered via ``--worker-extension-cls arctic_platform.inference.server.weight_sync.WeightSyncExtension``

Provides two entry points:
  1. ``sync_weights()``      — for the main (base) model.
  2. ``sync_spec_weights()`` — for the spec (drafter) model.

Each lazily creates / reuses a persistent ``NCCLEngine`` and receives
weight tensors via NCCL.  Loading strategies for the base model:
  - **direct_zero_copy** (BF16, TP=1): receive straight into parameter views
  - **direct** (BF16, TP=1, fallback): bucket receive + copy into param views
  - **fp8**: bucket receive + FP8 in-place quantization
  - **batched**: bucket receive + model.load_weights for TP slicing

Spec (drafter) weight sync defaults to the **hotswap** strategy (continue
serving while syncing) because the drafter model is small enough that sync
completes quickly, and updating spec weights never affects the base model's
output correctness — at worst, speculative acceptance rate degrades
transiently.
"""

from __future__ import annotations

import logging
import time

import torch

from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.inference.utils import require_supported_vllm_version

logger = logging.getLogger(__name__)


def _tensor_l2_sq(value: torch.Tensor, chunk_numel: int = 1 << 20) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=value.device)
    # reshape (not view) so non-contiguous parameters still checksum.
    for chunk in value.detach().reshape(-1).split(chunk_numel):
        total.add_(chunk.float().square().sum(dtype=torch.float64))
    return total


def _model_parameter_l2(
    model,
    device,
    loaded_destinations: set[str] | None = None,
    *,
    collect: bool = False,
) -> tuple[
    float,
    float | None,
    dict[str, float] | None,
    dict[str, float] | None,
]:
    """Measure the full model and an optional unique destination subset."""
    full_l2_sq = torch.zeros((), dtype=torch.float64, device=device)
    loaded_l2_sq = torch.zeros((), dtype=torch.float64, device=device) if loaded_destinations is not None else None
    names: list[str] = []
    values: list[torch.Tensor] = []
    found: set[str] = set()

    for name, parameter in model.named_parameters():
        value = _tensor_l2_sq(parameter)
        full_l2_sq.add_(value)
        if collect:
            names.append(name)
            values.append(value)
        if loaded_destinations is not None and name in loaded_destinations:
            loaded_l2_sq.add_(value)
            found.add(name)

    missing = loaded_destinations - found if loaded_destinations is not None else set()
    if missing:
        sample = sorted(missing)[:8]
        raise RuntimeError(f"Weight sync resolved destination names that are not model parameters: {sample}")

    param_l2 = None
    loaded_param_l2 = None
    if collect:
        param_l2 = dict(zip(names, torch.stack(values).tolist() if values else []))
        if loaded_destinations is not None:
            loaded_param_l2 = {name: value for name, value in param_l2.items() if name in loaded_destinations}
    return (
        full_l2_sq.item(),
        loaded_l2_sq.item() if loaded_l2_sq is not None else None,
        param_l2,
        loaded_param_l2,
    )


def _canonicalize_fused_moe_destination(name: str, param_names: set[str]) -> str:
    """Map FusedMoE wire / load_weights names onto runtime Parameter names.

    vLLM MiniMax (and Qwen MoE) store experts at ``experts.routed_experts.w13_weight``,
    but ``model.load_weights`` reports ``experts.w13_weight``. Destination
    validation compares against ``named_parameters()`` and would otherwise fail.
    """
    if name in param_names:
        return name
    for short, long in (
        (".experts.w13_weight", ".experts.routed_experts.w13_weight"),
        (".experts.w2_weight", ".experts.routed_experts.w2_weight"),
    ):
        if short in name:
            alt = name.replace(short, long, 1)
            if alt in param_names:
                return alt
    return name


def _canonicalize_destinations(destinations: list[str], param_names: set[str]) -> list[str]:
    return sorted({_canonicalize_fused_moe_destination(name, param_names) for name in destinations})


def _loaded_destination_names(
    source_to_destinations: dict[str, list[str]],
) -> set[str]:
    return {destination for destinations in source_to_destinations.values() for destination in destinations}


def _served_lm_module_prefix(model) -> str:
    """Prefix under which ``model`` registers its decoder layers: ``model.`` for
    dense LMs, ``language_model.model.`` for multimodal wrappers (e.g.
    ``Qwen3_5MoeForConditionalGeneration``)."""
    if model is None:
        return "model."
    try:
        for name, _ in model.named_modules():
            i = name.find("layers.")
            if i > 0 and name[:i].endswith("model."):
                return name[:i]
    except Exception:
        pass
    return "model."


def _remap_lora_key(name: str, served_prefix: str) -> str:
    """Normalize one synced LoRA key to the served model's modules: strip
    DeepSpeed AC wraps, then remap the trainer LM root to ``served_prefix``.

    Dense PEFT emits ``base_model.model.model.layers.…``. Multimodal / MoE
    trainers (e.g. Qwen3.6) often wrap a nest and emit
    ``base_model.model.model.language_model.layers.…``. Served wrappers use
    ``language_model.model.layers.…``, so both trainer roots must map onto
    ``served_prefix``; matching ``model.`` before ``model.language_model.``
    would double the nest and bind the adapter to zero modules (silent no-op).
    No-op for dense LMs (``served_prefix`` is ``model.``).
    """
    name = name.replace("._checkpoint_wrapped_module", "")
    if served_prefix == "model.":
        return name
    head = "base_model.model." if name.startswith("base_model.model.") else ""
    body = name[len(head) :]
    # Longer trainer root first so ``model.language_model.`` is not
    # partially rewritten via the ``model.`` branch.
    for trainer_root in ("model.language_model.", "model."):
        if body.startswith(trainer_root):
            body = served_prefix + body[len(trainer_root) :]
            break
    return head + body


_FUSED_EXPERT_PROJS = ("w1", "w2", "w3")


def _pack_fused_expert_loras(lora_model) -> int:
    """Collapse stacked ``experts.w{1,2,3}`` into one ``pack_moe_stacked`` entry.

    Trainer broadcasts one 3-D tensor per projection instead of one 2-D
    tensor per expert. ``add_adapter`` then finds the packed ``...experts``
    key and skips the per-expert ``pack_moe`` expansion.
    """
    from vllm.lora.lora_weights import PackedLoRALayerWeights

    grouped: dict[str, dict[str, tuple[str, object]]] = {}
    for name, lora in lora_model.loras.items():
        for proj in _FUSED_EXPERT_PROJS:
            suffix = f".experts.{proj}"
            if name.endswith(suffix):
                experts_mod = name[: -len(f".{proj}")]
            elif name == f"experts.{proj}":
                experts_mod = "experts"
            else:
                continue
            grouped.setdefault(experts_mod, {})[proj] = (name, lora)
            break

    packed = 0
    for experts_mod, parts in grouped.items():
        missing = [p for p in _FUSED_EXPERT_PROJS if p not in parts]
        if missing:
            have = sorted(parts)
            raise RuntimeError(f"Fused expert LoRA for {experts_mod!r} is missing {missing}; have {have}")
        lora_model.loras[experts_mod] = PackedLoRALayerWeights.pack_moe_stacked(
            [parts[p][1] for p in _FUSED_EXPERT_PROJS],
            experts_mod,
        )
        for proj in _FUSED_EXPERT_PROJS:
            lora_model.loras.pop(parts[proj][0], None)
        packed += 1
    return packed


def _build_lora_model(
    lora_int_id: int,
    tensors: "dict[str, torch.Tensor]",
    lora_config: dict,
    device: torch.device,
    dtype,
    model=None,
):
    """Build a vLLM ``LoRAModel`` from received adapter tensors + a small config.

    ``lora_config`` is validated before weight transfer begins.

    MoE: tag ``is_3d_lora_weight`` only for PEFT 3D gate_up/down keys; leave
    False for stacked ``experts.w{1,2,3}`` (``pack_moe_stacked``) and for
    per-expert 2D ``gate_proj|down_proj|up_proj`` (``pack_moe``).
    """
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    # Central point for all LoRA key rewriting: strip AC wraps and remap the LM
    # root to the served model's namespace (see ``_remap_lora_key``).
    prefix = _served_lm_module_prefix(model)
    tensors = {_remap_lora_key(name, prefix): tensor for name, tensor in tensors.items()}

    # Dropout affects training only. target_parameters is represented by the
    # received tensor names and vLLM's mixed-MoE LoRA mode, not PEFTHelper.
    peft_helper = PEFTHelper.from_dict(
        {
            "peft_type": lora_config["peft_type"],
            "task_type": lora_config["task_type"],
            "r": lora_config["r"],
            "lora_alpha": lora_config["lora_alpha"],
            "bias": lora_config["bias"],
            "target_modules": lora_config["target_modules"],
        }
    )
    # Unstacked mapper keeps PEFT names (q_a_proj, …) for PackedLoRA packing.
    # vLLM 0.29 renamed get_unstacked_mapper to get_rename_mapper.
    weights_mapper = None
    if model is not None:
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        if hf_to_vllm_mapper is not None:
            weights_mapper = hf_to_vllm_mapper.get_rename_mapper()
    lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None) if model is not None else None
    lora_model = LoRAModel.from_lora_tensors(
        lora_model_id=lora_int_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device=str(device),
        dtype=dtype,
        weights_mapper=weights_mapper,
        skip_prefixes=lora_skip_prefixes,
    )
    packed_layers = _pack_fused_expert_loras(lora_model)
    if packed_layers:
        logger.info(
            "Packed %d fused expert LoRA layer(s) via pack_moe_stacked",
            packed_layers,
        )
    # 3D only for single-nest experts.base_layer (not ParamWrapper nests / 2D).
    has_double_base_nest = any(".base_layer.base_layer." in name for name in tensors)
    has_3d_gate_up = any(".experts.base_layer." in name for name in tensors)
    has_per_expert_2d = any(
        (".experts." in name) and any(f".{p}." in name for p in ("gate_proj", "down_proj", "up_proj"))
        for name in tensors
    )
    if has_3d_gate_up and not has_double_base_nest and not has_per_expert_2d:
        lora_model.is_3d_lora_weight = True
    return lora_model


def _install_lora_adapter(adapter_manager, lora_model, lora_int_id: int) -> tuple[bool, bool]:
    """Register and activate one adapter, failing if vLLM rejects either step."""
    added = bool(adapter_manager.add_adapter(lora_model))
    if not added:
        raise RuntimeError(f"vLLM failed to add synced LoRA adapter id {lora_int_id}")
    activated = bool(adapter_manager.activate_adapter(lora_int_id))
    if not activated:
        raise RuntimeError(f"vLLM failed to activate synced LoRA adapter id {lora_int_id}")
    return added, activated


def _ensure_engine_process_vllm_patches() -> None:
    require_supported_vllm_version("WeightSyncExtension")

    from arctic_platform.inference.vllm.router_replay import ensure_router_replay_vllm_patches
    from arctic_platform.inference.vllm.xgrammar_stop_mask import ensure_xgrammar_stop_mask_fix

    ensure_router_replay_vllm_patches()
    ensure_xgrammar_stop_mask_fix()


_ensure_engine_process_vllm_patches()


class WeightSyncExtension:
    """vLLM worker extension for NCCL weight sync.

    Usage:
        --worker-extension-cls arctic_platform.inference.server.weight_sync.WeightSyncExtension
    """

    def _arl_cuda_sync(self) -> dict:
        torch.cuda.synchronize(self.device)
        return {"status": "ok"}

    def cleanup_router_replay_shm(self) -> dict:
        return {
            "status": "unsupported",
            "reason": "vllm_0_23plus_uses_routed_experts_manager",
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_or_create_engine(
        self,
        master_addr: str,
        master_port: int,
        bucket_size: int,
        reverse: bool,
        *,
        engine_attr: str,
        key_attr: str,
        label: str = "",
    ):
        """Return a (possibly cached) NCCLEngine for the given config.

        *engine_attr* / *key_attr* are the instance attribute names used
        to cache the engine and its key, allowing separate engines for
        base and spec models.
        """
        from vllm.distributed.parallel_state import get_world_group

        from arctic_platform.inference.server.weight_sync.engine import NCCLEngine

        tp_rank = get_world_group().rank
        my_port = master_port + tp_rank
        engine_key = (master_addr, my_port)

        engine = getattr(self, engine_attr, None)
        if engine is None or getattr(self, key_attr, None) != engine_key:
            if engine is not None:
                engine.destroy()
            logger.info(
                "Creating %sNCCLEngine rank=1 ws=2 port=%d tp_rank=%d bucket=%dMB",
                label,
                my_port,
                tp_rank,
                bucket_size // (1024 * 1024),
            )
            engine = NCCLEngine(
                master_addr=master_addr,
                master_port=my_port,
                rank=1,
                world_size=2,
                device=self.device,
                bucket_size=bucket_size,
                reverse=reverse,
            )
            setattr(self, engine_attr, engine)
            setattr(self, key_attr, engine_key)

        return engine

    def _build_result(self, start: float, mem_before: int, path: str, loaded: int) -> dict:
        """Synchronize CUDA and return timing / memory metrics."""
        torch.cuda.synchronize(self.device)
        peak_mem = torch.cuda.max_memory_allocated(self.device)
        mem_after = torch.cuda.memory_allocated(self.device)
        elapsed = time.time() - start

        return {
            "status": "done",
            "params_loaded": loaded,
            "elapsed": elapsed,
            "update_path": path,
            "mem_before_bytes": mem_before,
            "mem_peak_bytes": peak_mem,
            "mem_after_bytes": mem_after,
            "mem_extra_bytes": peak_mem - mem_before,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sync_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        direct_mode: bool = False,
        reverse: bool = False,
    ) -> dict:
        """Receive weights via NCCL and load them into the model.

        On the first call, a persistent ``NCCLEngine`` is created (NCCL
        rendezvous blocks until the sender also creates its engine).
        Subsequent calls with the same config reuse the existing engine.

        If *engine_only* is True, only the NCCL rendezvous is performed.
        If *direct_mode* is True, per-weight send/recv is used (BF16 TP=1).
        """
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

        engine = self._get_or_create_engine(
            master_addr,
            master_port,
            bucket_size,
            reverse,
            engine_attr="_ws_engine",
            key_attr="_ws_engine_key",
        )

        if engine_only:
            return {"status": "engine_ready", "rank": 1, "world_size": 2}

        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        tp = get_tensor_model_parallel_world_size()

        if self.model_config.quantization:
            path = "fp8"
            loaded, _ = self._load_fp8(model, engine)
        elif tp == 1 and direct_mode:
            path = "direct_zero_copy"
            loaded = self._load_direct_zero_copy(model, engine)
        elif tp == 1:
            path = "direct"
            loaded, _, _, _, _ = self._load_direct(model, engine)
        else:
            path = "batched"
            loaded, _, _, _, _ = self._load_batched(model, engine)

        return self._build_result(start, mem_before, path, loaded)

    # ------------------------------------------------------------------
    # Loading strategies
    # ------------------------------------------------------------------

    def _load_fp8(self, model, engine) -> tuple[int, float]:
        from arctic_platform.inference.server.weight_sync.utils import _FP8InplaceUpdater

        updater = _FP8InplaceUpdater(
            model,
            self.model_config.dtype,
            self.device,
        )
        loaded = 0
        recv_l2_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, tensor in engine.receive_weights():
            recv_l2_sq.add_(_tensor_l2_sq(tensor))
            updater.feed(name, tensor)
            loaded += 1
        if updater.pending:
            logger.warning(
                "FP8 updater has %d incomplete modules",
                updater.pending,
            )
        return loaded, recv_l2_sq.item()

    def _load_direct_zero_copy(self, model, engine) -> int:
        """True zero-copy: receive each weight directly into its parameter view.

        Uses ``engine.receive_weights_direct()`` so that NCCL writes bytes
        straight into the model's parameter storage — no intermediate buffers
        or copies.  Requires BF16, TP=1.
        """
        from arctic_platform.inference.server.weight_sync.utils import _DirectParamWriter

        writer = _DirectParamWriter(model, self.device)
        param_views: dict[str, torch.Tensor] = {}
        for name in writer.all_keys():
            v = writer.get_view(name)
            if v is not None:
                param_views[name] = v

        result = engine.receive_weights_direct(param_views)
        orphan_count = int(result.get("orphan", 0))
        if orphan_count:
            raise AssertionError(
                "Weight sync (_load_direct_zero_copy) received "
                f"{orphan_count}/{result.get('params_loaded', 0)} tensor(s) "
                "whose names do not match any vLLM parameter view"
            )
        return result.get("params_loaded", 0)

    def _load_direct(
        self,
        model,
        engine,
    ) -> tuple[int, float, float, dict[str, list[str]], list[str]]:
        """Bucket path for TP=1 non-quantized models.

        Copies from the engine's bucket buffer into pre-computed parameter views.
        Returns the received count, all-byte and applied-source squared-L2
        norms, canonical destination mapping, and skipped names. Direct loading
        never skips a source: unresolved names fail after the receive completes.
        """
        from arctic_platform.inference.server.weight_sync.utils import _DirectParamWriter

        writer = _DirectParamWriter(model, self.device)
        loaded = 0
        orphans: list[str] = []
        source_to_destinations: dict[str, list[str]] = {}
        recv_l2_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, tensor in engine.receive_weights():
            recv_l2_sq.add_(_tensor_l2_sq(tensor))
            view = writer.get_view(name)
            destination = writer.get_destination(name)
            if view is not None and destination is not None:
                view.copy_(tensor)
                source_to_destinations[name] = [destination]
            else:
                orphans.append(name)
            loaded += 1

        if orphans:
            sample = "\n".join(f"  - {n}" for n in orphans[:10])
            more = f"\n  ... and {len(orphans) - 10} more" if len(orphans) > 10 else ""
            raise AssertionError(
                f"Weight sync (_load_direct) received {len(orphans)}/{loaded} "
                "tensor(s) whose names do not match any vLLM parameter view -- "
                "they were dropped on the floor, which corrupts the inference "
                "model.\n\n"
                f"Sample orphan names:\n{sample}{more}\n\n"
            )
        received_value = recv_l2_sq.item()
        return (
            loaded,
            received_value,
            received_value,
            source_to_destinations,
            [],
        )

    def _load_batched(
        self,
        model,
        engine,
    ) -> tuple[int, float, float, dict[str, list[str]], list[str]]:
        """Batched path for TP>1 models (including FP8+TP>1) and HF-named sync.

        Each received tensor is a full (un-sharded) weight.  Most params are
        handed to ``model.load_weights``, which does TP slicing, HF ->
        vLLM-internal name/shape conversion, and quantization.

        The vLLM-*fused* families that ``model.load_weights`` cannot place when
        fed already-fused names (Gated DeltaNet ``in_proj_qkvz``/``in_proj_ba``
        and FusedMoE ``experts.w13_weight``/``w2_weight``) are intercepted by
        :class:`_ShardAwareFusedWriter`, which drives each param's own
        ``weight_loader`` to shard correctly.  Non-fused names fall through
        untouched, so the HF-named (``weight_format="hf"``) path is unaffected.

        Returns the received count, all-byte and applied-source squared-L2
        norms, canonical destination mapping, and skipped source names.
        """
        from arctic_platform.inference.server.weight_sync.utils import _ShardAwareFusedWriter

        writer = _ShardAwareFusedWriter(model, self.device)
        loaded = 0
        source_to_destinations: dict[str, list[str]] = {}
        skipped_source_names: list[str] = []
        param_names = {pname for pname, _ in model.named_parameters()}
        recv_l2_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        applied_source_l2_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, tensor in engine.receive_weights():
            source_l2_sq = _tensor_l2_sq(tensor)
            recv_l2_sq.add_(source_l2_sq)
            destination = writer.destination_name(name)
            if destination is not None:
                writer.feed(name, tensor)
                destinations = [destination]
            else:
                loaded_names = model.load_weights([(name, tensor)])
                if isinstance(loaded_names, str):
                    destinations = [loaded_names]
                else:
                    destinations = sorted(set(loaded_names or ()))
            destinations = _canonicalize_destinations(destinations, param_names)
            if destinations:
                source_to_destinations[name] = destinations
                applied_source_l2_sq.add_(source_l2_sq)
            else:
                skipped_source_names.append(name)
            loaded += 1
        if skipped_source_names:
            logger.warning(
                "Weight sync skipped %d/%d received source tensor(s) because "
                "model.load_weights reported no destination on this rank",
                len(skipped_source_names),
                loaded,
            )
        return (
            loaded,
            recv_l2_sq.item(),
            applied_source_l2_sq.item(),
            source_to_destinations,
            skipped_source_names,
        )

    # ------------------------------------------------------------------
    # Broadcast-mode weight sync (single world_size > 2 NCCL group)
    # ------------------------------------------------------------------

    def _get_or_create_broadcast_engine(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        bucket_size: int,
    ):
        from arctic_platform.inference.server.weight_sync.broadcast import BroadcastNCCLEngine

        key = (master_addr, master_port, rank, world_size)
        engine = getattr(self, "_ws_bcast_engine", None)
        if engine is None or getattr(self, "_ws_bcast_engine_key", None) != key:
            if engine is not None:
                engine.destroy()
            engine = BroadcastNCCLEngine(
                master_addr=master_addr,
                master_port=master_port,
                rank=rank,
                world_size=world_size,
                device=self.device,
                bucket_size=bucket_size,
            )
            self._ws_bcast_engine = engine
            self._ws_bcast_engine_key = key
        return engine

    def sync_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        weight_format: str = "vllm",
    ) -> dict:
        """Join the broadcast group and load weights from rank 0.

        ``weight_format`` selects how received tensors are written into the
        running vLLM model:

        - ``"vllm"`` (default): tensor names match
          ``vllm_model.named_parameters()``
        - ``"hf"``: tensor names match the HuggingFace checkpoint layout

        When ``envs.ARCTIC_INFERENCE_DUMP_PARAM_L2`` is set, the result also
        includes full-model and loaded-only per-parameter L2 values, the
        source-to-destination mapping, and skipped source names.
        """
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
        from vllm.distributed.parallel_state import get_world_group

        if weight_format not in ("vllm", "hf"):
            raise ValueError(f"weight_format must be 'vllm' or 'hf'; got {weight_format!r}")

        rank = rank_offset + get_world_group().rank
        engine = self._get_or_create_broadcast_engine(
            master_addr,
            master_port,
            rank,
            world_size,
            bucket_size,
        )

        if engine_only:
            return {
                "status": "engine_ready",
                "rank": rank,
                "world_size": world_size,
                "weight_format": weight_format,
            }

        start = time.time()
        model = self.model_runner.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        # Opt-in weight-sync correctness diagnostic, env-gated (default off).
        from arctic_platform.inference import envs

        dump_param_l2 = envs.ARCTIC_INFERENCE_DUMP_PARAM_L2
        loaded_destination_validation_applicable = not bool(self.model_config.quantization)

        model_l2_sq_before, _, _, _ = _model_parameter_l2(model, self.device)

        recv_l2_sq: float | None = None
        applied_source_l2_sq: float | None = None
        source_to_destinations: dict[str, list[str]] | None = None
        skipped_source_names: list[str] | None = None
        if weight_format == "hf":
            path = "batched"
            (
                loaded,
                recv_l2_sq,
                applied_source_l2_sq,
                source_to_destinations,
                skipped_source_names,
            ) = self._load_batched(model, engine)
        elif self.model_config.quantization:
            path = "fp8"
            loaded, recv_l2_sq = self._load_fp8(model, engine)
        elif get_tensor_model_parallel_world_size() == 1:
            path = "direct"
            (
                loaded,
                recv_l2_sq,
                applied_source_l2_sq,
                source_to_destinations,
                skipped_source_names,
            ) = self._load_direct(model, engine)
        else:
            path = "batched"
            (
                loaded,
                recv_l2_sq,
                applied_source_l2_sq,
                source_to_destinations,
                skipped_source_names,
            ) = self._load_batched(model, engine)

        if loaded_destination_validation_applicable and source_to_destinations is not None:
            loaded_destinations = _loaded_destination_names(source_to_destinations)
            (
                model_l2_sq_after,
                loaded_model_l2_sq_after,
                param_l2_after,
                loaded_param_l2_after,
            ) = _model_parameter_l2(
                model,
                self.device,
                loaded_destinations,
                collect=dump_param_l2,
            )
        else:
            # Quantized sync is outside loaded-destination validation.
            loaded_destinations = None
            loaded_model_l2_sq_after = None
            (
                model_l2_sq_after,
                _,
                param_l2_after,
                loaded_param_l2_after,
            ) = _model_parameter_l2(
                model,
                self.device,
                collect=dump_param_l2,
            )

        loaded_destination_trace_collected = bool(
            dump_param_l2
            and loaded_destination_validation_applicable
            and source_to_destinations is not None
            and skipped_source_names is not None
        )

        result = self._build_result(start, mem_before, path, loaded)
        result["weight_format"] = weight_format
        result["model_l2_sq_before"] = model_l2_sq_before
        result["model_l2_sq_after"] = model_l2_sq_after
        result["loaded_destination_validation_applicable"] = loaded_destination_validation_applicable
        result["loaded_destination_trace_collected"] = loaded_destination_trace_collected
        if loaded_destinations is not None:
            result["loaded_model_l2_sq_after"] = loaded_model_l2_sq_after
            result["loaded_parameter_count"] = len(loaded_destinations)
            result["applied_source_l2_sq"] = applied_source_l2_sq
            result["skipped_source_count"] = len(skipped_source_names or ())
        if dump_param_l2:
            result["param_l2"] = param_l2_after
        if loaded_destination_trace_collected:
            result["loaded_param_l2"] = loaded_param_l2_after
            result["source_to_destinations"] = source_to_destinations
            result["skipped_source_names"] = skipped_source_names
        if recv_l2_sq is not None:
            result["recv_l2_sq"] = recv_l2_sq
        return result

    # ------------------------------------------------------------------
    # LoRA adapter weight sync (broadcast)
    # ------------------------------------------------------------------

    def _lora_adapter_manager(self):
        """vLLM's ``LoRAModelManager``. Present only when the engine started with
        ``enable_lora`` (sampling job set ``inference_config.peft_config``).
        """
        lora_manager = getattr(self.model_runner, "lora_manager", None)
        if lora_manager is None:
            raise RuntimeError(
                "LoRA sync called but the engine wasn't started with LoRA "
                "enabled. Set inference_config.peft_config on the sampling job."
            )
        adapter_manager = getattr(lora_manager, "_adapter_manager", None)
        if adapter_manager is None:
            raise RuntimeError("vLLM LoRA manager not created (model not loaded yet?).")
        return adapter_manager

    def sync_lora_weights_broadcast(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        lora_int_id: int,
        lora_name: str,
        lora_config: dict,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        staging: str = "cpu",
        evict_first: bool = True,
    ) -> dict:
        """Receive adapter tensors, build a ``LoRAModel`` under ``lora_int_id``,
        and inject it into the resident LoRA manager (no disk round-trip).
        Serving still needs a matching ``LoRARequest`` (see ``_active_lora_request``).

        The stable adapter id is removed and repopulated on every sync, matching
        vLLM's in-place LoRA reload semantics without loading from disk.
        ``evict_first=False`` (hotswap) replaces the adapter after receive.
        """
        from vllm.distributed.parallel_state import get_world_group

        lora_config = normalize_lora_peft_config(lora_config, location="lora_config")
        if staging not in {"cpu", "gpu"}:
            raise ValueError(f"lora_sync_staging must be 'cpu' or 'gpu', got {staging!r}")
        rank = rank_offset + get_world_group().rank
        engine = self._get_or_create_broadcast_engine(
            master_addr,
            master_port,
            rank,
            world_size,
            bucket_size,
        )

        if engine_only:
            return {
                "status": "engine_ready",
                "rank": rank,
                "world_size": world_size,
                "weight_format": "lora",
            }

        start = time.time()

        adapter_manager = self._lora_adapter_manager()
        replaced = False
        if evict_first:
            replaced = bool(adapter_manager.remove_adapter(lora_int_id))
            torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        # receive_weights() yields views into reusable NCCL buffers. Keep a
        # permanent copy before the next bucket reuses them. CPU staging avoids
        # the pack_moe GPU peak; GPU staging is faster when enough HBM is free.
        pin_memory = False
        if staging == "cpu":
            try:
                from vllm.utils.platform_utils import is_pin_memory_available

                pin_memory = bool(is_pin_memory_available())
            except ImportError:
                pass
        tensors: dict[str, torch.Tensor] = {}
        recv_l2_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        for name, tensor in engine.receive_weights():
            recv_l2_sq.add_(_tensor_l2_sq(tensor))
            if staging == "cpu":
                dest = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                dest.copy_(tensor)
                tensors[name] = dest
            else:
                tensors[name] = tensor.detach().clone()

        n_tensors = len(tensors)
        lora_model = _build_lora_model(
            lora_int_id=lora_int_id,
            tensors=tensors,
            lora_config=lora_config,
            device="cpu" if staging == "cpu" else self.device,
            dtype=self.model_config.dtype,
            model=getattr(self.model_runner, "model", None),
        )
        del tensors

        if not evict_first:
            replaced = bool(adapter_manager.remove_adapter(lora_int_id))
        added, activated = _install_lora_adapter(
            adapter_manager,
            lora_model,
            lora_int_id,
        )

        result = self._build_result(start, mem_before, "lora", n_tensors)
        result["weight_format"] = "lora"
        result["lora_int_id"] = lora_int_id
        result["lora_name"] = lora_name
        result["lora_added"] = bool(added)
        result["lora_activated"] = bool(activated)
        result["lora_replaced"] = replaced
        result["lora_removed_int_ids"] = [lora_int_id] if replaced else []
        result["lora_sync_staging"] = staging
        result["lora_evict_first"] = evict_first
        result["is_3d_lora_weight"] = bool(getattr(lora_model, "is_3d_lora_weight", False))
        result["recv_l2_sq"] = recv_l2_sq.item()

        # Free 2×bucket staging; keep the NCCL communicator for the next sync.
        engine.release_staging_buffers()
        return result

    # ------------------------------------------------------------------
    # Spec (drafter) model weight sync
    # ------------------------------------------------------------------

    def sync_spec_weights(
        self,
        master_addr: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        bucket_size: int = 256 * 1024 * 1024,
        engine_only: bool = False,
        reverse: bool = False,
    ) -> dict:
        """Receive weights via NCCL and load them into the drafter model.

        Mirrors :meth:`sync_weights` but targets the spec (drafter) model.
        Uses a separate NCCL engine instance so it can operate on a
        different port/connection.

        Spec weight sync defaults to the **hotswap** strategy: weights are
        received and loaded while inference continues uninterrupted.  This
        is safe because the drafter model is small (sync completes quickly)
        and updating it never affects the base model's output correctness —
        at worst, speculative acceptance rate degrades transiently.
        """
        drafter = getattr(self.model_runner, "drafter", None)
        if drafter is None or getattr(drafter, "model", None) is None:
            raise RuntimeError("sync_spec_weights called but no drafter model is loaded")

        engine = self._get_or_create_engine(
            master_addr,
            master_port,
            bucket_size,
            reverse,
            engine_attr="_ws_spec_engine",
            key_attr="_ws_spec_engine_key",
            label="spec ",
        )

        if engine_only:
            return {"status": "engine_ready", "rank": 1, "world_size": 2}

        start = time.time()
        spec_model = drafter.model
        torch.cuda.reset_peak_memory_stats(self.device)
        mem_before = torch.cuda.memory_allocated(self.device)

        all_weights = [(n, t.cpu()) for n, t in engine.receive_weights()]
        loaded = len(all_weights)
        spec_model.load_weights(all_weights)

        return self._build_result(start, mem_before, "batched", loaded)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close_weight_sync(self) -> dict:
        """Destroy persistent NCCLEngine instances."""
        for attr, key_attr in [
            ("_ws_engine", "_ws_engine_key"),
            ("_ws_spec_engine", "_ws_spec_engine_key"),
            ("_ws_bcast_engine", "_ws_bcast_engine_key"),
        ]:
            engine = getattr(self, attr, None)
            if engine is not None:
                engine.destroy()
                setattr(self, attr, None)
                setattr(self, key_attr, None)
        return {"status": "ok"}
