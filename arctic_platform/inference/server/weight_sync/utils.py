"""Shared utilities for weight synchronization.

Contains lightweight helpers, parameter descriptors, and model-aware
writers/updaters used by both sender and receiver code paths.
"""

from __future__ import annotations

import json
import logging
import struct

import torch
from torch import nn

logger = logging.getLogger(__name__)

_SF_DTYPE_MAP = {
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64,
    "U8": torch.uint8, "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


# ---------------------------------------------------------------------------
# NCCL group creation (shared by sender + receiver)
# ---------------------------------------------------------------------------

def stateless_init_nccl(master_addr, master_port, rank, world_size, device,
                        *, is_server=None):
    """Create an independent PyNcclCommunicator via StatelessProcessGroup.

    When *is_server* is ``None`` (default), ``rank == 0`` creates the TCP
    listener (standard behaviour).  Pass an explicit bool to decouple the
    TCP listener role from the NCCL rank — needed when the network only
    allows one direction of connectivity (e.g. training → inference).
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup, create_tcp_store

    if is_server is None:
        pg = StatelessProcessGroup.create(
            host=master_addr, port=master_port, rank=rank, world_size=world_size
        )
    else:
        import socket
        from datetime import timedelta

        listen_socket = None
        if is_server:
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind(("0.0.0.0", master_port))
            listen_socket.listen()

        store = create_tcp_store(
            master_addr,
            master_port,
            world_size=world_size,
            is_master=is_server,
            timeout=timedelta(seconds=300),
            use_libuv=False,
            listen_socket=listen_socket,
        )
        pg = StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=3600,
        )

    return PyNcclCommunicator(pg, device=device)


# ---------------------------------------------------------------------------
# WeightInfo — lightweight parameter descriptor
# ---------------------------------------------------------------------------

_DTYPE_BYTES = {
    torch.float16: 2, torch.bfloat16: 2,
    torch.float32: 4, torch.float64: 8,
    torch.int8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8,
    torch.uint8: 1, torch.bool: 1,
    torch.float8_e4m3fn: 1, torch.float8_e5m2: 1,
}


class WeightInfo:
    __slots__ = ("name", "shape", "dtype")

    def __init__(self, name: str, shape: torch.Size, dtype: torch.dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    @property
    def nbytes(self) -> int:
        numel = 1
        for s in self.shape:
            numel *= s
        return numel * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "dtype": str(self.dtype)}

    @classmethod
    def from_dict(cls, d: dict) -> WeightInfo:
        return cls(d["name"], torch.Size(d["shape"]),
                   getattr(torch, d["dtype"].replace("torch.", "")))


def build_weights_info(model_path: str) -> list[WeightInfo]:
    """Build WeightInfo list from safetensors file headers (zero-copy)."""
    import glob
    from pathlib import Path

    p = Path(model_path)
    if not p.is_dir():
        from huggingface_hub import snapshot_download
        p = Path(snapshot_download(model_path))

    sf_files = sorted(glob.glob(str(p / "*.safetensors")))
    if not sf_files:
        raise FileNotFoundError(f"No safetensors files in {p}")

    infos: list[WeightInfo] = []
    for sf in sf_files:
        with open(sf, "rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_size))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dtype = _SF_DTYPE_MAP.get(v["dtype"], torch.float32)
            infos.append(WeightInfo(k, torch.Size(v["shape"]), dtype))
    return infos


# ---------------------------------------------------------------------------
# FP8 in-place weight update helpers
# ---------------------------------------------------------------------------

_STACKED_PARAMS = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "q_a_proj": "fused_qkv_a_proj",
    "kv_a_proj_with_mqa": "fused_qkv_a_proj",
}

_SHARD_IDS = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v",
    "gate_proj": 0, "up_proj": 1,
    "q_a_proj": 0, "kv_a_proj_with_mqa": 1,
}

_SHARD_COUNTS = {"qkv_proj": 3, "gate_up_proj": 2, "fused_qkv_a_proj": 2}

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

# Wrapper levels that change a parameter's path without changing the parameter: vLLM's and PEFT's LoRA wrappers keep
# the wrapped layer as ``base_layer``, and activation checkpointing nests ``_checkpoint_wrapped_module``.
_WRAPPER_SEGMENTS = frozenset({"base_layer", "_checkpoint_wrapped_module"})


def _canonical_name(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def _canonical_index(names) -> dict[str, str]:
    """Map each canonical name to the one real name it stands for; a canonical name shared by two real names is
    ambiguous and left out."""
    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for name in names:
        key = _canonical_name(name)
        if index.setdefault(key, name) != name:
            ambiguous.add(key)
    for key in ambiguous:
        del index[key]
    return index


def _is_lora_adapter_name(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def _dequant_fp8_block(weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a block-FP8 weight with its per-block inverse scale, as vLLM's GLM-5 checkpoint loader does.

    The block edge is ``in_features / scale columns``; a partial last block on either axis is zero-padded for the
    scale broadcast and trimmed afterwards (``kv_a_proj_with_mqa`` has 576 rows = 4 * 128 + 64).
    """
    out_dim, in_dim = weight.shape
    block = in_dim // scale_inv.shape[1]
    if block <= 0 or scale_inv.shape != (-(-out_dim // block), -(-in_dim // block)):
        raise ValueError(
            f"block-FP8 scale shape {tuple(scale_inv.shape)} does not tile weight shape {tuple(weight.shape)}"
        )
    padded = torch.nn.functional.pad(weight.to(torch.float32), (0, (-in_dim) % block, 0, (-out_dim) % block))
    scale = scale_inv.to(torch.float32).repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return (padded * scale).to(dtype)[:out_dim, :in_dim].contiguous()


class _FP8InplaceUpdater:
    """Accumulates BF16 weight shards and quantizes them back to FP8 in-place.

    Preserves the existing FP8 tensor GPU addresses so that CUDA graphs
    remain valid — no enforce_eager required.  Peak temporary memory is
    one module's BF16 buffer (~100-200 MB) rather than the full model.
    """

    def __init__(self, model: nn.Module, target_dtype: torch.dtype, device):
        from vllm.model_executor.parameter import BasevLLMParameter
        from vllm.model_executor.layers.linear import (
            ColumnParallelLinear, MergedColumnParallelLinear,
            QKVParallelLinear, RowParallelLinear,
        )
        self._linear_types = (ColumnParallelLinear, MergedColumnParallelLinear,
                              QKVParallelLinear, RowParallelLinear)
        self._model = model
        self._device = device
        self._dtype = target_dtype

        self._fp8_modules: dict[str, nn.Module] = {}
        self._params: dict[str, nn.Parameter] = dict(model.named_parameters())

        for name, mod in model.named_modules():
            if not isinstance(mod, self._linear_types):
                continue
            w = getattr(mod, "weight", None)
            if (
                w is not None
                and w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and not isinstance(w, BasevLLMParameter)
            ):
                self._fp8_modules[name] = mod

        self._modules: dict[str, nn.Module] = dict(model.named_modules())
        self._param_index = _canonical_index(self._params)
        self._fp8_module_index = _canonical_index(self._fp8_modules)
        self._bufs: dict[str, nn.Parameter] = {}
        self._shards_left: dict[str, int] = {}
        # Canonical wire projection path -> the block-FP8 weight or scale received so far, its unquantized
        # destination, and the fused shard it fills.
        self._dequant: dict[str, dict] = {}
        self._unplaced: list[str] = []
        self._fused_writer = _ShardAwareFusedWriter(model, device)

    def _param_name(self, name: str) -> str | None:
        """The vLLM parameter a wire name denotes, matching across LoRA and checkpoint wrapper levels."""
        if name in self._params:
            return name
        return self._param_index.get(_canonical_name(name))

    def _fp8_module_path(self, path: str) -> str | None:
        if path in self._fp8_modules:
            return path
        return self._fp8_module_index.get(_canonical_name(path))

    def _unquantized_target(self, wire_module: str) -> tuple[str, object] | None:
        """The unquantized vLLM parameter, and fused shard, that a block-FP8 wire projection loads into."""
        name = self._param_name(f"{wire_module}.weight")
        if name is not None:
            return (name, None) if self._params[name].dtype not in _FP8_DTYPES else None
        prefix, _, leaf = _canonical_name(wire_module).rpartition(".")
        merged = _STACKED_PARAMS.get(leaf)
        if not merged or not prefix:
            return None
        name = self._param_name(f"{prefix}.{merged}.weight")
        if name is None or self._params[name].dtype in _FP8_DTYPES:
            return None
        return name, _SHARD_IDS[leaf]

    def _stage_dequant(self, wire_name: str, wire_module: str, part: str, tensor: torch.Tensor) -> bool:
        target = self._unquantized_target(wire_module)
        if target is None:
            return False
        key = _canonical_name(wire_module)
        entry = self._dequant.setdefault(key, {"names": [], "destination": target[0], "shard": target[1]})
        if part in entry:
            raise ValueError(f"block-FP8 {part} for {key!r} arrived twice before its partner")
        # Allocate, then copy: the received tensor is a view of the transport buffer.
        entry[part] = torch.empty_like(tensor)
        entry[part].copy_(tensor)
        entry["names"].append(wire_name)
        if "weight" in entry and "scale" in entry:
            del self._dequant[key]
            self._load_dequantized(entry)
        return True

    def _load_dequantized(self, entry: dict) -> None:
        param = self._params[entry["destination"]]
        weight = _dequant_fp8_block(entry["weight"], entry["scale"], param.dtype)
        shard = entry["shard"]
        if shard is None:
            self._copy_param(param, weight)
            return
        # vLLM's loader zero-fills the rows a NoPE checkpoint omits (``kv_a`` lacks the rope rows of its shard).
        module = self._modules[entry["destination"].rpartition(".")[0]]
        rows = module.output_sizes[shard]
        if weight.shape[0] < rows:
            weight = torch.cat([weight, weight.new_zeros((rows - weight.shape[0], weight.shape[1]))])
        param.weight_loader(param, weight, shard)

    def _ensure_buf(self, mod_path: str) -> nn.Parameter:
        if mod_path in self._bufs:
            return self._bufs[mod_path]
        from vllm.model_executor.layers.linear import RowParallelLinear
        mod = self._fp8_modules[mod_path]
        w = mod.weight
        orig_shape = (w.shape[1], w.shape[0]) if w.ndim == 2 else w.shape
        buf = nn.Parameter(
            torch.empty(orig_shape, dtype=self._dtype, device=self._device),
            requires_grad=False)
        buf.weight_loader = mod.weight_loader
        if isinstance(mod, RowParallelLinear):
            buf.input_dim = 1
        else:
            buf.output_dim = 0
        merged_name = _canonical_name(mod_path).rsplit(".", 1)[-1]
        self._shards_left[mod_path] = _SHARD_COUNTS.get(merged_name, 1)
        self._bufs[mod_path] = buf
        return buf

    def _flush_module(self, mod_path: str):
        from vllm._custom_ops import scaled_fp8_quant
        buf = self._bufs.pop(mod_path)
        mod = self._fp8_modules[mod_path]
        qweight, scale = scaled_fp8_quant(buf.data, scale=None)
        mod.weight.data.copy_(qweight.t().contiguous())
        mod.weight_scale.data.copy_(scale)
        del buf

    def _copy_param(self, param: nn.Parameter, tensor: torch.Tensor):
        """Copy tensor into param, using weight_loader for TP sharding if needed."""
        loader = getattr(param, "weight_loader", None)
        if loader is not None and param.data.shape != tensor.shape:
            loader(param, tensor)
        else:
            param.data.copy_(tensor)

    def _copy_native_fp8_param(
        self, module: nn.Module, param: nn.Parameter, tensor: torch.Tensor
    ) -> None:
        loader = getattr(module, "weight_loader", None)
        if loader is None or tensor.ndim != 2 or param.ndim != 2:
            param.data.copy_(tensor)
            return

        from vllm.model_executor.layers.linear import RowParallelLinear

        if isinstance(module, RowParallelLinear):
            logical_shape = (tensor.shape[0], param.numel() // tensor.shape[0])
            shard_dim = "input_dim"
            shard_axis = 1
        else:
            logical_shape = (param.numel() // tensor.shape[1], tensor.shape[1])
            shard_dim = "output_dim"
            shard_axis = 0

        local = nn.Parameter(
            torch.empty(logical_shape, dtype=tensor.dtype, device=tensor.device),
            requires_grad=False,
        )
        setattr(local, shard_dim, shard_axis)
        loader(local, tensor)
        if param.shape == local.shape:
            param.data.copy_(local)
        elif param.shape == local.T.shape:
            param.data.copy_(local.T.contiguous())
        else:
            raise ValueError(
                f"native FP8 local shape {tuple(local.shape)} cannot be copied "
                f"to vLLM parameter shape {tuple(param.shape)}"
            )

    def _copy_native_fp8_scale(
        self, module: nn.Module, param: nn.Parameter, tensor: torch.Tensor
    ) -> None:
        from vllm.model_executor.layers.linear import RowParallelLinear

        # Shard from the full tensor: ``param`` may be stored transposed.
        tp_rank, tp_size = module.tp_rank, module.tp_size
        if isinstance(module, RowParallelLinear):
            if tensor.shape[1] % tp_size:
                raise ValueError(
                    f"native FP8 scale shape {tuple(tensor.shape)} does not "
                    f"split across {tp_size} input shards"
                )
            shard_size = tensor.shape[1] // tp_size
            local = tensor.narrow(1, tp_rank * shard_size, shard_size)
        else:
            output_sizes = getattr(module, "output_partition_sizes", None) or (1,)
            total = sum(output_sizes)
            offset = 0
            shards = []
            for output_size in output_sizes:
                scale_rows, rows_rem = divmod(tensor.shape[0] * output_size, total)
                if rows_rem or scale_rows % tp_size:
                    raise ValueError(
                        f"native FP8 scale shape {tuple(tensor.shape)} does not "
                        f"split into output sizes {list(output_sizes)} across "
                        f"{tp_size} shards"
                    )
                shard_size = scale_rows // tp_size
                shards.append(
                    tensor.narrow(0, offset + tp_rank * shard_size, shard_size)
                )
                offset += scale_rows
            local = torch.cat(shards, dim=0)

        if param.shape == local.shape:
            param.data.copy_(local)
        elif param.shape == local.T.shape:
            param.data.copy_(local.T.contiguous())
        else:
            raise ValueError(
                f"native FP8 scale local shape {tuple(local.shape)} cannot be "
                f"copied to vLLM parameter shape {tuple(param.shape)}"
            )

    def _quant_and_copy(self, param: nn.Parameter, tensor: torch.Tensor,
                        scale_param: nn.Parameter | None = None):
        """FP8-quantize a BF16 tensor and copy into an FP8-transposed param."""
        from vllm._custom_ops import scaled_fp8_quant
        qweight, scale = scaled_fp8_quant(
            tensor.to(dtype=self._dtype, device=self._device), scale=None,
        )
        if param.data.ndim == 2 and param.data.shape == (qweight.shape[1], qweight.shape[0]):
            param.data.copy_(qweight.t().contiguous())
        else:
            param.data.copy_(qweight)
        if scale_param is not None:
            scale_param.data.copy_(scale)

    def _feed_fp8_module(self, mod_path: str, tensor: torch.Tensor,
                         shard_id=None) -> None:
        buf = self._ensure_buf(mod_path)
        if shard_id is not None:
            buf.weight_loader(buf, tensor, shard_id)
        else:
            buf.weight_loader(buf, tensor)
        self._shards_left[mod_path] -= 1
        if self._shards_left[mod_path] <= 0:
            self._flush_module(mod_path)

    def feed(self, sf_key: str, tensor: torch.Tensor):
        """Route one received BF16 tensor to the correct module."""
        if not self._route(sf_key, tensor):
            self._unplaced.append(sf_key)

    def _route(self, sf_key: str, tensor: torch.Tensor) -> bool:
        """Write or stage ``tensor``; ``False`` when no vLLM parameter takes it."""
        if sf_key.endswith(".weight"):
            mod_candidate = sf_key[: -len(".weight")]
            prefix, _, leaf = _canonical_name(mod_candidate).rpartition(".")

            param_name = self._param_name(sf_key)
            mod_path = self._fp8_module_path(mod_candidate)
            if tensor.dtype in _FP8_DTYPES and param_name is not None and mod_path is not None:
                self._copy_native_fp8_param(
                    self._fp8_modules[mod_path], self._params[param_name], tensor
                )
                return True

            merged = _STACKED_PARAMS.get(leaf)
            if merged and prefix:
                merged_path = self._fp8_module_path(f"{prefix}.{merged}")
                if merged_path is not None:
                    self._feed_fp8_module(merged_path, tensor, _SHARD_IDS[leaf])
                    return True

            if mod_path is not None:
                self._feed_fp8_module(mod_path, tensor)
                return True

            if tensor.dtype in _FP8_DTYPES and self._stage_dequant(sf_key, mod_candidate, "weight", tensor):
                return True

        if sf_key.endswith(".weight_scale_inv"):
            mod_candidate = sf_key[: -len(".weight_scale_inv")]
            param_name = self._param_name(sf_key)
            mod_path = self._fp8_module_path(mod_candidate)
            if param_name is not None and mod_path is not None:
                self._copy_native_fp8_scale(
                    self._fp8_modules[mod_path], self._params[param_name], tensor
                )
                return True
            if param_name is None and self._stage_dequant(sf_key, mod_candidate, "scale", tensor):
                return True

        family = self._fused_writer.family(sf_key)
        if (
            family is not None
            and (family.startswith("moe_") or tensor.dtype not in _FP8_DTYPES)
            and self._fused_writer.feed(sf_key, tensor)
        ):
            return True

        param_name = self._param_name(sf_key)
        if param_name is None:
            return self._feed_stacked_unquantized(sf_key, tensor)
        param = self._params[param_name]
        if param.data.shape == tensor.shape:
            self._copy_param(param, tensor)
            return True
        # Shape mismatch — the param is likely FP8-transposed.
        # Quantize the BF16 tensor to FP8 and copy with transposition.
        if param.data.dtype == torch.float8_e4m3fn:
            scale_name = self._param_name(sf_key.replace(".weight", ".weight_scale"))
            scale_param = self._params[scale_name] if scale_name is not None else None
            self._quant_and_copy(param, tensor, scale_param)
            return True
        self._copy_param(param, tensor)
        return True

    def _feed_stacked_unquantized(self, sf_key: str, tensor: torch.Tensor) -> bool:
        """Load one unfused projection (``gate_proj.bias``, ``up_proj.weight``, ...) into the shard of the unquantized
        merged vLLM parameter that holds it."""
        module_path, _, suffix = _canonical_name(sf_key).rpartition(".")
        prefix, _, leaf = module_path.rpartition(".")
        merged = _STACKED_PARAMS.get(leaf)
        if tensor.dtype in _FP8_DTYPES or not merged or not prefix:
            return False
        name = self._param_name(f"{prefix}.{merged}.{suffix}")
        if name is None or self._params[name].dtype in _FP8_DTYPES:
            return False
        param = self._params[name]
        param.weight_loader(param, tensor, _SHARD_IDS[leaf])
        return True

    @property
    def pending(self) -> int:
        return len(self._bufs) + len(self._dequant)

    @property
    def unplaced(self) -> list[str]:
        """Received names that reached no vLLM parameter, including block-FP8 halves whose partner never arrived."""
        stranded = [name for entry in self._dequant.values() for name in entry["names"]]
        return [*self._unplaced, *stranded]

    def staged(self):
        """Each staging tensor this updater holds across received tensors, with the parameters it will write."""
        for path, buffer in self._bufs.items():
            yield buffer, [f"{path}.weight", f"{path}.weight_scale"]
        for entry in self._dequant.values():
            for part in ("weight", "scale"):
                if part in entry:
                    yield entry[part], [entry["destination"]]


# Param-name patterns that are legitimately NOT shipped by a BF16 sender:
# rotary frequency buffers, and FP8 / GPTQ quantization metadata that only
# exists on a quantized receiver.
_NON_SYNCED_HF_NAME_PATTERNS = (
    ".inv_freq",
    ".weight_scale",
    ".input_scale",
    ".weight_zero_point",
    ".weight_qzeros",
    ".weight_g_idx",
    ".weight_packed",
    ".weight_shape",
)


def _name_is_non_synced(name: str) -> bool:
    return any(pat in name for pat in _NON_SYNCED_HF_NAME_PATTERNS)


def compute_expected_hf_param_names(model: nn.Module) -> set[str]:
    """HF-style names a vLLM model can accept from a BF16 sender.

    Fused ``QKVParallelLinear`` and ``gate_up_proj`` modules are expanded to
    the unfused HF names. The walk uses module types, not ``param.data``, so
    it still works when weights are offloaded.
    """
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.linear import QKVParallelLinear

    expected: set[str] = set()
    fused_names: set[str] = set()

    def _parent_of(mod_path: str) -> str:
        return mod_path.rsplit(".", 1)[0] if "." in mod_path else ""

    for mod_path, mod in model.named_modules():
        if isinstance(mod, QKVParallelLinear):
            parent = _parent_of(mod_path)
            for orig in ("q_proj", "k_proj", "v_proj"):
                key = f"{parent}.{orig}.weight" if parent else f"{orig}.weight"
                expected.add(key)
            fused_names.add(f"{mod_path}.weight")
        elif isinstance(mod, MergedColumnParallelLinear):
            leaf = mod_path.rsplit(".", 1)[-1] if "." in mod_path else mod_path
            if leaf == "gate_up_proj":
                parent = _parent_of(mod_path)
                for orig in ("gate_proj", "up_proj"):
                    key = f"{parent}.{orig}.weight" if parent else f"{orig}.weight"
                    expected.add(key)
                fused_names.add(f"{mod_path}.weight")

    for name, _ in model.named_parameters():
        if name in fused_names or _name_is_non_synced(name):
            continue
        expected.add(name)

    return expected


# ---------------------------------------------------------------------------
# Non-quantized direct-to-parameter writer (zero temp allocation for TP=1)
# ---------------------------------------------------------------------------

class _DirectParamWriter:
    """Pre-computes views into model parameter storage for zero-copy writes.

    For TP=1 non-quantized models, every safetensor weight maps exactly to
    either a model parameter or a narrow slice of a merged parameter
    (qkv_proj, gate_up_proj).
    """

    def __init__(self, model: nn.Module, device):
        from vllm.model_executor.layers.linear import (
            QKVParallelLinear, MergedColumnParallelLinear,
        )
        self._device = device
        self._views: dict[str, torch.Tensor] = {}
        self._destinations: dict[str, str] = {}

        params = dict(model.named_parameters())

        for mod_path, mod in model.named_modules():
            if not isinstance(mod, (QKVParallelLinear, MergedColumnParallelLinear)):
                continue
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue

            output_dim = getattr(weight, "output_dim", None)
            if output_dim is None:
                output_dim = 0
            output_sizes = mod.output_sizes
            tp_size = mod.tp_size

            prefix = mod_path.rsplit(".", 1)[0] if "." in mod_path else ""
            leaf = mod_path.rsplit(".", 1)[-1] if "." in mod_path else mod_path

            if isinstance(mod, QKVParallelLinear):
                originals = [("q_proj", 0), ("k_proj", 1), ("v_proj", 2)]
            elif leaf == "gate_up_proj":
                originals = [("gate_proj", 0), ("up_proj", 1)]
            else:
                continue

            for orig_name, shard_idx in originals:
                sf_key = f"{prefix}.{orig_name}.weight" if prefix else f"{orig_name}.weight"
                offset = sum(output_sizes[:shard_idx]) // tp_size
                size = output_sizes[shard_idx] // tp_size
                self._views[sf_key] = weight.data.narrow(output_dim, offset, size)
                self._destinations[sf_key] = f"{mod_path}.weight"

        for name, param in params.items():
            if name not in self._views:
                self._views[name] = param.data
                self._destinations[name] = name

        aliases: dict[str, torch.Tensor] = {}
        destination_aliases: dict[str, str] = {}
        for name, view in self._views.items():
            alias = self._strip_routed_experts_module(name)
            if alias != name and alias not in self._views:
                aliases[alias] = view
                destination_aliases[alias] = self._destinations[name]
        self._views.update(aliases)
        self._destinations.update(destination_aliases)

        aliases = {}
        destination_aliases = {}
        for name, view in self._views.items():
            alias = self._add_checkpoint_wrapper_alias(name)
            if alias != name and alias not in self._views:
                aliases[alias] = view
                destination_aliases[alias] = self._destinations[name]
        self._views.update(aliases)
        self._destinations.update(destination_aliases)

    @staticmethod
    def _strip_routed_experts_module(sf_key: str) -> str:
        return sf_key.replace(".experts.routed_experts.", ".experts.")

    @staticmethod
    def _strip_checkpoint_wrapper(sf_key: str) -> str:
        return sf_key.replace("._checkpoint_wrapped_module", "")

    @staticmethod
    def _add_checkpoint_wrapper_alias(sf_key: str) -> str:
        parts = sf_key.split(".")
        for idx in range(len(parts) - 2):
            if (
                parts[idx] == "layers"
                and parts[idx + 1].isdigit()
                and parts[idx + 2] != "_checkpoint_wrapped_module"
            ):
                return ".".join(
                    parts[: idx + 2]
                    + ["_checkpoint_wrapped_module"]
                    + parts[idx + 2 :]
                )
        return sf_key

    def get_view(self, sf_key: str) -> torch.Tensor | None:
        view = self._views.get(sf_key)
        if view is not None:
            return view
        return self._views.get(self._strip_checkpoint_wrapper(sf_key))

    def get_destination(self, sf_key: str) -> str | None:
        destination = self._destinations.get(sf_key)
        if destination is not None:
            return destination
        return self._destinations.get(self._strip_checkpoint_wrapper(sf_key))

    def all_keys(self) -> list[str]:
        return list(self._views.keys())


# ---------------------------------------------------------------------------
# TP>1 shard-aware writer for vLLM-fused param families (batched sync)
# ---------------------------------------------------------------------------

class _ShardAwareFusedWriter:
    """Loads the fused param families that ``model.load_weights`` cannot place
    when fed vLLM-*internal* (already-fused) names -- i.e. the DSS
    ``weight_format="vllm"`` broadcast at ``sampling_tensor_parallel_size > 1``.

    Two families break (empirically enumerated on vLLM 0.26, Qwen3.5-MoE):

    * **Gated DeltaNet input projections** ``in_proj_qkvz`` / ``in_proj_ba``
      (``MergedColumnParallelLinear``).  ``model.load_weights`` re-applies the
      HF->vLLM stacked mapping to the *already-fused* name; because
      ``in_proj_qkv`` is a prefix of ``in_proj_qkvz`` the substring rename
      yields ``in_proj_qkvzz`` (and ``in_proj_baa``) -> hard ``ValueError``.
    * **FusedMoE expert weights** ``experts.w13_weight`` / ``experts.w2_weight``
      (``RoutedExperts``).  The fused 3-D vLLM tensor matches neither the
      per-expert HF names nor the pre-fused checkpoint names, so vLLM's loader
      returns an empty set -> the weights are *silently dropped* (no error).

    For each family we resolve the target module/param once and drive its
    **own** ``weight_loader`` -- exactly the entry point vLLM's offline
    checkpoint load uses -- which performs the correct TP (and, for MoE, EP)
    sharding internally.  This bypasses only the broken name-mapping layer;
    every other (non-fused) param is left to ``model.load_weights``.

    Usage mirrors :class:`_FP8InplaceUpdater`::

        writer = _ShardAwareFusedWriter(model, device)
        for name, tensor in engine.receive_weights():
            if not writer.feed(name, tensor):
                model.load_weights([(name, tensor)])
    """

    def __init__(self, model: nn.Module, device):
        self._device = device
        self._params: dict[str, nn.Parameter] = dict(model.named_parameters())
        modules = dict(model.named_modules())

        # wire-name -> {"family", "module", "param", "destination"}
        self._handlers: dict[str, dict] = {}

        self._register_gdn(modules)
        self._register_sparse_indexer(modules)
        self._register_moe(modules)
        # A LoRA wrapper moves the wrapped layer under ``base_layer``; the wire names it without that level.
        for name, canonical in [(name, _canonical_name(name)) for name in self._handlers]:
            self._handlers.setdefault(canonical, self._handlers[name])

    # -- registration ------------------------------------------------------

    def _register_gdn(self, modules: dict[str, nn.Module]) -> None:
        """Fused GDN input projections: ``MergedColumnParallelLinear`` whose
        parent module is a Gated DeltaNet attention block.  Anchoring on the
        parent type (not a hard-coded leaf name) captures exactly
        ``in_proj_qkvz`` + ``in_proj_ba`` and nothing else (``conv1d`` is a
        plain ``ColumnParallelLinear``; ``out_proj`` is row-parallel)."""
        try:
            from vllm.model_executor.layers.linear import (
                MergedColumnParallelLinear,
            )
        except Exception:
            return
        try:
            from vllm.model_executor.layers.mamba.gdn.base import (
                GatedDeltaNetAttention,
            )
        except Exception:
            return

        for mod_path, mod in modules.items():
            if not isinstance(mod, MergedColumnParallelLinear):
                continue
            canonical_path = _canonical_name(mod_path)
            parent_path = canonical_path.rsplit(".", 1)[0] if "." in canonical_path else ""
            parent = modules.get(parent_path)
            if parent is None or not isinstance(parent, GatedDeltaNetAttention):
                continue
            wname = f"{mod_path}.weight"
            param = self._params.get(wname)
            if param is None:
                continue
            self._handlers[wname] = {
                "family": "gdn_merged",
                "module": mod,
                "param": param,
                "destination": wname,
            }

    def _register_sparse_indexer(self, modules: dict[str, nn.Module]) -> None:
        try:
            from vllm.model_executor.layers.linear import (
                MergedColumnParallelLinear,
            )
        except Exception:
            return

        for mod_path, mod in modules.items():
            if not (
                isinstance(mod, MergedColumnParallelLinear)
                and _canonical_name(mod_path).endswith(".indexer.wk_weights_proj")
            ):
                continue
            wname = f"{mod_path}.weight"
            param = self._params.get(wname)
            if param is None:
                continue
            self._handlers[wname] = {
                "family": "gdn_merged",
                "module": mod,
                "param": param,
                "destination": wname,
            }

    def _register_moe(self, modules: dict[str, nn.Module]) -> None:
        """FusedMoE expert weights.  Duck-typed on the ``RoutedExperts`` API
        (``w13_weight`` + ``w2_weight`` params, per-expert ``weight_loader``,
        and the global->local expert map) so we don't couple to a specific
        module class.  Both the registry name (``...experts.routed_experts.
        w13_weight``) and the DSS wire name (``...experts.w13_weight``, which
        drops the ``routed_experts`` level) are registered."""
        for mod_path, mod in modules.items():
            if not (hasattr(mod, "w13_weight") and hasattr(mod, "w2_weight")
                    and hasattr(mod, "_map_global_expert_id_to_local_expert_id")
                    and callable(getattr(mod, "weight_loader", None))):
                continue
            for leaf, family in (("w13_weight", "moe_w13"),
                                 ("w2_weight", "moe_w2"),
                                 ("w13_weight_scale_inv", "moe_w13"),
                                 ("w2_weight_scale_inv", "moe_w2")):
                reg_name = f"{mod_path}.{leaf}"
                param = self._params.get(reg_name)
                if param is None:
                    continue
                handler = {
                    "family": family,
                    "module": mod,
                    "param": param,
                    "destination": reg_name,
                    "weight_name": leaf,
                }
                self._handlers[reg_name] = handler
                wire_name = reg_name.replace(".routed_experts.", ".")
                if wire_name != reg_name:
                    self._handlers[wire_name] = handler

    # -- loading -----------------------------------------------------------

    def _handler_for(self, name: str) -> dict | None:
        handler = self._handlers.get(name)
        if handler is not None:
            return handler
        return self._handlers.get(_canonical_name(name))

    def family(self, name: str) -> str | None:
        handler = self._handler_for(name)
        return handler["family"] if handler is not None else None

    def destination_name(self, name: str) -> str | None:
        handler = self._handler_for(name)
        return handler["destination"] if handler is not None else None

    def feed(self, name: str, tensor: torch.Tensor) -> bool:
        """Load one received *full* (un-sharded) fused tensor into its param.

        Returns ``True`` if this writer handled *name* (a fused family), or
        ``False`` if the caller should fall back to ``model.load_weights``.
        """
        handler = self._handler_for(name)
        if handler is None:
            return False

        family = handler["family"]
        param = handler["param"]
        module = handler["module"]

        if family == "gdn_merged":
            replicated_shard_ids = getattr(module, "replicated_shard_ids", None)
            if replicated_shard_ids:
                output_dim = getattr(param, "output_dim", 0)
                offset = 0
                for shard_id, output_size in enumerate(module.output_sizes):
                    if shard_id in replicated_shard_ids:
                        output_size //= module.tp_size
                    loaded_shard = tensor.narrow(output_dim, offset, output_size)
                    module.weight_loader(param, loaded_shard, shard_id)
                    offset += output_size
                if offset != tensor.shape[output_dim]:
                    raise ValueError(
                        f"Fused GDN tensor has {tensor.shape[output_dim]} rows, "
                        f"but registered shards consume {offset}"
                    )
                return True
            # MergedColumnParallelLinear.weight_loader(param, full_weight,
            # loaded_shard_id=None) splits the full fused tensor by
            # ``output_sizes`` ([q,k,v,z] / [b,a]) and TP-narrows each shard
            # (respecting disable_tp) -- one call does the whole projection.
            module.weight_loader(param, tensor)
            return True

        if family == "moe_w13":
            # Gated FusedMoE w13 is [w1(gate) | w3(up)] on the intermediate dim.
            # Non-gated MoE (Nemotron-H) stores only w1 in w13_weight; splitting
            # in half then loading shard_id=w3 does expert_data.narrow(shard_size)
            # past the TP-local intermediate (e.g. 672/672 on Super TP=4).
            # NOTE: RoutedExperts.weight_loader routes by *substring* of
            # ``weight_name``; the model-weight copy branch only runs when it
            # contains "weight" (and none of scale/zero/offset/g_idx/shape), so
            # we pass the param's own name ("w13_weight") like vLLM's oracle.
            moe_config = getattr(module, "moe_config", None)
            gated = True if moe_config is None else bool(
                getattr(moe_config, "is_act_and_mul", True)
            )
            weight_name = handler["weight_name"]
            tensor = self._refined_block_scale(module, param, weight_name, tensor)
            for expert_id in range(tensor.shape[0]):
                if gated:
                    inter = tensor.shape[1] // 2
                    module.weight_loader(
                        param, tensor[expert_id, :inter, :],
                        weight_name, "w1", expert_id, return_success=True,
                    )
                    module.weight_loader(
                        param, tensor[expert_id, inter:, :],
                        weight_name, "w3", expert_id, return_success=True,
                    )
                else:
                    module.weight_loader(
                        param, tensor[expert_id],
                        weight_name, "w1", expert_id, return_success=True,
                    )
            return True

        if family == "moe_w2":
            weight_name = handler["weight_name"]
            tensor = self._refined_block_scale(module, param, weight_name, tensor)
            for expert_id in range(tensor.shape[0]):
                module.weight_loader(
                    param, tensor[expert_id],
                    weight_name, "w2", expert_id, return_success=True,
                )
            return True

        return False

    @staticmethod
    def _refined_block_scale(module, param, weight_name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Upsample checkpoint block scales to a TP-refined scale grid the expert loader can no longer see.

        ``RoutedExperts.weight_loader`` upsamples block scales by ``quant_method.weight_scale_refine``. vLLM's LoRA
        wrapper replaces that quant method with a modular wrapper which keeps the attribute on ``old_quant_method``, so
        the loader would slice unrefined scales into a refined parameter.
        """
        quant_method = getattr(module, "quant_method", None)
        if (
            "scale" not in weight_name
            or getattr(param, "quant_method", None) != "block"
            or getattr(quant_method, "weight_scale_refine", None) is not None
        ):
            return tensor
        refine = getattr(getattr(quant_method, "old_quant_method", None), "weight_scale_refine", None)
        if refine is None:
            return tensor
        return tensor.repeat_interleave(refine[0], dim=-2).repeat_interleave(refine[1], dim=-1)


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------

def load_spec_checkpoint(
    model_path: str,
) -> list[tuple[str, torch.Tensor]]:
    """Load all spec-model weights from a checkpoint directory.

    Supports both safetensors and pytorch_model.bin formats.
    Returns a list of ``(name, tensor)`` pairs with tensors on CPU.
    """
    import glob as _glob
    import os

    weights: list[tuple[str, torch.Tensor]] = []
    st_files = sorted(_glob.glob(os.path.join(model_path, "*.safetensors")))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            for name, tensor in load_file(f, device="cpu").items():
                weights.append((name, tensor))
    else:
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(bin_file):
            state = torch.load(bin_file, map_location="cpu", weights_only=True)
            for name, tensor in state.items():
                weights.append((name, tensor))
    return weights


def spec_bucket_size(
    model_path: str,
    min_bucket_size: int = 256 * 1024 * 1024,
) -> int:
    """Compute a bucket size large enough for the largest spec-model weight.

    Reads only ``config.json`` from *model_path*; no tensors are loaded.
    Falls back to *min_bucket_size* when the config is absent.
    """
    import os

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return min_bucket_size
    with open(config_path) as f:
        cfg = json.load(f)

    def _parse_dim(val) -> list[int]:
        if isinstance(val, str):
            return [int(x) for x in val.split(".")]
        if isinstance(val, list):
            return [int(x) for x in val]
        if isinstance(val, int):
            return [val]
        return []

    vocab = cfg.get("vocab_size", 0)
    dims: list[int] = []
    for key in ("emb_dim", "inner_dim", "proj_dim", "input_hidden_dim"):
        dims.extend(_parse_dim(cfg.get(key, 0)))
    max_dim = max(dims) if dims else 0

    max_bytes = vocab * max_dim * 4  # float32
    return max(min_bucket_size, max_bytes)
