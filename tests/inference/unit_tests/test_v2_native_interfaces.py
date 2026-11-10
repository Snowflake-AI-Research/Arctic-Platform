from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import vllm.config
import vllm.distributed.parallel_state as parallel_state

from arctic_platform.inference.vllm import ulysses
from arctic_platform.inference.vllm.swiftkv import llama_swiftkv
from arctic_platform.inference.vllm.spec_dec.vocab_parallel_embedding import SpeculatorTPInit


def test_native_parallel_groups_keep_ulysses_head_and_replica_ownership(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2, ulysses_sequence_parallel_size=2,
            enable_elastic_ep=False, enable_eplb=True, use_all2all=False),
        model_config=SimpleNamespace(is_moe=True), engram_config=None,
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 5)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda *args: "gloo")
    world = SimpleNamespace(local_rank=5, device_group=None)
    monkeypatch.setattr(ulysses, "get_world_group", lambda: world)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    for name in ("_TP", "_ETP", "_ENGRAM_DP", "_DP", "_EP", "_EPLB", "_PP", "_PCP", "_DCP", "_SP", "_SP_TP"):
        monkeypatch.setattr(parallel_state, name, None, raising=False)
    groups = {}

    def init_group(ranks, *args, group_name, **kwargs):
        owned = next(group for group in ranks if 5 in group)
        groups[group_name] = owned
        return SimpleNamespace(rank_in_group=owned.index(5), world_size=len(owned))

    monkeypatch.setattr(parallel_state, "init_model_parallel_group", init_group)
    monkeypatch.setattr(ulysses, "init_model_parallel_group", init_group)
    monkeypatch.setattr(parallel_state, "_orig_initialize_model_parallel",
        ulysses.UlyssesParallelState._orig_initialize_model_parallel, raising=False)
    ulysses.UlyssesParallelState.initialize_model_parallel(2, 2, 1, 1, "gloo")
    assert groups == {"tp": [4, 5], "dcp": [5], "pcp": [5], "pp": [1, 5],
        "dp": [5, 13], "ep": [4, 5, 12, 13], "eplb": [4, 5, 12, 13],
        "sp": [5, 7], "sp_tp": [4, 6, 5, 7]}
    assert parallel_state._ETP is parallel_state._TP
    assert parallel_state.init_model_parallel_group is init_group


def test_speculator_shards_follow_the_selected_group_rank(monkeypatch):
    tp = SimpleNamespace(world_size=2, rank=17, rank_in_group=1)
    sp = SimpleNamespace(world_size=4, rank=17, rank_in_group=2)
    monkeypatch.setattr(parallel_state, "_TP", tp)
    monkeypatch.setattr(parallel_state, "_SP", sp, raising=False)
    speculator = SpeculatorTPInit()
    assert speculator.TP_GROUP is sp
    assert (speculator.tp_size, speculator.tp_rank) == (4, 2)


@pytest.mark.parametrize("failure_stage", ["selection", "decode"])
def test_swiftkv_restores_native_forward_context_on_error(monkeypatch, failure_stage):
    original_metadata = {"attention": SimpleNamespace(swiftkv_logits_indices=torch.tensor([0]))}
    descriptor = SimpleNamespace(num_tokens=2, num_active_loras=0)
    mode = vllm.config.CUDAGraphMode.FULL
    context = SimpleNamespace(attn_metadata=original_metadata,
        batch_descriptor=descriptor, cudagraph_runtime_mode=mode)
    monkeypatch.setattr(llama_swiftkv, "get_forward_context", lambda: context)
    monkeypatch.setattr(llama_swiftkv.model_runner, "set_shift_parallel_mode", lambda *_: nullcontext())
    tensor = torch.ones(2, 2)

    def fail(*args):
        raise RuntimeError("late layer failed")

    model = SimpleNamespace(
        prefill_runner=lambda *_: (tensor, tensor, tensor, tensor, tensor),
        swiftkv_select=fail if failure_stage == "selection" else lambda *args: tuple(t[:1] for t in args),
        decode_runner=fail,
    )
    with pytest.raises(RuntimeError, match="late layer failed"):
        llama_swiftkv.LlamaSwiftKVModel.forward(model, tensor, tensor)
    assert context.attn_metadata is original_metadata
    assert context.batch_descriptor is descriptor
    assert context.cudagraph_runtime_mode is mode
