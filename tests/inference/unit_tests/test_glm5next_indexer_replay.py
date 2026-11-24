"""GLM-5.3 sampler capture in vLLM."""

import types

import torch
from vllm.distributed.aux_output_connector import worker as aux_worker
from vllm.model_executor.layers.fused_moe import routed_experts_capturer
from vllm.v1.core.sched import scheduler
from vllm.v1.core.sched import utils

from arctic_platform.inference.vllm.router_replay import ensure_router_replay_vllm_patches


def test_sampler_captures_each_sparse_layers_pools_past_the_decoder_layers(monkeypatch):
    text = types.SimpleNamespace(
        model_type="glm5_next_text",
        layer_types=["linear_attention", "deepseek_sparse_attention"] * 2,
        index_topk=8,
        index_kpool=2,
        num_experts_per_tok=2,
    )
    layers, rows = len(text.layer_types), 2
    model_config = types.SimpleNamespace(
        hf_text_config=text,
        max_model_len=1024,
        get_total_num_hidden_layers=lambda: layers,
        get_num_experts=lambda: 40,
        get_num_experts_per_tok=lambda: text.num_experts_per_tok,
    )
    vllm_config = types.SimpleNamespace(
        model_config=model_config,
        additional_config={"arctic_router_replay_indexer": True},
        parallel_config=types.SimpleNamespace(data_parallel_rank=0, tensor_parallel_size=1),
    )
    selected = torch.tensor([[0, 1, 2, 3, 4, -1, -1, -1, -1], [2, 3, 8, 9, 6, 7, 0, 1, -1]], dtype=torch.int32)

    class Op(torch.nn.Module):  # a CustomOp calls the forward method it bound at construction
        def __init__(self):
            super().__init__()
            self._forward_method = lambda hidden_states, *args, **kwargs: selected

        def forward(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

    class Indexer(torch.nn.Module):  # the attributes of vLLM's glm5next Indexer the glue reads
        def __init__(self, layer):
            super().__init__()
            self.vllm_config, self.config, self.indexer_op = vllm_config, text, Op()
            self.prefix = f"language_model.model.layers.{layer}.self_attn.indexer"
            self.topk_tokens, self.index_kpool = text.index_topk, text.index_kpool

    for owner, name in (
        (routed_experts_capturer, "_get_routed_experts_shape"),
        (routed_experts_capturer, "bind_routed_experts_capturer"),
        (aux_worker, "bind_routed_experts_capturer"),
        (scheduler, "check_stop"),
        (utils, "check_stop"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    ensure_router_replay_vllm_patches()
    monkeypatch.setattr(scheduler, "check_stop", lambda *args: False)  # full plugin also replaces this callback
    ensure_router_replay_vllm_patches()
    assert routed_experts_capturer._get_routed_experts_shape(vllm_config) == (layers + 2 * rows, 512, 2)
    off = types.SimpleNamespace(model_config=model_config, additional_config={})
    assert routed_experts_capturer._get_routed_experts_shape(off) == (layers, 40, 2)

    model = torch.nn.ModuleList([Indexer(1), Indexer(3), Indexer(layers)])  # the last one is an MTP layer's
    model.layer_id, model.capture_fn = 0, None  # a routed-expert source for the original binder
    monkeypatch.setattr(routed_experts_capturer, "current_platform", types.SimpleNamespace(device_type="cpu"))
    monkeypatch.setattr(routed_experts_capturer, "dbo_current_ubatch_token_offset", lambda: 1)
    ctx = types.SimpleNamespace(dp_metadata=None, batch_descriptor=types.SimpleNamespace(num_tokens=2))
    calls = []

    def forward_context():
        calls.append(True)
        return ctx

    monkeypatch.setattr(routed_experts_capturer, "get_forward_context", forward_context)
    capturer = routed_experts_capturer.RoutedExpertsCapturer(4, vllm_config)
    aux_worker.bind_routed_experts_capturer(model, capturer)
    for indexer in model:
        torch.testing.assert_close(indexer.indexer_op(torch.zeros(2, 4)), selected)

    writes = capturer.snapshot_routing_data(4).int()
    pools = torch.tensor([[1, 2, 0, 0], [2, 5, 4, 1]], dtype=torch.int32)
    torch.testing.assert_close(writes[1:3, layers:].flatten(1), pools.repeat(1, 2))
    assert writes[[0, 3]].eq(0).all() and writes[:, :layers].eq(0).all()
    assert len(calls) == 2  # restore the local token batch once per main-model indexer
