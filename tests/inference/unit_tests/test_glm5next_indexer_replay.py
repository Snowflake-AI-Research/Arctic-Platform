"""GLM-5.3 sampler capture in vLLM."""

import types

import torch
from vllm.distributed.aux_output_connector import worker as aux_worker
from vllm.model_executor.layers.fused_moe import routed_experts_capturer

from arctic_platform.common.glm5next_indexer_replay import pool_rows
from arctic_platform.inference.vllm import glm5next_indexer_replay as replay


def test_sampler_captures_each_sparse_layers_pools_past_the_decoder_layers(monkeypatch):
    text = types.SimpleNamespace(
        layer_types=["linear_attention", "deepseek_sparse_attention"] * 2,
        index_topk=8,
        index_kpool=2,
        num_experts_per_tok=2,
    )
    layers, rows = len(text.layer_types), pool_rows(text)
    model_config = types.SimpleNamespace(
        hf_text_config=text,
        max_model_len=1024,
        get_total_num_hidden_layers=lambda: layers,
        get_num_experts=lambda: 40,
        get_num_experts_per_tok=lambda: text.num_experts_per_tok,
    )
    vllm_config = types.SimpleNamespace(model_config=model_config, additional_config={replay.INDEXER_REPLAY_KEY: True})
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
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    replay.ensure_glm5next_indexer_replay()
    assert routed_experts_capturer._get_routed_experts_shape(vllm_config) == (layers + 2 * rows, 512, 2)
    off = types.SimpleNamespace(model_config=model_config, additional_config={})
    assert routed_experts_capturer._get_routed_experts_shape(off) == (layers, 40, 2)

    model = torch.nn.ModuleList([Indexer(1), Indexer(3), Indexer(layers)])  # the last one is an MTP layer's
    model.layer_id, model.capture_fn = 0, None  # a routed-expert source for the original binder
    writes = torch.full((2, layers + 2 * rows, 2), -1, dtype=torch.int32)

    def capture(row, ids):
        writes[:, row] = ids

    aux_worker.bind_routed_experts_capturer(model, types.SimpleNamespace(capture=capture))
    for indexer in model:
        indexer.indexer_op(torch.zeros(2, 4))

    pools = torch.tensor([[1, 2, 0, 0], [2, 5, 4, 1]], dtype=torch.int32)
    torch.testing.assert_close(writes[:, layers:].flatten(1), pools.repeat(1, 2))
    assert writes[:, :layers].eq(-1).all()
