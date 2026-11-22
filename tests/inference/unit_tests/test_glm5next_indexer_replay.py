"""The sampler half of GLM-5.3 sparse-attention selection replay; ``tests/model/test_glm53_indexer_replay.py`` covers
the trainer half."""

import types

import torch
from vllm.distributed.aux_output_connector import worker as aux_worker
from vllm.model_executor.layers.fused_moe import routed_experts_capturer

from arctic_platform.inference.vllm import glm5next_indexer_replay as replay
from arctic_platform.inference.vllm.glm5next_indexer_replay import pool_rows


def test_sampler_captures_each_sparse_layers_pools_past_the_decoder_layers(monkeypatch):
    """The vLLM glue: with the key set, the capture grows by the pool rows, and binding the capturer makes each main-model
    glm5next indexer op write the pools it selected at its sparse layer's first row (MTP indexers stay unbound)."""
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
    selected = torch.tensor([[4, 5, 0, 1, 10, 11, -1, -1, 12], [2, 3, 8, 9, 6, 7, 0, 1, -1]], dtype=torch.int32)

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
    assert routed_experts_capturer._get_routed_experts_shape(vllm_config) == (layers + 2 * rows, 257, 2)
    off = types.SimpleNamespace(model_config=model_config, additional_config={})
    assert routed_experts_capturer._get_routed_experts_shape(off) == (layers, 40, 2)

    model = torch.nn.ModuleList([Indexer(1), Indexer(3), Indexer(layers)])  # the last one is an MTP layer's
    writes = {}
    aux_worker.bind_routed_experts_capturer(
        model, types.SimpleNamespace(capture=lambda row, ids: writes.update({row: ids.tolist()}))
    )
    for indexer in model:
        indexer.indexer_op(torch.zeros(2, 4))

    pools = [[2, 0, 5, -1], [1, 4, 3, 0]]
    expected = {
        layers + s * rows + r: [pools[q][r * 2 : r * 2 + 2] for q in range(2)] for s in range(2) for r in range(rows)
    }
    assert writes == expected
