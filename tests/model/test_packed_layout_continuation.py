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

"""Prefix continuation through real attention modules on CPU.

The flash kernels are replaced by ``flash_attn_varlen_reference`` (pure torch, bottom-right causal), so these
tests check the wiring -- which K/V each continuation sees, which segments the model uses, what the layout
stash does across checkpoint recompute -- not the CUDA kernels. Kernel numerics are the on-prem gate's job.

Each model call runs three ways on one batch: prefix sharing on, a sharing-off layout, and every row on its own.
Response log-probs and every parameter gradient must agree in float64; disabling the prefix gather must not.
"""

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM
from transformers import Qwen3MoeConfig
from transformers import Qwen3MoeForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from arctic_platform.common.packing import PackedLayout
from arctic_platform.common.packing import pack_with_layout
from arctic_platform.common.packing import using_layout
from arctic_platform.common.packing.layout import _hf_prefix_attention_fn
from arctic_platform.common.packing.layout import unshared_groups
from arctic_platform.model.implementations.layers import attn as attn_module
from arctic_platform.model.implementations.layers.attn import AttentionConfig
from arctic_platform.model.implementations.layers.attn import FlashAttention
from arctic_platform.model.implementations.layers.attn import SDPAAttention
from arctic_platform.model.implementations.layers.sequence import get_cu_seqlens_from_position_ids
from arctic_platform.testing_utils import flash_attn_varlen_reference
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

PROMPT_LEN = 5
VOCAB = 48
HEAD_DIM = 8
# float64 end to end: segment lengths differ between the three runs, which only reorders reductions (~1e-15
# relative). 1e-10 is far below any real wiring error, which moves log-probs by >1e-3 (see the negative control).
F64_ATOL = 1e-10
# Share-on gradients only. Both RMSNorm implementations normalize in float32 and cast back, so the root's K
# gradient, which sums every row's contribution before that cast, is rounded to float32 once instead of once per
# row: |diff| <= 2**-24 * |grad|, and grads here are <= 0.5. Dropping the gather is off by > 1e-3.
SHARED_GRAD_ATOL = 1e-7


def _batch():
    """Rows 0-2 share a prompt; row 3 is its own group. Right padding, positions 0..L-1."""
    lengths = [9, 8, 10, 7]
    width = max(lengths)
    ids = torch.randint(1, VOCAB, (len(lengths), width))
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    ids[1:3, :PROMPT_LEN] = ids[0, :PROMPT_LEN]
    ids = ids * mask
    positions = torch.arange(width).expand_as(mask) * mask
    scored = mask.clone()
    scored[:, :PROMPT_LEN] = 0
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}, [[0, 1, 2], [3]], scored


def _layouts(batch, groups, **build_kwargs):
    shared = PackedLayout.build(
        batch["attention_mask"],
        groups,
        [PROMPT_LEN - 1 if len(rows) > 1 else 0 for rows in groups],
        batch["position_ids"],
        input_ids=batch["input_ids"],
        **build_kwargs,
    )
    unshared = PackedLayout.build(
        batch["attention_mask"], *unshared_groups(len(batch["input_ids"])), batch["position_ids"]
    )
    assert shared.has_sharing and not unshared.has_sharing
    return shared, unshared


def _response_logprobs(logits_rows, batch, scored):
    logprobs = torch.log_softmax(logits_rows[:, :-1], dim=-1)
    logprobs = logprobs.gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1)
    return logprobs[scored[:, 1:].bool()]


def _run(model, logits_fn, batch, scored, layout=None):
    """Token-mean NLL over response tokens; forward, loss and backward all inside ``using_layout``."""
    model.zero_grad()
    if layout is None:
        width = batch["input_ids"].shape[1]
        rows = []
        for row in range(batch["input_ids"].shape[0]):
            length = int(batch["attention_mask"][row].sum())
            out = logits_fn(
                model, batch["input_ids"][row : row + 1, :length], batch["position_ids"][row : row + 1, :length]
            )
            rows.append(torch.cat([out, out.new_zeros(width - length, VOCAB)]))
        logits_rows = torch.stack(rows)
        logprobs = _response_logprobs(logits_rows, batch, scored)
        (-logprobs.mean()).backward()
    else:
        packed = pack_with_layout(batch, layout)
        with using_layout(model, layout):
            logits = logits_fn(model, packed["input_ids"], packed["position_ids"])
            logprobs = _response_logprobs(layout.unpack(logits), batch, scored)
            (-logprobs.mean()).backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    return logprobs.detach(), grads


def _assert_runs_match(got, want, grad_atol=F64_ATOL):
    torch_assert_close(got[0], want[0], rtol=0, atol=F64_ATOL, msg="logprobs")
    assert got[1].keys() == want[1].keys()
    for name in want[1]:
        torch_assert_close(got[1][name], want[1][name], rtol=0, atol=grad_atol, msg=name)


def _rope(position_ids):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    freqs = position_ids.reshape(-1, 1).double() * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)[None]
    return emb.cos(), emb.sin()


def _attention_config(use_qk_norm=True):
    return AttentionConfig(
        hidden_size=32,
        head_dim=HEAD_DIM,
        num_attention_heads=4,
        num_key_value_heads=2,
        is_causal=True,
        attention_bias=False,
        use_qk_norm=use_qk_norm,
        rms_norm_eps=1e-6,
    )


def _ap_attention(sliding_window=None):
    module = FlashAttention(_attention_config(), flash_attn_version=2)
    module._flash_attn_call = flash_attn_varlen_reference
    if sliding_window is not None:
        module.sliding_window = sliding_window
    return module


class APStack(nn.Module):
    """Embedding, Arctic-Platform flash attention + MLP blocks, LM head. Segments come from the model-stack helper."""

    def __init__(self, layers=2, sliding_window=None, use_checkpoint=False):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32)
        self.attn = nn.ModuleList(_ap_attention(sliding_window) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32)) for _ in range(layers))
        self.head = nn.Linear(32, VOCAB)
        self.use_checkpoint = use_checkpoint

    def _block(self, i, hidden, cos, sin, cu_seqlens, max_seqlen):
        hidden = hidden + self.attn[i](hidden, (cos, sin), cu_seqlens, max_seqlen)[0]
        return hidden + self.mlp[i](hidden)

    def forward(self, input_ids, position_ids):
        cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)
        cos, sin = _rope(position_ids)
        hidden = self.embed(input_ids)
        for i in range(len(self.attn)):
            args = (i, hidden, cos, sin, cu_seqlens, max_seqlen)
            hidden = checkpoint(self._block, *args, use_reentrant=False) if self.use_checkpoint else self._block(*args)
        return self.head(hidden)[0]


def _ap_logits(model, input_ids, position_ids):
    return model(input_ids, position_ids)


def _ap_stack(**kwargs):
    set_seed(0)
    return APStack(**kwargs).double()


def test_ap_get_cu_seqlens_reads_active_layout():
    batch, groups, _ = _batch()
    shared, _ = _layouts(batch, groups)
    packed = pack_with_layout(batch, shared)
    with using_layout(_ap_attention(), shared):
        cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(packed["position_ids"])
    torch_assert_equal(cu_seqlens, shared.cu_seqlens)
    assert max_seqlen == shared.max_len_q


@pytest.mark.parametrize("sliding_window", [None, 3])
def test_ap_stack_share_on_matches_share_off_and_rows(sliding_window):
    set_seed(1)
    batch, groups, scored = _batch()
    shared, unshared = _layouts(batch, groups)
    model = _ap_stack(sliding_window=sliding_window)
    rows = _run(model, _ap_logits, batch, scored)
    _assert_runs_match(_run(model, _ap_logits, batch, scored, unshared), rows)
    _assert_runs_match(_run(model, _ap_logits, batch, scored, shared), rows, SHARED_GRAD_ATOL)


def test_ap_stack_without_prefix_gather_does_not_match(monkeypatch):
    set_seed(1)
    batch, groups, scored = _batch()
    shared, _ = _layouts(batch, groups)
    model = _ap_stack()
    rows = _run(model, _ap_logits, batch, scored)

    def no_gather(k, v, cu_seqlens, max_seqlen, layout):
        cu = layout.cu_seqlens.to(torch.int32)
        return k, v, cu, cu, layout.max_len_q, layout.max_len_q

    monkeypatch.setattr(attn_module, "prefix_varlen_args", no_gather)
    broken = _run(model, _ap_logits, batch, scored, shared)
    assert (broken[0] - rows[0]).abs().max() > 1e-3


def test_ap_stack_checkpoint_recompute_matches():
    set_seed(1)
    batch, groups, scored = _batch()
    shared, _ = _layouts(batch, groups)
    plain = _run(_ap_stack(), _ap_logits, batch, scored, shared)
    recomputed = _run(_ap_stack(use_checkpoint=True), _ap_logits, batch, scored, shared)
    torch_assert_equal(recomputed[0], plain[0])
    for name in plain[1]:
        torch_assert_equal(recomputed[1][name], plain[1][name], msg=name)


def test_ap_sdpa_is_rejected_under_sharing():
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    module = SDPAAttention(_attention_config(use_qk_norm=False))
    with pytest.raises(ValueError, match="SDPAAttention"):
        with using_layout(module, shared):
            pass
    with using_layout(module, unshared):
        pass


def _hf_flash_reference(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_length_q=None,
    max_length_k=None,
    position_ids=None,
    **kwargs,
):
    """CPU stand-in for transformers' ``flash_attention_forward`` on packed ``[1, H, T, D]`` inputs."""
    assert attention_mask is None
    q, k, v = (t.transpose(1, 2).reshape(-1, t.shape[1], t.shape[-1]) for t in (query, key, value))
    if cu_seq_lens_q is None:
        # transformers' own packed-sequence inference: a segment starts wherever position_ids is 0.
        starts = (position_ids.reshape(-1) == 0).nonzero().flatten()
        cu_seq_lens_q = cu_seq_lens_k = torch.cat([starts, starts.new_tensor([q.shape[0]])]).to(torch.int32)
    window = (-1, -1) if sliding_window is None else (sliding_window - 1, sliding_window - 1)
    out = flash_attn_varlen_reference(
        q, k, v, cu_seq_lens_q, cu_seq_lens_k, causal=module.is_causal, window_size=window, softmax_scale=scaling
    )
    return out.view(query.shape[0], -1, *out.shape[-2:]), None


@pytest.fixture
def hf_flash_reference():
    """Point transformers' ``flash_attention_2`` entry at the CPU reference for one test, then remove it."""
    assert "flash_attention_2" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _hf_flash_reference
    yield
    del ALL_ATTENTION_FUNCTIONS["flash_attention_2"]


def _hf_config(cls, **kwargs):
    config = cls(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        use_cache=False,
        **kwargs,
    )
    return config


def _hf_model(cls, config, gradient_checkpointing=False):
    set_seed(0)
    model = cls(config).double()
    # Built as sdpa (transformers refuses flash on a CPU-only host), then dispatched to the flash entry.
    model.config._attn_implementation = "flash_attention_2"
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model


def _hf_logits(model, input_ids, position_ids):
    return model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits[0]


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_hf_qwen3_share_on_matches_share_off_and_rows(hf_flash_reference, gradient_checkpointing):
    set_seed(2)
    batch, groups, scored = _batch()
    shared, unshared = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config), gradient_checkpointing)
    rows = _run(model, _hf_logits, batch, scored)
    _assert_runs_match(_run(model, _hf_logits, batch, scored, unshared), rows)
    _assert_runs_match(_run(model, _hf_logits, batch, scored, shared), rows, SHARED_GRAD_ATOL)
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is _hf_flash_reference


def test_hf_registry_wrapper_is_scoped_and_not_stacked(hf_flash_reference):
    set_seed(2)
    batch, groups, scored = _batch()
    shared, _ = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    other = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    packed = pack_with_layout(batch, shared)
    expected = _hf_logits(model, packed["input_ids"], packed["position_ids"]).detach()
    with using_layout(model, shared):
        wrapper = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        assert wrapper is not _hf_flash_reference
        with using_layout(model, shared):
            assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is wrapper
            nested = _hf_logits(model, packed["input_ids"], packed["position_ids"]).detach()
        # A model without a stash in the same process still runs plain flash attention.
        other_out = _hf_logits(other, packed["input_ids"], packed["position_ids"]).detach()
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is _hf_flash_reference
    with using_layout(model, shared):
        once = _hf_logits(model, packed["input_ids"], packed["position_ids"]).detach()
    torch_assert_equal(nested, once)
    torch_assert_equal(other_out, expected)
    assert (once - expected).abs().max() > 1e-3


def test_hf_registry_restores_global_entry():
    batch, groups, _ = _batch()
    shared, _ = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    original = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    with pytest.raises(RuntimeError, match="boom"):
        with using_layout(model, shared):
            assert "flash_attention_2" in ALL_ATTENTION_FUNCTIONS._local_mapping
            raise RuntimeError("boom")
    assert "flash_attention_2" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original


def test_hf_sliding_window_fails_closed(hf_flash_reference):
    batch, groups, _ = _batch()
    shared, _ = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    attention = model.model.layers[0].self_attn
    attention.sliding_window = 4
    with pytest.raises(ValueError, match="sliding window"):
        with using_layout(model, shared):
            pass
    attention.sliding_window = None
    wrapper = _hf_prefix_attention_fn(_hf_flash_reference)
    q = torch.zeros(1, 4, shared.num_tokens, HEAD_DIM, dtype=torch.float64)
    k = torch.zeros(1, 2, shared.num_tokens, HEAD_DIM, dtype=torch.float64)
    with using_layout(model, shared):
        with pytest.raises(ValueError, match="sliding-window"):
            wrapper(attention, q, k, k, None, sliding_window=4)
        with pytest.raises(ValueError, match="already gathered"):
            wrapper(attention, q, k.index_select(2, shared.kv_index), k, None)


def test_hf_qwen3_moe_needs_router_replay_decision(hf_flash_reference):
    set_seed(3)
    batch, groups, scored = _batch()
    config = _hf_config(
        Qwen3MoeConfig, moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2, experts_implementation="eager"
    )
    model = _hf_model(Qwen3MoeForCausalLM, config)
    unstated, _ = _layouts(batch, groups)
    with pytest.raises(ValueError, match="MoE model"):
        _run(model, _hf_logits, batch, scored, unstated)
    shared, unshared = _layouts(batch, groups, router_replay=False)
    rows = _run(model, _hf_logits, batch, scored)
    _assert_runs_match(_run(model, _hf_logits, batch, scored, unshared), rows)
    _assert_runs_match(_run(model, _hf_logits, batch, scored, shared), rows, SHARED_GRAD_ATOL)


def test_position_id_boundary_helpers_fail_closed_under_sharing():
    try:
        from arctic_platform.model.implementations.deepseek_v4.modeling_deepseek_v4 import (
            _seq_lens_from_position_ids as v4_seq_lens,
        )
        from arctic_platform.model.implementations.deepseek_v4.sequence import (
            get_cu_seqlens_from_position_ids as v4_cu,
        )
        from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import (
            _seq_lens_from_position_ids as v41_seq_lens,
        )
        from arctic_platform.model.implementations.dflash2.online import row_spans
    except Exception as exc:
        pytest.skip(f"DeepSeek/DFlash2 helpers unavailable: {exc}")
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    positions = pack_with_layout(batch, shared)["position_ids"]
    helpers = (v4_cu, v4_seq_lens, v41_seq_lens, row_spans)
    with using_layout(_ap_attention(), shared):
        for helper in helpers:
            with pytest.raises(ValueError, match="position_ids == 0"):
                helper(positions)
    with using_layout(_ap_attention(), unshared):
        for helper in helpers:
            helper(pack_with_layout(batch, unshared)["position_ids"])
