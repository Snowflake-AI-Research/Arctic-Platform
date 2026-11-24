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
Hugging Face models run transformers' own ``flash_attention_forward``; only the kernels it imports are swapped.

Each model call runs three ways on one batch: prefix sharing on, a sharing-off layout, and every row on its own.
Response log-probs and every parameter gradient must agree in float64; disabling the prefix gather must not.
"""

import functools
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM
from transformers import Qwen3MoeConfig
from transformers import Qwen3MoeForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from arctic_platform.common.packing import PackedLayout
from arctic_platform.common.packing import layout as layout_module
from arctic_platform.common.packing import pack_with_layout
from arctic_platform.common.packing import using_layout
from arctic_platform.common.packing.layout import _hf_prefix_attention_fn
from arctic_platform.common.packing.layout import unshared_groups
from arctic_platform.model.implementations.afmoe import modeling_afmoe as afmoe_module
from arctic_platform.model.implementations.afmoe.modeling_afmoe import AfmoeAttentionConfig
from arctic_platform.model.implementations.afmoe.modeling_afmoe import AfmoeFlashAttention
from arctic_platform.model.implementations.afmoe.modeling_afmoe import AfmoeSDPAAttention
from arctic_platform.model.implementations.gpu.sp import gated_delta_net
from arctic_platform.model.implementations.layers import attn as attn_module
from arctic_platform.model.implementations.layers.attn import AttentionConfig
from arctic_platform.model.implementations.layers.attn import FlashAttention
from arctic_platform.model.implementations.layers.attn import SDPAAttention
from arctic_platform.model.implementations.layers.sequence import get_cu_seqlens_from_position_ids
from arctic_platform.testing_utils import flash_attn_varlen_reference
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal
from tests.model.test_gated_delta_continuation import reference_rule

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


@pytest.fixture(autouse=True)
def _reference_flash_attn(monkeypatch):
    """The reference kernel stands in for flash-attn 2.8, FA3 and FA4 here, installed or not."""
    monkeypatch.setattr(
        layout_module, "_installed_version", {"flash_attn": (2, 8), "flash_attn_3": (3, 0), "flash_attn_4": (4, 0)}.get
    )
    monkeypatch.setattr(layout_module, "_fa4_varlen_backward_available", lambda: True)


def _batch(lengths=(9, 8, 10, 7), prompt_len=PROMPT_LEN):
    """Rows 0-2 share a prompt; row 3 is its own group. Right padding, positions 0..L-1."""
    width = max(lengths)
    ids = torch.randint(1, VOCAB, (len(lengths), width))
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    ids[1:3, :prompt_len] = ids[0, :prompt_len]
    ids = ids * mask
    positions = torch.arange(width).expand_as(mask) * mask
    scored = mask.clone()
    scored[:, :prompt_len] = 0
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}, [[0, 1, 2], [3]], scored


def _layouts(batch, groups, prompt_len=PROMPT_LEN, **build_kwargs):
    shared = PackedLayout.build(
        batch["attention_mask"],
        groups,
        [prompt_len - 1 if len(rows) > 1 else 0 for rows in groups],
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
        packed = pack_with_layout({**batch, "loss_mask": scored}, layout)
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


def _ap_attention(sliding_window=None, kind="ap"):
    """``kind="afmoe"`` builds AFMoE's gated attention, which rotates only its sliding-window (local) layers."""
    if kind == "afmoe":
        config = AfmoeAttentionConfig(
            hidden_size=32,
            head_dim=HEAD_DIM,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            is_local_attention=sliding_window is not None,
            sliding_window=sliding_window,
        )
        module = AfmoeFlashAttention(config, flash_attn_version=2)
    else:
        module = FlashAttention(_attention_config(), flash_attn_version=2)
        if sliding_window is not None:
            module.sliding_window = sliding_window
    module._flash_attn_call = flash_attn_varlen_reference
    return module


class APStack(nn.Module):
    """Embedding, Arctic-Platform flash attention + MLP blocks, LM head. Segments come from the model-stack helper."""

    def __init__(self, layers=2, sliding_window=None, use_checkpoint=False, kind="ap"):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 32)
        self.attn = nn.ModuleList(_ap_attention(sliding_window, kind) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32)) for _ in range(layers))
        self.head = nn.Linear(32, VOCAB)
        self.use_checkpoint = use_checkpoint

    def _block(self, i, hidden, cos, sin, cu_seqlens, max_seqlen):
        hidden = hidden + self.attn[i](hidden, (cos, sin), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)[0]
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


@pytest.mark.parametrize("kind", ["ap", "afmoe"])
@pytest.mark.parametrize("sliding_window", [None, 3])
def test_ap_stack_share_on_matches_share_off_and_rows(sliding_window, kind):
    set_seed(1)
    batch, groups, scored = _batch()
    shared, unshared = _layouts(batch, groups)
    model = _ap_stack(sliding_window=sliding_window, kind=kind)
    rows = _run(model, _ap_logits, batch, scored)
    _assert_runs_match(_run(model, _ap_logits, batch, scored, unshared), rows)
    _assert_runs_match(_run(model, _ap_logits, batch, scored, shared), rows, SHARED_GRAD_ATOL)


@pytest.mark.parametrize("kind", ["ap", "afmoe"])
def test_ap_stack_without_prefix_gather_does_not_match(monkeypatch, kind):
    set_seed(1)
    batch, groups, scored = _batch()
    shared, _ = _layouts(batch, groups)
    model = _ap_stack(kind=kind)
    rows = _run(model, _ap_logits, batch, scored)

    def no_gather(k, v, cu_seqlens, max_seqlen, layout):
        cu = layout.cu_seqlens.to(torch.int32)
        return k, v, cu, cu, layout.max_len_q, layout.max_len_q

    monkeypatch.setattr(attn_module if kind == "ap" else afmoe_module, "prefix_varlen_args", no_gather)
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


def test_ap_stack_needs_its_own_unnested_context():
    set_seed(1)
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    model, other = _ap_stack(), _ap_stack()
    packed = pack_with_layout(batch, shared)
    packed_off = pack_with_layout(batch, unshared)
    with using_layout(model, shared):
        # Same width as the active layout, so its segments reach other's attention, which has no stash.
        with pytest.raises(ValueError, match="no PackedLayout stashed"):
            other(packed["input_ids"], packed["position_ids"])
        # The model-stack segment helper cannot tell which of two nested layouts it serves.
        with using_layout(other, unshared):
            with pytest.raises(ValueError, match="nested using_layout contexts"):
                other(packed_off["input_ids"], packed_off["position_ids"])
    with using_layout(other, unshared):
        other(packed_off["input_ids"], packed_off["position_ids"])


def test_fa4_gets_max_seqlen_only_under_sharing():
    calls = []

    def fa4(q, k, v, **kwargs):
        calls.append(kwargs)
        return q

    q = torch.zeros(4, 1, 8)
    cu_q, cu_k = torch.tensor([0, 1, 4]), torch.tensor([0, 3, 6])
    for sharing in (False, True):
        attn_module.flash_varlen_call(fa4, 4, q, q, q, cu_q, cu_k, 3, 3, sharing=sharing, causal=True)
    assert calls[0] == {"causal": True, "cu_seqlens_q": cu_q, "cu_seqlens_k": cu_k}
    assert calls[1] == {**calls[0], "max_seqlen_q": 3, "max_seqlen_k": 3}


def test_ap_sdpa_is_rejected_under_sharing():
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    module = SDPAAttention(_attention_config(use_qk_norm=False))
    with pytest.raises(ValueError, match="SDPAAttention"):
        with using_layout(module, shared):
            pass
    with using_layout(module, unshared):
        pass


def test_sdpa_refuses_to_run_on_a_packed_layout():
    """SDPA masks one causal rectangle per row, so packed segments would attend to each other."""
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    afmoe = AfmoeSDPAAttention(
        AfmoeAttentionConfig(
            hidden_size=32,
            head_dim=HEAD_DIM,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            is_local_attention=False,
        )
    )
    # SDPAAttention repeats its 2 KV heads itself; AFMoE's SDPA core expects them already repeated.
    cases = ((SDPAAttention(_attention_config(use_qk_norm=False)), 2), (afmoe, 4))
    for module, kv_heads in cases:
        q = torch.zeros(1, 4, unshared.num_tokens, HEAD_DIM)
        k = torch.zeros(1, kv_heads, unshared.num_tokens, HEAD_DIM)
        module._attention_core(q, k, k)
        with using_layout(module, unshared):
            with pytest.raises(ValueError, match="no varlen path"):
                module._attention_core(q, k, k)
        with using_layout(_ap_attention(), shared):
            with pytest.raises(ValueError, match="no PackedLayout stashed"):
                module._attention_core(q, k, k)


def _reference_flash_fn(q, k, v, causal=True, softmax_scale=None, window_size=(-1, -1), **unused):
    """Dense ``[B, S, H, D]`` flash call: each batch row is one segment of the varlen reference."""
    batch_size, len_q, len_k = q.shape[0], q.shape[1], k.shape[1]
    cu_q = torch.arange(0, (batch_size + 1) * len_q, len_q, dtype=torch.int32)
    cu_k = torch.arange(0, (batch_size + 1) * len_k, len_k, dtype=torch.int32)
    out = flash_attn_varlen_reference(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_q,
        cu_k,
        causal=causal,
        window_size=window_size,
        softmax_scale=softmax_scale,
    )
    return out.view(q.shape)


def _flatten(items):
    for item in items:
        if isinstance(item, tuple):
            yield from _flatten(item)
        else:
            yield item


@pytest.fixture
def hf_flash_reference(monkeypatch):
    """Run transformers' own ``flash_attention_forward`` with only the flash kernels swapped for the CPU reference.

    transformers caches the kernels it imports in ``modeling_flash_attention_utils`` globals (4.57:
    ``_flash_fn`` .. ``_process_flash_kwargs_fn``; 5.x also ``_loaded_implementation`` and the paged
    entries). Seeding that cache keeps its kwarg routing -- ``cu_seq_lens_*`` / ``max_length_*`` to the varlen
    kernel, segment inference from ``position_ids``, the window mapping -- on the path under test. Each varlen
    kernel call's keyword arguments plus its K/V token count are recorded in ``calls``.
    """
    from transformers import modeling_flash_attention_utils as fa_utils
    from transformers.integrations.flash_attention import flash_attention_forward

    needed = ("lazy_import_flash_attention", "_lazy_define_process_function", "_pad_input", "_unpad_input")
    missing = [name for name in needed if not hasattr(fa_utils, name)]
    if missing:
        pytest.skip(f"transformers.modeling_flash_attention_utils has no {missing}; kernel cache layout changed")
    calls = []

    # ``wraps`` keeps the reference's signature, which transformers inspects to pick the kwargs it forwards.
    @functools.wraps(flash_attn_varlen_reference)
    def varlen(q, k, v, *args, **kwargs):
        calls.append({**kwargs, "k_tokens": k.shape[0]})
        return flash_attn_varlen_reference(q, k, v, *args, **kwargs)

    cache = {
        "_flash_fn": _reference_flash_fn,
        "_flash_varlen_fn": varlen,
        "_pad_fn": fa_utils._pad_input,
        "_unpad_fn": fa_utils._unpad_input,
        "_process_flash_kwargs_fn": fa_utils._lazy_define_process_function(varlen),
        "_flash_with_kvcache_fn": None,
        "_process_with_kvcache_kwargs_fn": dict,
        "_loaded_implementation": "flash_attention_2",
    }
    for name, value in cache.items():
        monkeypatch.setattr(fa_utils, name, value, raising=False)
    if varlen not in list(_flatten(fa_utils.lazy_import_flash_attention("flash_attention_2"))):
        pytest.skip("transformers' lazy_import_flash_attention ignores the seeded kernel cache")
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is flash_attention_forward
    assert "flash_attention_2" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    return SimpleNamespace(forward=flash_attention_forward, calls=calls)


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


def _hf_model(cls, config, use_reentrant=None):
    """``use_reentrant=None`` leaves gradient checkpointing off; transformers 4.x defaults it to ``True``, 5.x to ``False``."""
    set_seed(0)
    model = cls(config).double()
    # Built as sdpa (transformers refuses flash on a CPU-only host), then dispatched to the flash entry.
    model.config._attn_implementation = "flash_attention_2"
    if use_reentrant is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    model.train()
    return model


def _hf_logits(model, input_ids, position_ids):
    return model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits[0]


@pytest.mark.parametrize("use_reentrant", [None, False, True], ids=["no_checkpoint", "non_reentrant", "reentrant"])
def test_hf_qwen3_share_on_matches_share_off_and_rows(hf_flash_reference, use_reentrant):
    set_seed(2)
    batch, groups, scored = _batch()
    shared, unshared = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config), use_reentrant)
    rows = _run(model, _hf_logits, batch, scored)
    _assert_runs_match(_run(model, _hf_logits, batch, scored, unshared), rows)
    _assert_runs_match(_run(model, _hf_logits, batch, scored, shared), rows, SHARED_GRAD_ATOL)
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is hf_flash_reference.forward


def test_hf_flash_forward_hands_layout_segments_to_varlen_kernel(hf_flash_reference):
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    layers = model.config.num_hidden_layers
    for layout in (shared, unshared):
        hf_flash_reference.calls.clear()
        packed = pack_with_layout(batch, layout)
        with using_layout(model, layout):
            _hf_logits(model, packed["input_ids"], packed["position_ids"])
        # Share-on: the wrapper's cu_seq_lens / max_length kwargs. Share-off: transformers' own inference from
        # position_ids, which matches the unshared layout because every row restarts at position 0.
        assert len(hf_flash_reference.calls) == layers
        for call in hf_flash_reference.calls:
            assert call["cu_seqlens_q"].tolist() == layout.cu_seqlens.tolist()
            assert call["cu_seqlens_k"].tolist() == layout.cu_seqlens_k.tolist()
            assert int(call["max_seqlen_q"]) == layout.max_len_q
            assert int(call["max_seqlen_k"]) == layout.max_len_k
            assert call["k_tokens"] == layout.kv_index.numel()
            assert call["causal"] is True
    assert shared.kv_index.numel() > shared.num_tokens


def test_hf_registry_wrapper_is_scoped_and_not_stacked(hf_flash_reference):
    set_seed(2)
    batch, groups, scored = _batch()
    shared, unshared = _layouts(batch, groups)
    model = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    other = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    packed = pack_with_layout(batch, shared)
    packed_off = pack_with_layout(batch, unshared)
    expected = _hf_logits(model, packed["input_ids"], packed["position_ids"]).detach()
    other_expected = _hf_logits(other, packed_off["input_ids"], packed_off["position_ids"]).detach()
    with using_layout(model, shared):
        wrapper = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        assert wrapper is not hf_flash_reference.forward
        with using_layout(model, shared):
            assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is wrapper
            nested = _hf_logits(model, packed["input_ids"], packed["position_ids"]).detach()
        # A model with no stash of its own refuses to run inside another model's sharing context ...
        with pytest.raises(ValueError, match="no PackedLayout stashed"):
            _hf_logits(other, packed_off["input_ids"], packed_off["position_ids"])
        # ... and runs as it does on its own once wrapped in its own unshared layout.
        with using_layout(other, unshared):
            other_out = _hf_logits(other, packed_off["input_ids"], packed_off["position_ids"]).detach()
    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is hf_flash_reference.forward
    with using_layout(model, shared):
        once = _hf_logits(model, packed["input_ids"], packed["position_ids"]).detach()
    torch_assert_equal(nested, once)
    torch_assert_equal(other_out, other_expected)
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
    wrapper = _hf_prefix_attention_fn(hf_flash_reference.forward)
    q = torch.zeros(1, 4, shared.num_tokens, HEAD_DIM, dtype=torch.float64)
    k = torch.zeros(1, 2, shared.num_tokens, HEAD_DIM, dtype=torch.float64)
    with using_layout(model, shared):
        with pytest.raises(ValueError, match="sliding-window"):
            wrapper(attention, q, k, k, None, sliding_window=4)
        with pytest.raises(ValueError, match="already gathered"):
            wrapper(attention, q, k.index_select(2, shared.kv_index), k, None)
        # One dropout mask on a shared root would stand in for every row's own.
        with pytest.raises(ValueError, match="dropout"):
            wrapper(attention, q, k, k, None, dropout=0.1)
        with pytest.raises(ValueError, match="needs a flash attention implementation, got 'sdpa'"):
            _hf_prefix_attention_fn(hf_flash_reference.forward, "sdpa")(attention, q, k, k, None)


@pytest.mark.parametrize("impl", ["flash_attention_2", "sdpa"])
def test_hf_attention_refuses_another_models_sharing_context(hf_flash_reference, impl):
    """The registry wrapper covers every implementation, not just the ones the sharing model uses."""
    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    hf = _hf_model(Qwen3ForCausalLM, _hf_config(Qwen3Config))
    hf.config._attn_implementation = impl
    packed_off = pack_with_layout(batch, unshared)
    original = ALL_ATTENTION_FUNCTIONS[impl]
    with using_layout(_ap_stack(), shared):
        with pytest.raises(ValueError, match="Qwen3Attention ran with no PackedLayout stashed"):
            _hf_logits(hf, packed_off["input_ids"], packed_off["position_ids"])
    assert ALL_ATTENTION_FUNCTIONS[impl] is original
    _hf_logits(hf, packed_off["input_ids"], packed_off["position_ids"])


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


class _PastSegmentGuard(Exception):
    pass


def test_qwen3_5_moe_segment_boundaries_come_from_the_stash():
    try:
        from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeModel,
        )
    except Exception as exc:
        pytest.skip(f"Qwen3.5-MoE model unavailable: {exc}")

    def first_layer(hidden_states, residual, *, cu_seqlens, max_seqlen, **unused):
        raise _PastSegmentGuard(cu_seqlens.tolist(), max_seqlen)

    # Only the code up to the first decoder layer runs: config, embedding, the segment split, rotary embeddings.
    def fake(layout=None):
        model = SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2"),
            embed_tokens=lambda ids: torch.zeros(*ids.shape, 4),
            rotary_emb=lambda hidden, positions: None,
            layers=[first_layer],
        )
        if layout is not None:
            setattr(model, layout_module.LAYOUT_ATTR, layout)
        return model

    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    for layout in (shared, unshared):
        packed = pack_with_layout(batch, layout)
        with using_layout(_ap_attention(), layout):
            # position_ids == 0 would glue each continuation onto the root before it.
            with pytest.raises(_PastSegmentGuard) as stopped:
                Qwen3_5MoeModel.forward(
                    fake(layout), input_ids=packed["input_ids"], position_ids=packed["position_ids"]
                )
            assert stopped.value.args == (layout.cu_seqlens.tolist(), layout.max_len_q)
            with pytest.raises(ValueError, match="Qwen3.5-MoE segment boundaries ran with no PackedLayout stashed"):
                Qwen3_5MoeModel.forward(fake(), input_ids=packed["input_ids"], position_ids=packed["position_ids"])
    with pytest.raises(_PastSegmentGuard) as stopped:
        packed = pack_with_layout(batch, unshared)
        Qwen3_5MoeModel.forward(fake(), input_ids=packed["input_ids"], position_ids=packed["position_ids"])
    assert stopped.value.args[0] == unshared.cu_seqlens.tolist()


# GatedDeltaNet hybrids. The CPU references stand in for FLA's chunk rule and causal-conv1d (whose pinned
# versions and CUDA numerics are the GPU gate's job): in transformers' stock forward, which the per-row runs take,
# and in the prefix-continuation adapter, which every run inside using_layout takes.
GDN_LENGTHS = (70, 67, 72, 66)
# The shared root is the first 64 tokens: GatedDeltaNet roots must fill whole 64-token chunks.
GDN_PROMPT_LEN = 65
_MOE = dict(
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    num_experts=4,
    num_experts_per_tok=2,
    experts_implementation="eager",
)
_GDN_MODELS = {
    "qwen3_5": ("Qwen3_5TextConfig", "Qwen3_5ForCausalLM", {}),
    "qwen3_5_moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM", _MOE),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextForCausalLM", _MOE),
}
# These gradients are summed in float32, so they differ between runs that sum different token sets: transformers
# computes g from A_log.float() and a.float(), and the Qwen3.5-family RMSNorm scales by (1 + weight.float()).
_GDN_FP32_GRADS = ("A_log", "in_proj_a.weight", "in_proj_ba.weight", "norm.weight", "layernorm.weight")


def _reference_conv(x, weight, bias=None, activation=None, seq_idx=None, **unused):
    """causal-conv1d's contract: the convolution restarts wherever ``seq_idx`` ``[1, T]`` changes."""
    if seq_idx is None:
        counts = torch.tensor([x.shape[-1]])
    else:
        counts = torch.unique_consecutive(seq_idx[0], return_counts=True)[1]
    return gated_delta_net._fallback_depthwise_causal_convolution(
        x, weight=weight, bias=bias, activation=activation, global_cu_seqlens=F.pad(counts.cumsum(0), (1, 0))
    )


def _reference_stock_rule(
    query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, **kwargs
):
    """transformers' torch chunk rule in float64; it only ever sees one row here."""
    assert kwargs.pop("cu_seqlens", None) is None and query.shape[0] == 1
    return reference_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=torch.tensor([0, query.shape[1]]),
        use_qk_l2norm_in_kernel=kwargs.get("use_qk_l2norm_in_kernel", False),
    )


@pytest.fixture
def gdn_reference(monkeypatch):
    for name in _GDN_MODELS:
        modeling = pytest.importorskip(f"transformers.models.{name}.modeling_{name}")
        monkeypatch.setattr(modeling, "causal_conv1d_fn", _reference_conv)
        monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", _reference_stock_rule)
    kernels = {"causal_conv1d_fn": _reference_conv, "chunk_gated_delta_rule": reference_rule}
    monkeypatch.setattr(gated_delta_net, "_resolve_gated_delta_net_kernel", lambda module, name: kernels[name])
    monkeypatch.setattr(gated_delta_net, "prefix_continuation_kernel_blocker", lambda conv, rule: None)


def _gdn_hf_model(name):
    config_name, model_name, extra = _GDN_MODELS[name]
    configuration = importlib.import_module(f"transformers.models.{name}.configuration_{name}")
    modeling = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
    config = getattr(configuration, config_name)(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        use_cache=False,
        **extra,
    )
    model = _hf_model(getattr(modeling, model_name), config)
    assert gated_delta_net.apply_gated_delta_net_prefix_continuation(model) == 1
    return model


def _gdn_batch(name):
    batch, groups, scored = _batch(GDN_LENGTHS, GDN_PROMPT_LEN)
    # Every model with experts must state its router-replay decision under sharing; these run without replay.
    build = {} if name == "qwen3_5" else {"router_replay": False}
    return batch, groups, scored, _layouts(batch, groups, GDN_PROMPT_LEN, **build)


def _assert_gdn_runs_match(got, want, grad_atol=F64_ATOL):
    torch.testing.assert_close(got[0], want[0], rtol=0, atol=F64_ATOL, msg=lambda m: f"outputs: {m}")
    assert got[1].keys() == want[1].keys()
    for name in want[1]:
        rtol = 1e-5 if name.endswith(_GDN_FP32_GRADS) else 0
        torch.testing.assert_close(
            got[1][name], want[1][name], rtol=rtol, atol=grad_atol, msg=lambda m, name=name: f"{name}: {m}"
        )


@pytest.mark.parametrize("name", list(_GDN_MODELS))
def test_hf_gated_delta_net_share_on_matches_share_off_and_rows(hf_flash_reference, gdn_reference, name):
    set_seed(4)
    batch, groups, scored, (shared, unshared) = _gdn_batch(name)
    model = _gdn_hf_model(name)
    rows = _run(model, _hf_logits, batch, scored)
    _assert_gdn_runs_match(_run(model, _hf_logits, batch, scored, unshared), rows)
    _assert_gdn_runs_match(_run(model, _hf_logits, batch, scored, shared), rows, SHARED_GRAD_ATOL)


def test_hf_gated_delta_net_without_the_root_handoff_does_not_match(hf_flash_reference, gdn_reference, monkeypatch):
    set_seed(4)
    batch, groups, scored, (shared, _) = _gdn_batch("qwen3_5")
    model = _gdn_hf_model("qwen3_5")
    rows = _run(model, _hf_logits, batch, scored)
    # Continuations start from a zero state and an empty conv window, as if each were its own row.
    monkeypatch.setattr(gated_delta_net, "gated_delta_continuation", lambda layout, kernel_size, device: None)
    assert (_run(model, _hf_logits, batch, scored, shared)[0] - rows[0]).abs().max() > 1e-4


def _custom_qwen3_5_moe(monkeypatch):
    monkeypatch.setenv("TILELANG_CACHE_DIR", "/tmp/tilelang-cache")
    try:
        from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import configuration_qwen3_5_moe
        from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe import modeling_qwen3_5_moe
    except Exception as exc:
        pytest.skip(f"Qwen3.5-MoE model unavailable: {exc}")
    return configuration_qwen3_5_moe, modeling_qwen3_5_moe


class _Float64GatedNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states, gate):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + 1e-6) * F.silu(gate)


def test_custom_qwen3_5_moe_gated_delta_net_continues_shared_prefixes(gdn_reference, monkeypatch):
    configuration, modeling = _custom_qwen3_5_moe(monkeypatch)
    config = configuration.Qwen3_5MoeConfig(
        hidden_size=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        fp32_parity=False,
    )
    set_seed(5)
    module = modeling.Qwen3_5MoeGatedDeltaNet(config)
    # Qwen3_5MoeRMSNormGated normalizes in float32, which rounds the row and packed runs differently (~1e-8).
    module.norm = _Float64GatedNorm(module.head_v_dim)
    module.double()
    module._causal_conv1d_fn, module._chunk_gated_delta_rule = _reference_conv, reference_rule

    batch, groups, _ = _batch(GDN_LENGTHS, GDN_PROMPT_LEN)
    # Any class named *Moe* counts as an MoE model, which must state its router-replay decision.
    shared, unshared = _layouts(batch, groups, GDN_PROMPT_LEN, router_replay=False)
    lengths = batch["attention_mask"].sum(1).tolist()
    hidden = torch.randn(len(lengths), max(lengths), config.hidden_size, dtype=torch.float64)
    hidden[1:3, :GDN_PROMPT_LEN] = hidden[0, :GDN_PROMPT_LEN]
    weights = [torch.randn(length, config.hidden_size, dtype=torch.float64) for length in lengths]

    def run(layout=None, stock_packed=False):
        module.zero_grad()
        if stock_packed:
            packed = module(unshared.pack(hidden).unsqueeze(0), cu_seqlens=unshared.cu_seqlens)[0]
            outputs = [packed[unshared.row_index[row, :length]] for row, length in enumerate(lengths)]
        elif layout is None:
            outputs = [
                module(hidden[row : row + 1, :length], cu_seqlens=torch.tensor([0, length], dtype=torch.int32))[0]
                for row, length in enumerate(lengths)
            ]
        else:
            with using_layout(module, layout):
                # Rows glued at position_ids == 0 boundaries must not be read under a layout.
                packed = module(layout.pack(hidden).unsqueeze(0), cu_seqlens=unshared.cu_seqlens)[0]
            outputs = [packed[layout.row_index[row, :length]] for row, length in enumerate(lengths)]
        sum((output * weight).sum() for output, weight in zip(outputs, weights)).backward()
        return torch.cat(outputs).detach(), {name: p.grad.clone() for name, p in module.named_parameters()}

    rows = run()
    _assert_gdn_runs_match(run(stock_packed=True), rows)
    _assert_gdn_runs_match(run(unshared), rows)
    _assert_gdn_runs_match(run(shared), rows, SHARED_GRAD_ATOL)


def test_custom_qwen3_5_moe_attention_reads_the_stash(monkeypatch):
    _, modeling = _custom_qwen3_5_moe(monkeypatch)
    flash, sdpa = modeling.Qwen3_5MoeGatedFlashAttention, modeling.Qwen3_5MoeGatedSDPAAttention
    assert flash in layout_module._NATIVE_PREFIX_ATTENTION
    calls = []

    def compute(q, k, v, cu_q, max_q, cu_k, max_k):
        calls.append((k.shape[0], cu_q.tolist(), cu_k.tolist(), max_q, max_k))
        return q

    batch, groups, _ = _batch()
    shared, unshared = _layouts(batch, groups)
    for layout in (shared, unshared):
        stale = torch.tensor([0, layout.num_tokens], dtype=torch.int32)
        q = torch.zeros(1, layout.num_tokens, 4, HEAD_DIM)
        k = torch.zeros(1, layout.num_tokens, 2, HEAD_DIM)
        stub = SimpleNamespace(_compute_attention=compute, **{layout_module.LAYOUT_ATTR: layout})
        with using_layout(_ap_attention(), layout):
            flash._attention_core(stub, q, k, k, stale, layout.num_tokens)
            with pytest.raises(ValueError, match="no varlen path"):
                sdpa._attention_core(stub, q, k, k)
        assert calls.pop() == (
            layout.kv_index.numel(),
            layout.cu_seqlens.tolist(),
            layout.cu_seqlens_k.tolist(),
            layout.max_len_q,
            layout.max_len_k,
        )
    unstashed = SimpleNamespace(_compute_attention=compute)
    tokens = unshared.num_tokens
    with using_layout(_ap_attention(), shared):
        with pytest.raises(ValueError, match="no PackedLayout stashed"):
            flash._attention_core(unstashed, q, k, k, stale, tokens)
    flash._attention_core(unstashed, q, k, k, stale, tokens)
    assert calls.pop() == (tokens, stale.tolist(), stale.tolist(), tokens, tokens)
