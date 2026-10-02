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

"""CPU checks for GLM-5.2 sparse MLA backend selection from model config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import _lse_to_ref_convention
from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import flash_mla_available
from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import set_sparse_mla_backend
from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import sparse_mla_backend
from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import sparse_mla_flashmla_apply
from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import sparse_mla_ref_apply


@pytest.fixture(autouse=True)
def _restore_sparse_mla_backend():
    previous = sparse_mla_backend()
    yield
    set_sparse_mla_backend(previous)


def test_set_sparse_mla_backend_ref():
    assert set_sparse_mla_backend("ref") == "ref"
    assert sparse_mla_backend() == "ref"


def test_set_sparse_mla_backend_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown sparse_mla_backend"):
        set_sparse_mla_backend("bogus")


def test_flashmla_raises_when_kernel_missing(monkeypatch):
    import arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla as kernel

    monkeypatch.setattr(kernel, "flash_mla_sparse_fwd", None)

    def _missing():
        raise ImportError(kernel._FLASHMLA_INSTALL_HINT)

    monkeypatch.setattr(kernel, "_load_flash_mla_sparse_fwd", _missing)
    with pytest.raises(ImportError, match="flash_mla is not installed"):
        set_sparse_mla_backend("flashmla")


def test_flashmla_load_error_is_not_hidden_as_missing(monkeypatch):
    import arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla as kernel

    monkeypatch.setattr(kernel, "flash_mla_sparse_fwd", None)

    def _broken():
        raise RuntimeError("cannot load libflashmla.so")

    monkeypatch.setattr(kernel, "_load_flash_mla_sparse_fwd", _broken)
    with pytest.raises(RuntimeError, match="libflashmla"):
        set_sparse_mla_backend("flashmla")
    with pytest.raises(RuntimeError, match="libflashmla"):
        flash_mla_available()


def test_tilelang_keeps_local_row_causal_mask_and_inf_grad_merge_flag():
    """sp=1 TileLang still masks with local s_i; only the Inf-grad flag is new.

    Reading the kernel sources (not a hand-built tensor) so a revert of either
    expression fails this test. Context-parallel global-index masking lives on
    the stacked CP PR.
    """
    kernels = Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/glm52/models/kernels"
    fwd = (kernels / "sparse_mla_fwd.py").read_text()
    bwd = (kernels / "sparse_mla_bwd.py").read_text()
    flash = (kernels / "sparse_mla_flashmla.py").read_text()
    assert "mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i" in fwd
    assert "max_kv_i = q_i" in fwd
    assert "mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] <= max_kv_i" in bwd
    assert "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False" in bwd
    assert "flash_indices > positions" in flash


def test_lse_to_ref_convention_maps_posinf_to_neginf():
    lse = torch.tensor([[0.5, float("inf")], [float("-inf"), 1.25]])
    got = _lse_to_ref_convention(lse)
    assert got[0, 0].item() == pytest.approx(0.5)
    assert torch.isneginf(got[0, 1])
    assert torch.isneginf(got[1, 0])
    assert got[1, 1].item() == pytest.approx(1.25)


def _stub_flashmla_on_cpu(monkeypatch):
    import arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla as kernel

    def _fake_flashmla_fwd_one(q, kv, flash_indices, sm_scale, d_v):
        return kernel._ref_sparse_mla_fwd_one(
            q,
            kv,
            flash_indices,
            sm_scale,
            q.shape[0],
            d_v,
        )

    monkeypatch.setattr(kernel, "_ensure_flash_mla_sparse_fwd", lambda: object())
    monkeypatch.setattr(kernel, "_flashmla_fwd_one", _fake_flashmla_fwd_one)
    return kernel


def _cpu_sparse_mla_tensors(batch: int = 2, seq_len: int = 5, topk: int = 7):
    dim_qk = 576
    heads = 2
    q = torch.randn(batch, seq_len, heads, dim_qk, requires_grad=True)
    kv = torch.randn(batch, seq_len + 1, 1, dim_qk)
    kv[:, -1].zero_()
    kv.requires_grad_()

    positions = torch.arange(seq_len)
    keys = torch.arange(topk).unsqueeze(0)
    indices = keys.expand(seq_len, topk).clamp(max=seq_len).to(torch.int64).clone()
    indices.masked_fill_(keys > positions.unsqueeze(1), seq_len)
    indices = indices.unsqueeze(0).unsqueeze(2).expand(batch, -1, -1, -1).clone()
    return q, kv, indices, 192.0**-0.5


def test_flashmla_saves_only_backward_tensors(monkeypatch):
    kernel = _stub_flashmla_on_cpu(monkeypatch)
    q, kv, indices, sm_scale = _cpu_sparse_mla_tensors()
    packed = []

    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: packed.append(tensor) or tensor,
        lambda tensor: tensor,
    ):
        out = sparse_mla_flashmla_apply(q, kv, indices, sm_scale)

    assert len(packed) == 3 + q.shape[0]
    saved_q, saved_kv, saved_lse, *saved_flash_indices = packed
    assert saved_q.data_ptr() == q.data_ptr()
    assert saved_kv.data_ptr() == kv.data_ptr()
    assert all(tensor.data_ptr() != indices.data_ptr() for tensor in packed)
    assert saved_lse.shape == q.shape[:3]
    assert not hasattr(out.grad_fn, "flash_indices_list")
    assert not hasattr(out.grad_fn, "lse")

    for b, saved_flash_indices_b in enumerate(saved_flash_indices):
        assert saved_flash_indices_b.dtype == torch.int32
        assert saved_flash_indices_b.shape == (q.shape[1], 1, indices.shape[-1])
        expected_flash_indices_b = kernel._prepare_flashmla_indices(
            indices[b, :, 0, :],
            q.shape[1],
            q.shape[1],
        )
        torch.testing.assert_close(saved_flash_indices_b, expected_flash_indices_b)


def test_flashmla_cpu_stub_matches_ref_backward_across_batches_and_chunks(monkeypatch):
    torch.manual_seed(0)
    kernel = _stub_flashmla_on_cpu(monkeypatch)
    monkeypatch.setattr(kernel, "_SPARSE_MLA_BWD_CHUNK", 2)
    q, kv, indices, sm_scale = _cpu_sparse_mla_tensors()
    q_ref = q.detach().clone().requires_grad_(True)
    kv_ref = kv.detach().clone().requires_grad_(True)

    out = sparse_mla_flashmla_apply(q, kv, indices, sm_scale)
    out_ref = sparse_mla_ref_apply(q_ref, kv_ref, indices, sm_scale)
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(q.grad, q_ref.grad)
    torch.testing.assert_close(kv.grad, kv_ref.grad)


def _prime_rl_sparse_mla_tensors(seq_len: int = 32, topk: int = 128, heads: int = 64):
    """Batched MQA layout used by GlmMoeDsaAttention (sentinel KV row).

    SM90 FlashMLA sparse prefill requires ``topk % 128 == 0`` (2 * B_TOPK).
    """
    dim_qk = 576
    batch = 1
    q = torch.randn(batch, seq_len, heads, dim_qk, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(batch, seq_len + 1, 1, dim_qk, device="cuda", dtype=torch.bfloat16)
    kv[:, -1].zero_()

    positions = torch.arange(seq_len, device="cuda")
    keys = torch.arange(topk, device="cuda").unsqueeze(0)
    indices = keys.expand(seq_len, topk).clamp(max=seq_len).to(torch.int32).clone()
    indices.masked_fill_(keys > positions.unsqueeze(1), seq_len)
    indices = indices.unsqueeze(0).unsqueeze(2)
    sm_scale = 192.0**-0.5
    return q, kv, indices, sm_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashMLA parity needs a GPU")
def test_flashmla_matches_ref_forward_and_backward():
    """Numeric lock: kernel fwd + tiled ref bwd must match full ref on short seqs."""
    if not flash_mla_available():
        pytest.skip("flash_mla / vllm._flashmla_C not installed")
    torch.manual_seed(0)
    q, kv, indices, sm_scale = _prime_rl_sparse_mla_tensors()
    q_ref = q.detach().clone().requires_grad_(True)
    kv_ref = kv.detach().clone().requires_grad_(True)
    q_fm = q.detach().clone().requires_grad_(True)
    kv_fm = kv.detach().clone().requires_grad_(True)

    set_sparse_mla_backend("ref")
    out_ref = sparse_mla_ref_apply(q_ref, kv_ref, indices, sm_scale)
    dout = torch.randn_like(out_ref)
    out_ref.backward(dout)

    set_sparse_mla_backend("flashmla")
    out_fm = sparse_mla_flashmla_apply(q_fm, kv_fm, indices, sm_scale)
    out_fm.backward(dout)

    torch.testing.assert_close(out_fm, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(q_fm.grad, q_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(kv_fm.grad, kv_ref.grad, atol=4e-2, rtol=4e-2)
