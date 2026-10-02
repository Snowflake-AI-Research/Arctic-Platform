"""FlashMLA sparse MLA: kernel forward + PyTorch reference backward.

Prime-rl tensors use batched MQA layout with a sentinel KV row. FlashMLA sparse
prefill expects per-batch tensors:

  q:       [s_q, h_q, d_qk]
  kv:      [s_kv, h_kv, d_qk]   (no sentinel; s_kv = seq_len)
  indices: [s_q, h_kv, topk]     (invalid = -1; causal enforced in adapter)

Forward uses ``flash_mla_sparse_fwd``. Backward follows FlashMLA ``tests/ref.py``
softmax semantics (natural-log LSE) with ``index_add`` into the KV cache.

FlashMLA ``tests/ref.py`` maps empty-row LSE to ``+inf`` so the corresponding
output is zero. Our tiled backward uses ``torch.logsumexp`` convention (``-inf``
for an empty softmax). The adapter converts ``+inf`` LSE to ``-inf`` after the
kernel so backward does not emit NaNs from ``exp(-inf - +inf)``.
"""

import torch

_FLASHMLA_INSTALL_HINT = (
    "flash_mla is not installed. Install flash_mla or a vLLM build that ships vllm._flashmla_C."
)


def _load_flash_mla_sparse_fwd():
    try:
        from flash_mla import flash_mla_sparse_fwd as _fn

        return _fn
    except ImportError:
        pass
    try:
        # Same sparse-prefill op as standalone flash_mla; shipped in the DSS vLLM build.
        # Imported lazily so GLM training does not pull vLLM unless flashmla is selected.
        import vllm._flashmla_C  # noqa: F401

        _op = torch.ops._flashmla_C.sparse_prefill_fwd

        def _fn(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None):
            return _op(q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out)

        return _fn
    except ImportError as err:
        raise ImportError(_FLASHMLA_INSTALL_HINT) from err


try:
    from flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None  # type: ignore


_VALID_SPARSE_MLA_BACKENDS = ("flashmla", "ref", "tilelang", "dense")
_DEFAULT_SPARSE_MLA_BACKEND = "ref"
_SPARSE_MLA_BWD_CHUNK = 64
_configured_sparse_mla_backend: str | None = None


def _ensure_flash_mla_sparse_fwd():
    global flash_mla_sparse_fwd
    if flash_mla_sparse_fwd is None:
        flash_mla_sparse_fwd = _load_flash_mla_sparse_fwd()
    return flash_mla_sparse_fwd


def flash_mla_available() -> bool:
    try:
        return _ensure_flash_mla_sparse_fwd() is not None
    except ImportError:
        return False


def set_sparse_mla_backend(backend: str) -> str:
    """Set the process-wide sparse MLA implementation.

    Requesting ``flashmla`` loads the kernel and raises on failure. Silent
    fallback to ``ref`` previously hid missing installs and broken extensions.
    """
    global _configured_sparse_mla_backend
    requested = (backend or _DEFAULT_SPARSE_MLA_BACKEND).lower()
    if requested not in _VALID_SPARSE_MLA_BACKENDS:
        raise ValueError(
            f"Unknown sparse_mla_backend={backend!r}, "
            f"expected one of {_VALID_SPARSE_MLA_BACKENDS}"
        )
    if requested == "flashmla":
        _ensure_flash_mla_sparse_fwd()
    _configured_sparse_mla_backend = requested
    return requested


def sparse_mla_backend() -> str:
    if _configured_sparse_mla_backend is not None:
        return _configured_sparse_mla_backend
    return set_sparse_mla_backend(_DEFAULT_SPARSE_MLA_BACKEND)


def _prepare_flashmla_indices(
    indices: torch.Tensor,
    seq_len: int,
    sentinel_idx: int,
) -> torch.Tensor:
    """Convert prime-rl indices [S, topk] to FlashMLA [S, 1, topk] with masking."""
    flash_indices = indices.to(torch.int32).contiguous()
    positions = torch.arange(seq_len, device=indices.device, dtype=torch.int32).unsqueeze(1)
    flash_indices = flash_indices.masked_fill(flash_indices > positions, -1)
    flash_indices = flash_indices.masked_fill(flash_indices == sentinel_idx, -1)
    flash_indices = flash_indices.masked_fill(flash_indices < 0, -1)
    return flash_indices.unsqueeze(1)


def _ref_sparse_mla_fwd_one(
    q: torch.Tensor,
    kv: torch.Tensor,
    flash_indices: torch.Tensor,
    sm_scale: float,
    seq_len: int,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference forward for one sequence; matches FlashMLA tests/ref.py."""
    idx = flash_indices.squeeze(1)
    topk = idx.shape[-1]
    invalid = (idx < 0) | (idx >= seq_len)
    idx_safe = idx.clamp_min(0)

    q_f = q.float()
    kv_body = kv[:seq_len, 0].float()
    gathered = kv_body.index_select(0, idx_safe.reshape(-1)).view(seq_len, topk, q.shape[-1])

    scores = torch.einsum("shd,std->sht", q_f, gathered) * sm_scale
    scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)

    lonely = lse == float("-inf")
    lse_for_softmax = lse.masked_fill(lonely, 0.0)
    probs = torch.exp(scores - lse_for_softmax.unsqueeze(-1))
    probs.masked_fill_(lonely.unsqueeze(-1), 0.0)

    out = torch.einsum("sht,std->shd", probs, gathered[..., :d_v])
    return out.to(q.dtype), lse


def _sparse_mla_bwd_chunk() -> int:
    # Query rows per reference-bwd tile. Full-sequence fp32 tensors are
    # [S, index_topk=2048, 576] and OOM on packed RL microbatches (~200 tokens)
    # after AdamW + LoRA sync. SFT overfit used ~80-token unpacked rows.
    return _SPARSE_MLA_BWD_CHUNK


def _ref_sparse_mla_bwd_one(
    q: torch.Tensor,
    kv: torch.Tensor,
    flash_indices: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float,
    seq_len: int,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward for one sequence (softmax over topk, scatter dKV).

    Tiled over the query axis so peak intermediates stay O(chunk × topk × dim)
    instead of O(S × topk × dim).
    """
    idx = flash_indices.squeeze(1)
    topk = idx.shape[-1]
    dim_qk = q.shape[-1]
    kv_body = kv[:seq_len, 0].float()

    dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dkv = torch.zeros(seq_len + 1, 1, dim_qk, device=kv.device, dtype=torch.float32)
    chunk = _sparse_mla_bwd_chunk()

    for s0 in range(0, seq_len, chunk):
        s1 = min(s0 + chunk, seq_len)
        sl = s1 - s0
        idx_c = idx[s0:s1]
        invalid = (idx_c < 0) | (idx_c >= seq_len)
        idx_safe = idx_c.clamp_min(0)

        q_f = q[s0:s1].float()
        do_f = do[s0:s1].float()
        gathered = kv_body.index_select(0, idx_safe.reshape(-1)).view(sl, topk, dim_qk)
        v = gathered[..., :d_v]

        scores = torch.einsum("shd,std->sht", q_f, gathered) * sm_scale
        scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))

        lse_c = lse[s0:s1]
        lse_f = lse_c.float().unsqueeze(-1)
        # FlashMLA ref.py uses +inf for empty rows; logsumexp uses -inf.
        lonely = torch.isneginf(lse_c) | torch.isposinf(lse_c)
        lse_safe = lse_f.masked_fill(lonely.unsqueeze(-1), 0.0)
        probs = torch.exp(scores - lse_safe)
        probs = probs.masked_fill(invalid.unsqueeze(1) | lonely.unsqueeze(-1), 0.0)

        d_probs = torch.einsum("shd,std->sht", do_f, v)
        d_scores = probs * (d_probs - (d_probs * probs).sum(dim=-1, keepdim=True))

        dq[s0:s1] = torch.einsum("sht,std->shd", d_scores, gathered) * sm_scale
        d_slot = torch.einsum("sht,shd->std", d_scores, q_f) * sm_scale
        dv_grad = torch.einsum("sht,shd->std", probs, do_f)
        d_slot[..., :d_v].add_(dv_grad)

        dkv[:seq_len, 0].index_add_(0, idx_safe.reshape(-1), d_slot.reshape(-1, dim_qk))

    return dq.to(q.dtype), dkv.to(kv.dtype)


def _lse_to_ref_convention(lse: torch.Tensor) -> torch.Tensor:
    """Map FlashMLA empty-row LSE (+inf) onto logsumexp empty-row LSE (-inf)."""
    return lse.masked_fill(torch.isposinf(lse), float("-inf"))


def _flashmla_fwd_one(
    q: torch.Tensor,
    kv: torch.Tensor,
    flash_indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fwd = _ensure_flash_mla_sparse_fwd()
    topk = int(flash_indices.shape[-1])
    if topk % 128 != 0:
        raise ValueError(
            f"FlashMLA sparse prefill requires topk % 128 == 0 (SM90 2*B_TOPK), got topk={topk}"
        )
    out, _, lse = fwd(q, kv, flash_indices, sm_scale, d_v=d_v)
    return out, _lse_to_ref_convention(lse)


class _SparseMLAFlashMLA(torch.autograd.Function):
    """FlashMLA sparse prefill forward + reference backward."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        _ensure_flash_mla_sparse_fwd()

        batch, seq_len, _, dim_qk = q.shape
        _, seq_len_kv, kv_group, _ = kv.shape
        sentinel_idx = seq_len
        d_v = 512

        assert kv_group == 1
        assert dim_qk == 576
        assert seq_len_kv == seq_len + 1

        if sm_scale is None:
            sm_scale = dim_qk**-0.5

        outs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        flash_indices_list: list[torch.Tensor] = []

        for b in range(batch):
            flash_idx = _prepare_flashmla_indices(indices[b, :, 0, :], seq_len, sentinel_idx)
            out_b, lse_b = _flashmla_fwd_one(
                q[b], kv[b, :seq_len], flash_idx, sm_scale, d_v
            )
            outs.append(out_b.unsqueeze(0))
            lses.append(lse_b.unsqueeze(0))
            flash_indices_list.append(flash_idx)

        out = torch.cat(outs, dim=0)
        lse = torch.cat(lses, dim=0)

        # Activation offload hooks only see tensors retained via save_for_backward.
        ctx.save_for_backward(q, kv, lse, *flash_indices_list)
        ctx.sm_scale = sm_scale
        ctx.seq_len = seq_len
        ctx.d_v = d_v
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, lse, *flash_indices_list = ctx.saved_tensors
        batch = q.shape[0]
        seq_len = ctx.seq_len
        sm_scale = ctx.sm_scale
        d_v = ctx.d_v

        dq_parts: list[torch.Tensor] = []
        dkv_parts: list[torch.Tensor] = []

        for b in range(batch):
            dq_b, dkv_b = _ref_sparse_mla_bwd_one(
                q[b],
                kv[b],
                flash_indices_list[b],
                lse[b],
                do[b],
                sm_scale,
                seq_len,
                d_v,
            )
            dq_parts.append(dq_b.unsqueeze(0))
            dkv_parts.append(dkv_b.unsqueeze(0))

        return torch.cat(dq_parts, dim=0), torch.cat(dkv_parts, dim=0), None, None


def sparse_mla_flashmla_apply(q, kv, indices, sm_scale):
    """Batched sparse MLA with FlashMLA forward and reference backward."""
    return _SparseMLAFlashMLA.apply(q, kv, indices, sm_scale)


def sparse_mla_ref_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float | None = None,
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-only forward (for tests / debugging)."""
    batch, seq_len, _, dim_qk = q.shape
    if sm_scale is None:
        sm_scale = dim_qk**-0.5
    sentinel_idx = seq_len

    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for b in range(batch):
        flash_idx = _prepare_flashmla_indices(indices[b, :, 0, :], seq_len, sentinel_idx)
        out_b, lse_b = _ref_sparse_mla_fwd_one(q[b], kv[b], flash_idx, sm_scale, seq_len, d_v)
        outs.append(out_b.unsqueeze(0))
        lses.append(lse_b.unsqueeze(0))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)


def sparse_mla_ref_bwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference backward for batched prime-rl sparse MLA tensors."""
    batch, seq_len = q.shape[:2]
    sentinel_idx = seq_len
    dq_parts: list[torch.Tensor] = []
    dkv_parts: list[torch.Tensor] = []
    for b in range(batch):
        flash_idx = _prepare_flashmla_indices(indices[b, :, 0, :], seq_len, sentinel_idx)
        dq_b, dkv_b = _ref_sparse_mla_bwd_one(
            q[b], kv[b], flash_idx, lse[b], do[b], sm_scale, seq_len, d_v
        )
        dq_parts.append(dq_b.unsqueeze(0))
        dkv_parts.append(dkv_b.unsqueeze(0))
    return torch.cat(dq_parts, dim=0), torch.cat(dkv_parts, dim=0)


class _SparseMLARefFn(torch.autograd.Function):
    """Pure PyTorch reference sparse MLA (FlashMLA ref.py semantics)."""

    @staticmethod
    def forward(ctx, q, kv, indices, sm_scale):
        out, lse = sparse_mla_ref_fwd_interface(q, kv, indices, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, indices)
        ctx.lse = lse
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        # Unpack once: under activation checkpointing, ctx.saved_tensors may only
        # be read a single time (torch.utils.checkpoint unpack hooks).
        q, kv, indices = ctx.saved_tensors
        dq, dkv = sparse_mla_ref_bwd_interface(
            q,
            kv,
            indices,
            ctx.lse,
            do.contiguous(),
            ctx.sm_scale,
        )
        return dq, dkv, None, None


def sparse_mla_ref_apply(q, kv, indices, sm_scale):
    return _SparseMLARefFn.apply(q, kv, indices, sm_scale)


def sparse_mla_flashmla_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float | None = None,
    d_v: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-only FlashMLA path (natural-log LSE). Prefer ``sparse_mla_flashmla_apply``."""
    _ensure_flash_mla_sparse_fwd()

    batch, seq_len, heads, dim_qk = q.shape
    assert heads in (64, 128)
    if sm_scale is None:
        sm_scale = dim_qk**-0.5
    sentinel_idx = seq_len

    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for b in range(batch):
        flash_idx = _prepare_flashmla_indices(indices[b, :, 0, :], seq_len, sentinel_idx)
        out_b, lse_b = _flashmla_fwd_one(q[b], kv[b, :seq_len], flash_idx, sm_scale, d_v)
        outs.append(out_b.unsqueeze(0))
        lses.append(lse_b.unsqueeze(0))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)
