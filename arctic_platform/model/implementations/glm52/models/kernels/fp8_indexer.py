"""Triton FP8 indexer kernel for DSA sparse token selection.

Scoring formula: I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)

FP8 quantization vendored from vLLM (Apache 2.0).
"""

import functools

import torch
import triton
import triton.language as tl

try:
    from flashinfer import top_k as _flashinfer_top_k
    from flashinfer.topk import TopKTieBreak
except ImportError:  # optional: torch.topk selects the same keys, more slowly
    _flashinfer_top_k = None

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    y_num_columns,
    y_row_stride,
    eps,
    fp8_min,
    fp8_max,
    use_ue8m0: tl.constexpr,
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) * group_size)
    y_ptr += y_ptr_offset

    y_q_ptr_offset = g_id.to(tl.int64) * group_size
    y_q_ptr += y_q_ptr_offset
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    scale_raw = _absmax / fp8_max
    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, use_ue8m0: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1

    x_q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    shape = x.shape[:-1] + (x.shape[-1] // group_size,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    M = x.numel() // group_size
    N = group_size
    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_group_quant_fp8[(M,)](
        x,
        x_q,
        x_s,
        group_size,
        x.shape[-1],
        x.stride(-2),
        FP8_EPS,
        FP8_MIN,
        FP8_MAX,
        use_ue8m0=use_ue8m0,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, x_s


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["N_BUCKET"],
)
@triton.jit
def _triton_fp8_indexer_kernel(
    Q_fp8,
    K_fp8,
    K_scales,
    W,
    Out,
    KS,
    KE,
    M,
    N,
    N_BUCKET,
    stride_qh,
    stride_qs,
    stride_ks,
    stride_ws,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    mask_m = offs_m < M
    mask_n = offs_n < N

    ks_vals = tl.load(KS + offs_m, mask=mask_m, other=N)
    ke_vals = tl.load(KE + offs_m, mask=mask_m, other=0)

    offs_m_64 = offs_m.to(tl.int64)
    offs_n_64 = offs_n.to(tl.int64)
    out_ptrs = Out + offs_m_64[:, None] * N + offs_n_64[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]

    ks_min = tl.min(ks_vals)
    ke_max = tl.max(ke_vals)
    n_lo = pid_n * BLOCK_N

    if n_lo >= ke_max or n_lo + BLOCK_N <= ks_min:
        tl.store(out_ptrs, tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32), mask=out_mask)
        return

    # Clamp indices for FP8 loads (can't use mask+other with fp8 dtype)
    offs_n_safe = tl.minimum(offs_n, N - 1)
    offs_m_safe = tl.minimum(offs_m, M - 1)

    k_block = tl.load(K_fp8 + offs_n_safe[:, None] * stride_ks + offs_d[None, :])
    k_sc = tl.load(K_scales + offs_n_safe, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h in range(H):
        q_h = tl.load(Q_fp8 + h * stride_qh + offs_m_safe[:, None] * stride_qs + offs_d[None, :])
        w_h = tl.load(W + offs_m * stride_ws + h, mask=mask_m, other=0.0).to(tl.float32)

        scores = tl.dot(q_h, tl.trans(k_block)).to(tl.float32)
        scores = tl.maximum(scores, 0.0) * k_sc[None, :]
        acc += scores * w_h[:, None]

    valid = (offs_n[None, :] >= ks_vals[:, None]) & (offs_n[None, :] < ke_vals[:, None])
    acc = tl.where(valid, acc, float("-inf"))

    tl.store(out_ptrs, acc, mask=out_mask)


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Column indices of each row's ``k`` largest scores, in no particular order.

    FlashInfer's radix select is about 4x faster than ``torch.topk`` on 128K-wide rows. Ties at the
    selection boundary go to the smaller column, so the selected set is deterministic.
    """
    if _flashinfer_top_k is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1).indices
    return _flashinfer_top_k(scores, k, sorted=False, deterministic=True, tie_break=TopKTieBreak.SMALL)[1]


@functools.cache
def _deep_gemm_mqa_logits(capability_major: int):
    """DeepGEMM's FP8 MQA-logits kernel, bundled with vLLM, on SM90/SM100; else ``None``.

    It scores exactly what ``_triton_fp8_indexer_kernel`` does, 2-3x faster on B200/B300, and
    skips key blocks outside each row's ``[ks, ke)``.
    """
    if capability_major not in (9, 10):
        return None
    try:
        from vllm.third_party.deep_gemm import fp8_fp4_mqa_logits
    except ImportError:
        return None
    return fp8_fp4_mqa_logits


def weighted_relu_scores(q, k, k_scales, w, ks, ke, width: int) -> torch.Tensor:
    """FP32 ``[M, width]`` scores ``sum_h w[m, h] * relu(q[h, m] . k[n]) * k_scales[n]``.

    ``q`` is ``[H, M, D]`` (row stride free) and ``k`` is ``[>= width, D]``, both FP8 or BF16; products
    accumulate in FP32. Columns outside each row's ``[ks, ke)`` are ``-inf``.
    """
    H, M, D = q.shape
    out = torch.empty(M, width, dtype=torch.float32, device=q.device)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(width, meta["BLOCK_N"]))

    _triton_fp8_indexer_kernel[grid](
        q,
        k,
        k_scales,
        w,
        out,
        ks,
        ke,
        M,
        width,
        triton.next_power_of_2(width),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        w.stride(0),
        H=H,
        D=D,
    )
    return out


# Scores are FP32 [queries, keys]; at long context a query chunk keeps that buffer bounded.
_SCORE_BUDGET_BYTES = 1 << 30


def fp8_indexer(q, k, w, ks, ke, topk, weight_scale=1.0):
    """FP8 indexer: UE8M0 quantization, scoring (DeepGEMM, else a fused Triton kernel), and top-k.

    Queries and keys may differ in number: under context parallelism ``q`` holds
    a rank's queries and ``k`` every key, and ``ks``/``ke`` index ``k`` globally.
    Queries are scored in chunks whose FP32 score block stays under
    ``_SCORE_BUDGET_BYTES``, each against only the keys its rows can select.

    Args:
        q: [M, H, D] bf16 query vectors per head
        k: [N, D] bf16 key vectors (shared across heads)
        w: [M, H] bf16 per-head weights
        ks: [M] int32 first selectable key (sequence start) per query
        ke: [M] int32 one past the last selectable key (causal end) per query
        topk: number of top indices to return
        weight_scale: constant scaling factor for weights

    Returns:
        [M, topk] int32 key indices per query; unselectable slots hold the sentinel ``N``
    """
    # NOTE: We don't use weight scale in this kernel as it produces higher KL mismatch for some reason
    # This is not a problem result-wise, as it is a constant multiplier
    _weight_scale = weight_scale
    M, H, D = q.shape
    N = k.shape[0]
    device = q.device

    q_flat = q.reshape(M * H, D).contiguous()
    q_fp8, q_scales = per_token_group_quant_fp8(q_flat, group_size=D, use_ue8m0=True)
    k_fp8, k_scales = per_token_group_quant_fp8(k.contiguous(), group_size=D, use_ue8m0=True)

    # Fold q_scale into weights
    q_scales = q_scales.view(M, H)
    w = (w * q_scales).contiguous()
    k_scales = k_scales.view(N)

    mqa_logits = _deep_gemm_mqa_logits(torch.cuda.get_device_capability(device)[0]) if q.is_cuda else None
    if mqa_logits is None:
        q_fp8 = q_fp8.view(M, H, D).permute(1, 0, 2).contiguous()
    else:
        q_fp8 = q_fp8.view(M, H, D)

    indices = torch.empty(M, topk, dtype=torch.int32, device=device)
    rows = max(64, min(M, _SCORE_BUDGET_BYTES // (4 * max(N, 1))) // 64 * 64)
    for start in range(0, M, rows):
        stop = min(start + rows, M)
        ks_c, ke_c = ks[start:stop], ke[start:stop]
        # DeepGEMM needs 16-byte-aligned key scales; keys before a row's ``ks`` are masked anyway.
        lo, hi = int(ks_c.min()) // 64 * 64, int(ke_c.max())
        width = max(hi - lo, 1)
        if mqa_logits is not None:
            logits = mqa_logits(
                (q_fp8[start:stop], None),
                (k_fp8[lo : lo + width], k_scales[lo : lo + width]),
                w[start:stop],
                ks_c - lo,
                ke_c - lo,
                clean_logits=True,
            )
            # DeepGEMM pads each row past ``width``. Top-k reads whole rows, so take the padded rows
            # with the padding masked instead of copying the block; picks there become the sentinel.
            logits = logits.as_strided((stop - start, logits.stride(0)), (logits.stride(0), 1))
            logits[:, width:] = float("-inf")
        else:
            logits = weighted_relu_scores(
                q_fp8[:, start:stop], k_fp8[lo:], k_scales[lo:], w[start:stop], ks_c - lo, ke_c - lo, width
            )
        actual_topk = min(topk, width)
        chosen = topk_indices(logits, actual_topk) + lo
        # Keys outside a row's [ks, ke) -- other sequences, or the causal future -- become the sentinel N.
        out_of_range = (chosen < ks_c.unsqueeze(1)) | (chosen >= ke_c.unsqueeze(1))
        indices[start:stop, :actual_topk] = chosen.masked_fill(out_of_range, N).to(torch.int32)
        indices[start:stop, actual_topk:] = N
    return indices
