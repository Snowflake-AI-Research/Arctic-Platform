"""All-to-all context parallelism for Mamba-2 SSM layers.

Transposes between sequence-parallel and head-parallel layouts so that each
CP rank processes the full sequence on a subset of heads, then transposes
back. This avoids materializing the full [S, full_heads] tensor on any GPU.

Communication pattern per Mamba layer:
  1. in_proj (token-parallel, no comm)
  2. all-to-all: [B, S/cp, D] -> [B, S, D/cp]   (seq-to-head)
  3. conv1d + SSM on full sequence, local heads
  4. all-to-all: [B, S, D/cp] -> [B, S/cp, D]    (head-to-seq)
  5. out_proj (token-parallel, no comm)
"""

from __future__ import annotations

import functools
import os
import weakref

import torch
import torch.distributed as dist


class _SeqToHeadParallel(torch.autograd.Function):
    """[B, S/cp, D] -> [B, S, D/cp] via all-to-all."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        return _all_to_all_seq_to_head(x, cp_group, cp_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return _all_to_all_head_to_seq(grad_output, ctx.cp_group, ctx.cp_size), None, None


class _HeadToSeqParallel(torch.autograd.Function):
    """[B, S, D/cp] -> [B, S/cp, D] via all-to-all."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        return _all_to_all_head_to_seq(x, cp_group, cp_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return _all_to_all_seq_to_head(grad_output, ctx.cp_group, ctx.cp_size), None, None


def _all_to_all_seq_to_head(x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
    """Transpose [B, S_local, D] -> [B, S_global, D_local] via all-to-all.

    Each rank sends its S_local tokens split into cp_size chunks along D,
    and receives one D_local chunk from each of the cp_size ranks.
    """
    B, S_local, D = x.shape
    D_local = D // cp_size

    # Split D into cp_size chunks, put cp dim first for all_to_all scatter
    # [B, S_local, D] -> [B, S_local, cp, D_local] -> [cp, B, S_local, D_local]
    x = x.reshape(B, S_local, cp_size, D_local).permute(2, 0, 1, 3).contiguous()
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=cp_group)
    # output[i] = data from rank i (its S_local tokens, our D_local slice)
    # [cp, B, S_local, D_local] -> [B, cp, S_local, D_local] -> [B, cp*S_local, D_local]
    return output.permute(1, 0, 2, 3).reshape(B, cp_size * S_local, D_local).contiguous()


def _all_to_all_head_to_seq(x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
    """Transpose [B, S_global, D_local] -> [B, S_local, D] via all-to-all.

    Inverse of _all_to_all_seq_to_head.
    """
    B, S_global, D_local = x.shape
    S_local = S_global // cp_size

    # Split S_global into cp_size chunks (one per rank), put cp dim first for scatter
    # [B, S_global, D_local] -> [B, cp, S_local, D_local] -> [cp, B, S_local, D_local]
    x = x.reshape(B, cp_size, S_local, D_local).permute(1, 0, 2, 3).contiguous()
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=cp_group)
    # Each rank now has cp chunks of [B, S_local, D_local] — one D_local slice from each rank
    # [cp, B, S_local, D_local] -> [B, S_local, cp, D_local] -> [B, S_local, D]
    return output.permute(1, 2, 0, 3).reshape(B, S_local, cp_size * D_local).contiguous()


def seq_to_head_parallel(x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
    """[B, S/cp, D] -> [B, S, D/cp] with correct backward."""
    return _SeqToHeadParallel.apply(x, cp_group, cp_size)


def head_to_seq_parallel(x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
    """[B, S, D/cp] -> [B, S/cp, D] with correct backward."""
    return _HeadToSeqParallel.apply(x, cp_group, cp_size)


def _seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """(1, seq_len) int32 ids. mamba_chunk_scan_combined resets SSM state when the id changes."""
    seq_idx = torch.zeros(seq_len, dtype=torch.int32, device=device)
    boundaries = cu_seqlens.tolist()
    for index in range(len(boundaries) - 1):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        if end > start:
            seq_idx[start:end] = index
    return seq_idx.unsqueeze(0)


def _causal_conv_packed(
    hidden_states_b_c: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    groups: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Depthwise causal conv. Each cu_seqlens segment is convolved alone so state does not cross rows."""
    length = hidden_states_b_c.shape[1]
    channels_first = hidden_states_b_c.transpose(1, 2)
    padding = weight.shape[2] - 1
    if cu_seqlens is None:
        convolved = torch.nn.functional.conv1d(
            channels_first, weight, bias, groups=groups, padding=padding
        )
        return torch.nn.functional.silu(convolved.transpose(1, 2)[:, :length])

    if int(cu_seqlens[-1]) != length:
        raise ValueError(
            f"cu_seqlens covers {int(cu_seqlens[-1])} tokens, but the gathered Mamba sequence has {length}"
        )
    segments = []
    boundaries = cu_seqlens.tolist()
    for index in range(len(boundaries) - 1):
        start, end = int(boundaries[index]), int(boundaries[index + 1])
        if start == end:
            continue
        segment = channels_first[:, :, start:end]
        convolved = torch.nn.functional.conv1d(segment, weight, bias, groups=groups, padding=padding)
        segments.append(convolved[:, :, : end - start])
    return torch.nn.functional.silu(torch.cat(segments, dim=-1).transpose(1, 2))


def mamba_cp_forward(
    mixer,
    hidden_states: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
    cu_seqlens: torch.Tensor | None = None,
):
    """CP-aware Mamba-2 forward: in_proj -> all-to-all -> conv+SSM(local heads) -> all-to-all -> out_proj.

    Replaces the mixer's cuda_kernels_forward when CP > 1. Parameters are
    sliced at runtime so FSDP / checkpoint shapes are unchanged.
    """
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    from arctic_platform.model.implementations.gpu.core_replay import keep_core

    batch_size, seq_len, _ = hidden_states.shape  # seq_len = S/cp here

    # ── 1. in_proj (token-parallel, no communication needed) ──
    projected_states = mixer.in_proj(hidden_states)
    A = -torch.exp(mixer.A_log.float())

    gate, hidden_states_B_C, time_step = torch.split(
        projected_states,
        [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads],
        dim=-1,
    )

    # ── 2. All-to-all: seq-parallel -> head-parallel ──
    # gate:              [B, S/cp, nheads * hdim]         -> [B, S, (nheads/cp) * hdim]
    # hidden_states_B_C: [B, S/cp, nheads*hdim + 2*ngroups*dstate] -> [B, S, ...]
    # time_step:         [B, S/cp, nheads]                -> [B, S, nheads/cp]
    gate = seq_to_head_parallel(gate, cp_group, cp_size)
    time_step = seq_to_head_parallel(time_step, cp_group, cp_size)

    # hidden_states_B_C needs special handling: x part shards by heads, B/C parts shard by groups
    # conv_dim = intermediate_size + 2 * ngroups * dstate
    groups_time_state_size = mixer.n_groups * mixer.ssm_state_size
    x_part, B_part, C_part = torch.split(
        hidden_states_B_C,
        [mixer.intermediate_size, groups_time_state_size, groups_time_state_size],
        dim=-1,
    )
    x_part = seq_to_head_parallel(x_part, cp_group, cp_size)
    B_part = seq_to_head_parallel(B_part, cp_group, cp_size)
    C_part = seq_to_head_parallel(C_part, cp_group, cp_size)
    hidden_states_B_C = torch.cat([x_part, B_part, C_part], dim=-1)

    full_seq_len = hidden_states_B_C.shape[1]  # S (full sequence)

    # ── 3. Local head dimensions ──
    local_num_heads = mixer.num_heads // cp_size
    local_intermediate_size = local_num_heads * mixer.head_dim
    local_n_groups = mixer.n_groups // cp_size

    # Slice parameters for local heads
    head_start = cp_rank * local_num_heads
    head_end = head_start + local_num_heads

    local_A = A[head_start:head_end]
    local_D = mixer.D[head_start:head_end]
    local_dt_bias = mixer.dt_bias[head_start:head_end]

    # Slice conv1d weight/bias for local channels
    # conv1d channels = intermediate_size + 2 * ngroups * dstate (same as conv_dim)
    # After all-to-all, we have local channels: local_intermediate + 2 * local_ngroups * dstate
    local_conv_dim = local_intermediate_size + 2 * local_n_groups * mixer.ssm_state_size
    conv_x_start = cp_rank * local_intermediate_size
    conv_x_end = conv_x_start + local_intermediate_size
    conv_b_start = mixer.intermediate_size + cp_rank * local_n_groups * mixer.ssm_state_size
    conv_b_end = conv_b_start + local_n_groups * mixer.ssm_state_size
    conv_c_start = mixer.intermediate_size + groups_time_state_size + cp_rank * local_n_groups * mixer.ssm_state_size
    conv_c_end = conv_c_start + local_n_groups * mixer.ssm_state_size

    conv_indices = torch.cat(
        [
            torch.arange(conv_x_start, conv_x_end, device=mixer.conv1d.weight.device),
            torch.arange(conv_b_start, conv_b_end, device=mixer.conv1d.weight.device),
            torch.arange(conv_c_start, conv_c_end, device=mixer.conv1d.weight.device),
        ]
    )
    local_conv_weight = mixer.conv1d.weight[conv_indices]
    local_conv_bias = mixer.conv1d.bias[conv_indices] if mixer.conv1d.bias is not None else None

    # ── 4. Conv1d on the gathered sequence, local heads. Packed rows are convolved per segment. ──
    hidden_states_B_C = _causal_conv_packed(
        hidden_states_B_C, local_conv_weight, local_conv_bias, local_conv_dim, cu_seqlens
    )

    local_groups_time_state_size = local_n_groups * mixer.ssm_state_size
    hidden_states_local, B_local, C_local = torch.split(
        hidden_states_B_C,
        [local_intermediate_size, local_groups_time_state_size, local_groups_time_state_size],
        dim=-1,
    )

    # ── 5. SSM scan on the gathered sequence, local heads. seq_idx resets state at row boundaries. ──
    dt_limit_kwargs = {} if mixer.time_step_limit is None else {"dt_limit": mixer.time_step_limit}
    seq_idx = (
        None
        if cu_seqlens is None
        else _seq_idx_from_cu_seqlens(cu_seqlens, full_seq_len, hidden_states_local.device)
    )

    scan_output, _ = keep_core(
        getattr(mixer, "_arl_core_replay", None),
        mamba_chunk_scan_combined,
        hidden_states_local.view(batch_size, full_seq_len, local_num_heads, mixer.head_dim),
        time_step,
        local_A,
        B_local.view(batch_size, full_seq_len, local_n_groups, -1),
        C_local.view(batch_size, full_seq_len, local_n_groups, -1),
        chunk_size=mixer.chunk_size,
        D=local_D,
        z=None,
        seq_idx=seq_idx,
        return_final_states=True,
        dt_bias=local_dt_bias,
        dt_softplus=True,
        **dt_limit_kwargs,
    )
    scan_output = scan_output.view(batch_size, full_seq_len, local_intermediate_size)

    # ── 6. Gated RMSNorm with local weights ──
    # Zamba2RMSNormGated normalizes within groups of size `group_size`.
    # In head-parallel mode, we have local_intermediate_size features.
    # The group_size stays the same, but group_count is reduced.
    local_norm_weight = mixer.norm.weight[cp_rank * local_intermediate_size : (cp_rank + 1) * local_intermediate_size]
    scan_output = _gated_rms_norm(
        scan_output, gate, local_norm_weight, mixer.norm.variance_epsilon, mixer.norm.group_size
    )

    # ── 7. All-to-all: head-parallel -> seq-parallel ──
    scan_output = head_to_seq_parallel(scan_output, cp_group, cp_size)

    # ── 8. out_proj (token-parallel, no communication needed) ──
    return mixer.out_proj(scan_output)


def _gated_rms_norm(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float, group_size: int
) -> torch.Tensor:
    """Gated RMSNorm matching Zamba2RMSNormGated: gate -> group-wise RMSNorm -> weight."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    if gate is not None:
        x = x * torch.nn.functional.silu(gate.to(torch.float32))
    *prefix_dims, last_dim = x.shape
    group_count = last_dim // group_size
    x_grouped = x.view(*prefix_dims, group_count, group_size)
    variance = x_grouped.pow(2).mean(-1, keepdim=True)
    x_grouped = x_grouped * torch.rsqrt(variance + eps)
    x = x_grouped.view(*prefix_dims, last_dim)
    return (weight * x).to(input_dtype)


@functools.cache
def fused_mamba_available() -> bool:
    """Whether mamba_ssm's fused conv + scan + gated-norm function and its causal_conv1d kernels are importable, and
    not turned off with ``ARL_FUSED_MAMBA=0`` (worker environment)."""
    if os.environ.get("ARL_FUSED_MAMBA", "1") == "0":
        return False
    try:
        import causal_conv1d  # noqa: F401
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa: F401
    except ImportError:
        return False
    return True


# The last cu_seqlens' row ids: every Mamba layer of a forward (and its recompute) reads the same cu_seqlens tensor.
_seq_idx_cache: tuple[weakref.ref, int, torch.Tensor] | None = None


def packed_seq_idx(cu_seqlens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(1, seq_len) int32 row ids of a packed sequence, built on the device without a host sync, once per
    ``cu_seqlens`` tensor. ``cu_seqlens[-1]`` must be ``seq_len``."""
    global _seq_idx_cache
    if _seq_idx_cache is not None:
        ref, length, seq_idx = _seq_idx_cache
        if ref() is cu_seqlens and length == seq_len:
            return seq_idx
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    ids = torch.arange(lengths.numel(), dtype=torch.int32, device=cu_seqlens.device)
    seq_idx = torch.repeat_interleave(ids, lengths, output_size=seq_len).unsqueeze(0)
    _seq_idx_cache = (weakref.ref(cu_seqlens), seq_len, seq_idx)
    return seq_idx


def fused_mamba_mixer(
    mixer, zxbcdt: torch.Tensor, seq_idx: torch.Tensor | None, rank: int = 0, size: int = 1
) -> torch.Tensor:
    """Conv1d, SiLU, the chunked scan, the D skip and the gated RMSNorm of a Mamba-2 mixer in one autograd node
    (mamba_ssm's ``mamba_split_conv1d_scan_combined``: causal_conv1d's kernel with ``seq_idx`` boundaries, the
    Triton scan and gated norm, the conv recomputed in its backward). ``zxbcdt`` is the in_proj output
    ``[z | x | B | C | dt]``, of this rank's ``1/size`` heads (and groups) under context parallelism; returns the
    normed ``[batch, seq, heads * head_dim]`` before out_proj."""
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

    from arctic_platform.model.implementations.gpu.core_replay import keep_core

    heads = mixer.num_heads // size
    dim = heads * mixer.head_dim
    groups = mixer.n_groups // size
    state = groups * mixer.ssm_state_size
    head_slice = slice(rank * heads, (rank + 1) * heads)
    weight, bias = mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias
    if size > 1:
        start_b = mixer.intermediate_size + rank * state
        start_c = mixer.intermediate_size + mixer.n_groups * mixer.ssm_state_size + rank * state
        channels = torch.cat(
            [
                torch.arange(rank * dim, (rank + 1) * dim, device=weight.device),
                torch.arange(start_b, start_b + state, device=weight.device),
                torch.arange(start_c, start_c + state, device=weight.device),
            ]
        )
        weight = weight[channels]
        bias = None if bias is None else bias[channels]
    norm_weight = mixer.norm.weight[rank * dim : (rank + 1) * dim]
    return keep_core(
        getattr(mixer, "_arl_core_replay", None),
        mamba_split_conv1d_scan_combined,
        zxbcdt,
        weight,
        bias,
        mixer.dt_bias[head_slice],
        -torch.exp(mixer.A_log[head_slice].float()),
        mixer.D[head_slice],
        mixer.chunk_size,
        seq_idx=seq_idx,
        dt_limit=(0.0, float("inf")) if mixer.time_step_limit is None else tuple(mixer.time_step_limit),
        activation=mixer.activation,
        rmsnorm_weight=norm_weight,
        rmsnorm_eps=mixer.norm.variance_epsilon,
        headdim=mixer.head_dim,
        ngroups=groups,
        norm_before_gate=False,
    )


class _PackedSeqToHead(torch.autograd.Function):
    """``[B, S/cp, sum(widths)]`` -> ``[B, S, sum(widths) / cp]`` in one all-to-all: each column block of ``widths``
    (z, x, B, C, dt of the in_proj output) is split over the ranks by heads, and every rank receives its share of
    each block side by side, in the block order."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, widths: tuple[int, ...], group: dist.ProcessGroup, size: int) -> torch.Tensor:
        batch, local_len, _ = x.shape
        parts = x.split(list(widths), dim=-1)
        send = torch.cat([p.reshape(batch, local_len, size, -1).permute(2, 0, 1, 3) for p in parts], dim=-1)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        ctx.shape, ctx.widths, ctx.group, ctx.size = (batch, local_len), widths, group, size
        return recv.permute(1, 0, 2, 3).reshape(batch, size * local_len, -1)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (batch, local_len), size = ctx.shape, ctx.size
        send = grad.reshape(batch, size, local_len, -1).permute(1, 0, 2, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=ctx.group)
        parts = recv.split([w // size for w in ctx.widths], dim=-1)
        grad_x = torch.cat([p.permute(1, 2, 0, 3).reshape(batch, local_len, -1) for p in parts], dim=-1)
        return grad_x, None, None, None


def mamba_cp_forward_fused(
    mixer,
    hidden_states: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_rank: int,
    cp_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """``mamba_cp_forward`` with one all-to-all for the in_proj output (instead of five) and the fused mixer
    (``fused_mamba_mixer``) on the gathered sequence's local heads."""
    groups_state = mixer.n_groups * mixer.ssm_state_size
    widths = (mixer.intermediate_size, mixer.intermediate_size, groups_state, groups_state, mixer.num_heads)
    zxbcdt = _PackedSeqToHead.apply(mixer.in_proj(hidden_states), widths, cp_group, cp_size)
    seq_idx = None if cu_seqlens is None else packed_seq_idx(cu_seqlens, zxbcdt.shape[1])
    scan_output = fused_mamba_mixer(mixer, zxbcdt, seq_idx, cp_rank, cp_size)
    return mixer.out_proj(head_to_seq_parallel(scan_output, cp_group, cp_size))
