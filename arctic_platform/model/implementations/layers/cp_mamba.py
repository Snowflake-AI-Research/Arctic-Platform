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

import torch
import torch.distributed as dist

from arctic_platform.model.implementations.gpu.sp.collectives import _exchange_barrier_group
from arctic_platform.model.implementations.gpu.sp.collectives import _finish_exchange


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
    if cp_size == 1:
        return x
    barrier_group = _exchange_barrier_group(cp_group)
    B, S_local, D = x.shape
    D_local = D // cp_size

    # Split D into cp_size chunks, put cp dim first for all_to_all scatter
    # [B, S_local, D] -> [B, S_local, cp, D_local] -> [cp, B, S_local, D_local]
    x = x.reshape(B, S_local, cp_size, D_local).permute(2, 0, 1, 3).contiguous()
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=cp_group)
    _finish_exchange(barrier_group)
    # output[i] = data from rank i (its S_local tokens, our D_local slice)
    # [cp, B, S_local, D_local] -> [B, cp, S_local, D_local] -> [B, cp*S_local, D_local]
    return output.permute(1, 0, 2, 3).reshape(B, cp_size * S_local, D_local).contiguous()


def _all_to_all_head_to_seq(x: torch.Tensor, cp_group: dist.ProcessGroup, cp_size: int) -> torch.Tensor:
    """Transpose [B, S_global, D_local] -> [B, S_local, D] via all-to-all.

    Inverse of _all_to_all_seq_to_head.
    """
    if cp_size == 1:
        return x
    barrier_group = _exchange_barrier_group(cp_group)
    B, S_global, D_local = x.shape
    S_local = S_global // cp_size

    # Split S_global into cp_size chunks (one per rank), put cp dim first for scatter
    # [B, S_global, D_local] -> [B, cp, S_local, D_local] -> [cp, B, S_local, D_local]
    x = x.reshape(B, cp_size, S_local, D_local).permute(1, 0, 2, 3).contiguous()
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x, group=cp_group)
    _finish_exchange(barrier_group)
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

    scan_output, _ = mamba_chunk_scan_combined(
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
