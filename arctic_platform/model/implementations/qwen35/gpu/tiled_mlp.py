"""Tiled MLP (ALST): shard a dense FFN's forward over the token dimension to cut its activation high-water.

A dense FFN (e.g. ``w2(silu(w1(x)) * w3(x))``) over all tokens materializes two ``[n_tokens, intermediate]``
intermediates at once -- tens of GiB at long sequence lengths. DeepSpeed's ``TiledMLP`` computes the FFN
shard-by-shard over the token dim (recomputing in backward) for a ``~1/shards`` intermediate, and defers ZeRO
grad reduction to the last shard via ``ds_grad_is_ready`` (ZeRO stage 1/2) so gradients stay correct.

Model-agnostic: the caller supplies ``is_target`` (which submodules to tile), ``mlp_forward`` (un-tiled FFN
on a shard) and ``compute_params`` (weights in the reduction). Shard count is derived per forward as
``ceil(n_tokens / token_chunk_size)``; ``0`` or unset disables tiling.
"""

from __future__ import annotations

import types
from typing import Callable, List

import torch
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

# ``mlp_forward(module, hidden_states) -> output``: the un-tiled FFN compute, run on each token shard.
MlpForward = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
# ``compute_params(module) -> [weights]``: params whose ZeRO grad reduction TiledMLP defers to the last shard.
ComputeParams = Callable[[torch.nn.Module], List[torch.Tensor]]


def make_tiled_forward(mlp_forward: MlpForward, compute_params: ComputeParams, token_chunk_size: int):
    def forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        # Tile over the token dim (-2). Shard count = ceil(tokens/chunk), clamped to num_tokens so the last
        # shard flips ds_grad_is_ready on.
        num_tokens = int(hidden_states.shape[-2] if hidden_states.dim() >= 2 else hidden_states.shape[0])
        if num_tokens <= token_chunk_size:
            return mlp_forward(self, hidden_states)
        num_shards = min((num_tokens + token_chunk_size - 1) // token_chunk_size, num_tokens)
        if num_shards <= 1:
            return mlp_forward(self, hidden_states)
        return TiledMLP.apply(mlp_forward, self, hidden_states, num_shards, compute_params(self))

    return forward


def apply_tiled_mlp(
    model: torch.nn.Module,
    *,
    is_target: Callable[[torch.nn.Module], bool],
    mlp_forward: MlpForward,
    compute_params: ComputeParams,
    token_chunk_size: int,
    logger=None,
) -> int:
    """Wrap every submodule where ``is_target(module)`` is true with a token-tiled forward.

    ``token_chunk_size`` is the target tokens-per-shard (shard count = ``ceil(n_tokens / token_chunk_size)``);
    ``0`` or ``None`` disables tiling (no-op), a negative value is rejected. Returns the number of modules
    patched. Safe before or after activation-checkpoint wrapping.
    """
    if token_chunk_size is None or token_chunk_size == 0:
        return 0
    if token_chunk_size < 0:
        raise ValueError(f"token_chunk_size must be 0/None (disable) or positive, got {token_chunk_size}")

    tiled_forward = make_tiled_forward(mlp_forward, compute_params, int(token_chunk_size))
    patched = 0
    for module in model.modules():
        if is_target(module):
            module.forward = types.MethodType(tiled_forward, module)
            patched += 1

    if logger is not None:
        logger.info(
            "Applied Tiled MLP to %d module(s) (token_chunk_size=%d, shards derived per step as "
            "ceil(tokens/chunk))",
            patched,
            int(token_chunk_size),
        )
    return patched
