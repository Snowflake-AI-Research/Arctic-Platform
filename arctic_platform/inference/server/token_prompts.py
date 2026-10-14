"""Token-id prompts passed as 1-D integer torch tensors.

``/generate`` accepts a token-id prompt either as ``list[int]`` or as a 1-D
integer CPU tensor. A tensor crosses Ray as one buffer instead of one object
per token, and the worker turns it into the ``list[int]`` vLLM needs, once per
prompt, in its own process rather than in the scheduler's.

Importing this module does not import torch: a value can only be a tensor if
torch is already loaded.
"""

from __future__ import annotations

import sys
from typing import Any


def is_token_tensor(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and torch.is_tensor(value)


def check_token_tensor(prompt: Any) -> Any:
    """Raise unless *prompt* is a non-empty 1-D integer CPU tensor."""
    import torch

    if prompt.dim() != 1:
        raise ValueError(f"A token-id tensor prompt must be 1-D, got shape {tuple(prompt.shape)}")
    if prompt.dtype.is_floating_point or prompt.dtype.is_complex or prompt.dtype == torch.bool:
        raise TypeError(f"A token-id tensor prompt must have an integer dtype, got {prompt.dtype}")
    if prompt.device.type != "cpu":
        raise ValueError(f"A token-id tensor prompt must be on the CPU, got {prompt.device}")
    if prompt.numel() == 0:
        raise ValueError("A token-id tensor prompt must not be empty")
    return prompt


def compact_token_tensor(prompt: Any) -> Any:
    """Check *prompt* and return it backed by exactly its own tokens.

    Pickling a view (a row of a batch, a slice) pickles its whole storage, so a
    prompt sent to a worker is copied out of any larger buffer first.
    """
    import torch

    check_token_tensor(prompt)
    if prompt.is_contiguous() and prompt.untyped_storage().nbytes() == prompt.numel() * prompt.element_size():
        return prompt
    return prompt.clone(memory_format=torch.contiguous_format)
