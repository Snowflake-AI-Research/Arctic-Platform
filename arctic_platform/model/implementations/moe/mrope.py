"""Multimodal rotary positions (M-RoPE) for Qwen3.5-family vision-language models.

A Qwen3.5 decoder rotates queries and keys by three position streams: temporal, height and width. Text tokens carry
the same position in all three, so for text the result equals 1-D RoPE. The tokens of an image carry their patch grid
coordinates, and the text after an image continues from ``max(h, w)`` past the image start rather than from the
image's token count. These helpers reproduce ``Qwen3_5MoeModel.get_rope_index`` and
``Qwen3_5MoeTextRotaryEmbedding`` in transformers, with one difference: a row may hold several packed sequences.
Each sequence starts where ``position_ids`` is 0 and is numbered independently, as the processor would number it on
its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

DEFAULT_MROPE_SECTION = (11, 11, 10)


def _segment_starts(row_position_ids: Tensor) -> list[int]:
    starts = torch.nonzero(row_position_ids == 0, as_tuple=False).flatten().tolist()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    return starts


def _image_positions(start: int, grid_thw: Tensor, spatial_merge_size: int, device: torch.device) -> Tensor:
    t, h, w = (int(v) for v in grid_thw.tolist())
    grid_t, grid_h, grid_w = t, h // spatial_merge_size, w // spatial_merge_size
    temporal, height, width = torch.meshgrid(
        torch.arange(grid_t, device=device),
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    return torch.stack([temporal, height, width]).reshape(3, -1) + start


def mrope_position_ids(
    input_ids: Tensor,
    position_ids: Tensor,
    image_grid_thw: Tensor,
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> Tensor:
    """Return ``[3, B, S]`` temporal, height and width positions for ``[B, S]`` token rows.

    ``image_grid_thw`` lists one ``(t, h, w)`` patch grid per image, in the order the images appear across the rows.
    Each image is a run of ``t * h * w / spatial_merge_size**2`` image tokens. Raises if the image tokens and the
    grids disagree, so a wrong split of the images across microbatches cannot train on misplaced embeddings.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B, S], got {tuple(input_ids.shape)}")
    position_ids = position_ids.expand_as(input_ids) if position_ids.shape != input_ids.shape else position_ids
    grids = image_grid_thw.reshape(-1, 3).to("cpu")
    out = torch.empty((3, *input_ids.shape), dtype=torch.long, device=input_ids.device)
    is_image_all = (input_ids == image_token_id).to("cpu")
    positions_all = position_ids.to("cpu")
    next_grid = 0

    for row in range(input_ids.shape[0]):
        is_image = is_image_all[row]
        starts = _segment_starts(positions_all[row])
        ends = starts[1:] + [input_ids.shape[1]]
        for seg_start, seg_end in zip(starts, ends):
            current = 0
            index = seg_start
            while index < seg_end:
                if bool(is_image[index]):
                    if next_grid >= grids.shape[0]:
                        raise ValueError("input_ids hold more image tokens than image_grid_thw describes")
                    grid = grids[next_grid]
                    next_grid += 1
                    count = int(grid.prod()) // spatial_merge_size**2
                    if index + count > seg_end or not bool(is_image[index : index + count].all()):
                        raise ValueError(
                            f"image {next_grid - 1} with grid {grid.tolist()} needs {count} consecutive image tokens "
                            f"at position {index} of row {row}"
                        )
                    out[:, row, index : index + count] = _image_positions(
                        current, grid, spatial_merge_size, input_ids.device
                    )
                    current += max(int(grid[1]), int(grid[2])) // spatial_merge_size
                    index += count
                else:
                    image_after = torch.nonzero(is_image[index:seg_end], as_tuple=False)
                    text_end = index + int(image_after[0]) if image_after.numel() else seg_end
                    length = text_end - index
                    out[:, row, index:text_end] = torch.arange(current, current + length, device=input_ids.device)
                    current += length
                    index = text_end

    if next_grid != grids.shape[0]:
        raise ValueError(f"image_grid_thw describes {grids.shape[0]} images but input_ids hold {next_grid}")
    return out


def mrope_section(config) -> Sequence[int]:
    """The ``(temporal, height, width)`` frequency split of a Qwen3.5 text config, with the transformers default."""
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    return tuple(rope_parameters.get("mrope_section", DEFAULT_MROPE_SECTION))


@torch.no_grad()
def mrope_cos_sin(
    inv_freq: Tensor,
    attention_scaling: float,
    position_ids: Tensor,
    section: Sequence[int],
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Interleaved M-RoPE ``cos`` and ``sin`` of shape ``[B, S, 2 * len(inv_freq)]`` for ``[3, B, S]`` positions.

    Frequency ``i`` takes its angle from the height stream when ``i % 3 == 1`` and ``i < 3 * section[1]``, from the
    width stream when ``i % 3 == 2`` and ``i < 3 * section[2]``, and from the temporal stream otherwise, as
    ``Qwen3_5MoeTextRotaryEmbedding`` does. The angles are computed in fp32.
    """
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(f"M-RoPE position_ids must be [3, B, S], got {tuple(position_ids.shape)}")
    device_type = position_ids.device.type if position_ids.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        freqs = (inv_freq_expanded.to(position_ids.device) @ position_ids[:, :, None, :].float()).transpose(2, 3)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
    return _interleave(cos, section).to(dtype), _interleave(sin, section).to(dtype)


def _interleave(freqs: Tensor, section: Sequence[int]) -> Tensor:
    combined = freqs[0].clone()
    for stream in (1, 2):
        index = slice(stream, section[stream] * 3, 3)
        combined[..., index] = freqs[stream, ..., index]
    return torch.cat((combined, combined), dim=-1)
