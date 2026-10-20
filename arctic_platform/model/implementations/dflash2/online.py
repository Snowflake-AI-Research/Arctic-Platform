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
"""Online DFlash2 training module: a frozen target plus SpecForge's trainable draft (written here).

SpecForge's ``OnlineDFlashModel`` (vendored unchanged) trains on padded ``[B, S]`` batches whose target hidden states
were captured by a separate server. Here the target runs inside the training step instead, and the DSS data plane
hands the model a packed ``[1, T]`` window with ``position_ids`` restarting per row. This module is the glue:

1. split the window back into rows at the ``position_ids`` resets,
2. run the frozen target on each row to get the hidden states of ``target_layer_ids`` (what the capture server gives
   SpecForge),
3. pad the rows into the ``[B, S]`` batch ``OnlineDFlashModel`` expects and call it, and
4. export the draft the way SpecForge does (``save_pretrained`` + ``apply_legacy_rope_scaling``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Optional

import torch
from torch import nn

from .config import build_draft_config
from .specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from .specforge.export.checkpoint_io import apply_legacy_rope_scaling
from .specforge.modeling.draft.dflash2 import DFlash2DraftModel

logger = logging.getLogger(__name__)

DRAFT_WEIGHTS = "model.safetensors"


def row_spans(position_ids: torch.Tensor) -> list[tuple[int, int]]:
    """Row boundaries of a packed ``[1, T]`` window: a new row starts wherever ``position_ids`` is 0."""
    flat = position_ids.reshape(-1)
    starts = (flat == 0).nonzero(as_tuple=False).reshape(-1).tolist()
    if not starts or starts[0] != 0:
        starts = [0] + starts
    ends = starts[1:] + [flat.numel()]
    return list(zip(starts, ends))


def text_backbone(target: nn.Module) -> nn.Module:
    """The decoder stack of a causal LM, skipping any vision tower of composite checkpoints."""
    inner = getattr(target, "model", target)
    return getattr(inner, "language_model", inner)


class DFlash2OnlineModel(nn.Module):
    def __init__(self, target: nn.Module, online_model: OnlineDFlashModel, *, seed: int = 0) -> None:
        super().__init__()
        self.target = target
        self.online_model = online_model
        # The DSS worker reads ``model.config`` for a few generic purposes; expose the target's.
        self.config = target.config
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.target.eval()
        self._seed = seed
        self._seeded = False

    @property
    def draft(self) -> DFlash2DraftModel:
        return self.online_model.draft_model

    def train(self, mode: bool = True) -> "DFlash2OnlineModel":
        super().train(mode)
        self.target.eval()  # the target is frozen and must never leave eval mode
        return self

    def _seed_anchor_sampling(self, device: torch.device) -> None:
        # ``OnlineDFlashModel`` samples anchors with the default generator. Give every rank its own stream so ranks
        # holding rows of equal length do not draw identical positions.
        if self._seeded:
            return
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = self._seed + 1_000_003 * rank
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        self._seeded = True

    @torch.no_grad()
    def target_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Concatenated hidden states of ``target_layer_ids`` for one row, aligned with ``input_ids``."""
        self.target.eval()
        out = text_backbone(self.target)(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        # hidden_states[0] is the embedding output, hidden_states[i + 1] the output of layer i (SpecForge's
        # ``extract_context_feature`` uses the same offset).
        return torch.cat([out.hidden_states[i + 1] for i in self.draft.target_layer_ids], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        selector_loss_alpha: Optional[float] = None,
        collect_detailed_metrics: bool = True,
        per_position_metrics: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        if loss_mask is None:
            raise ValueError("DFlash2 training needs a loss_mask marking the supervised tokens")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        self._seed_anchor_sampling(input_ids.device)

        rows = []
        for start, end in row_spans(position_ids):
            row_mask = loss_mask[:, start:end]
            if end - start < 3 or int(((row_mask[:, :-1] > 0.5) & (row_mask[:, 1:] > 0.5)).sum()) == 0:
                continue  # no anchor with a supervised successor: this row cannot form a block
            rows.append((input_ids[:, start:end], row_mask, self.target_context(input_ids[:, start:end])))

        if not rows:
            # No usable row in this window (e.g. a padding window): a zero that stays connected to the draft so every
            # rank still takes part in the backward pass.
            zero = sum(p.float().sum() for p in self.draft.parameters() if p.requires_grad) * 0.0
            return {"loss": zero, "dflash2_terms": {}}

        # Pad into the [B, S] batch SpecForge trains on. Padded positions carry loss_mask 0, so they are never
        # anchors or targets, and the block mask only exposes context before each anchor.
        longest = max(ids.shape[1] for ids, _, _ in rows)

        def pad(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(x, (0, 0, 0, longest - x.shape[1])) if x.ndim == 3 else torch.nn.functional.pad(x, (0, longest - x.shape[1]))

        ids = torch.cat([pad(r[0]) for r in rows], dim=0)
        mask = torch.cat([pad(r[1]) for r in rows], dim=0)
        hidden = torch.cat([pad(r[2]) for r in rows], dim=0)

        loss, _accuracy, metrics = self.online_model(
            ids,
            hidden,
            mask,
            selector_loss_alpha=selector_loss_alpha,
            collect_detailed_metrics=collect_detailed_metrics,
        )
        terms = {
            name: (num.detach(), den.detach())
            for name, (num, den) in metrics["ratio_metrics"].items()
            if num.ndim == 0 and den.ndim == 0 and (per_position_metrics or not name.startswith("position_"))
        }
        return {"loss": loss, "dflash2_terms": terms}

    # --- serving-ready export, as SpecForge's `specforge export --to hf` ------------------------------------
    def save_pretrained(self, save_directory: str | Path, **_: Any) -> None:
        """Write only the draft: ``config.json`` and ``model.safetensors`` in the layout the serving stack loads."""
        self.draft.save_pretrained(str(save_directory))
        apply_legacy_rope_scaling(str(save_directory))

    def load_draft(self, directory: str | Path) -> None:
        """Weights-only warm start from a draft directory (SpecForge's or this module's export)."""
        from safetensors.torch import load_file

        state = load_file(str(Path(directory) / DRAFT_WEIGHTS))
        missing, unexpected = self.draft.load_state_dict(state, strict=False)
        missing = [key for key in missing if "rotary_emb" not in key]
        if missing or unexpected:
            raise ValueError(f"draft checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def build_online_model(
    target: nn.Module,
    draft_config: dict[str, Any],
    *,
    dtype: torch.dtype,
    attention_backend: str = "flex_attention",
    num_anchors: int = 512,
    loss_decay_gamma: Optional[float] = 7.0,
    objective_chunk_blocks: int = 128,
    selector_loss_alpha: float = 1.0,
    selector_stop_gradient: bool = False,
    seed: int = 0,
    init_from: str | Path | None = None,
) -> DFlash2OnlineModel:
    """Mirror of SpecForge's ``build_dflash_draft`` + ``build_dflash_model`` with the target's own embedding and head."""
    config = build_draft_config(draft_config)
    head, embed = target.get_output_embeddings(), target.get_input_embeddings()
    if head is None or config.vocab_size != head.weight.shape[0]:
        raise ValueError(f"draft vocab_size={config.vocab_size} does not match the target LM head")
    config._attn_implementation = attention_backend

    torch.manual_seed(seed)
    draft = DFlash2DraftModel(config).to(dtype=dtype)
    method_config = config.dflash_config
    method_config["target_layer_ids"] = list(draft.target_layer_ids)
    online = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=int(method_config["mask_token_id"]),
        block_size=int(draft.block_size),
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        objective_chunk_blocks=objective_chunk_blocks,
        selector_loss_alpha=selector_loss_alpha,
        selector_stop_gradient=selector_stop_gradient,
        teacher_metrics=False,  # needs the target's last hidden state, which is only a diagnostic
    )
    model = DFlash2OnlineModel(target, online, seed=seed)
    if init_from is not None:
        model.load_draft(init_from)
    return model
