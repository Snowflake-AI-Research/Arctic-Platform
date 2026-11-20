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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pack UltraChat rows into one sequence of an exact token length."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

DATASET = "HuggingFaceH4/ultrachat_200k"
SPLIT = "train_sft"


def _row_text(row: dict) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    for key in ("text", "content", "prompt"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"UltraChat row has no text fields: keys={sorted(row)}")


def load_ultrachat_texts(*, seed: int, limit: int = 4096) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT)
    ds = ds.shuffle(seed=seed)
    texts = []
    for i in range(min(limit, len(ds))):
        texts.append(_row_text(ds[i]))
    if not texts:
        raise RuntimeError(f"{DATASET} {SPLIT} produced no texts")
    return texts


def pack_exact_length(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    target_len: int,
    *,
    pad_token_id: int,
) -> dict:
    """Concatenate tokenized docs with position_ids resets until ``target_len`` tokens.

    The final document is truncated so the packed row is exactly ``target_len``.
    Labels equal input_ids (full SFT loss on the packed row).
    """
    if target_len < 1:
        raise ValueError(f"target_len must be positive, got {target_len}")
    ids: list[int] = []
    position_ids: list[int] = []
    for text in texts:
        if len(ids) >= target_len:
            break
        piece = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if not piece:
            continue
        room = target_len - len(ids)
        piece = piece[:room]
        ids.extend(piece)
        position_ids.extend(range(len(piece)))
    if len(ids) != target_len:
        raise RuntimeError(
            f"could not fill packed row to {target_len} tokens from {len(texts)} UltraChat rows (got {len(ids)})"
        )
    input_ids = torch.tensor([ids], dtype=torch.long)
    pos = torch.tensor([position_ids], dtype=torch.long)
    labels = input_ids.clone()
    return {
        "batch": {
            "input_ids": input_ids,
            "position_ids": pos,
            "labels": labels,
        },
        "meta": {"pad_token_id": pad_token_id, "sample_packing": True},
        "processing": {"loss_fn": "sft"},
    }
