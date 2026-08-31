#!/usr/bin/env python
# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""BIRD parquet + SQLite exec-match reward for TRL async-GRPO.

``sql_reward`` must stay a module-level function (spawned ``AsyncRolloutWorker``
pickles it by path). ``compute_score`` is loaded from the sibling verl recipe.
"""

from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Sibling verl recipe; loaded by file path (not an installed package).
_BIRD_REWARD_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "verl",
        "txt2sql",
        "bird_reward.py",
    )
)


def _load_compute_score():
    if not os.path.exists(_BIRD_REWARD_PATH):
        raise FileNotFoundError(
            f"BIRD reward module not found at {_BIRD_REWARD_PATH}; expected the txt2sql recipe under Arctic-Platform."
        )
    spec = importlib.util.spec_from_file_location("bird_reward", _BIRD_REWARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.compute_score


# Loaded once per process (re-loaded in the spawned child when it re-imports this module).
compute_score = _load_compute_score()


def _completion_text(completion: Any) -> str:
    """Assistant text from a TRL completion (chat list or plain string)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", "") or "")
        return str(last)
    if isinstance(completion, dict):
        return str(completion.get("content", "") or "")
    return str(completion or "")


def _score_one(
    completion: Any,
    ground_truth: Any,
    db_path: Any,
    data_source: Any,
) -> dict[str, float]:
    """One ``compute_score`` call; always returns the three verl keys as floats."""
    text = _completion_text(completion)
    out = compute_score(
        data_source or "bird",
        text,
        ground_truth,
        extra_info={"db_path": db_path},
    )
    if not isinstance(out, dict):
        return {"score": float(out), "format_correct": 0.0, "execution_success": 0.0}
    return {
        "score": float(out.get("score", 0.0)),
        "format_correct": float(out.get("format_correct", 0.0)),
        "execution_success": float(out.get("execution_success", 0.0)),
    }


def sql_reward_detailed(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    completion_ids: list[Any] | None = None,
    ground_truth: list[Any] | None = None,
    db_path: list[Any] | None = None,
    data_source: list[Any] | None = None,
    **kwargs: Any,
) -> list[dict[str, float]]:
    """SQLite exec-match; keep the full ``compute_score`` dict (val). Train uses ``sql_reward``."""
    del prompts, completion_ids, kwargs
    if completions is None:
        return []
    n = len(completions)
    if n == 0:
        return []
    gts = ground_truth if ground_truth is not None else [None] * n
    dbs = db_path if db_path is not None else [None] * n
    srcs = data_source if data_source is not None else ["bird"] * n

    def _one(i: int) -> dict[str, float]:
        return _score_one(
            completions[i],
            gts[i] if i < len(gts) else None,
            dbs[i] if i < len(dbs) else None,
            srcs[i] if i < len(srcs) else "bird",
        )

    if n == 1:
        return [_one(0)]
    workers = min(16, n)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sql-reward") as pool:
        return list(pool.map(_one, range(n)))


def sql_reward(
    prompts: list[Any] | None = None,
    completions: list[Any] | None = None,
    completion_ids: list[Any] | None = None,
    ground_truth: list[Any] | None = None,
    db_path: list[Any] | None = None,
    data_source: list[Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """One exec-match score per completion (floats only)."""
    return [
        float(d["score"])
        for d in sql_reward_detailed(
            prompts=prompts,
            completions=completions,
            completion_ids=completion_ids,
            ground_truth=ground_truth,
            db_path=db_path,
            data_source=data_source,
            **kwargs,
        )
    ]


def load_bird_dataset(parquet_path: str, num_prompts: int = -1):
    """Load a verl BIRD parquet and flatten to TRL columns."""
    from datasets import load_dataset

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"BIRD parquet not found at {parquet_path}; run preprocess_bird.py (README step 4) first."
        )
    ds = load_dataset("parquet", data_files=parquet_path, split="train")

    def _flatten(row: dict) -> dict:
        reward_model = row.get("reward_model") or {}
        extra_info = row.get("extra_info") or {}
        return {
            "prompt": row["prompt"],
            "ground_truth": reward_model.get("ground_truth", ""),
            "db_path": extra_info.get("db_path", ""),
            "data_source": row.get("data_source", "bird"),
            "db_id": extra_info.get("db_id", ""),
        }

    ds = ds.map(_flatten, remove_columns=ds.column_names)
    if num_prompts and num_prompts > 0:
        ds = ds.select(range(min(num_prompts, len(ds))))
    return ds
