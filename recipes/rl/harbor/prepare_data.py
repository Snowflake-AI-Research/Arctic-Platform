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
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Build train.parquet and validation.parquet from train and eval Harbor task directories.

The two inputs are:
1. ``--train-tasks-dir`` (or ``TRAIN_HARBOR_TASKS``)
2. ``--eval-tasks-dir`` (or ``EVAL_HARBOR_TASKS``; defaults to ``--train-tasks-dir``)

Also supports ``--export-swe-smith-to DIR`` to export the SWE-smith code
localization dataset into Harbor task directories (``train/`` and ``eval/``)
so the exact code-localization task can be evaluated and trained as Harbor tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

DATASET = "OpenHands/SWE-smith-py-code-search"
REVISION = "b1e6864f1c847455ea64c7ebdb4164da2b902bb6"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train-00000-of-00001.parquet"
VALIDATION_ROWS = 100
SEED = 42


def discover_harbor_tasks(tasks_root: Path, task_names: list[str] | None = None) -> list[Path]:
    """Find Harbor task directories under ``tasks_root`` (each containing ``instruction.md`` and ``tests/test.sh``)."""
    root = tasks_root.expanduser().resolve()
    if (root / "instruction.md").is_file() and (root / "tests" / "test.sh").is_file():
        found = [root]
    else:
        found = sorted(
            {instr.parent for instr in root.rglob("instruction.md") if (instr.parent / "tests" / "test.sh").is_file()},
            key=lambda p: p.name,
        )
    if task_names:
        allow = {name.strip() for name in task_names if name.strip()}
        by_name = {p.name: p for p in found}
        ordered = [by_name[name] for name in task_names if name in by_name]
        return ordered if ordered else [p for p in found if p.name in allow]
    return found


def _rows_from_task_dirs(task_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        if not instruction:
            continue
        row: dict[str, Any] = dict(
            instance_id=task_dir.name,
            problem_statement=instruction,
            prompt=[{"role": "user", "content": instruction}],
            harbor_task_dir=str(task_dir),
        )
        truth_path = task_dir / "tests" / "ground_truth.json"
        if truth_path.is_file():
            try:
                truth = json.loads(truth_path.read_text(encoding="utf-8"))
                if "file_changes" in truth:
                    row["file_changes"] = truth["file_changes"]
                    row["target"] = truth["file_changes"]
            except Exception:
                pass
        records.append(row)
    return records


def build(
    train_tasks_root: Path,
    eval_tasks_root: Path | None = None,
    *,
    task_names: list[str] | None = None,
    repeat: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``(train_df, val_df)`` from ``train_tasks_root`` and ``eval_tasks_root``."""
    train_dirs = discover_harbor_tasks(train_tasks_root, task_names=task_names)
    if not train_dirs:
        raise ValueError(f"No Harbor tasks (with instruction.md and tests/test.sh) found under {train_tasks_root}")
    eval_root = eval_tasks_root if eval_tasks_root is not None else train_tasks_root
    eval_dirs = discover_harbor_tasks(eval_root, task_names=task_names)
    if not eval_dirs:
        raise ValueError(f"No Harbor tasks (with instruction.md and tests/test.sh) found under {eval_root}")

    train_records = _rows_from_task_dirs(train_dirs)
    eval_records = _rows_from_task_dirs(eval_dirs)
    if not train_records:
        raise ValueError(f"All Harbor tasks under {train_tasks_root} had empty instruction.md")
    if not eval_records:
        raise ValueError(f"All Harbor tasks under {eval_root} had empty instruction.md")

    train_df = pd.DataFrame(train_records * max(int(repeat), 1)).reset_index(drop=True)
    val_df = pd.DataFrame(eval_records).reset_index(drop=True)
    return train_df, val_df


def export_swe_smith_harbor_tasks(
    out_dir: Path,
    *,
    source_parquet_dir: Path | None = None,
    max_train_tasks: int = 256,
    max_eval_tasks: int = 32,
) -> tuple[Path, Path]:
    """Export SWE-smith code-localization rows into ``out_dir/train`` and ``out_dir/eval`` Harbor task directories."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from harbor_rl.entrypoint import materialize_harbor_task_from_row

    if source_parquet_dir is not None and (source_parquet_dir / "train.parquet").is_file():
        train_df = pd.read_parquet(source_parquet_dir / "train.parquet")
        val_df = pd.read_parquet(source_parquet_dir / "validation.parquet")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.parquet"
            print(f"Downloading {DATASET}@{REVISION[:8]}")
            urllib.request.urlretrieve(URL, source)
            raw = pd.read_parquet(source)
        rows = raw[raw["problem_statement"].str.strip().astype(bool)].copy()
        rows["target"] = rows["file_changes"]
        rows["prompt"] = [[{"role": "user", "content": text}] for text in rows["problem_statement"]]
        rows["use_patch"] = True
        rows["base_commit"] = None
        rows = rows.sample(frac=1, random_state=SEED).reset_index(drop=True)
        train_df, val_df = rows.iloc[:-VALIDATION_ROWS], rows.iloc[-VALIDATION_ROWS:]

    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for _, row in train_df.head(max_train_tasks).iterrows():
        materialize_harbor_task_from_row(row.to_dict(), train_dir)
    for _, row in val_df.head(max_eval_tasks).iterrows():
        materialize_harbor_task_from_row(row.to_dict(), eval_dir)

    return train_dir, eval_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--train-tasks-dir",
        "--harbor-tasks-dir",
        dest="train_tasks_dir",
        type=Path,
        default=None,
        help="Directory of training Harbor tasks (each with instruction.md and tests/test.sh).",
    )
    parser.add_argument(
        "--eval-tasks-dir",
        type=Path,
        default=None,
        help="Directory of evaluation Harbor tasks (defaults to --train-tasks-dir).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Optional comma-separated list of Harbor task names to include (in order).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the training task list in train.parquet (default: 1).",
    )
    parser.add_argument(
        "--export-swe-smith-to",
        type=Path,
        default=None,
        help="Export SWE-smith code localization tasks into <DIR>/train and <DIR>/eval Harbor task directories.",
    )
    parser.add_argument(
        "--swe-smith-parquet-dir",
        type=Path,
        default=None,
        help="Optional existing SWE-smith parquet directory to export from without downloading.",
    )
    parser.add_argument("--max-train-tasks", type=int, default=256)
    parser.add_argument("--max-eval-tasks", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path.home() / "data" / "harbor-rl")
    args = parser.parse_args()

    train_tasks_dir = args.train_tasks_dir
    eval_tasks_dir = args.eval_tasks_dir

    if args.export_swe_smith_to is not None:
        train_tasks_dir, eval_tasks_dir = export_swe_smith_harbor_tasks(
            args.export_swe_smith_to,
            source_parquet_dir=args.swe_smith_parquet_dir,
            max_train_tasks=args.max_train_tasks,
            max_eval_tasks=args.max_eval_tasks,
        )
        print(f"Exported SWE-smith Harbor tasks to {train_tasks_dir} and {eval_tasks_dir}")

    if train_tasks_dir is None:
        parser.error("Provide --train-tasks-dir (or --export-swe-smith-to)")

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    train, validation = build(
        train_tasks_dir,
        eval_tasks_dir,
        task_names=task_names,
        repeat=args.repeat,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out / "train.parquet")
    validation.to_parquet(args.out / "validation.parquet")
    print(f"Wrote {len(train)} train and {len(validation)} validation rows to {args.out}")


if __name__ == "__main__":
    main()
