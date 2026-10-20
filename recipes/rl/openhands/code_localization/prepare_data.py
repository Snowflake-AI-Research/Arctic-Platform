# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow"]
# ///
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
"""Write train.parquet and validation.parquet from the SWE-smith code-search set.

Adapted from ``src/build_dataset.py --use_patch`` in
https://github.com/OpenHands/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2
The output matches CodeScout's ``data/swe_smith`` row for row.

A SWE-smith patch introduces the bug, so every row sets ``use_patch`` and the
rollout applies the patch after cloning the repo.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

DATASET = "OpenHands/SWE-smith-py-code-search"
REVISION = "b1e6864f1c847455ea64c7ebdb4164da2b902bb6"
URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/train-00000-of-00001.parquet"
VALIDATION_ROWS = 100
SEED = 42


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, validation): the shuffled rows, last 100 held out."""
    rows = raw[raw["problem_statement"].str.strip().astype(bool)].copy()
    rows["target"] = rows["file_changes"]
    rows["prompt"] = [[{"role": "user", "content": text}] for text in rows["problem_statement"]]
    rows["use_patch"] = True
    rows["base_commit"] = None
    rows = rows.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return rows.iloc[:-VALIDATION_ROWS], rows.iloc[-VALIDATION_ROWS:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path.home() / "data" / "swe-smith-localization")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.parquet"
        print(f"Downloading {DATASET}@{REVISION[:8]}")
        urllib.request.urlretrieve(URL, source)
        train, validation = build(pd.read_parquet(source))

    args.out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out / "train.parquet")
    validation.to_parquet(args.out / "validation.parquet")
    print(f"Wrote {len(train)} train and {len(validation)} validation rows to {args.out}")


if __name__ == "__main__":
    main()
