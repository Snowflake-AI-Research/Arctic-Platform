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

"""Download LoongRL-Train-Data from HuggingFace and save as verl-compatible parquets.

Subsets:
  - hotpotqa_qwen_0_2500 + hotpotqa_distractor_2500_5000
  - musique_qwen_0_2500 + musique_distractor_2500_5000
  - 2wikipedia_qwen_0_2500 + 2wikipedia_distractor_2500_5000

Each subset pair is merged into a single dataset per task, then split into
train/test parquets. A merged version across all tasks is also produced.
"""

import argparse
import os

from datasets import concatenate_datasets
from datasets import load_dataset

HF_REPO = "OldKingMeister/LoongRL-Train-Data"

SUBSET_PAIRS = {
    "hotpotqa": [
        "hotpotqa_qwen_0_2500",
        "hotpotqa_distractor_2500_5000",
    ],
    "musique": [
        "musique_qwen_0_2500",
        "musique_distractor_2500_5000",
    ],
    "2wikimqa": [
        "2wikipedia_qwen_0_2500",
        "2wikipedia_distractor_2500_5000",
    ],
}

VERL_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves "
    "it. The Assistant first thinks about the reasoning process in the mind and then provides the "
    "User with the answer. The reasoning process is enclosed within <think> </think> and answer is "
    "enclosed within \\boxed{} tags, respectively, i.e., <think> reasoning process here </think> "
    "\\boxed{answer here}."
)


def add_system_prompt(example):
    """Prepend a system message to the prompt."""
    prompt = list(example["prompt"])
    prompt.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"prompt": prompt}


def download_and_save(output_dir: str, test_ratio: float = 0.05, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    all_train = []
    all_test = []

    for task_name, subsets in SUBSET_PAIRS.items():
        print(f"\n{'='*60}")
        print(f"Processing task: {task_name}")
        print(f"{'='*60}")

        task_datasets = []
        for subset_name in subsets:
            print(f"  Downloading subset: {subset_name} ...")
            ds = load_dataset(HF_REPO, subset_name, split="train")
            print(f"  -> {len(ds)} rows")
            task_datasets.append(ds)

        merged = concatenate_datasets(task_datasets)
        print(f"  Merged: {len(merged)} rows")

        existing_cols = set(merged.column_names)
        keep_cols = [c for c in VERL_COLUMNS if c in existing_cols]
        drop_cols = [c for c in existing_cols if c not in keep_cols]
        if drop_cols:
            merged = merged.remove_columns(drop_cols)

        merged = merged.map(add_system_prompt)
        print("  Added system prompt to all rows")

        split = merged.train_test_split(test_size=test_ratio, seed=seed)
        train_ds = split["train"]
        test_ds = split["test"]

        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        train_path = os.path.join(task_dir, "train.parquet")
        test_path = os.path.join(task_dir, "test.parquet")
        train_ds.to_parquet(train_path)
        test_ds.to_parquet(test_path)
        print(f"  Saved: {train_path} ({len(train_ds)} rows)")
        print(f"  Saved: {test_path} ({len(test_ds)} rows)")

        all_train.append(train_ds)
        all_test.append(test_ds)

    merged_train = concatenate_datasets(all_train)
    merged_test = concatenate_datasets(all_test)

    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_train.to_parquet(os.path.join(merged_dir, "train.parquet"))
    merged_test.to_parquet(os.path.join(merged_dir, "test.parquet"))
    print("\nMerged all tasks:")
    print(f"  Train: {os.path.join(merged_dir, 'train.parquet')} ({len(merged_train)} rows)")
    print(f"  Test:  {os.path.join(merged_dir, 'test.parquet')} ({len(merged_test)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/data/snowflakesql/long-context",
    )
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_and_save(args.output_dir, args.test_ratio, args.seed)
