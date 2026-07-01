# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download LoongRL-Train-Data from HuggingFace and save as SkyRL-format parquets.

Subsets (same as recipes/rl/verl/long_context_qa/download_data.py — keeps the two
recipes pointing at byte-identical training distributions):

  - hotpotqa_qwen_0_2500 + hotpotqa_distractor_2500_5000
  - musique_qwen_0_2500  + musique_distractor_2500_5000
  - 2wikipedia_qwen_0_2500 + 2wikipedia_distractor_2500_5000

Each task is concatenated + train/test split, then per-task and merged parquets are
written.  Rows are converted to SkyRL's standard schema:

    data_source, prompt, env_class="long_context_qa", reward_spec={method, ground_truth}, extra_info

The vendored arctic_rl LongContextQAEnv accepts both ``reward_model.ground_truth`` (verl
schema) and ``reward_spec.ground_truth`` (SkyRL schema), so the launcher works against
either; we write the SkyRL shape here for consistency with the simple/ recipe.

Output layout::

    <output_dir>/
    ├── hotpotqa/{train,test}.parquet
    ├── musique/{train,test}.parquet
    ├── 2wikimqa/{train,test}.parquet
    └── merged/
        ├── train.parquet      ~14k rows (all three tasks concatenated)
        └── test.parquet       ~750 rows
"""

import argparse
import os

from datasets import concatenate_datasets, load_dataset

HF_REPO = "OldKingMeister/LoongRL-Train-Data"

SUBSET_PAIRS = {
    "hotpotqa": ["hotpotqa_qwen_0_2500", "hotpotqa_distractor_2500_5000"],
    "musique": ["musique_qwen_0_2500", "musique_distractor_2500_5000"],
    "2wikimqa": ["2wikipedia_qwen_0_2500", "2wikipedia_distractor_2500_5000"],
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the "
    "Assistant solves it. The Assistant first thinks about the reasoning process in the "
    "mind and then provides the User with the answer. The reasoning process is enclosed "
    "within <think> </think> and answer is enclosed within \\boxed{} tags, respectively, "
    "i.e., <think> reasoning process here </think> \\boxed{answer here}."
)


def _to_skyrl_row(example, idx: int, split: str, task_name: str):
    """Convert a verl-format LoongRL row to SkyRL schema."""
    prompt = list(example["prompt"])
    prompt.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    reward_model = example.get("reward_model") or {}
    ground_truth = reward_model.get("ground_truth")

    extra_info = dict(example.get("extra_info") or {})
    extra_info.update({"split": split, "task": task_name, "index": idx})

    return {
        "data_source": example.get("data_source") or f"loongrl/{task_name}",
        "prompt": prompt,
        "env_class": "long_context_qa",
        "reward_spec": {"method": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
    }


def download_and_save(output_dir: str, test_ratio: float = 0.05, seed: int = 42):
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_train, all_test = [], []

    for task_name, subsets in SUBSET_PAIRS.items():
        print(f"\n=== Task: {task_name} ===")
        task_datasets = []
        for subset in subsets:
            print(f"  Downloading subset: {subset}")
            ds = load_dataset(HF_REPO, subset, split="train")
            print(f"    -> {len(ds)} rows")
            task_datasets.append(ds)

        merged = concatenate_datasets(task_datasets)
        print(f"  Merged: {len(merged)} rows")

        split = merged.train_test_split(test_size=test_ratio, seed=seed)
        train_ds = split["train"].map(
            lambda ex, idx, t=task_name: _to_skyrl_row(ex, idx, "train", t),
            with_indices=True,
            remove_columns=split["train"].column_names,
        )
        test_ds = split["test"].map(
            lambda ex, idx, t=task_name: _to_skyrl_row(ex, idx, "test", t),
            with_indices=True,
            remove_columns=split["test"].column_names,
        )

        task_dir = os.path.join(output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        train_ds.to_parquet(os.path.join(task_dir, "train.parquet"))
        test_ds.to_parquet(os.path.join(task_dir, "test.parquet"))
        print(f"  Saved: {task_dir}/{{train,test}}.parquet "
              f"({len(train_ds)} / {len(test_ds)} rows)")

        all_train.append(train_ds)
        all_test.append(test_ds)

    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_train = concatenate_datasets(all_train)
    merged_test = concatenate_datasets(all_test)
    merged_train.to_parquet(os.path.join(merged_dir, "train.parquet"))
    merged_test.to_parquet(os.path.join(merged_dir, "test.parquet"))
    print(f"\nMerged:")
    print(f"  train: {len(merged_train)} rows -> {merged_dir}/train.parquet")
    print(f"  test:  {len(merged_test)} rows -> {merged_dir}/test.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/loongrl")
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    download_and_save(args.output_dir, args.test_ratio, args.seed)
