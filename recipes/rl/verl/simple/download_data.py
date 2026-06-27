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

"""Download GSM8K from HuggingFace and save as verl-compatible parquets.

GSM8K ships its own train/test splits. Each row is converted to verl's standard schema with
``data_source="openai/gsm8k"``, which verl scores with its built-in GSM8K reward (exact match on the
``#### <number>`` final answer) — no custom reward function needed.
"""

import argparse
import os
import re

from datasets import load_dataset

DATA_SOURCE = "openai/gsm8k"
INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def extract_solution(answer_raw: str) -> str:
    """Pull the gold numeric answer out of a GSM8K `#### <number>` answer field."""
    match = re.search("#### (\\-?[0-9\\.\\,]+)", answer_raw)
    assert match is not None, f"no '#### <answer>' found in: {answer_raw!r}"
    return match.group(0).split("#### ")[1].replace(",", "")


def make_row(example, idx, split):
    question_raw = example["question"]
    answer_raw = example["answer"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": question_raw + " " + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": extract_solution(answer_raw)},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        },
    }


def download_and_save(output_dir: str):
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_dataset(DATA_SOURCE, "main")
    for split in ("train", "test"):
        ds = dataset[split].map(
            lambda ex, idx: make_row(ex, idx, split),
            with_indices=True,
            remove_columns=dataset[split].column_names,
        )
        out_path = os.path.join(output_dir, f"{split}.parquet")
        ds.to_parquet(out_path)
        print(f"Saved: {out_path} ({len(ds)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/gsm8k")
    args = parser.parse_args()

    download_and_save(args.output_dir)
