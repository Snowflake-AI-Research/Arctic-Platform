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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkExample:
    prompt_text: str
    answer: str
    example_id: str


@dataclass
class BenchmarkResult:
    name: str
    score: float
    num_examples: int
    num_correct: int
    metrics: dict[str, float]


def load_math500(max_examples: int | None = None) -> list[BenchmarkExample]:
    from datasets import load_dataset
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    examples: list[BenchmarkExample] = []
    for idx, row in enumerate(ds):
        try:
            expected = extract_boxed(row["solution"])
        except ValueError:
            continue
        examples.append(
            BenchmarkExample(
                prompt_text=row["problem"],
                answer=expected,
                example_id=f"math500-{idx}",
            )
        )
    return examples


def run_math500(
    *,
    client: Any,
    job_id: str,
    renderer: Any,
    max_examples: int | None,
    sampling_params: dict[str, Any],
    generate_batch_size: int,
    max_seq_len: int,
) -> BenchmarkResult:
    from recipes.rl.math_grpo.train import build_prompt
    from recipes.rl.math_grpo.train import score_response

    examples = load_math500(max_examples)
    if len(examples) == 0:
        raise ValueError("MATH-500 produced no examples")
    prompts: list[list[int]] = []
    for example in examples:
        tokens = build_prompt(example.prompt_text, renderer)
        if len(tokens) >= max_seq_len:
            raise ValueError(
                f"{example.example_id} prompt has {len(tokens)} tokens; raise max_seq_len (currently {max_seq_len})"
            )
        prompts.append(tokens)

    results = generate_results(client, job_id, prompts, sampling_params, generate_batch_size)
    n_correct = 0
    format_sum = 0.0
    max_tokens = sampling_params.get("max_tokens")
    for example, result in zip(examples, results):
        text = result.get("text") or ""
        _reward, metrics = score_response(
            text,
            example.answer,
            result=result,
            max_tokens=max_tokens,
        )
        n_correct += int(metrics["correct"] >= 1.0)
        format_sum += float(metrics.get("format") or 0.0)
    n = len(examples)
    score = n_correct / n
    return BenchmarkResult(
        name="math500",
        score=score,
        num_examples=n,
        num_correct=n_correct,
        metrics={
            "math500/correct": score,
            "math500/format": format_sum / n,
            "math500/num_examples": float(n),
            "test/env/all/correct": score,
            "test/env/all/format": format_sum / n,
            "test/env/all/num_examples": float(n),
        },
    )


def generate_results(
    client: Any,
    job_id: str,
    prompts: list[list[int]],
    sampling_params: dict[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    width = max(1, batch_size)
    for start in range(0, len(prompts), width):
        batch = prompts[start : start + width]
        request_id = client.generate(job_id, prompts=batch, sampling_params=sampling_params)
        payload = client.poll_request(job_id, request_id)
        batch_results = payload.get("results") or []
        if len(batch_results) != len(batch):
            raise RuntimeError(f"asked for {len(batch)} completions, got {len(batch_results)}")
        for result in batch_results:
            results.append(result if isinstance(result, dict) else {"text": str(result)})
    return results
