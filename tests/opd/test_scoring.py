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

import pytest

from arctic_platform.opd.scoring import score_teacher
from arctic_platform.opd.scoring import score_teacher_topk


class Teacher:
    def generate_teacher(self, prompts, sampling_params):
        assert prompts == [[1, 2, 3, 4]]
        assert sampling_params["prompt_logprobs"] == 0
        return [
            {
                "prompt_logprobs": [
                    None,
                    {2: -0.1},
                    {"3": {"logprob": -0.2}},
                    {4: -0.3},
                ]
            }
        ]


def test_teacher_scores_exact_completion_token_ids():
    scored = score_teacher(
        Teacher(),
        [{"prompt_ids": [1, 2], "completion_ids": [3, 4], "sampler_logprobs": [-0.4, -0.5]}],
    )
    assert scored[0]["teacher_logprobs"] == [-0.2, -0.3]


def test_teacher_topk_keeps_support_and_tail():
    class TopKTeacher:
        def generate_teacher(self, prompts, sampling_params):
            assert sampling_params["prompt_logprobs"] == 2
            return [
                {
                    "prompt_logprobs": [
                        None,
                        {2: -0.1},
                        {3: -0.2, 9: -1.0},
                        {4: -0.3, 8: -2.0},
                    ]
                }
            ]

    scored = score_teacher_topk(
        TopKTeacher(),
        [{"prompt_ids": [1, 2], "completion_ids": [3, 4]}],
        teacher_top_k=2,
    )
    row = scored[0]
    assert row["teacher_token_ids"] == [[3, 9], [4, 8]]
    assert row["teacher_logprobs"] == [[-0.2, -1.0], [-0.3, -2.0]]
    assert len(row["teacher_tail_logprob"]) == 2


def test_teacher_topk_requires_realized_token():
    class MissingRealized:
        def generate_teacher(self, prompts, sampling_params):
            return [{"prompt_logprobs": [None, {9: -0.1}]}]

    with pytest.raises(RuntimeError, match="absent"):
        score_teacher_topk(
            MissingRealized(),
            [{"prompt_ids": [1], "completion_ids": [2]}],
            teacher_top_k=1,
        )


def test_teacher_alignment_mismatch_fails():
    class BadTeacher:
        def generate_teacher(self, prompts, sampling_params):
            return [{"prompt_logprobs": [None]}]

    with pytest.raises(RuntimeError, match="length"):
        score_teacher(BadTeacher(), [{"prompt_ids": [1], "completion_ids": [2]}])
