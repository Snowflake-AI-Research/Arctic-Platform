from __future__ import annotations

import pytest

from arctic_platform.opd.scoring import score_teacher


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


def test_teacher_alignment_mismatch_fails():
    class BadTeacher:
        def generate_teacher(self, prompts, sampling_params):
            return [{"prompt_logprobs": [None]}]

    with pytest.raises(RuntimeError, match="length"):
        score_teacher(BadTeacher(), [{"prompt_ids": [1], "completion_ids": [2]}])
