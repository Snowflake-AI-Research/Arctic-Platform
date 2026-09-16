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

from arctic_platform.integrations.trl_distill import ArcticOPDRolloutWorker


class FakeOPD:
    def generate(self, prompts, sampling_params):
        assert sampling_params["logprobs"] == 0
        assert len(prompts) == 1
        return [{"token_ids": [3, 4], "logprobs": [-0.4, -0.5]}]

    def generate_teacher(self, prompts, sampling_params):
        assert sampling_params["prompt_logprobs"] == 2
        assert prompts == [[1, 2, 3, 4]]
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


def test_rollout_worker_generate_and_score():
    worker = ArcticOPDRolloutWorker(FakeOPD(), teacher_top_k=2)
    samples = worker.generate_and_score([[1, 2]], max_tokens=8)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.prompt_ids == [1, 2]
    assert sample.completion_ids == [3, 4]
    assert sample.sampler_logprobs == [-0.4, -0.5]
    assert sample.teacher_token_ids == [[3, 9], [4, 8]]
    assert worker.rollout_buffer.empty()


def test_start_enqueues_prompt_source():
    worker = ArcticOPDRolloutWorker(
        FakeOPD(), teacher_top_k=2, prompt_source=[[1, 2]], max_tokens=8
    )
    worker.start()
    assert not worker.rollout_buffer.empty()
    sample = worker.rollout_buffer.get_nowait()
    assert sample.input_ids == [1, 2, 3, 4]
    worker.stop()
