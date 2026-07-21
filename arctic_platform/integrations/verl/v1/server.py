# Copyright 2026 Snowflake Inc.
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
"""V1-compatible Arctic LLM server.

Emits ``TokenOutput.extra_fields["global_steps"]`` (V1's real field; V0
used ``extra_info`` which pydantic silently drops) and exposes
:meth:`set_global_steps` so :class:`ArcticV1Replica` can fan the current
policy version out after each ``CheckpointEngineManager`` weight sync.
The V0 server file stays untouched.
"""

from __future__ import annotations

from typing import Any, Optional

from verl.workers.rollout.replica import TokenOutput

from arctic_platform.integrations.verl.rollout import ArcticLLMServer


class ArcticV1LLMServer(ArcticLLMServer):
    """V1-compat Arctic LLM server: publishes ``global_steps`` on every generation."""

    async def set_global_steps(self, global_steps: int) -> None:
        """Publish the current policy version onto this server so subsequent
        :meth:`generate` calls tag their outputs with the right version.
        """
        self.global_steps = global_steps

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        out: TokenOutput = await super().generate(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
            video_data=video_data,
            priority=priority,
        )
        # Re-publish global_steps under V1's real field name.
        extra_fields = dict(out.extra_fields or {})
        extra_fields["global_steps"] = self.global_steps
        return out.model_copy(update={"extra_fields": extra_fields})
