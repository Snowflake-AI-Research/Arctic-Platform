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
"""An OpenAI-compatible ``/v1`` surface over an Arctic sampling job.

Point any OpenAI client at it -- the ``openai`` SDK, LiteLLM, an eval harness,
curl -- and change nothing but ``base_url``. Non-streaming: ``stream=true`` is
refused rather than faked.

See ``README.md`` in this package for the parameter-by-parameter compatibility
table.
"""

from typing import Any

__all__ = ["OpenAIGateway", "app_for_client", "backend_for", "build_app", "router"]


def __getattr__(name: str) -> Any:
    # fastapi/uvicorn/transformers are only needed when someone actually serves,
    # so importing this package stays cheap for callers that just want the
    # translation helpers.
    if name in ("build_app", "app_for_client", "OpenAIGateway"):
        from arctic_platform.openai_compat import server

        return getattr(server, name)
    if name == "router":
        from arctic_platform.openai_compat.router import router

        return router
    if name == "backend_for":
        from arctic_platform.openai_compat.backend import backend_for

        return backend_for
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
