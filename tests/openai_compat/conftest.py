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
"""Fixtures for the OpenAI-compat suite.

The tokenizer is the real Qwen3 one rather than a stub: a stub proves the
routes emit the right JSON, but only a real ``chat_template`` proves that
``tools`` actually reach the prompt and that tool turns render.
"""

from __future__ import annotations

from typing import Any

import pytest
from openai_harness import MODEL
from openai_harness import RecordingBackend
from openai_harness import http_client
from openai_harness import make_app


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{MODEL} tokenizer is not in the local HF cache: {exc}")


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def app(backend: RecordingBackend, tokenizer: Any) -> Any:
    return make_app(backend, tokenizer)


@pytest.fixture
def http(app: Any):
    pytest.importorskip("httpx")
    with http_client(app) as client:
        yield client
