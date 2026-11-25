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

"""The media formats the sampling worker decodes, checked without the worker (``common.media_formats``)."""

import subprocess
import sys

import pytest

from arctic_platform.common.media_formats import MEDIA_FORMATS
from arctic_platform.common.media_formats import sniff_media_format


@pytest.mark.parametrize(
    "data, expected",
    [
        (b"\x89PNG\r\n\x1a\n" + b"\0" * 8, "png"),
        (b"\xff\xd8\xff\xe0" + b"\0" * 8, "jpeg"),
        (b"\0\0\0\x18ftypisom" + b"\0" * 8, "mp4"),
        (b"\0\0\0\x18ftypavc1" + b"\0" * 8, "mp4"),
        (b"\0\0\0\x18ftyphvc1" + b"\0" * 8, None),  # HEVC: not a brand decode_media takes
        (b"\0\0\0\x18ftypav01" + b"\0" * 8, None),
        (b"\0\0\0\x08free\0\0\0\x18ftypisom", None),  # ftyp must be the first box
        (b"GIF89a" + b"\0" * 8, None),
        (b"", None),
    ],
)
def test_sniff_media_format_names_what_the_worker_decodes(data, expected):
    assert sniff_media_format(data) == expected
    assert sniff_media_format(memoryview(bytearray(data))) == expected
    allowed = {
        modality
        for modality, signatures in MEDIA_FORMATS.items()
        if any(data[offset : offset + len(magic)] == magic for offset, magic in signatures)
    }
    assert allowed == ({"image"} if expected in ("png", "jpeg") else {"video"} if expected == "mp4" else set())


def test_importing_it_pulls_in_neither_vllm_nor_torch():
    code = (
        "import sys, arctic_platform.common.media_formats; "
        "print(sorted(m for m in ('vllm', 'torch') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "[]"
