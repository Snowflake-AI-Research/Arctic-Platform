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

"""The encoded media formats the sampling worker decodes, by their leading bytes.

``inference.server.media_prompts.decode_media`` refuses any other bytes before a loader sees them. This module
imports nothing beyond the standard library, so a service in front of the worker (with no vLLM) can refuse the
same bytes at its edge.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MEDIA_FORMATS", "sniff_media_format"]

# The encoded formats ``decode_media`` hands to vLLM's loaders, by modality, as
# (offset, leading bytes) signatures: PNG and JPEG images, and MP4 video (an
# ISO-BMFF file whose first box is ``ftyp`` with one of these major brands).
# Bytes in any other format are refused before a loader sees them, so Pillow's
# other plugins and the video codec's other demuxers never parse request bytes.
_MP4_BRANDS = (b"isom", b"iso2", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1")
# The same signatures by format name, for a caller that checks bytes against a declared format.
_FORMAT_SIGNATURES: dict[str, tuple[tuple[int, bytes], ...]] = {
    "png": ((0, b"\x89PNG\r\n\x1a\n"),),
    "jpeg": ((0, b"\xff\xd8\xff"),),
    "mp4": tuple((4, b"ftyp" + brand) for brand in _MP4_BRANDS),
}
MEDIA_FORMATS: dict[str, tuple[tuple[int, bytes], ...]] = {
    "image": _FORMAT_SIGNATURES["png"] + _FORMAT_SIGNATURES["jpeg"],
    "video": _FORMAT_SIGNATURES["mp4"],
}


def sniff_media_format(data: Any) -> str | None:
    """``"png"``, ``"jpeg"`` or ``"mp4"`` when *data* opens with a signature ``decode_media`` accepts, else None.

    The worker's own check, by format name, so a service in front of it can refuse at its edge exactly the bytes
    the worker would refuse after queueing. *data* is any bytes-like object; only its first 16 bytes are read.
    """
    head = bytes(memoryview(data).cast("B")[:16])
    for name, signatures in _FORMAT_SIGNATURES.items():
        if any(head[offset : offset + len(magic)] == magic for offset, magic in signatures):
            return name
    return None
