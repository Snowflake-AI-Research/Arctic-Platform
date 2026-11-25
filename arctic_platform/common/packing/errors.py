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

"""Errors for a malformed media batch, shared by the packing modules that check one."""

from __future__ import annotations

from typing import Tuple


class MediaBatchError(ValueError):
    """The caller's media batch is malformed: its media leaves disagree with each other, its rows or its tokens.

    A ValueError, so callers that catch ValueError still do; a caller that must tell a bad batch from an
    internal error catches this instead of matching the message.
    """


class MediaStraddleError(MediaBatchError):
    """A media item's tokens straddle a window boundary, so no window can hold the item whole.

    A MediaBatchError; a caller that must tell this refusal from other malformed media (to change its layout
    and retry, say) catches this instead of matching the message.
    ``kind`` is the media kind's name (``"image"`` or ``"video"``), ``index`` the item's position among that
    kind's items in the call, ``span`` its ``(start, end)`` token range and ``boundary`` the token it crosses.
    """

    def __init__(self, kind: str, index: int, span: Tuple[int, int], boundary: int) -> None:
        start, end = span
        super().__init__(
            f"{kind} {index}'s tokens [{start}, {end}) straddle the window boundary at token {boundary}; "
            "each window would get all its pixels but only part of its tokens. Move the boundary off "
            "the item: pad or reorder the text before it so it sits inside one window"
        )
        self.kind = kind
        self.index = index
        self.span = (start, end)
        self.boundary = boundary

    def __reduce__(self):
        # Rebuild from the attributes, not the message, so the error survives a trip across processes.
        return type(self), (self.kind, self.index, self.span, self.boundary)
