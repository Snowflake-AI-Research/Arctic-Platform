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

"""Source guard for zero-token UCCL-EP combine stream ownership.

``combine`` and ``internode_combine`` substitute a fresh ``src_idx`` or ``src_meta``
when a rank receives zero tokens, then pass that pointer to an async kernel. The
substitution is not reachable through ``handle``, so it has to be recorded on its
own. This reads the implementation as text because the substitution only triggers
on a zero-token rank in a live multi-node EP job, and importing the module needs
the ``uccl`` runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest

BUFFER_PATH = (
    Path(__file__).resolve().parents[2] / "arctic_platform/model/implementations/moe/distributed/uccl_ep/buffer.py"
)


def test_combine_paths_record_the_substituted_index_tensor():
    """Both combine paths must list the substituted tensor alongside ``handle``."""
    src = BUFFER_PATH.read_text()

    for substituted in ("src_idx", "src_meta"):
        marker = f"            handle,\n            {substituted},\n"
        assert marker in src, (
            f"{substituted} is rebound by a zero-token substitution and must be recorded "
            "explicitly; recording only 'handle' covers the pre-substitution tensor"
        )


def _record_stream_safe():
    pytest.importorskip("uccl", reason="uccl ships the C++ EP runtime")
    from arctic_platform.model.implementations.moe.distributed.uccl_ep.buffer import _record_stream_safe as fn

    return fn


def _recording_tensor_cls(torch):
    class _RecordingTensor(torch.Tensor):
        """Tensor that logs ``record_stream`` calls instead of requiring a real stream."""

        @staticmethod
        def __new__(cls, log):
            inst = torch.Tensor._make_subclass(cls, torch.zeros(1, device="cuda"))
            inst._log = log
            return inst

        def record_stream(self, stream):  # type: ignore[override]
            self._log.append(self)

    return _RecordingTensor


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("record_stream needs CUDA")
    return torch


def test_record_stream_safe_recurses_exactly_one_level():
    """One level of nesting is recorded; two are not.

    This is the property that made the bug possible: a tensor nested inside the
    ``handle`` tuple is covered, but nothing deeper is.
    """
    torch = _cuda_or_skip()
    record = _record_stream_safe()
    tensor_cls = _recording_tensor_cls(torch)

    log: list = []
    top = tensor_cls(log)
    nested = tensor_cls(log)
    deep = tensor_cls(log)

    record((top, (nested, (deep,))), torch.cuda.current_stream())

    # Identity, not ``in``: equal-valued tensors compare elementwise.
    recorded = [id(t) for t in log]
    assert id(top) in recorded, "a directly passed tensor must be recorded"
    assert id(nested) in recorded, "a tensor one level deep (as in handle) must be recorded"
    assert id(deep) not in recorded, "recursion is one level only; deeper tensors need to be passed explicitly"


def test_record_stream_safe_skips_none_and_non_tensors():
    torch = _cuda_or_skip()
    record = _record_stream_safe()
    tensor = _recording_tensor_cls(torch)(log := [])

    record((None, 7, "handle", tensor), torch.cuda.current_stream())

    assert [id(t) for t in log] == [id(tensor)]


def test_record_stream_safe_no_stream_is_a_noop():
    torch = _cuda_or_skip()
    record = _record_stream_safe()
    log: list = []

    record((_recording_tensor_cls(torch)(log),), None)

    assert log == []
