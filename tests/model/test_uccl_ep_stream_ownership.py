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
