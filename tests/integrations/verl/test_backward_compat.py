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

"""Backward-compat: legacy import path still resolves.

The verl-shaped GRPO loss moved from
``arctic_platform/rl/processors/verl_grpo.py`` to
``arctic_platform/integrations/verl/grpo_loss.py``. The old module
remains as a thin re-export shim so downstream users doing
``from arctic_platform.rl.processors.verl_grpo import ...`` continue to
work.
"""

from __future__ import annotations

import pytest

try:
    # The old and new module both register the loss on
    # `arctic_platform.rl.processors.pipeline`, which pulls in ray via
    # `arctic_platform.rl.__init__`. Skip if that's not installed.
    import arctic_platform.rl.processors  # noqa: F401
except ImportError:
    pytest.skip(
        "arctic_platform.rl.processors is not importable (missing [rl] extras). This test requires the full RL stack.",
        allow_module_level=True,
    )


def test_legacy_verl_grpo_module_still_imports() -> None:
    from arctic_platform.rl.processors import verl_grpo as legacy

    assert hasattr(legacy, "verl_grpo_loss"), "shim must re-export verl_grpo_loss"
    assert callable(legacy.verl_grpo_loss)


def test_new_grpo_loss_module_imports_directly() -> None:
    from arctic_platform.integrations.verl import grpo_loss as new

    assert hasattr(new, "verl_grpo_loss")
    assert callable(new.verl_grpo_loss)


def test_shim_and_new_module_share_same_function_object() -> None:
    """The shim must re-export the same function object, not a copy."""
    from arctic_platform.integrations.verl.grpo_loss import verl_grpo_loss as new_fn
    from arctic_platform.rl.processors.verl_grpo import verl_grpo_loss as shim_fn

    assert shim_fn is new_fn


def test_verl_grpo_registered_in_loss_fns_registry() -> None:
    """Importing the shim (or the new module) must register the loss on the pipeline registry."""
    # Force a re-import path that exercises the shim: importing the
    # processors package pulls the shim in via
    # `from .verl_grpo import verl_grpo_loss`.
    # Trigger shim import if the package init hasn't already.
    from arctic_platform.rl.processors import verl_grpo as _shim  # noqa: F401
    from arctic_platform.rl.processors.pipeline import LOSS_FNS

    assert "verl_grpo" in LOSS_FNS, (
        "importing arctic_platform.rl.processors.verl_grpo must register the "
        "'verl_grpo' loss fn (via the moved implementation)."
    )
