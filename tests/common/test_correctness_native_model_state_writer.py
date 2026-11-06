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

"""The native optimizer capture's rank-zero module payload under node-local checkpoint storage.

The correctness gateway sets ``checkpoint.use_node_local_storage``. DeepSpeed then selects the model-state writer by
``LOCAL_RANK``, which DSS sets to ``0`` in every Ray worker, so every rank of a node writes
``mp_rank_00_model_states.pt``. Under expert parallelism each rank's module holds different experts. CPU only:
``FakeEngine`` reproduces the two DeepSpeed attributes involved, and the save mirrors ``DeepSpeedEngine._save_checkpoint``
writing the module state when ``save_non_zero_checkpoint`` is set.
"""

from pathlib import Path

import pytest
import torch

from arctic_platform.correctness.harness.optimizer import expert_sidefile_module_mismatch
from arctic_platform.correctness.harness.optimizer import node_local_model_state_writer

EXPERT = "model.layers.1.mlp.experts.w1"


class FakeEngine:
    """DeepSpeed's node-local writer selection with ``LOCAL_RANK=0``: every rank is configured as a writer."""

    def __init__(self, experts: torch.Tensor, *, node_local: bool = True):
        self.module = {"model.embed_tokens.weight": torch.ones(4, 2), EXPERT: experts}
        self.save_non_zero_checkpoint = True
        self._node_local = node_local

    def use_node_local_storage(self):
        return self._node_local

    def save(self, path: Path):
        if self.save_non_zero_checkpoint:
            torch.save({"module": self.module}, path)


def node_engines(node_ranks: int) -> list[FakeEngine]:
    return [FakeEngine(torch.full((2, 3), float(rank), dtype=torch.bfloat16)) for rank in range(node_ranks)]


def test_first_node_rank_alone_writes_the_model_state(tmp_path):
    path = tmp_path / "mp_rank_00_model_states.pt"
    for engine in node_engines(4):
        engine.save(path)
    # Without the writer guard the last rank of the node leaves its own experts in the rank-zero file.
    assert torch.equal(torch.load(path)["module"][EXPERT], node_engines(4)[3].module[EXPERT])

    path.unlink()
    engines = node_engines(4)
    for node_rank, engine in enumerate(engines):
        with node_local_model_state_writer(engine, node_rank):
            engine.save(path)
    assert torch.equal(torch.load(path)["module"][EXPERT], engines[0].module[EXPERT])
    assert all(engine.save_non_zero_checkpoint for engine in engines)


@pytest.mark.parametrize("configured", [True, False])
def test_shared_storage_writer_selection_is_unchanged(configured):
    engine = FakeEngine(torch.zeros(2, 3, dtype=torch.bfloat16), node_local=False)
    engine.save_non_zero_checkpoint = configured
    with node_local_model_state_writer(engine, 3):
        assert engine.save_non_zero_checkpoint is configured
    assert engine.save_non_zero_checkpoint is configured


def test_writer_selection_is_restored_after_a_failed_save():
    engine = FakeEngine(torch.zeros(2, 3, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="save failed"):
        with node_local_model_state_writer(engine, 2):
            assert engine.save_non_zero_checkpoint is False
            raise RuntimeError("save failed")
    assert engine.save_non_zero_checkpoint is True


@pytest.mark.parametrize("node_rank", [-1, 0.0, None])
def test_writer_requires_an_integer_node_rank(node_rank):
    with pytest.raises(ValueError, match="nonnegative rank within its node"):
        with node_local_model_state_writer(FakeEngine(torch.zeros(1)), node_rank):
            pass


def test_mismatch_names_the_tensor_both_values_and_the_module_owner():
    dense = "model.layers.1.mlp.experts.w2"
    files = [Path("experts_ep_rank_000.pt"), Path("experts_ep_rank_001.pt")]
    experts = [
        {EXPERT: torch.zeros(2, 3, dtype=torch.bfloat16), dense: torch.ones(3, 2, dtype=torch.bfloat16)},
        {EXPERT: torch.full((2, 3), 0.25, dtype=torch.bfloat16), dense: torch.ones(3, 2, dtype=torch.bfloat16)},
    ]
    module = {EXPERT: experts[1][EXPERT].clone(), dense: experts[1][dense].clone()}

    message = expert_sidefile_module_mismatch(files, experts, 0, module)

    assert (
        message
        == "rank-zero expert sidefile/module mismatch: 1 of 2 tensors in experts_ep_rank_000.pt differ from the "
        f"mp_rank_00 module payload; first {EXPERT!r} (2, 3) torch.bfloat16: 6/6 elements differ, max |sidefile - "
        "module| 2.500000e-01 at (0, 0) (sidefile 0.0, module 0.25); module tensor equals experts_ep_rank_001.pt"
    )
    assert expert_sidefile_module_mismatch(files, experts, 1, module) is None
