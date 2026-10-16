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

"""Which GPU widths one config is measured at, and what a config placed at a wider one keeps."""

import json
from pathlib import Path

import pytest

from arctic_platform.correctness.harness.arms import arms_for
from arctic_platform.correctness.harness.arms import scaled_for_placement
from arctic_platform.correctness.harness.config import GPU_WIDTH_MULTIPLE_CAP
from arctic_platform.correctness.harness.config import config_checksum
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.config import placement_widths
from arctic_platform.correctness.selftest.config_factory import native_config

# One row per allocation size and declared width. A node holds eight GPUs, so the pools are one, two and
# four nodes. The declared width is always first; the second entry is the same config placed wider.
CASES = [
    # one node
    (8, 4, [4, 8]),
    (8, 8, [8]),
    (8, 16, []),
    # two nodes
    (16, 4, [4, 16]),
    (16, 8, [8, 16]),
    (16, 16, [16]),
    # four nodes
    (32, 4, [4, 16]),
    (32, 8, [8, 32]),
    (32, 16, [16, 32]),
]


@pytest.mark.parametrize("pool,declared,expected", CASES)
def test_widths_for_pool_and_declared_width(pool: int, declared: int, expected: list) -> None:
    assert placement_widths(declared, pool) == expected


def test_a_pool_narrower_than_the_config_offers_no_width() -> None:
    """The config is run as written, so a pool it does not fit in yields nothing rather than a narrower run."""
    assert placement_widths(16, 8) == []
    assert placement_widths(16, 15) == []


def test_the_declared_width_is_always_measured_first() -> None:
    """A run cut short after one width has measured the topology the spec was reviewed against."""
    for pool, declared, expected in CASES:
        if expected:
            assert expected[0] == declared, (pool, declared)


def test_the_multiple_is_capped_however_large_the_pool_is() -> None:
    """A very wide allocation does not turn one config into an arbitrarily wide job."""
    widths = placement_widths(4, 4096)

    assert widths == [4, 4 * GPU_WIDTH_MULTIPLE_CAP]


def test_a_pool_that_holds_the_width_only_once_measures_it_once() -> None:
    """With no room for a second copy there is no wider topology to compare against."""
    assert placement_widths(8, 8) == [8]
    assert placement_widths(8, 15) == [8]


def test_a_nonpositive_declared_width_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        placement_widths(0, 8)


def _config(tmp_path: Path, n_gpus: int = 4, sp_size: int = 2, train_batch_size: int | None = 2):
    ds_config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1}
    training = {
        "n_gpus": n_gpus,
        "sp_size": sp_size,
        "max_seq_len": 4096,
        "attn_implementation": "flash_attention_3",
        "ds_config": ds_config,
    }
    # A config states the global batch in both places, and the checks read the training config's copy
    # while DeepSpeed reads the one under ``ds_config``.
    if train_batch_size is not None:
        ds_config["train_batch_size"] = train_batch_size
        training["train_batch_size"] = train_batch_size
    path = tmp_path / "train-sft-4gpus-64k.config"
    path.write_text(json.dumps(native_config(training, model_name="some/model")))
    return load_config(path)


def test_a_wider_placement_is_not_the_body_the_spec_froze(tmp_path: Path) -> None:
    """``n_gpus`` is part of the hashed training body, so a derived width hashes to something else.

    This is why the spec gate is applied to the config as loaded from the file and never to a config
    returned by ``at_gpu_width``: hashing the derived one would refuse every wider run as drifted, and
    relaxing the hash to ignore ``n_gpus`` would stop the gate noticing a real edit to that field.
    """
    cfg = _config(tmp_path)

    assert config_checksum(cfg.at_gpu_width(16)) != config_checksum(cfg)
    assert config_checksum(cfg) == config_checksum(load_config(cfg.path))


def test_only_the_gpu_count_differs(tmp_path: Path) -> None:
    """Every other training value stays as written, which is what makes the two widths comparable."""
    cfg = _config(tmp_path)

    wider = cfg.at_gpu_width(16)

    moved = {"n_gpus", "ds_config", "train_batch_size"}
    assert (wider.n_gpus, wider.sp_size) == (16, 2)
    assert {k: v for k, v in wider.training.items() if k not in moved} == {
        k: v for k, v in cfg.training.items() if k not in moved
    }
    assert (wider.config_id, wider.path, wider.max_seq_len) == (cfg.config_id, cfg.path, cfg.max_seq_len)


def test_the_config_it_was_derived_from_is_untouched(tmp_path: Path) -> None:
    """The declared width is measured too, so deriving must not overwrite it."""
    cfg = _config(tmp_path)

    cfg.at_gpu_width(16)

    assert cfg.n_gpus == 4
    assert cfg.training["n_gpus"] == 4


def test_the_extra_gpus_widen_data_parallelism(tmp_path: Path) -> None:
    """Sequence-parallel width is a config property and stays put; the extra ranks carry more rows."""
    cfg = _config(tmp_path, n_gpus=4, sp_size=2)

    wider = cfg.at_gpu_width(16)

    assert (wider.sp_size, wider.dp_size) == (2, 8)


def test_the_batch_grows_with_the_width(tmp_path: Path) -> None:
    """DeepSpeed asserts ``train_batch_size == micro * grad_acc * dp_size``, so the batch tracks the width."""
    cfg = _config(tmp_path, n_gpus=4, sp_size=2, train_batch_size=2)

    assert cfg.at_gpu_width(8).training["ds_config"]["train_batch_size"] == 4
    assert cfg.at_gpu_width(16).training["ds_config"]["train_batch_size"] == 8


def test_the_batch_grows_wherever_the_config_states_it(tmp_path: Path) -> None:
    """The multi-step checks size their dataset slice from the training config's own copy of the batch.

    The data plane needs at least one row per data-parallel shard, so a copy left at the declared value
    is short by the placement multiple and the first dispatch refuses the batch.
    """
    cfg = _config(tmp_path, n_gpus=4, sp_size=1, train_batch_size=4)

    assert cfg.at_gpu_width(8).training["train_batch_size"] == 8
    assert cfg.at_gpu_width(8).training["train_batch_size"] >= cfg.at_gpu_width(8).dp_size


def test_the_declared_width_repairs_a_batch_that_does_not_match_deepspeed(tmp_path: Path) -> None:
    """A 16-GPU file written with an 8-GPU batch still places: DeepSpeed needs 16, the file stays 8."""
    cfg = _config(tmp_path, n_gpus=16, sp_size=1, train_batch_size=8)

    placed = cfg.at_gpu_width(16)

    assert placed.training["train_batch_size"] == 16
    assert placed.training["ds_config"]["train_batch_size"] == 16
    assert cfg.training["train_batch_size"] == 8
    assert config_checksum(placed) != config_checksum(cfg)


def test_a_consistent_declared_width_leaves_the_batch_put(tmp_path: Path) -> None:
    """When the file already satisfies the identity, the declared-width copy is a no-op on the batch."""
    cfg = _config(tmp_path, n_gpus=4, sp_size=1, train_batch_size=4)

    placed = cfg.at_gpu_width(4)

    assert placed.training["train_batch_size"] == 4
    assert config_checksum(placed) == config_checksum(cfg)


def test_the_deepspeed_batch_assertion_holds_at_every_width(tmp_path: Path) -> None:
    """The invariant DeepSpeed checks at engine initialization, evaluated for each width this produces."""
    for sp_size in (1, 2, 4):
        cfg = _config(tmp_path, n_gpus=4, sp_size=sp_size, train_batch_size=4 // sp_size)
        for pool in (8, 16, 32):
            for width in placement_widths(cfg.n_gpus, pool):
                wider = cfg.at_gpu_width(width)
                ds_config = wider.training["ds_config"]
                assert ds_config["train_batch_size"] == (
                    ds_config["train_micro_batch_size_per_gpu"]
                    * ds_config["gradient_accumulation_steps"]
                    * wider.dp_size
                ), (sp_size, width)


def test_a_config_stating_no_batch_has_none_written_for_it(tmp_path: Path) -> None:
    """Nothing is invented: a config without ``train_batch_size`` is placed wider without acquiring one."""
    cfg = _config(tmp_path, n_gpus=4, sp_size=2, train_batch_size=None)

    wider = cfg.at_gpu_width(16)

    assert "train_batch_size" not in wider.training.get("ds_config", {})
    assert "train_batch_size" not in wider.training


def test_a_width_that_is_not_a_whole_multiple_is_refused(tmp_path: Path) -> None:
    cfg = _config(tmp_path, n_gpus=4, sp_size=2)

    with pytest.raises(ValueError, match="whole multiple"):
        cfg.at_gpu_width(6)


def test_every_width_this_dimension_produces_is_placeable(tmp_path: Path) -> None:
    """A multiple of the declared width is always divisible by a sequence-parallel size that divides it."""
    cfg = _config(tmp_path, n_gpus=4, sp_size=4)

    for pool in (8, 16, 32, 64):
        for width in placement_widths(cfg.n_gpus, pool):
            assert cfg.at_gpu_width(width).n_gpus == width


def test_a_wider_placement_carries_proportionally_more_rows() -> None:
    """Only the row count moves; the padded width and the case names are the reviewed ones."""
    declared = arms_for(8192, 4)

    wider = scaled_for_placement(declared, 4)

    assert [a.name for a in wider] == [a.name for a in declared]
    assert [a.max_seq_len for a in wider] == [a.max_seq_len for a in declared]
    assert [a.global_batch_size for a in wider] == [a.global_batch_size * 4 for a in declared]


def test_the_declared_width_leaves_the_cases_untouched() -> None:
    """A multiple of one is the config as written, so its cases are the reviewed ones exactly."""
    declared = arms_for(8192, 8)

    assert scaled_for_placement(declared, 1) == declared


def test_every_data_parallel_shard_receives_a_row_at_every_width(tmp_path: Path) -> None:
    """The condition Arctic Platform enforces when it dispatches a batch: rows must cover the data-parallel width.

    Without scaling the cases this fails at any width above the declared one, because the reviewed case
    holds one row per declared GPU and a wider placement has more shards than rows.
    """
    for sp_size in (1, 2, 4):
        cfg = _config(tmp_path, n_gpus=4, sp_size=sp_size, train_batch_size=4 // sp_size)
        declared = arms_for(cfg.max_seq_len, cfg.n_gpus)
        for pool in (8, 16, 32):
            for width in placement_widths(cfg.n_gpus, pool):
                wider = cfg.at_gpu_width(width)
                for arm in scaled_for_placement(declared, width // cfg.n_gpus):
                    assert arm.global_batch_size >= wider.dp_size, (sp_size, width, arm.name)


def test_a_multiple_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        scaled_for_placement(arms_for(8192, 4), 0)
