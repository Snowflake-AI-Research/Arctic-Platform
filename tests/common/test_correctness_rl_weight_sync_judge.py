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

import copy
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

# Load before tests replace deepspeed in sys.modules; the export imports this module lazily.
import arctic_platform.model.weight_export  # noqa: F401
from arctic_platform.correctness.checks.rl_weight_sync import judge
from arctic_platform.debug import weight_sync
from arctic_platform.debug.weight_sync import ShadowVerifier
from arctic_platform.debug.weight_sync import checkpoint_shard_manifests
from arctic_platform.debug.weight_sync import checkpoint_shard_rows
from arctic_platform.debug.weight_sync import global_inventory
from arctic_platform.debug.weight_sync import tensor_manifest

# Four checkpoint shards of two rows; trainer EP and sampler TP each hold four rows per rank.
TRAINER_PARAMETER = "model.language_model.layers.1._checkpoint_wrapped_module.ple.ple_embedding.ngram_embedding.weight"
TRAINER_TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
SAMPLER_TABLE = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight"
ROWS, PARTS, HALF = 8, 4, 4
TABLE = torch.arange(ROWS * 3, dtype=torch.bfloat16).reshape(ROWS, 3)
HEAD = torch.ones(4, dtype=torch.bfloat16)
MAPPER = SimpleNamespace(orig_to_new_prefix={"model.language_model.": "language_model.model.", "mtp.": None})


def _half(table, position):
    return table[position * HALF : (position + 1) * HALF]


def _shards(table, *indices):
    size = ROWS // PARTS
    return {str(index): tensor_manifest(table[index * size : (index + 1) * size]) for index in indices}


def _names(failed):
    return [mismatch.name for mismatch in failed]


def _sync(sampler_table=TABLE):
    """One transmitted trainable tensor and a frozen table proved unchanged instead of transmitted."""
    before = [
        dict(
            rank=rank,
            world_size=2,
            parameters={
                "head.weight": dict(**tensor_manifest(HEAD), requires_grad=True, ep_ranks=None),
                TRAINER_PARAMETER: dict(
                    **tensor_manifest(_half(TABLE, rank)), requires_grad=False, ep_ranks=[0, 1], skip_weight_sync=True
                ),
            },
            buffers={},
        )
        for rank in range(2)
    ]
    after = [
        dict(
            rank=rank,
            world_size=2,
            error=None,
            parameters={TRAINER_PARAMETER: tensor_manifest(_half(TABLE, rank))},
            checkpoint_shards={
                TRAINER_PARAMETER: dict(checkpoint_name=TRAINER_TABLE, shards=_shards(TABLE, 2 * rank, 2 * rank + 1))
            },
        )
        for rank in range(2)
    ]
    sent = {"head.weight": tensor_manifest(HEAD)}
    samplers = [
        {
            "status": "ok",
            "received": {"head.weight": {**sent["head.weight"], "destinations": ["head.weight"]}},
            "destinations": {"head.weight": {"uncovered": 0, "mismatched": 0}},
            "untouched_parameters": {SAMPLER_TABLE: HALF * 3},
            "unverifiable_destinations": [],
            "non_parameter_destinations": [],
            "changed_after_verification": [],
            "checkpoint_shards": {
                SAMPLER_TABLE: dict(
                    checkpoint_name=TRAINER_TABLE, shards=_shards(sampler_table, 2 * rank, 2 * rank + 1)
                )
            },
        }
        for rank in range(2)
    ]
    send = {
        "sent_manifest": sent,
        "trainer_parameters": {"count": 2, "numel": HEAD.numel() + TABLE.numel()},
        "trainer_inventory": global_inventory(before),
        "trainer_skipped_parameters": dict(schema="frozen-weight-sync-skip-v1", before=before, after=after),
    }
    return {
        "weight_format": "vllm",
        "targets": [{"send": send, "recv": {"all_rank_weight_sync_verification": samplers}}],
    }


def _sampler(sync, rank):
    return sync["targets"][0]["recv"]["all_rank_weight_sync_verification"][rank]


def _trainer(sync, rank):
    return sync["targets"][0]["send"]["trainer_skipped_parameters"]["after"][rank]


def test_equal_checkpoint_shard_digests_pass():
    failed, metrics = judge(_sync())
    assert failed == []
    assert metrics["trainer_skipped_frozen_elements"] == TABLE.numel()


def test_one_differing_checkpoint_shard_fails_naming_the_parameter_and_the_shard():
    changed = TABLE.clone()
    changed[7, 0] += 1
    failed, _ = judge(_sync(sampler_table=changed))
    assert _names(failed) == [
        f"sampler rank 1: parameter {SAMPLER_TABLE} checkpoint shard 3 differs from the trainer's"
    ]


@pytest.mark.parametrize("missing", ["sampler record", "trainer records", "sampler shard", "trainer shard"])
def test_missing_checkpoint_shard_digests_fail(missing):
    sync = _sync()
    expected = [f"sampler rank 1: parameter {SAMPLER_TABLE} written by no tensor"]
    if missing == "sampler record":
        del _sampler(sync, 1)["checkpoint_shards"]
    elif missing == "trainer records":
        for rank in range(2):
            del _trainer(sync, rank)["checkpoint_shards"]
        expected = [f"sampler rank {rank}: parameter {SAMPLER_TABLE} written by no tensor" for rank in range(2)]
    elif missing == "sampler shard":
        _sampler(sync, 1)["checkpoint_shards"][SAMPLER_TABLE]["shards"]["2"] = None
        expected = [f"sampler rank 1: parameter {SAMPLER_TABLE} checkpoint shard 2 has no sampler digest"]
    else:
        del _trainer(sync, 1)["checkpoint_shards"][TRAINER_PARAMETER]["shards"]["3"]
        expected = [f"sampler rank 1: parameter {SAMPLER_TABLE} checkpoint shard 3 has no trainer digest"]
    failed, _ = judge(sync)
    assert _names(failed) == expected


@pytest.mark.parametrize("digests", ["absent", "under a checkpoint name the trainer did not skip"])
def test_an_unmarked_unwritten_parameter_still_fails_as_written_by_no_tensor(digests):
    sync = _sync()
    sampler = _sampler(sync, 0)
    sampler["untouched_parameters"]["visual.w"] = 5
    if digests != "absent":
        table = copy.deepcopy(sampler["checkpoint_shards"][SAMPLER_TABLE])
        sampler["checkpoint_shards"]["visual.w"] = dict(table, checkpoint_name="model.visual.w")
    failed, _ = judge(sync)
    assert _names(failed) == ["sampler rank 0: parameter visual.w written by no tensor"]


@pytest.mark.parametrize("defect", ["parameter not marked skip-sync", "shard reported twice", "manifest field added"])
def test_malformed_trainer_checkpoint_shards_invalidate_the_proof(defect):
    sync = _sync()
    first, second = (_trainer(sync, rank)["checkpoint_shards"] for rank in range(2))
    if defect == "parameter not marked skip-sync":
        first["head.weight"] = copy.deepcopy(first[TRAINER_PARAMETER])
    elif defect == "shard reported twice":
        second[TRAINER_PARAMETER]["shards"]["1"] = copy.deepcopy(first[TRAINER_PARAMETER]["shards"]["1"])
    else:
        first[TRAINER_PARAMETER]["shards"]["0"]["rows"] = 2
    with pytest.raises(RuntimeError, match="invalid skipped frozen checkpoint shards"):
        judge(sync)


def test_checkpoint_shard_rows_follow_the_ceiling_layout_the_sampler_loader_requires():
    assert checkpoint_shard_rows(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert checkpoint_shard_rows(10, 4) == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_checkpoint_shard_manifests_hash_whole_shards_and_leave_partial_ones_unhashed():
    assert checkpoint_shard_manifests(TABLE[3:], 3, ROWS, PARTS) == {"1": None, **_shards(TABLE, 2, 3)}
    assert checkpoint_shard_manifests(TABLE, 0, ROWS, PARTS) == {
        str(index): tensor_manifest(shard) for index, shard in enumerate(TABLE.tensor_split(PARTS))
    }


class _Sampler(torch.nn.Module):
    def load_weights(self, weights):
        raise AssertionError("these tests load no weights")


def _module_at(root, path):
    module = root
    for part in path.split("."):
        if part not in module._modules:
            module.add_module(part, torch.nn.Module())
        module = module._modules[part]
    return module


def _sampler_model(rank, *, table=TABLE, mapper=MAPPER):
    """One TP rank's vocabulary-parallel rows, one padding row, and an unrelated parameter."""
    model = _Sampler()
    if mapper is not None:
        model.hf_to_vllm_mapper = mapper
    owner = _module_at(model, SAMPLER_TABLE.removesuffix(".ngram_embedding.weight"))
    owner.split_ngram_parts = PARTS
    embedding = _module_at(owner, "ngram_embedding")
    embedding.weight = torch.nn.Parameter(torch.cat([_half(table, rank), torch.zeros(1, 3, dtype=table.dtype)]))
    embedding.shard_indices = SimpleNamespace(org_vocab_start_index=rank * HALF, org_vocab_end_index=(rank + 1) * HALF)
    embedding.org_vocab_size = ROWS
    _module_at(model, "visual").w = torch.nn.Parameter(torch.ones(5, dtype=torch.bfloat16))
    return model


def test_sampler_reports_the_checkpoint_shards_of_its_unwritten_table_rows():
    record = ShadowVerifier(_sampler_model(1), max_shadow_bytes=1024).finish()
    assert record["untouched_parameters"] == {SAMPLER_TABLE: (HALF + 1) * 3, "visual.w": 5}
    assert record["checkpoint_shards"] == {
        SAMPLER_TABLE: dict(checkpoint_name=TRAINER_TABLE, shards=_shards(TABLE, 2, 3))
    }


def test_sampler_reports_no_checkpoint_shards_without_a_checkpoint_name_mapper():
    assert "checkpoint_shards" not in ShadowVerifier(_sampler_model(1, mapper=None), max_shadow_bytes=1024).finish()


def _trainer_model(position, parts):
    """One EP rank's frozen half, marked as EPShardedEmbedding marks it."""
    model = torch.nn.Module()
    model.config = SimpleNamespace(text_config=SimpleNamespace())
    if parts is not None:
        model.config.text_config.split_ngram_parts = parts
    _module_at(model, "head").weight = torch.nn.Parameter(HEAD.clone())
    weight = torch.nn.Parameter(_half(TABLE, position).clone(), requires_grad=False)
    weight._dss_skip_weight_sync = True
    weight.allreduce = False
    weight.group_name = "ep_size_2"
    _module_at(model, TRAINER_PARAMETER.removesuffix(".weight")).weight = weight
    return model


def _export_rank(monkeypatch, rank, world, *, parts=PARTS):
    """Export one rank of EP pairs; return its bound inventory and completion."""
    import torch.distributed as dist

    groups_of = {member: [member - member % 2, member - member % 2 + 1] for member in range(world)}
    deepspeed, utils, groups = (
        ModuleType(name) for name in ("deepspeed", "deepspeed.utils", "deepspeed.utils.groups")
    )
    groups._get_expert_parallel_group = lambda name: groups_of[rank]
    deepspeed.utils, utils.groups = utils, groups
    for module in (deepspeed, utils, groups):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: list(group))
    gathered = []
    created, released = [], []

    def create_group(*, backend):
        assert backend == "gloo"
        group = object()
        created.append(group)
        return group

    def release_group(group):
        released.append(group)

    monkeypatch.setattr(dist, "new_group", create_group)
    monkeypatch.setattr(dist, "destroy_process_group", release_group)

    def all_gather_object(records, local, *, group):
        assert group is created[-1]
        records[:] = [copy.deepcopy(local) for _ in range(world)]
        for member, record in enumerate(records):
            record["rank"] = member
            held = tensor_manifest(_half(TABLE, member % 2))
            if "buffers" in record:
                record["parameters"][TRAINER_PARAMETER].update(held, ep_ranks=groups_of[member])
            else:
                record["parameters"][TRAINER_PARAMETER] = held
        gathered.append((local, list(records)))

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    model = _trainer_model(rank % 2, parts)
    token = weight_sync._SEND.set({"trainer_parameters": None} if rank == 0 else None)
    try:
        weight_sync.maybe_bind_broadcast_trainer_inventory(model, is_master=rank == 0, is_zero3=False)
        weight_sync.maybe_complete_broadcast_trainer_inventory(model, is_master=rank == 0)
    finally:
        weight_sync._SEND.reset(token)
        weight_sync._EXPORT.set(None)
    assert len(created) == 2 and released == created
    (_, bound), (completed, _) = gathered
    return bound, completed


@pytest.mark.parametrize(
    "world, rank, indices", [(2, 0, (0, 1)), (2, 1, (2, 3)), (4, 1, (2, 3)), (4, 2, None), (4, 3, None)]
)
def test_only_the_exported_ep_replica_reports_the_checkpoint_shards_of_its_rows(monkeypatch, world, rank, indices):
    _, completed = _export_rank(monkeypatch, rank, world)
    if indices is None:
        assert "checkpoint_shards" not in completed
    else:
        assert completed["checkpoint_shards"] == {
            TRAINER_PARAMETER: dict(checkpoint_name=TRAINER_TABLE, shards=_shards(TABLE, *indices))
        }


def test_trainer_reports_no_checkpoint_shards_without_split_ngram_parts(monkeypatch):
    _, completed = _export_rank(monkeypatch, 0, 2, parts=None)
    assert "checkpoint_shards" not in completed


@pytest.mark.parametrize("sampler_table", ["equal", "row 7 changed"])
def test_trainer_and_sampler_digests_meet_in_the_judge(monkeypatch, sampler_table):
    table = TABLE.clone()
    if sampler_table != "equal":
        table[7, 0] += 1
    sync = _sync()
    for rank in range(2):
        _, completed = _export_rank(monkeypatch, rank, 2)
        _trainer(sync, rank)["checkpoint_shards"] = completed["checkpoint_shards"]
        record = ShadowVerifier(_sampler_model(rank, table=table), max_shadow_bytes=1024).finish()
        _sampler(sync, rank)["checkpoint_shards"] = record["checkpoint_shards"]
    failed, _ = judge(sync)
    differs = f"sampler rank 1: parameter {SAMPLER_TABLE} checkpoint shard 3 differs from the trainer's"
    assert _names(failed) == ([] if sampler_table == "equal" else [differs])
