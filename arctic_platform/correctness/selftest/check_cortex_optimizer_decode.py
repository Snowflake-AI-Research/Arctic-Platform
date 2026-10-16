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

"""Synthetic CPU checkpoint decoding with the public resumable ZeRO2 schema."""

import json
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address

from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.harness import cortex_optimizer_decode as decoder

OPTIMIZER = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0}
LR = 0.01


def _pair(tmp_path, *, world_size=2, before_step=0, prefix="bf16_", lazy_baseline=True):
    shapes = [
        OrderedDict([("layer.weight", torch.Size([2, 3])), ("layer.bias", torch.Size([3]))]),
        OrderedDict([("layer.norm.weight", torch.Size([5]))]),
    ]
    config = {
        "zero_optimization": {"stage": 2},
        "bf16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {**OPTIMIZER, "lr": 2e-5}},
    }
    baseline = {}
    expected = {}
    for group_index, group in enumerate(shapes):
        for name, shape in group.items():
            before = torch.full(shape, 1.0 + group_index, dtype=torch.float32)
            # The update is visible in FP32 masters, but both BF16 module snapshots round identically.
            after = before + 0.0001
            baseline[name] = before
            expected[name] = {
                "parameter_update": after - before,
                "exp_avg": torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + 1,
                "exp_avg_sq": torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + 2,
            }
    before_root, after_root = tmp_path / "before", tmp_path / "after"
    for root, step, post in ((before_root, before_step, False), (after_root, before_step + 1, True)):
        tag = root / f"global_step{step}"
        tag.mkdir(parents=True)
        (root / "metadata.json").write_text("{}")
        (root / "export_receipt.json").write_text("{}")
        module = {name: (value + (0.0001 if post else 0)).to(torch.bfloat16) for name, value in baseline.items()}
        torch.save(
            {
                "module": module,
                "param_shapes": shapes,
                "global_steps": step,
                "dp_world_size": world_size,
                "mp_world_size": 1,
                "ds_config": config,
                "shared_params": {},
            },
            tag / "mp_rank_00_model_states.pt",
        )
        for rank in range(world_size):
            masters, mappings, paddings, states, groups = [], [], [], {}, []
            for group_index, group in enumerate(shapes):
                total = sum(shape.numel() for shape in group.values())
                length = math.ceil(total / (2 * world_size)) * 2
                start, end = rank * length, (rank + 1) * length
                valid = max(0, min(length, total - start))
                values = [baseline[name].flatten() + (0.0001 if post else 0) for name in group]
                vector = torch.cat(values)
                masters.append(vector[start : start + valid].clone())
                paddings.append(length - valid)
                mapping = OrderedDict()
                offset = 0
                for name, shape in group.items():
                    fragment_start, fragment_end = max(offset, start), min(offset + shape.numel(), end)
                    if fragment_start < fragment_end:
                        mapping[name] = fragment_address(
                            numel=fragment_end - fragment_start, start=fragment_start - start
                        )
                    offset += shape.numel()
                mappings.append(mapping)
                param_id = 17 + group_index * 13
                groups.append(
                    {
                        "params": [param_id],
                        "betas": (0.9, 0.999),
                        "lr": LR if post else 0.001,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "bias_correction": True,
                        "step": 0,
                    }
                )
                if post or before_step or not lazy_baseline:
                    state = {"step": torch.tensor(float(step))}
                    for key in ("exp_avg", "exp_avg_sq"):
                        full = torch.cat([expected[name][key].flatten() for name in group])
                        partition = torch.zeros(length, dtype=torch.float32)
                        partition[:valid].copy_(full[start : start + valid] if post else torch.zeros(valid))
                        state[key] = partition
                    states[param_id] = state
            torch.save(
                {
                    "ds_config": config,
                    "ds_version": "synthetic",
                    "optimizer_state_dict": {
                        "zero_stage": ZeroStageEnum.gradients,
                        "base_optimizer_state": {"state": states, "param_groups": groups},
                        "single_partition_of_fp32_groups": masters,
                        "param_slice_mappings": mappings,
                        "partition_count": [world_size] * len(shapes),
                        "group_paddings": paddings,
                        "param_alignment_paddings": [[0] * len(group) for group in shapes],
                        "loss_scaler": LossScaler(1.0),
                        "dynamic_loss_scale": False,
                        "overflow": False,
                        "clip_grad": 1.0,
                        "ds_version": "synthetic",
                    },
                },
                tag / f"{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
            )
    return SimpleNamespace(before=before_root, after=after_root, expected=expected, baseline=baseline)


def _model(root):
    return next(root.rglob("*_model_states.pt"))


def _optim(root, rank=0):
    return next(root.rglob(f"*zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"))


def _rewrite(path, edit):
    state = decoder._load(path)
    edit(state)
    # Do not truncate the backing file of mmap-loaded fixture tensors.
    replacement = path.with_suffix(".replacement")
    torch.save(state, replacement)
    replacement.replace(path)


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_decode_requires_data_fast_before_checkpoint_load(tmp_path, monkeypatch, unsafe):
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    monkeypatch.setattr(
        decoder, "_snapshot", lambda *args, **kwargs: pytest.fail("unsafe output reached checkpoint load")
    )
    with pytest.raises(ValueError, match="/data-fast"):
        decoder.decode_optimizer_capture(tmp_path / "before", tmp_path / "after", destination)
    assert not (tmp_path / "before").exists()
    assert not (tmp_path / "after").exists()


@pytest.mark.parametrize("prefix", ["", "bf16_", "fp16_"])
@pytest.mark.parametrize("world_size", [1, 2, 4, 12])
def test_decode_named_fp32_update_moments_and_padding(tmp_path, prefix, world_size):
    pair = _pair(tmp_path, world_size=world_size, prefix=prefix)
    manifest = decoder.decode_optimizer_capture(
        pair.before, pair.after, tmp_path / "out", expected_optimizer_config=OPTIMIZER, expected_learning_rate=LR
    )
    metadata = json.loads(manifest.read_text())
    assert metadata["step"] == 1
    assert metadata["before"]["engine_global_step"] == 0
    assert metadata["after"]["engine_global_step"] == 1
    assert [item["name"] for item in metadata["parameters"]] == list(pair.expected)
    for entry in metadata["parameters"]:
        tensor = torch.load(manifest.parent / entry["file"], weights_only=True)
        assert set(tensor) == {"parameter_update", "exp_avg", "exp_avg_sq"}
        for key, expected in pair.expected[entry["name"]].items():
            assert tensor[key].dtype == torch.float32
            assert tuple(entry["shape"]) == tuple(expected.shape)
            assert torch.equal(tensor[key], expected)
        assert bool(torch.count_nonzero(tensor["parameter_update"]))
    before_module = decoder._load(_model(pair.before))["module"]
    after_module = decoder._load(_model(pair.after))["module"]
    assert all(torch.equal(before_module[name], after_module[name]) for name in before_module)
    comparison = compare_optimizer_artifacts(manifest, manifest, optimizer_config=OPTIMIZER, learning_rate=LR)
    assert len(comparison.deltas) == 3 * len(pair.expected)
    assert all(item.delta_norm == 0 for item in comparison.deltas if item.state in ("exp_avg", "exp_avg_sq"))
    assert not comparison.only_dss and not comparison.only_reference


def test_decode_uses_actual_state_step_and_group_lr_not_legacy_or_config_defaults(tmp_path):
    pair = _pair(tmp_path)
    state = decoder._load(_optim(pair.after))["optimizer_state_dict"]
    assert state["base_optimizer_state"]["param_groups"][0]["step"] == 0
    assert state["base_optimizer_state"]["state"][17]["step"] == 1
    assert decoder._load(_optim(pair.after))["ds_config"]["optimizer"]["params"]["lr"] != LR
    assert decoder.decode_optimizer_capture(
        pair.before, pair.after, tmp_path / "out", expected_optimizer_config=OPTIMIZER, expected_learning_rate=LR
    ).exists()


def test_decode_preserves_nonzero_actual_steps_without_relabeling_as_step_one(tmp_path):
    pair = _pair(tmp_path, before_step=10)
    manifest = decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert json.loads(manifest.read_text())["step"] == 11
    with pytest.raises(ValueError, match="expected=1"):
        compare_optimizer_artifacts(manifest, manifest, optimizer_config=OPTIMIZER, learning_rate=LR)


def test_decode_accepts_initialized_step_zero_and_tag_directories(tmp_path):
    pair = _pair(tmp_path, lazy_baseline=False)
    assert decoder.decode_optimizer_capture(
        _model(pair.before).parent, _model(pair.after).parent, tmp_path / "out"
    ).exists()


def test_decode_mmap_restricted_loads_and_does_not_mutate_source_tensors(tmp_path, monkeypatch):
    pair = _pair(tmp_path)
    before_bytes = {path: path.read_bytes() for root in (pair.before, pair.after) for path in root.rglob("*.pt")}
    original_load = torch.load
    calls = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert calls
    assert all(
        call["weights_only"] is True and call["mmap"] is True and call["map_location"] == "cpu" for call in calls
    )
    assert all(path.read_bytes() == data for path, data in before_bytes.items())


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("zero_stage", 1, "ZeRO2"),
        ("zero_stage", 3, "ZeRO2"),
        ("zero_stage", None, "ZeRO stage"),
        ("partition_count", [2], "groups"),
        ("partition_count", [1, 2], "partition_count"),
        ("partition_count", [3, 3], "partition_count"),
        ("group_paddings", [1, 0], "partition padding"),
        ("group_paddings", [0], "groups"),
        ("group_paddings", [-1, 0], "padding"),
        ("param_alignment_paddings", [[0, 1], [0]], "alignment padding"),
        ("param_alignment_paddings", [[0], [0]], "alignment padding"),
        ("overflow", True, "overflow"),
        ("base_optimizer_state", [], "nonelastic"),
        ("single_partition_of_fp32_groups", [], "groups"),
        ("param_slice_mappings", [], "groups"),
    ],
)
def test_decode_rejects_invalid_optimizer_schema(tmp_path, field, value, error):
    pair = _pair(tmp_path)
    _rewrite(_optim(pair.after), lambda state: state["optimizer_state_dict"].__setitem__(field, value))
    with pytest.raises(ValueError, match=error):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("key", ["fp32_master", "exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_decode_rejects_non_fp32_tensors(tmp_path, side, key, dtype):
    pair = _pair(tmp_path, lazy_baseline=False)

    def edit(state):
        opt = state["optimizer_state_dict"]
        if key == "fp32_master":
            opt["single_partition_of_fp32_groups"][0] = opt["single_partition_of_fp32_groups"][0].to(dtype)
        else:
            opt["base_optimizer_state"]["state"][17][key] = opt["base_optimizer_state"]["state"][17][key].to(dtype)

    _rewrite(_optim(getattr(pair, side)), edit)
    with pytest.raises(ValueError, match="FP32"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("key", ["exp_avg", "exp_avg_sq"])
@pytest.mark.parametrize("mutation", ["missing", "short", "nonvector"])
def test_decode_rejects_missing_or_malformed_moments(tmp_path, key, mutation):
    pair = _pair(tmp_path)

    def edit(state):
        moments = state["optimizer_state_dict"]["base_optimizer_state"]["state"][17]
        if mutation == "missing":
            del moments[key]
        elif mutation == "short":
            moments[key] = moments[key][:-1]
        else:
            moments[key] = moments[key].reshape(2, 3)

    _rewrite(_optim(pair.after), edit)
    with pytest.raises(ValueError, match="FP32"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("step", [-1, 0, 2, 1.5, float("nan"), True])
def test_decode_rejects_inconsistent_post_optimizer_step(tmp_path, step):
    pair = _pair(tmp_path)
    _rewrite(
        _optim(pair.after, 1),
        lambda state: state["optimizer_state_dict"]["base_optimizer_state"]["state"][17].__setitem__("step", step),
    )
    with pytest.raises(ValueError, match="step"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


def test_decode_requires_exactly_one_engine_step(tmp_path):
    pair = _pair(tmp_path)
    # A valid independently stamped snapshot is insufficient without exactly one step delta.
    for path in pair.after.rglob("*.pt"):
        if path == _model(pair.after):
            _rewrite(path, lambda state: state.__setitem__("global_steps", 3))
        else:
            _rewrite(
                path,
                lambda state: [
                    item.__setitem__("step", 3)
                    for item in state["optimizer_state_dict"]["base_optimizer_state"]["state"].values()
                ],
            )
    _model(pair.after).parent.rename(pair.after / "global_step3")
    with pytest.raises(ValueError, match="exactly one step"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["remove", "duplicate", "foreign", "symlink", "second_tag", "second_model"])
def test_decode_rejects_ambiguous_or_incomplete_checkpoint_files(tmp_path, mutation):
    pair = _pair(tmp_path)
    path = _optim(pair.after, 1)
    if mutation == "remove":
        path.unlink()
    elif mutation == "duplicate":
        (path.parent / path.name.replace("bf16_", "")).write_bytes(path.read_bytes())
    elif mutation == "foreign":
        (path.parent / "expert_0_model_states.pt").write_bytes(path.read_bytes())
    elif mutation == "symlink":
        original = tmp_path / "saved.pt"
        path.rename(original)
        path.symlink_to(original)
    elif mutation == "second_tag":
        another = pair.after / "global_step2"
        another.mkdir()
        (another / path.name).write_bytes(path.read_bytes())
    else:
        model = _model(pair.after)
        (model.parent / "mp_rank_01_model_states.pt").write_bytes(model.read_bytes())
    with pytest.raises(ValueError):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["shape", "order", "duplicate", "empty", "module_shape"])
def test_decode_rejects_shape_or_order_drift(tmp_path, mutation):
    pair = _pair(tmp_path)

    def edit(state):
        shapes = state["param_shapes"]
        if mutation == "shape":
            shapes[0]["layer.weight"] = torch.Size([3, 2])
            state["module"]["layer.weight"] = state["module"]["layer.weight"].reshape(3, 2)
        elif mutation == "order":
            shapes[0] = OrderedDict(reversed(list(shapes[0].items())))
        elif mutation == "duplicate":
            shapes[1]["layer.weight"] = torch.Size([2, 3])
        elif mutation == "empty":
            shapes[0] = {}
        else:
            state["module"]["layer.weight"] = torch.empty(2, 2)

    _rewrite(_model(pair.after), edit)
    with pytest.raises(ValueError, match="shape|order|parameter|mapping"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize("mutation", ["missing", "extra", "offset", "length"])
def test_decode_rejects_bad_named_fragment_mappings(tmp_path, mutation):
    pair = _pair(tmp_path)

    def edit(state):
        mapping = state["optimizer_state_dict"]["param_slice_mappings"][0]
        if mutation == "missing":
            mapping.pop("layer.weight")
        elif mutation == "extra":
            mapping["unknown"] = fragment_address(start=0, numel=1)
        elif mutation == "offset":
            mapping["layer.weight"].start += 1
        else:
            mapping["layer.weight"].numel -= 1

    _rewrite(_optim(pair.after), edit)
    with pytest.raises(ValueError, match="mapping"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("ep_config", "expert parallel"),
        ("ep_group", "EP"),
        ("tp", "parallel"),
        ("amsgrad", "AMSGrad"),
        ("missing_group", "groups"),
        ("extra_state", "state groups"),
        ("missing_post_state", "step"),
        ("rank_lr", "rank config"),
        ("rank_betas", "rank config"),
        ("lr", "learning rate"),
        ("betas", "config"),
        ("bias_correction", "bias_correction"),
    ],
)
def test_decode_rejects_unsupported_or_inconsistent_group_configuration(tmp_path, mutation, error):
    pair = _pair(tmp_path)

    def edit(state):
        config = state["ds_config"]
        opt = state["optimizer_state_dict"]
        base = opt["base_optimizer_state"]
        group = base["param_groups"][0]
        if mutation == "ep_config":
            config["expert_parallel"] = {"enabled": True, "autoep_size": 2}
        elif mutation == "ep_group":
            group["moe"] = True
        elif mutation == "tp":
            config["tensor_parallel"] = {"tp_size": 2}
        elif mutation == "amsgrad":
            group["amsgrad"] = True
        elif mutation == "missing_group":
            base["param_groups"].pop()
        elif mutation == "extra_state":
            base["state"][99] = base["state"][17]
        elif mutation == "missing_post_state":
            base["state"].pop(17)
        elif mutation in ("lr", "rank_lr"):
            group["lr"] = 0.002
        elif mutation in ("betas", "rank_betas"):
            group["betas"] = (0.8, 0.999)
        else:
            group["bias_correction"] = False

    _rewrite(_optim(pair.after, int(mutation.startswith("rank_"))), edit)
    kwargs = (
        {}
        if mutation.startswith("rank_")
        else {
            "expected_optimizer_config": OPTIMIZER,
            "expected_learning_rate": LR,
        }
    )
    with pytest.raises(ValueError, match=error):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out", **kwargs)


@pytest.mark.parametrize("key", ["fp32_master", "exp_avg", "exp_avg_sq"])
def test_decode_nonfinite_parameter_rolls_back_only_new_output(tmp_path, key):
    pair = _pair(tmp_path)

    def edit(state):
        opt = state["optimizer_state_dict"]
        vector = (
            opt["single_partition_of_fp32_groups"][1]
            if key == "fp32_master"
            else opt["base_optimizer_state"]["state"][30][key]
        )
        vector[0] = float("nan")

    _rewrite(_optim(pair.after), edit)
    with pytest.raises(ValueError, match="nonfinite"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert _model(pair.before).exists() and _model(pair.after).exists()


def test_decode_preserves_preexisting_output(tmp_path):
    pair = _pair(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel"
    sentinel.write_text("owned elsewhere")
    with pytest.raises(ValueError, match="already exist"):
        decoder.decode_optimizer_capture(pair.before, pair.after, out)
    assert sentinel.read_text() == "owned elsewhere"


def test_decode_rejects_output_nested_in_input(tmp_path):
    pair = _pair(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        decoder.decode_optimizer_capture(pair.before, pair.after, pair.before / "out")


def test_decode_does_not_allow_arbitrary_pickled_checkpoint_classes(tmp_path):
    pair = _pair(tmp_path)
    _rewrite(_optim(pair.after), lambda state: state.__setitem__("unknown", SimpleNamespace(value=1)))
    with pytest.raises(ValueError, match="unsupported checkpoint globals"):
        decoder.decode_optimizer_capture(pair.before, pair.after, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("source", ["model", "rank0", "rank1"])
@pytest.mark.parametrize("optimizer_type", ["Adam", "SGD", None, "FusedAdam"])
def test_decode_rejects_non_adamw_or_missing_model_and_shard_optimizer_types(tmp_path, side, source, optimizer_type):
    pair = _pair(tmp_path)
    root = getattr(pair, side)
    path = _model(root) if source == "model" else _optim(root, int(source[-1]))
    _rewrite(path, lambda state: state["ds_config"]["optimizer"].__setitem__("type", optimizer_type))
    with pytest.raises(ValueError, match="optimizer type AdamW"):
        decoder.decode_optimizer_capture(
            pair.before, pair.after, tmp_path / "out", expected_optimizer_config=OPTIMIZER
        )
    assert not (tmp_path / "out").exists()


def test_decode_rejects_unmatched_expected_optimizer_type(tmp_path):
    pair = _pair(tmp_path)
    with pytest.raises(ValueError, match="expected optimizer type"):
        decoder.decode_optimizer_capture(
            pair.before, pair.after, tmp_path / "out", expected_optimizer_config={**OPTIMIZER, "name": "Adam"}
        )
