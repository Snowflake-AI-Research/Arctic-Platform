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

"""ToyHybridMoE conformance on GPUs: every single-process tier on CUDA, and SP and EP over NCCL, including layouts
whose SP and EP degrees differ.

The CPU tier runs SP and EP over one gloo group, so it only covers ``sp == ep``. Here each case lays ranks out as
production does: contiguous sequence-parallel groups of ``sp`` ranks (one data-parallel replica each, with its own
rows), contiguous expert-parallel groups of ``ep`` ranks, and strided expert-data-parallel groups that sum an expert
block's gradient. Every case compares, against the unsharded model run on every replica's rows:

- each rank's logits shard, and the logits with the reference's routing replayed;
- every gradient: replicated weights summed over the world, expert blocks summed over the expert-data-parallel group;
- under EP, the exported Hugging Face weight stream.
"""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from toy_hybrid_moe import PLUGIN

from arctic_platform.model_plugin import conformance
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import prepare_model
from arctic_platform.model_plugin.parallel import shard_sequence
from arctic_platform.model_plugin.recompute import apply_activation_checkpointing
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import reserve_free_port

from .conftest import TOY_PLUGIN_ROOT

TARGET = "toy_hybrid_moe:PLUGIN"

# Every head count and the expert count divide by 8; 32 tokens give each of 8 SP ranks 4 tokens, at least the
# convolution halo of 2, and rows of 13, 9 and 10 tokens cross shard boundaries at every SP degree.
FIXTURE = dict(
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=8,
    recurrence_heads=8,
    recurrence_head_dim=4,
    conv_channels=32,
    num_experts=8,
    num_experts_per_tok=2,
)
SEQUENCE_LENGTHS = [13, 9, 10]

# (world, sp, ep): pure SP, pure EP, equal SP and EP, EP spanning SP groups (ep > sp), and SP spanning EP groups.
CASES = [
    (2, 2, 1),
    (2, 1, 2),
    (2, 2, 2),
    (4, 4, 1),
    (4, 1, 4),
    (4, 2, 2),
    (4, 2, 4),
    (4, 4, 2),
    (8, 8, 1),
    (8, 1, 8),
    (8, 2, 4),
    (8, 4, 2),
    (8, 2, 8),
    (8, 8, 8),
]

TOLERANCE = {torch.float32: 1e-5, torch.bfloat16: 5e-2}


def _spec(sp: int):
    base = conformance.load_conformance_spec(TOY_PLUGIN_ROOT / "conformance.toml")
    return base.model_copy(
        update=dict(fixture=base.fixture | FIXTURE, sequence_lengths=SEQUENCE_LENGTHS, sp_sizes=[sp], ep_sizes=[])
    )


def _contiguous_groups(world: int, size: int, rank: int):
    """Every rank creates every group, in the same order; returns this rank's group (None when ``size`` is 1)."""
    mine = None
    for start in range(0, world, size):
        ranks = list(range(start, start + size))
        group = dist.new_group(ranks)
        if rank in ranks:
            mine = group
    return mine if size > 1 else None


def _strided_groups(world: int, stride: int, rank: int):
    mine = None
    for offset in range(stride):
        ranks = list(range(offset, world, stride))
        group = dist.new_group(ranks)
        if rank in ranks:
            mine = group
    return mine


def _packed(config, spec, replica: int, device):
    input_ids, position_ids = pack_sequences(
        conformance.fixture_sequences(config, spec, offset=replica, device=device)
    )
    return input_ids, position_ids, next_token_labels(input_ids, position_ids)


class _Mismatches(list):
    """Collect mismatches instead of raising at the first, so no rank stops while others wait in a collective."""

    def close(self, actual, expected, atol, what):
        try:
            conformance._assert_close(actual, expected, atol, what)
        except AssertionError as error:
            self.append(str(error))


def _worker(rank, world, sp, ep, port, dtype, checkpointing):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world,
        rank=rank,
        timeout=timedelta(seconds=180),
        device_id=device,
    )
    try:
        _check(rank, world, sp, ep, device, dtype, checkpointing)
    finally:
        dist.destroy_process_group()


def _check(rank, world, sp, ep, device, dtype, checkpointing):
    atol = TOLERANCE[dtype]
    mismatches = _Mismatches()
    _close = mismatches.close
    spec = _spec(sp)
    sp_group = _contiguous_groups(world, sp, rank)
    ep_group = _contiguous_groups(world, ep, rank)
    edp_group = _strided_groups(world, ep, rank)
    replicas, replica = world // sp, rank // sp

    config, baseline = conformance.build_fixture(PLUGIN, spec, device)
    baseline = baseline.to(dtype)
    model = copy.deepcopy(baseline)
    prepare_model(model, PLUGIN, config, sp_group=sp_group, ep_group=ep_group)
    if checkpointing:
        apply_activation_checkpointing(model)

    # The unsharded model on every replica's rows: its gradient is the whole step's gradient.
    reference_logits, reference_routing = None, None
    for other in range(replicas):
        input_ids, position_ids, labels = _packed(config, spec, other, device)
        logits, routing = forward(baseline, input_ids, position_ids, return_routing=True)
        conformance._token_loss(logits, labels).backward()
        if other == replica:
            reference_logits, reference_routing = logits.detach(), routing

    input_ids, position_ids, labels = _packed(config, spec, replica, device)
    logits = forward(model, input_ids, position_ids, sp_group=sp_group)
    conformance._token_loss(logits, shard_sequence(labels, sp_group)).backward()
    where = f"world={world} sp={sp} ep={ep} {dtype} ac={checkpointing} rank {rank}"
    _close(logits.detach(), shard_sequence(reference_logits, sp_group), atol, f"{where}: logits")

    with torch.no_grad():
        replayed = forward(model, input_ids, position_ids, sp_group=sp_group, routed_experts=reference_routing)
    _close(replayed, shard_sequence(reference_logits, sp_group), atol, f"{where}: replayed routing")

    experts = {
        f"{path}.{name}"
        for path, module in model.named_modules()
        if PLUGIN.expert_layer.is_instance(module)
        for name in PLUGIN.expert_layer.expert_parameters(module)
    }
    expected = conformance._grads(baseline)
    for name, grad in conformance._grads(model).items():
        reference = expected[name]
        if name in experts and ep > 1:
            dist.all_reduce(grad, group=edp_group)
            per_rank = reference.shape[0] // ep
            ep_rank = dist.get_rank(ep_group)
            reference = reference[ep_rank * per_rank : (ep_rank + 1) * per_rank]
        else:
            dist.all_reduce(grad)
        _close(grad, reference, atol * replicas, f"{where}: grad {name}")

    if ep > 1:
        exported = dict(iter_hf_weights(model, PLUGIN, for_sampler=True, ep_group=ep_group))
        unsharded = dict(iter_hf_weights(baseline, PLUGIN, for_sampler=True))
        if set(exported) != set(unsharded):
            mismatches.append(f"{where}: export names differ: {sorted(set(exported) ^ set(unsharded))[:5]}")
        for name in set(exported) & set(unsharded):
            _close(exported[name], unsharded[name], 0.0, f"{where}: exported {name}")
    if len(mismatches) > 0:
        raise AssertionError(f"{len(mismatches)} mismatches, first: {mismatches[0]}")


def _run(world, sp, ep, dtype=torch.float32, checkpointing=False):
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    port = reserve_free_port(get_unique_port_number() + 1, span=7)
    mp.spawn(_worker, args=(world, sp, ep, port, dtype, checkpointing), nprocs=world, join=True)


SINGLE_PROCESS_TIERS = [
    conformance.check_reference_parity,
    conformance.check_packing,
    conformance.check_activation_checkpointing,
    conformance.check_router_replay,
    conformance.check_checkpoint_and_sync,
    conformance.check_rl_step,
]


@require_torch_gpu
@pytest.mark.integration
@pytest.mark.parametrize("check", SINGLE_PROCESS_TIERS, ids=lambda check: check.__name__)
def test_single_process_tier_on_cuda(check):
    check(PLUGIN, conformance.load_conformance_spec(TOY_PLUGIN_ROOT / "conformance.toml"), "cuda")


@require_torch_multi_gpu
@pytest.mark.integration
@pytest.mark.parametrize("mode", ["sp", "ep", "sp+ep"])
def test_parallel_tier_over_nccl(mode):
    conformance.check_parallel(
        TARGET, conformance.load_conformance_spec(TOY_PLUGIN_ROOT / "conformance.toml"), mode, 2, "cuda"
    )


@require_torch_multi_gpu
@pytest.mark.integration
@pytest.mark.parametrize("world, sp, ep", CASES, ids=[f"w{w}-sp{s}-ep{e}" for w, s, e in CASES])
def test_sp_ep_layouts_match_unsharded(world, sp, ep):
    _run(world, sp, ep)


@require_torch_multi_gpu
@pytest.mark.integration
@pytest.mark.parametrize(
    "world, sp, ep", [(4, 2, 2), (8, 2, 4), (8, 4, 2)], ids=["w4-sp2-ep2", "w8-sp2-ep4", "w8-sp4-ep2"]
)
def test_sp_ep_layouts_with_activation_checkpointing(world, sp, ep):
    _run(world, sp, ep, checkpointing=True)


@require_torch_multi_gpu
@pytest.mark.integration
@pytest.mark.parametrize("world, sp, ep", [(4, 2, 2), (8, 2, 4)], ids=["w4-sp2-ep2", "w8-sp2-ep4"])
def test_sp_ep_layouts_bf16(world, sp, ep):
    _run(world, sp, ep, dtype=torch.bfloat16, checkpointing=True)
