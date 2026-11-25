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

"""ToyHybridMoE trained by DeepSpeed with expert parallelism, through ``initialize_deepspeed``.

Four GPU ranks: data parallelism 4, expert parallelism 2, so each expert block has an expert-data-parallel group
of 2. One SGD step under ZeRO 0, 1 and 2 must equal one step of the unsharded model on every rank's rows.
DeepSpeed averages gradients over data parallelism, so the reference steps on the mean of the ranks' losses.
"""

import copy
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from toy_hybrid_moe import PLUGIN

from arctic_platform.model_plugin import conformance
from arctic_platform.model_plugin import initialize_deepspeed
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import prepare_model
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import reserve_free_port

from .conftest import TOY_PLUGIN_ROOT

WORLD, EP, LR = 4, 2, 0.5


def _spec():
    base = conformance.load_conformance_spec(TOY_PLUGIN_ROOT / "conformance.toml")
    return base.model_copy(update=dict(fixture=base.fixture | dict(num_experts=8), sp_sizes=[], ep_sizes=[]))


def _rows(config, spec, rank, device):
    input_ids, position_ids = pack_sequences(conformance.fixture_sequences(config, spec, offset=rank, device=device))
    return input_ids, position_ids, next_token_labels(input_ids, position_ids)


def _loss(model, rows, spec):
    """``model`` is the plain model or its DeepSpeed engine; logits come from the plain model's head."""
    input_ids, position_ids, labels = rows
    return conformance._token_loss(
        getattr(model, "module", model), forward(model, input_ids, position_ids), labels, spec
    )


def _worker(rank, port, stage, case):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD)
    )
    import deepspeed
    from deepspeed.utils import groups

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    deepspeed.init_distributed("nccl", timeout=timedelta(seconds=180))
    try:
        spec = _spec()
        config, baseline = conformance.build_fixture(PLUGIN, spec, device)
        model = copy.deepcopy(baseline)
        if case == "unregistered":
            binding = prepare_model(
                model, PLUGIN, config, ep_group=dist.new_group([0, 1]) if rank < 2 else dist.new_group([2, 3])
            )
            with pytest.raises(ValueError, match="DeepSpeed has not registered"):
                initialize_deepspeed(model, binding, config=dict(train_batch_size=WORLD))
            return
        groups._create_expert_and_data_parallel(EP)
        binding = prepare_model(model, PLUGIN, config, ep_group=groups._get_expert_parallel_group(f"ep_size_{EP}"))
        ds_config = dict(
            train_micro_batch_size_per_gpu=1,
            zero_optimization=dict(stage=stage),
            optimizer=dict(type="SGD", params=dict(lr=LR)),
            gradient_clipping=0.0,  # an exact comparison with the unclipped reference step
            zero_allow_untested_optimizer=True,
        )
        if stage == 3:
            with pytest.raises(AssertionError, match="MoE not supported with Stage 3"):
                initialize_deepspeed(model, binding, config=ds_config)
            return
        if case == "bf16":
            engine, *_ = initialize_deepspeed(model, binding, config=ds_config | dict(bf16=dict(enabled=True)))
            router = engine.module.get_parameter("model.layers.1.mlp.router.weight")
            assert router.dtype == torch.float32, "the fp32-marked router was cast"
            assert engine.module.get_parameter("model.layers.1.mlp.experts.w1").dtype == torch.bfloat16
            before = router.detach().clone()
            engine.backward(_loss(engine, _rows(config, spec, rank, device), spec))
            engine.step()
            assert router.dtype == torch.float32, "the router left fp32 in the optimizer step"
            assert not torch.equal(router.detach(), before), "the fp32 router did not train"
            return
        engine, *_ = initialize_deepspeed(model, binding, config=ds_config)
        engine.backward(_loss(engine, _rows(config, spec, rank, device), spec))
        engine.step()

        reference = torch.optim.SGD(baseline.parameters(), lr=LR)
        (sum(_loss(baseline, _rows(config, spec, other, device), spec) for other in range(WORLD)) / WORLD).backward()
        reference.step()

        expected = dict(baseline.named_parameters())
        ep_rank = groups._get_expert_parallel_rank(f"ep_size_{EP}")
        for name, param in engine.module.named_parameters():
            target = expected[name].detach()
            if getattr(param, "allreduce", True) is False:
                per_rank = target.shape[0] // EP
                target = target[ep_rank * per_rank : (ep_rank + 1) * per_rank]
            conformance._assert_close(param.detach(), target, 1e-6, f"ZeRO-{stage} rank {rank} {name} after a step")
    finally:
        dist.destroy_process_group()


def _run(stage, case="step"):
    if torch.cuda.device_count() < WORLD:
        pytest.skip(f"needs {WORLD} GPUs")
    port = reserve_free_port(get_unique_port_number() + 1, span=7)
    mp.spawn(_worker, args=(port, stage, case), nprocs=WORLD, join=True)


@require_deepspeed
@require_torch_multi_gpu
@pytest.mark.integration
@pytest.mark.parametrize("stage", [0, 1, 2])
def test_expert_parallel_step_matches_unsharded(stage):
    _run(stage)


@require_deepspeed
@require_torch_multi_gpu
@pytest.mark.integration
def test_zero3_with_expert_parallelism_is_refused():
    _run(3)


@require_deepspeed
@require_torch_multi_gpu
@pytest.mark.integration
def test_expert_group_deepspeed_did_not_register_is_refused():
    _run(0, "unregistered")


@require_deepspeed
@require_torch_multi_gpu
@pytest.mark.integration
def test_fp32_marks_survive_deepspeeds_bf16_cast():
    _run(1, "bf16")
