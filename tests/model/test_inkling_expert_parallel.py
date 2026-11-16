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

"""Inkling NCCL EP/SP training-step parity with checkpointing and activation offload."""

import copy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from arctic_platform.model import ActivationCheckpointConfig
from arctic_platform.model.implementations.inkling.attention import enable_flex_attention
from arctic_platform.model.implementations.inkling.modeling_inkling import InklingMoE
from arctic_platform.model.implementations.inkling.sequence_parallel import apply_sequence_parallelism
from arctic_platform.model.implementations.moe.deepspeed_integration import apply_ep_with_mesh
from arctic_platform.model.implementations.moe.deepspeed_integration import convert_dtensors_to_local
from arctic_platform.model.implementations.qwen35.model_builder import apply_ac
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from tests.model.test_inkling_training import initialized_prime_model
from tests.model.test_inkling_training import tiny_config


def _worker(rank, port):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4, timeout=timedelta(seconds=300)
    )
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    sp_group = groups[rank // 2]
    mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda", mesh=list(range(4)))
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for lora in (False, True):
                set_seed(947)
                config = tiny_config()
                config.head_dim = config.swa_head_dim = 16
                config.num_attention_heads = config.swa_num_attention_heads = 4
                config.inkling_mlp_token_chunk_size = 32
                reference = initialized_prime_model(config).cuda().to(dtype)
                for module in reference.modules():
                    if isinstance(module, InklingMoE):
                        module.ep_comm_backend = "local"
                    if hasattr(module, "conv1d"):
                        module.float()
                enable_flex_attention(reference, query_chunk_size=32)
                if lora:
                    reference = apply_peft(
                        reference,
                        dict(
                            peft_type="LORA",
                            task_type="CAUSAL_LM",
                            r=2,
                            lora_alpha=4,
                            target_modules=["q_proj", "v_proj"],
                        ),
                    )
                    for name, parameter in reference.named_parameters():
                        if "lora_B" in name:
                            torch.nn.init.normal_(parameter, std=0.01)
                actual = copy.deepcopy(reference)
                base = actual.base_model.model if lora else actual
                for module in base.modules():
                    if isinstance(module, InklingMoE):
                        module.set_ep_comm_backend("nccl")
                        module.deepep_token_chunk_size = 16
                apply_ep_with_mesh(base, SimpleNamespace(ep_comm_backend="nccl"), mesh)
                convert_dtensors_to_local(base)
                apply_sequence_parallelism(base, 2, sp_group)
                apply_ac(
                    base,
                    ActivationCheckpointConfig(
                        router_replay_recompute=False,
                        combine_replay=False,
                        offload_config=dict(enabled=True, tensor_size_threshold=0, keep_last_n=0, use_streams=False),
                    ),
                )
                ids = torch.randint(0, 60, (2, 64), device="cuda")
                labels = torch.cat((ids[:, 1:], ids[:, :1]), dim=1)
                labels[:, -1] = -100
                positions = torch.arange(64, device="cuda")[None].expand_as(ids)
                shard = slice((rank % 2) * 32, (rank % 2 + 1) * 32)
                optimizers = [
                    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
                    for m in (reference, actual)
                ]
                atol = 1e-6 if dtype == torch.float32 else 1e-3
                for step in range(2):
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    expected = reference(input_ids=ids, position_ids=positions).logits
                    count = (labels != -100).sum()
                    expected_loss = (
                        F.cross_entropy(expected.float().flatten(0, 1), labels.flatten(), reduction="sum") / count
                    )
                    expected_loss.backward()
                    output = actual(input_ids=ids[:, shard], position_ids=positions[:, shard]).logits
                    # Two SP groups consume the same batch; sum their half-weighted gradients.
                    loss = F.cross_entropy(
                        output.float().flatten(0, 1), labels[:, shard].flatten(), reduction="sum"
                    ) / (count * 2)
                    loss.backward()
                    dist.all_reduce(loss.detach())
                    torch_assert_close(output, expected[:, shard], atol=atol, rtol=0)
                    torch_assert_close(loss, expected_loss, atol=atol, rtol=0)
                    expected_params = dict(reference.named_parameters())
                    norm_sq = torch.zeros((), device="cuda")
                    expected_norm_sq = torch.zeros_like(norm_sq)
                    for name, parameter in actual.named_parameters():
                        reference_name = name.replace("._checkpoint_wrapped_module", "")
                        baseline = expected_params[reference_name]
                        expert = ".experts." in name
                        if not parameter.requires_grad:
                            continue
                        if not expert:
                            dist.all_reduce(parameter.grad)
                        expected_grad = baseline.grad[rank : rank + 1] if expert else baseline.grad
                        torch_assert_close(parameter.grad.float(), expected_grad.float(), atol=atol, rtol=0, msg=name)
                        norm_sq += parameter.grad.float().square().sum() / (1 if expert else 4)
                        expected_norm_sq += baseline.grad.float().square().sum()
                    dist.all_reduce(norm_sq)
                    torch_assert_close(norm_sq.sqrt(), expected_norm_sq.sqrt(), atol=atol, rtol=0)
                    for optimizer in optimizers:
                        optimizer.step()
                    for name, parameter in actual.named_parameters():
                        baseline = expected_params[name.replace("._checkpoint_wrapped_module", "")]
                        expected_value = baseline[rank : rank + 1] if ".experts." in name else baseline
                        torch_assert_close(parameter, expected_value, atol=atol, rtol=0, msg=name)
                        if parameter.requires_grad:
                            actual_state = optimizers[1].state[parameter]
                            expected_state = optimizers[0].state[baseline]
                            torch_assert_close(actual_state["step"], expected_state["step"], atol=0, rtol=0)
                            for key in ("exp_avg", "exp_avg_sq"):
                                expected_moment = expected_state[key]
                                if ".experts." in name:
                                    expected_moment = expected_moment[rank : rank + 1]
                                torch_assert_close(actual_state[key], expected_moment, atol=atol, rtol=0, msg=name)
                    if rank == 0:
                        print(f"EP4/SP2 passed dtype={dtype} lora={lora} step={step}", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(900)
class TestInklingNcclExpertParallel(TestCasePlus):
    @require_torch_gpu
    def test_ep4_sp2_full_and_lora_updates(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("requires four GPUs")
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        torch.multiprocessing.spawn(_worker, args=(port,), nprocs=4, join=True)
