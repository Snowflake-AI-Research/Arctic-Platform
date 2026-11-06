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

"""Real two-rank CPU capture/export/reload controls, not large-model or NCCL qualification."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file
from safetensors.torch import save_file
from torch import nn

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_equal


class _CheckpointFixture(nn.Module):
    """Small live EP tensor plus inactive sidecars; no training kernels are exercised."""

    def __init__(self, family, rank):
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                model_type="qwen3_5_moe_text" if family == "qwen" else "glm5_next_text",
                mtp_num_hidden_layers=1,
                num_hidden_layers=1,
                num_nextn_predict_layers=1,
            )
        )
        if family == "glm":
            from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

            self.config.text_config = Glm5NextTextConfig(
                vocab_size=32,
                pad_token_id=0,
                eos_token_id=1,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=1,
                num_nextn_predict_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                n_shared_experts=1,
                n_routed_experts=2,
                num_experts_per_tok=1,
                q_lora_rank=8,
                kv_lora_rank=4,
                qk_rope_head_dim=0,
                qk_nope_head_dim=4,
                v_head_dim=4,
                index_n_heads=2,
                index_head_dim=4,
                index_topk=4,
                index_kpool=2,
                linear_num_heads=2,
                linear_head_dim=4,
                hc_mult=2,
                layer_types=["deepseek_sparse_attention"],
                mlp_layer_types=["sparse"],
            )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        parameter = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) + rank * 10)
        parameter.allreduce = False
        parameter.group_name = "ep_size_2"
        layer.mlp.experts.register_parameter("w2", parameter)

    @staticmethod
    def convert_layer_to_hf(state, index):
        return state


def _sidecar(family, model):
    if family == "qwen":
        from arctic_platform.model.implementations.qwen35.mtp_weights import _configured_names

        names = sorted(_configured_names(model))
    else:
        from arctic_platform.model.implementations.glm53 import auxiliary_weights

        names = sorted(auxiliary_weights._required_layer_names(model.config, "model.language_model.layers.1.", set()))
    return {
        name: (
            (torch.arange(6, dtype=torch.float32).reshape(2, 3) + index).to(
                (torch.float32, torch.bfloat16, torch.float16)[index % 3]
            )
        )
        for index, name in enumerate(names)
    }


def _checkpoint_worker(rank, rendezvous, directory, family, fault):
    import deepspeed.comm as ds_dist
    import deepspeed.utils.groups as groups

    from arctic_platform.model import iter_model_weights
    from arctic_platform.model.implementations.glm53 import auxiliary_weights
    from arctic_platform.model.implementations.glm53 import deepspeed_integration as glm
    from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen
    from arctic_platform.model.implementations.qwen35 import mtp_weights
    from arctic_platform.model.weight_export import WeightExportContract
    from arctic_platform.model.weight_export import register_weight_export
    from arctic_platform.model.weight_export import save_exported_state_dict

    ds_dist.init_distributed(
        dist_backend="gloo",
        auto_mpi_discovery=False,
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        groups._EXPERT_PARALLEL_GROUP["ep_size_2"] = dist.group.WORLD
        model = _CheckpointFixture(family, rank)
        payload = _sidecar(family, model)
        source = Path(directory) / "source"
        capture = mtp_weights.capture_mtp_weights if family == "qwen" else auxiliary_weights.capture_auxiliary_weights
        builder = (
            qwen._build_iter_full_hf_weights_with_mtp
            if family == "qwen"
            else glm._build_iter_full_hf_weights_with_auxiliary
        )
        attribute = "_ap_preserved_mtp_weights" if family == "qwen" else auxiliary_weights.STATE_ATTRIBUTE
        missing = "mtp.fc.weight" if family == "qwen" else "model.language_model.layers.1.self_attn.o_proj.weight"
        if rank == 0:
            source.mkdir()
            if fault == "capture":
                payload.pop(missing)
            save_file(payload, source / "model.safetensors")
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(payload, "model.safetensors")})
            )
        dist.barrier()
        # Capture records failures on both ranks without preventing ordinary synchronization.
        # Peers intentionally have no usable source path: only rank zero may access sidecars.
        capture(model, source if rank == 0 else Path(directory) / "absent-on-peer")
        register_weight_export(model, WeightExportContract(hf=builder))
        synced = dict(iter_model_weights(model, "hf", is_master=rank == 0, is_zero3=False))
        assert not set(synced) & set(payload)
        live_name = "model.layers.0.mlp.experts.w2"
        expected_live = torch.cat([torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset for offset in (0, 10)])
        assert set(synced) == ({live_name} if rank == 0 else set())
        if rank == 0:
            torch_assert_equal(synced[live_name], expected_live)
        if rank == 0 and fault == "export":
            model.__dict__.pop(attribute)
        published = {}
        try:
            for name, tensor in iter_model_weights(
                model, "hf", is_master=rank == 0, is_zero3=False, include_skipped=True
            ):
                published[name] = tensor
        except RuntimeError as error:
            assert fault in ("capture", "export"), str(error)
            assert published == {}, "invalid sidecar published live tensors before validation"
            message = str(error)
            if fault == "capture":
                assert missing in message
            else:
                assert "requires captured original" in message
        else:
            assert fault == "none", "rank returned success for a failed full export"
            message = None
            if rank == 0:
                expected = {live_name: expected_live, **payload}
                assert set(published) == set(expected)
                for name, wanted in expected.items():
                    torch_assert_equal(published[name], wanted)
                destination = Path(directory) / "exported"
                save_exported_state_dict(published, str(destination))
                reloaded = {}
                for shard in destination.glob("*.safetensors"):
                    reloaded.update(load_file(shard))
                assert set(reloaded) == set(expected)
                for name, wanted in expected.items():
                    assert reloaded[name].dtype == wanted.dtype
                    torch_assert_equal(reloaded[name], wanted)
            else:
                assert published == {}
        # This real collective proves both ranks reach the same terminal status without a stuck peer.
        messages = [None, None]
        dist.all_gather_object(messages, message)
        assert messages[0] == messages[1], messages
        assert (messages[0] is None) == (fault == "none")
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps({"status": message, "fault": fault}))
    finally:
        dist.destroy_process_group()


class TestPredictionCheckpointDistributed(TestCasePlus):
    def _run(self, family, fault):
        directory = self.get_auto_remove_tmp_dir()
        mp.spawn(
            _checkpoint_worker,
            args=(f"file://{directory}/rendezvous", str(directory), family, fault),
            nprocs=2,
            join=True,
        )
        for rank in range(2):
            result = json.loads((directory / f"rank{rank}.json").read_text())
            self.assertEqual(result["fault"], fault)

    def test_qwen_full_checkpoint_roundtrip(self):
        self._run("qwen", "none")

    def test_glm_full_checkpoint_roundtrip(self):
        self._run("glm", "none")

    def test_qwen_capture_failure_reaches_both_ranks(self):
        self._run("qwen", "capture")

    def test_glm_capture_failure_reaches_both_ranks(self):
        self._run("glm", "capture")

    def test_qwen_invalid_export_reaches_both_ranks(self):
        self._run("qwen", "export")

    def test_glm_invalid_export_reaches_both_ranks(self):
        self._run("glm", "export")
