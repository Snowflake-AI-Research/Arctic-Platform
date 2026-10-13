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

"""Capture the frozen B200 gas1 head/norm and ZeRO boundaries without changing training."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch

from arctic_platform.correctness.harness.seeds import SEED

CONFIG = Path("arctic_platform/correctness/configs/qwen3.8-27b/b200/train-sft-8gpus-2k.config")
SPEC = Path("arctic_platform/correctness/specs/qwen3.8-27b-b200-train-sft-8gpus-2k.json")


def install_capture(model, directory: Path):
    """Record existing gradients through observational hooks, returning local parameter sums."""
    directory.mkdir(parents=True, exist_ok=False)
    local = {}
    current = {}
    calls = [0]
    handles = []
    norm_candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".norm") and ".layers." not in name and type(module).__name__ == "Qwen3_5RMSNorm"
    ]
    assert len(norm_candidates) == 1, [name for name, _ in norm_candidates]
    norm_name, norm = norm_candidates[0]
    embeddings = model.get_input_embeddings()

    def embedding_hook(module, inputs):
        current["ids"] = inputs[0].detach().cpu()

    def norm_hook(module, inputs, output):
        index = calls[0]
        calls[0] += 1
        values = {
            "ids": current["ids"].clone(),
            "input": inputs[0].detach().cpu(),
            "output": output.detach().cpu(),
            "norm_name": norm_name,
            "norm_weight": module.weight.detach().cpu(),
        }

        def output_gradient(gradient):
            values["output_gradient"] = gradient.detach().cpu()
            torch.save(values, directory / f"norm-call-{index:04d}.pt")

        output.register_hook(output_gradient)

    handles.append(embeddings.register_forward_pre_hook(embedding_hook))
    handles.append(norm.register_forward_hook(norm_hook))
    for name, parameter in model.named_parameters():
        if parameter is not norm.weight and not name.endswith("layers.0.linear_attn.out_proj.weight"):
            continue

        def record(gradient, name=name):
            value = gradient.detach().to(device="cpu", dtype=torch.float32)
            if name in local:
                local[name].add_(value)
            else:
                local[name] = value.clone()

        handles.append(parameter.register_hook(record))
    assert len(handles) == 4, len(handles)
    return local, handles


def reference(out: Path):
    from transformers import AutoConfig

    from arctic_platform.correctness.harness.batches import build_batch
    from arctic_platform.correctness.harness.batches import save
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.spec import TestSpec
    from arctic_platform.correctness.reference.hf_single_gpu import build_engine
    from arctic_platform.correctness.reference.hf_single_gpu import forward_backward
    from arctic_platform.correctness.reference.process_determinism import pin_reduction_order

    cfg, spec = load_config(CONFIG), TestSpec.read(SPEC)
    pin_reduction_order(spec.model.cache_path, cfg.attention_implementation)
    config = AutoConfig.from_pretrained(spec.model.cache_path, trust_remote_code=True)
    vocab = getattr(config, "text_config", config).vocab_size
    arm = spec.arms[0]
    assert arm.name == "gas1"
    batch = build_batch(arm.name, arm.global_batch_size, arm.max_seq_len, vocab, seed=SEED)
    assert batch.active_tokens == arm.active_tokens
    save(batch, out / "batch.pt")
    engine = build_engine(spec.model.cache_path, attn_implementation=cfg.attention_implementation, seed=SEED)
    local, _ = install_capture(engine.model, out / "reference")
    result = forward_backward(engine, batch, mixer_packing=True, lm_head_token_chunk_size=cfg.lm_head_token_chunk_size)
    torch.save(local, out / "reference" / "local.pt")
    print("REFERENCE_BOUNDARY_LOSS", result.loss, flush=True)
    print("REFERENCE_CAPTURE_COMPLETE", out, flush=True)


@contextlib.contextmanager
def capture_workers(out: Path):
    import ray

    from arctic_platform.common import deepspeed_worker
    from arctic_platform.common import ray_server
    from arctic_platform.correctness.harness.optimizer_capture import _patched_deepspeed_worker_bindings

    base = deepspeed_worker.DeepSpeedWorker.__ray_metadata__.modified_class

    @ray.remote
    class BoundaryWorker(base):
        def initialize(self, master_addr, job_config):
            result = super().initialize(master_addr, job_config)
            self.capture_dir = out / f"rank-{self.rank}"
            self.local_gradients, self.capture_handles = install_capture(self.engine.module, self.capture_dir)
            return result

        def step(self, learning_rate=None):
            from deepspeed.utils import safe_get_full_grad

            snapshots = {}
            for name, parameter in sorted(self.engine.module.named_parameters()):
                if name not in self.local_gradients:
                    continue
                hp_getter = getattr(parameter, "get_full_hp_grad", None)
                snapshots[name] = {
                    "safe": safe_get_full_grad(parameter).detach().cpu(),
                    "hp": hp_getter().detach().cpu() if callable(hp_getter) else None,
                    "param_grad": parameter.grad.detach().cpu() if parameter.grad is not None else None,
                }
            torch.save(self.local_gradients, self.capture_dir / "local.pt")
            torch.save(snapshots, self.capture_dir / "reduced.pt")
            optimizer = self.engine.optimizer
            print(
                "BOUNDARY_OPTIMIZER",
                self.rank,
                type(optimizer).__name__,
                getattr(optimizer, "gradient_accumulation_dtype", None),
                getattr(optimizer, "communication_data_type", None),
                flush=True,
            )
            return super().step(learning_rate=learning_rate)

    original = ray_server.ArcticRLRayServerState

    class BoundaryServer(original):
        async def initialize(self, job_config_dict):
            with _patched_deepspeed_worker_bindings(BoundaryWorker):
                return await super().initialize(job_config_dict)

    with _patched_deepspeed_worker_bindings(BoundaryWorker):
        ray_server.ArcticRLRayServerState = BoundaryServer
        try:
            yield
        finally:
            ray_server.ArcticRLRayServerState = original


def analyze(out: Path):
    reference_calls = [torch.load(path, weights_only=True) for path in sorted((out / "reference").glob("norm-call*"))]
    reference_local = torch.load(out / "reference" / "local.pt", weights_only=True)
    ranks = sorted(out.glob("rank-*"))
    assert len(ranks) == 8, ranks
    local_sums = {}
    summary = []
    for rank in ranks:
        local = torch.load(rank / "local.pt", weights_only=True)
        for name, gradient in local.items():
            local_sums[name] = local_sums.get(name, torch.zeros_like(gradient)) + gradient / len(ranks)
        for path in sorted(rank.glob("norm-call*")):
            values = torch.load(path, weights_only=True)
            ids = values["ids"].reshape(-1)
            matches = [ref for ref in reference_calls if torch.equal(ref["ids"].reshape(-1)[:16], ids[:16])]
            assert len(matches) == 1, (rank, path, len(matches))
            ref = matches[0]
            tokens = ids.numel()
            assert torch.equal(ref["ids"].reshape(-1)[:tokens], ids)
            item = {"rank": rank.name, "tokens": tokens}
            for key in ("input", "output", "output_gradient"):
                target = values[key].reshape(tokens, -1).float()
                expected = ref[key].reshape(-1, target.shape[-1])[:tokens].float()
                if key == "output_gradient":
                    target /= len(ranks)
                delta = target - expected
                item[key] = {
                    "l2": float(delta.norm()),
                    "max": float(delta.abs().max()),
                    "target_norm": float(target.norm()),
                    "reference_norm": float(expected.norm()),
                }
            summary.append(item)
            print("NORM_BOUNDARY", json.dumps(item), flush=True)
    reduced = torch.load(ranks[0] / "reduced.pt", weights_only=True)
    for name, local in local_sums.items():
        ref_name = next(key for key in reference_local if key.endswith(name.split("language_model.")[-1]))
        ref = reference_local[ref_name]
        item = {"parameter": name, "reference_norm": float(ref.norm()), "local_mean_norm": float(local.norm())}
        for key in ("safe", "hp", "param_grad"):
            value = reduced[name][key]
            if value is not None:
                value = value.float()
                item[key] = {
                    "norm": float(value.norm()),
                    "vs_local_l2": float((value - local).norm()),
                    "vs_reference_l2": float((value - ref).norm()),
                }
        print("REDUCTION_BOUNDARY", json.dumps(item), flush=True)
        summary.append(item)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BOUNDARY_DIAGNOSTIC_COMPLETE", out, flush=True)


def main(out: Path):
    from arctic_platform.correctness.harness.batches import load
    from arctic_platform.correctness.harness.config import load_config
    from arctic_platform.correctness.harness.dss_driver import build_payload
    from arctic_platform.correctness.harness.dss_driver import fwd_bwd_step
    from arctic_platform.correctness.harness.dss_driver import gateway
    from arctic_platform.correctness.harness.dss_driver import pack_microbatches
    from arctic_platform.correctness.harness.dss_driver import running_job
    from arctic_platform.correctness.harness.spec import TestSpec
    from arctic_platform.model.implementations.debug.determinism import CUBLAS_WORKSPACE

    assert "B200" in torch.cuda.get_device_name()
    out.mkdir(parents=True, exist_ok=False)
    # Use only the local eight-GPU node, even on a sixteen-GPU allocation.
    from arctic_platform.common import ray_cluster

    hostfile = out / "hostfile"
    hostfile.write_text(f"{socket.gethostname()} slots=8\n")
    os.environ["DSS_GATEWAY_HOSTFILE"] = str(hostfile)
    ray_cluster._HOSTFILE = str(hostfile)
    subprocess.run(
        [sys.executable, "-u", "-m", __spec__.name, "reference", str(out)],
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE},
        check=True,
    )
    cfg, spec = load_config(CONFIG), TestSpec.read(SPEC)
    arm = spec.arms[0]
    payload = build_payload(
        cfg.training,
        spec.model.cache_path,
        SEED,
        attn_implementation=cfg.attention_implementation,
        gradient_accumulation_steps=arm.dss_microbatches,
    )
    with capture_workers(out), gateway(out / "gateway", cfg.n_gpus) as url:
        with running_job(url, payload) as job_id:
            result = fwd_bwd_step(url, job_id, pack_microbatches(load(out / "batch.pt"), arm.dss_microbatches))
            print("AP_BOUNDARY_LOSS", result.avg_loss, flush=True)
    analyze(out)


if __name__ == "__main__":
    mode, directory = sys.argv[1:]
    {"reference": reference, "run": main, "analyze": analyze}[mode](Path(directory))
