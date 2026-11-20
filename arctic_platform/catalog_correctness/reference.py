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

"""Generate a single-GPU Hugging Face reference for a full-training catalog recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .checks import read_recipe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(tensor) -> str:
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw).hexdigest()


def _optimizer_norms(named, masters, adam, optimizer_config):
    from arctic_platform.correctness.checks.optimizer_state import optimizer_comparison_tensors
    from arctic_platform.correctness.checks.optimizer_state import optimizer_tensor_norm

    result = {}
    for (name, param), master in zip(named, masters, strict=True):
        state = adam.state[master]
        tensors = {
            "parameter_update": master.detach() - param.detach().float(),
            "exp_avg": state["exp_avg"].detach(),
            "exp_avg_sq": state["exp_avg_sq"].detach(),
        }
        values = optimizer_comparison_tensors(
            tensors,
            step=1,
            optimizer_config=optimizer_config,
            learning_rate=optimizer_config["lr"],
            backend="reference",
        )
        result[name] = {f"{key}_norm": optimizer_tensor_norm(value) for key, value in values.items()}
    return result


def _accumulation_dtype(config: dict, model_dtype: str, default=None) -> str:
    value = (config.get("data_types") or {}).get("grad_accum_dtype", default)
    if value is None:
        return model_dtype
    aliases = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp32": "float32", "float32": "float32"}
    if value not in aliases:
        raise ValueError(f"unsupported gradient accumulation dtype {value!r}")
    return aliases[value]


def generate(recipe_path: Path, model_path: Path, batch_path: Path, output: Path, *, attention: str, steps: int):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    recipe, recipe_hash = read_recipe(recipe_path, "training")
    sub = recipe["sub_job_configs"][0]
    training = sub["training_config"]
    optimizer = training["optimizer"]
    ds = training.get("ds_config") or {}
    if sub.get("dtype") != "bfloat16" or training.get("peft_config"):
        raise ValueError("this reference requires full-parameter bfloat16 training")
    if (
        training.get("n_gpus") != 1
        or training.get("train_batch_size") != 1
        or ds.get("gradient_accumulation_steps", 1) != 1
        or (ds.get("zero_optimization") or {}).get("stage") != 2
    ):
        raise ValueError("this reference requires one GPU, one training row, one accumulation step and ZeRO stage 2")
    if optimizer.get("name", "").lower() != "adamw" or optimizer.get("weight_decay", 0) != 0:
        raise ValueError("this reference requires AdamW with zero weight decay")
    if training.get("lr_scheduler") or ds.get("scheduler"):
        raise ValueError("learning-rate schedulers are not supported by this reference")
    if steps < 1 or not torch.cuda.is_available():
        raise ValueError("a CUDA device and at least one training step are required")
    batches = json.loads(batch_path.read_bytes())
    if not isinstance(batches, list) or len(batches) < max(steps + 1, 4):
        raise ValueError("supply one training batch per step plus a held-out batch, and at least four batches")
    batches = batches[: max(steps + 1, 4)]
    if any(
        not isinstance(batch, list)
        or len(batch) != 1
        or not isinstance(batch[0], list)
        or len(batch[0]) < 2
        or any(type(token) is not int or token < 0 for token in batch[0])
        for batch in batches
    ):
        raise ValueError("each batch must contain one sequence of at least two token IDs")
    if len({len(batch[0]) for batch in batches[:4]}) != 1:
        raise ValueError("the four accumulation rows must have equal lengths")

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(sub.get("seed", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attention, local_files_only=True
    ).cuda()
    model.train()
    model.config.use_cache = False
    named = list(model.named_parameters())
    if not named or any(not param.requires_grad for _, param in named):
        raise ValueError("reference expects all model parameters to be trainable")
    initial = {name: {"shape": list(param.shape), "sha256": _tensor_hash(param)} for name, param in named}
    masters = [torch.nn.Parameter(param.detach().float()) for _, param in named]
    adam = torch.optim.AdamW(
        masters,
        lr=optimizer["lr"],
        betas=tuple(optimizer.get("betas", (0.9, 0.999))),
        eps=optimizer.get("eps", 1e-8),
        weight_decay=0,
        fused=False,
        foreach=False,
    )
    files = [model_path / "config.json", *sorted(model_path.glob("*.safetensors"))]
    if len(files) < 2 or not all(path.is_file() for path in files):
        raise ValueError("model snapshot must contain config.json and safetensors weights")
    result = {
        "schema": "catalog-hf-reference-v2",
        "model": sub["model_name"],
        "recipe_sha256": recipe_hash,
        "inputs_sha256": hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest(),
        "model_files_sha256": {path.name: _sha256(path) for path in files},
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "reference_source_sha256": _sha256(Path(__file__)),
        "reference_attention": attention,
        "dtype": sub["dtype"],
        "optimizer": optimizer,
        "gradient_clipping": training.get("gradient_clipping"),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "initial_parameters": initial,
        "losses": [],
    }
    accumulation_dtype = _accumulation_dtype(ds, sub["dtype"])
    accumulators = [None] * len(named)
    token_count = sum(len(batch[0]) - 1 for batch in batches[:4])
    for batch in batches[:4]:
        ids = torch.tensor(batch, dtype=torch.long, device="cuda")
        weighted_loss = model(input_ids=ids, labels=ids, use_cache=False).loss * ((len(batch[0]) - 1) / token_count)
        weighted_loss.backward()
        for index, (_, param) in enumerate(named):
            if param.grad is None:
                raise ValueError("reference accumulation found a parameter without gradients")
            gradient = param.grad.detach().to(getattr(torch, accumulation_dtype))
            accumulators[index] = (
                gradient.clone() if accumulators[index] is None else accumulators[index].add_(gradient)
            )
            param.grad = None
    for master, accumulator in zip(masters, accumulators, strict=True):
        master.grad = accumulator.float()
    result["gas4"] = {
        "gradient_accumulation_dtype": accumulation_dtype,
        "gradient_norms": {
            name: float(torch.linalg.vector_norm(master.grad))
            for (name, _), master in zip(named, masters, strict=True)
        },
    }
    clip = training.get("gradient_clipping")
    if clip is not None and clip > 0:
        result["gas4"]["gradient_norm_before_clip"] = float(torch.nn.utils.clip_grad_norm_(masters, clip))
    adam.step()
    result["gas4"]["optimizer_step"] = _optimizer_norms(named, masters, adam, optimizer)
    with torch.no_grad():
        for (_, param), master in zip(named, masters, strict=True):
            master.copy_(param)
    adam.state.clear()
    adam.zero_grad(set_to_none=True)
    for index, batch in enumerate(batches[:steps], 1):
        ids = torch.tensor(batch, dtype=torch.long, device="cuda")
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        missing = [name for name, param in named if param.grad is None]
        if missing:
            raise ValueError(f"reference parameters without gradients: {missing}")
        if index == 1:
            result["gradient_norms"] = {
                name: float(torch.linalg.vector_norm(param.grad.float())) for name, param in named
            }
        for (_, param), master in zip(named, masters, strict=True):
            master.grad = param.grad.float()
            param.grad = None
        clip = training.get("gradient_clipping")
        if clip is not None and clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(masters, clip))
            if index == 1:
                result["gradient_norm_before_clip"] = grad_norm
        adam.step()
        if index == 1:
            result["optimizer_step"] = _optimizer_norms(named, masters, adam, optimizer)
        with torch.no_grad():
            for (_, param), master in zip(named, masters, strict=True):
                param.copy_(master)
        adam.zero_grad(set_to_none=True)
        result["losses"].append(float(loss.detach()))
        if index == 1 or index % 10 == 0 or index == steps:
            print(f"reference training step {index}/{steps}", flush=True)
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(batches[steps], dtype=torch.long, device="cuda")
        result["validation_loss"] = float(model(input_ids=ids, labels=ids, use_cache=False).loss)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention", default="sdpa", help="Hugging Face reference attention implementation")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    generate(
        args.training_config,
        args.model_snapshot,
        args.batches,
        args.output,
        attention=args.attention,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()
