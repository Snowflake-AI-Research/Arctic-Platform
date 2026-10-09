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

"""Does DeepSpeed alone reproduce the gradient-norm disagreement, on one GPU?

At sequence-parallel degree 1 the engine still disagrees with the single-GPU reference on 27 of 374
tensors, so the origin is in what one rank does. This removes everything else: no gateway, no sequence
parallelism, no distributed anything. One process builds the model twice from the same files, runs the
same batch through the same loss, and takes the backward once plainly and once through a DeepSpeed engine
configured from the job config's ``ds_config``.

The two arms share the loss code, so a difference between them is DeepSpeed's.
Run it under the DeepSpeed launcher, which supplies the one-rank world it needs:

    deepspeed --num_gpus 1 arctic_platform/correctness/diagnostics/deepspeed_vs_plain.py <config>
"""

from __future__ import annotations

import copy
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.config import LoadedConfig  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402

ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
CE_CHUNK = int(os.environ.get("PROBE_CE_CHUNK", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"
IGNORE_INDEX = -100
WATCH = ("lm_head.weight", "norm.weight")


def load_model(model_path: str, attn: str):
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attn, trust_remote_code=True, config=cfg
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model.cuda()


def _build_liger_fused_cross_entropy():
    """Build the same fp32-accumulating fused CE used by the correctness reference."""
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        reduction="mean",
        accum_dtype=torch.float32,
    )


def loss_settings(cfg: LoadedConfig) -> tuple[bool, bool | str, int | None, int]:
    """Resolve loss knobs the way the correctness harness passes them to the reference."""
    training = cfg.effective_training
    return (
        bool(training.get("fp32_lm_head", False)),
        cfg.fused_cross_entropy,
        cfg.lm_head_token_chunk_size,
        int(training.get("fused_lm_head_vocab_chunk_size", 8192)),
    )


def token_weighted_loss(
    model,
    input_ids,
    labels,
    active_tokens,
    fp32_lm_head: bool,
    fused_cross_entropy: bool | str,
    token_chunk_size: int | None,
    vocab_chunk_size: int,
    ce_chunk: int = CE_CHUNK,
):
    """The production loss selected by the config, identical in both arms."""
    from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs

    if fused_cross_entropy:
        if fused_cross_entropy not in {True, "liger"}:
            raise ValueError(f"diagnostic does not support fused_cross_entropy={fused_cross_entropy!r}")
        if token_chunk_size is not None:
            raise ValueError("fused cross-entropy and chunked LM-head logprobs are mutually exclusive")

    module = getattr(model, "module", model)
    hidden = module.get_decoder()(input_ids=input_ids, use_cache=False).last_hidden_state
    head = module.get_output_embeddings()
    if token_chunk_size is not None:
        logprobs = chunked_lm_head_logprobs(
            hidden,
            head.weight,
            labels,
            bias=getattr(head, "bias", None),
            token_chunk_size=token_chunk_size,
            vocab_chunk_size=vocab_chunk_size,
            fp32_lm_head=fp32_lm_head,
        )
        valid = labels != IGNORE_INDEX
        return -torch.where(valid, logprobs.float(), 0.0).sum() / active_tokens

    flat_labels = labels.reshape(-1)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    if fused_cross_entropy:
        valid_tokens = int((flat_labels != IGNORE_INDEX).sum())
        if valid_tokens == 0:
            return hidden.sum() * 0.0
        fused_ce = _build_liger_fused_cross_entropy()
        mean_loss = fused_ce(head.weight, flat_hidden, flat_labels)
        return mean_loss * (valid_tokens / active_tokens)

    total_loss = None
    for start in range(0, flat_labels.numel(), ce_chunk):
        chunk_labels = flat_labels[start : start + ce_chunk]
        if int((chunk_labels != IGNORE_INDEX).sum()) == 0:
            continue
        chunk_hidden = flat_hidden[start : start + ce_chunk]
        if fp32_lm_head:
            logits = F.linear(chunk_hidden.float(), head.weight.float())
        else:
            logits = F.linear(chunk_hidden, head.weight).float()
        loss = F.cross_entropy(logits, chunk_labels, ignore_index=IGNORE_INDEX, reduction="sum") / active_tokens
        total_loss = loss if total_loss is None else total_loss + loss
    if total_loss is None:
        return hidden.sum() * 0.0
    return total_loss


def grad_norms_plain(model):
    return {name: float(p.grad.detach().float().norm()) for name, p in model.named_parameters() if p.grad is not None}


def grad_norms_deepspeed(engine, model):
    from deepspeed.utils import safe_get_full_grad

    norms = {}
    for name, param in model.named_parameters():
        grad = safe_get_full_grad(param)
        if grad is not None:
            norms[name] = float(grad.detach().float().norm())
    return norms


def main(config_path: str, spec_path: str) -> int:
    import deepspeed
    from transformers import AutoConfig

    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    layers = LAYERS or spec.model.num_hidden_layers
    model_path = (
        spec.model.cache_path
        if layers == spec.model.num_hidden_layers
        else materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{layers}L", layers).cache_path
    )
    attn = os.environ.get("PROBE_ATTN") or cfg.training.get("attn_implementation", "flash_attention_3")
    fp32_lm_head, fused_cross_entropy, token_chunk_size, vocab_chunk_size = loss_settings(cfg)

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    input_ids = batch.input_ids.cuda()
    labels = batch.shifted_labels().cuda()
    active = int((labels != IGNORE_INDEX).sum())
    print(
        f"model {model_path} ({layers} layers), {ROW_TOKENS:,}-token sequence, {active:,} real tokens, "
        f"attn {attn}, one GPU, no gateway\n",
        flush=True,
    )
    print(
        f"loss fp32_lm_head={fp32_lm_head}, fused_cross_entropy={fused_cross_entropy!r}, "
        f"lm_head_token_chunk={token_chunk_size}, lm_head_vocab_chunk={vocab_chunk_size}, ce_chunk={CE_CHUNK}",
        flush=True,
    )

    print("[plain] forward/backward without DeepSpeed ...", flush=True)
    model = load_model(model_path, attn)
    loss = token_weighted_loss(
        model,
        input_ids,
        labels,
        active,
        fp32_lm_head,
        fused_cross_entropy,
        token_chunk_size,
        vocab_chunk_size,
    )
    loss.backward()
    plain_loss = float(loss.item())
    plain = grad_norms_plain(model)
    print(f"[plain] loss {plain_loss:.6f}, {len(plain)} gradients", flush=True)
    del model, loss
    torch.cuda.empty_cache()

    ds_config = copy.deepcopy(cfg.training.get("ds_config", {}))
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["train_batch_size"] = 1
    ds_config["gradient_accumulation_steps"] = 1
    print(
        f"\n[deepspeed] zero stage {ds_config.get('zero_optimization', {}).get('stage')}, "
        f"bf16 {ds_config.get('bf16', {}).get('enabled')}, "
        f"communication_data_type {ds_config.get('communication_data_type')}",
        flush=True,
    )

    model = load_model(model_path, attn)
    from deepspeed.ops.adam import DeepSpeedCPUAdam

    optimizer = DeepSpeedCPUAdam(
        model.parameters(),
        lr=cfg.training["optimizer"]["lr"],
        betas=tuple(cfg.training["optimizer"]["betas"]),
        eps=cfg.training["optimizer"]["eps"],
        weight_decay=cfg.training["optimizer"]["weight_decay"],
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config, model_parameters=model.parameters()
    )
    loss = token_weighted_loss(
        engine,
        input_ids,
        labels,
        active,
        fp32_lm_head,
        fused_cross_entropy,
        token_chunk_size,
        vocab_chunk_size,
    )
    engine.backward(loss)
    ds_loss = float(loss.item())
    wrapped = grad_norms_deepspeed(engine, model)
    print(f"[deepspeed] loss {ds_loss:.6f}, {len(wrapped)} gradients", flush=True)

    shared = sorted(set(plain) & set(wrapped))
    if not shared:
        print("no parameter names in common; nothing to compare")
        return 1
    diffs = [(abs(wrapped[k] - plain[k]), k) for k in shared]
    ratios = sorted(wrapped[k] / plain[k] for k in shared if plain[k])
    worst, worst_name = max(diffs)
    over = sum(1 for d, _ in diffs if d > STATED_CRITERION_ABS)
    print("")
    print(f"tensors compared              {len(shared)}")
    print(f"loss delta                    {abs(ds_loss - plain_loss):.3e}")
    print(f"median DeepSpeed/plain ratio  {ratios[len(ratios) // 2]:.6f}")
    print(f"ratios above 1.0              {sum(1 for r in ratios if r > 1.0)}/{len(ratios)}")
    print(f"over 1e-3 absolute            {over}")
    print(f"largest disagreement          {worst:.3e}  {worst_name}")
    print(f"median plain norm             {statistics.median(plain[k] for k in shared):.4f}")
    print("")
    print(f"{'tensor':28} {'deepspeed':>14} {'plain':>14} {'abs diff':>11}")
    for name in shared:
        short = name.split("model.")[-1]
        if any(short.endswith(w) or name.endswith(w) for w in WATCH):
            print(f"{short:28} {wrapped[name]:>14.6f} {plain[name]:>14.6f} {abs(wrapped[name] - plain[name]):>11.3e}")
    print("")
    print(f"{'worst 10 tensors':28} {'deepspeed':>14} {'plain':>14} {'abs diff':>11}")
    for d, name in sorted(diffs, reverse=True)[:10]:
        print(f"{name.split('model.')[-1]:28} {wrapped[name]:>14.6f} {plain[name]:>14.6f} {d:>11.3e}")
    return 0


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--local_rank=")]
    raise SystemExit(
        main(
            args[0] if args else "arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config",
            args[1] if len(args) > 1 else SPEC,
        )
    )
