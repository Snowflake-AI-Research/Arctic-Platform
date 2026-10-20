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

"""Add one piece of the engine at a time to a plain single-GPU backward, and see which one breaks it.

Everything cheaper has been ruled out: the attention kernel (running flash attention on both sides moved
the loss and left the gradient norms where they were), sequence parallelism (sp 1 disagrees on more tensors
than sp 8), full attention (removing those layers triples the failure rate), and the fixture's weights
(random initialization produced a systematic offset that checkpoint weights removed).

What is left is the difference between a plain HuggingFace backward and the one the engine runs on a single
rank. The arms below are cumulative, from the plainest computation to the job config as written:

    plain          one forward, one backward, output projection on float32 operands
    gc             + gradient checkpointing
    chunked        + cross-entropy in token chunks against a detached hidden state
    zero0          + DeepSpeed, ZeRO disabled
    zero2          + ZeRO stage 2
    zero2_offload  + the optimizer on CPU, which is the config as written

Each arm writes its per-parameter gradient norms; ``--compare`` reads them and reports every arm against
``plain``. The arm that first disagrees is the one that introduces the difference.

Run each arm under the DeepSpeed launcher so the engine arms have a one-rank world:

    deepspeed --num_gpus 1 arctic_platform/correctness/diagnostics/ladder.py --arm zero2 --out /tmp/zero2.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from arctic_platform.correctness.harness.batches import build_batch  # noqa: E402
from arctic_platform.correctness.harness.config import load_config  # noqa: E402
from arctic_platform.correctness.harness.seeds import SEED  # noqa: E402
from arctic_platform.correctness.harness.spec import STATED_CRITERION_ABS  # noqa: E402
from arctic_platform.correctness.harness.spec import TestSpec  # noqa: E402
from arctic_platform.correctness.onboarding.synth_model import materialize_pretrained  # noqa: E402
from arctic_platform.model.implementations.debug.determinism import configure_full_determinism  # noqa: E402
from arctic_platform.model.implementations.debug.determinism import determinism_worker_env  # noqa: E402
from arctic_platform.model.implementations.debug.determinism import resolve_seed  # noqa: E402

# The engine's own determinism switch, applied to every arm so a difference between two of them is the
# arithmetic and not the order the GPU happened to accumulate in.
DETERMINISM = {"debug": {"full_determinism": True, "full_determinism_must_comply": False}}

ROW_TOKENS = int(os.environ.get("PROBE_ROW_TOKENS", 2048))
LAYERS = int(os.environ.get("PROBE_LAYERS", 4))
SOURCE = "/data-fast/base-models/Qwen/Qwen3.8-27B"
CACHE_ROOT = "/data-fast/base-models/synthetic"
SPEC = "arctic_platform/correctness/specs/qwen3.8-27b-h200-train-sft-8gpus-2k.json"
IGNORE_INDEX = -100
CE_CHUNK = 2048

ARMS = ("plain", "gc", "chunked", "zero0", "zero2", "zero2_offload")


def wants(arm: str, feature: str) -> bool:
    """Cumulative membership: an arm has every feature introduced at or before it."""
    return ARMS.index(arm) >= ARMS.index(feature)


def deep_merge(base: dict, over: dict) -> dict:
    """Merge ``over`` into ``base`` in place, descending into nested dicts rather than replacing them."""
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_model(spec: TestSpec) -> tuple:
    layers = LAYERS or spec.model.num_hidden_layers
    if layers == spec.model.num_hidden_layers:
        return spec.model.cache_path, layers
    return materialize_pretrained(SOURCE, f"{CACHE_ROOT}/Qwen3.8-27B-{layers}L", layers).cache_path, layers


def build(model_path: str, attn: str, gradient_checkpointing: bool):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation=attn, trust_remote_code=True
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.cuda()


def decoder_and_head(model):
    """The module producing hidden states, and the output projection, for the arms that split them."""
    head = model.get_output_embeddings()
    inner = getattr(model, "model", model)
    return inner, head


def loss_fp32_head(model, input_ids, labels, active):
    """``fp32_lm_head`` is required by the configuration under test, so every arm carries it."""
    inner, head = decoder_and_head(model)
    hidden = inner(input_ids=input_ids).last_hidden_state
    logits = F.linear(hidden.view(-1, hidden.shape[-1]).float(), head.weight.float())
    return F.cross_entropy(logits, labels.view(-1), ignore_index=IGNORE_INDEX, reduction="sum") / active


def backward_chunked(model, input_ids, labels, active, backward):
    """The reference's shape: detach the hidden state, take the projection in chunks, re-enter the stack."""
    inner, head = decoder_and_head(model)
    hidden = inner(input_ids=input_ids).last_hidden_state
    detached = hidden.detach().requires_grad_(True)
    flat_labels = labels.reshape(-1)
    hidden_size = hidden.shape[-1]
    total = 0.0
    for start in range(0, flat_labels.numel(), CE_CHUNK):
        chunk_labels = flat_labels[start : start + CE_CHUNK]
        if int((chunk_labels != IGNORE_INDEX).sum()) == 0:
            continue
        chunk_hidden = detached.view(-1, hidden_size)[start : start + CE_CHUNK]
        logits = F.linear(chunk_hidden.float(), head.weight.float())
        loss = F.cross_entropy(logits, chunk_labels, ignore_index=IGNORE_INDEX, reduction="sum") / active
        backward(loss)
        total += float(loss.item())
    hidden.backward(detached.grad)
    return total


def run_arm(arm: str, config_path: str, spec_path: str, out_path: str) -> int:
    seed = resolve_seed(SEED, DETERMINISM)
    env = determinism_worker_env(DETERMINISM, seed)
    # Flash attention 3 refuses a deterministic backward at head dim 256, which this architecture uses.
    # The rest of the set still applies: the pinned cuBLAS workspace and the deterministic ATen kernels,
    # which is what fixes the embedding gradient's scatter-add.
    env.pop("FLASH_ATTENTION_DETERMINISTIC", None)
    os.environ.update(env)
    configure_full_determinism(DETERMINISM)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = load_config(Path(config_path))
    spec = TestSpec.read(Path(spec_path))
    model_path, layers = resolve_model(spec)
    # Flash attention's backward accumulates the query gradient with atomics and ignores
    # ``torch.use_deterministic_algorithms``, so the kernel is the one part of this path that can differ
    # between two identical runs. ``PROBE_ATTN`` swaps in a kernel that cannot.
    attn = os.environ.get("PROBE_ATTN") or cfg.training.get("attn_implementation", "flash_attention_3")

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab = getattr(model_cfg, "text_config", model_cfg).vocab_size

    batch = build_batch("gas1", 1, ROW_TOKENS, vocab, seed=SEED)
    # Logit-aligned targets, the same conversion both engines apply at dispatch.
    input_ids, labels = batch.input_ids.cuda(), batch.shifted_labels().cuda()
    active = batch.active_tokens

    model = build(model_path, attn, wants(arm, "gc"))
    engine = None
    if wants(arm, "zero0"):
        import deepspeed

        ds_config = copy.deepcopy(cfg.training.get("ds_config", {}))
        ds_config["train_micro_batch_size_per_gpu"] = 1
        ds_config["train_batch_size"] = 1
        ds_config["gradient_accumulation_steps"] = 1
        zero = ds_config.setdefault("zero_optimization", {})
        zero["stage"] = 0 if arm == "zero0" else 2
        optimizer = {"type": "AdamW", "params": {"lr": 0.0}}
        if wants(arm, "zero2_offload"):
            zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        else:
            zero.pop("offload_optimizer", None)
        ds_config["optimizer"] = optimizer
        # One ZeRO knob at a time, so a difference against ``plain`` names the part of the gradient path that
        # produced it. Only the gradient side can matter here: the comparison reads gradients and never steps.
        deep_merge(ds_config, json.loads(os.environ.get("PROBE_DS_OVERRIDE", "{}")))
        engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())

    module = engine if engine is not None else model
    grad_dtypes: set = set()
    backward = engine.backward if engine is not None else (lambda t: t.backward())

    if wants(arm, "chunked"):
        loss_value = backward_chunked(module, input_ids, labels, active, backward)
    else:
        loss = loss_fp32_head(module, input_ids, labels, active)
        backward(loss)
        loss_value = float(loss.item())

    if engine is not None:
        from deepspeed.utils import safe_get_full_grad

        norms = {}
        for name, param in model.named_parameters():
            grad = safe_get_full_grad(param)
            if grad is not None:
                grad_dtypes.add(str(grad.dtype))
                norms[name] = float(grad.detach().float().norm())
    else:
        norms = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_dtypes.add(str(param.grad.dtype))
            norms[name] = float(param.grad.detach().float().norm())

    dump = os.environ.get("PROBE_DUMP")
    if dump:
        dump_embedding(model, engine, dump)

    label = os.environ.get("PROBE_LABEL", arm)
    if int(os.environ.get("RANK", "0")) == 0:
        Path(out_path).write_text(
            json.dumps(
                {
                    "arm": label,
                    "layers": layers,
                    "loss": loss_value,
                    "grad_dtype": sorted(grad_dtypes),
                    "model": model_path,
                    "norms": norms,
                },
                indent=1,
            )
        )
        print(
            f"[{label}] loss {loss_value:.6f}, {len(norms)} gradients in "
            f"{'/'.join(sorted(grad_dtypes))} -> {out_path}",
            flush=True,
        )
    return 0


def dump_embedding(model, engine, path: str) -> None:
    """Save the rows of the input-embedding gradient that received a contribution.

    A norm says two gradients differ; the rows say how. A per-element difference at the width of one
    bfloat16 step, spread over every row, is rounding. A handful of rows that disagree by orders of
    magnitude is a contribution that one path dropped or added.
    """
    name = next(n for n, _ in model.named_parameters() if n.endswith("embed_tokens.weight"))
    param = dict(model.named_parameters())[name]
    if engine is not None:
        from deepspeed.utils import safe_get_full_grad

        grad = safe_get_full_grad(param)
    else:
        grad = param.grad
    grad = grad.detach().float()
    rows = torch.nonzero(grad.abs().sum(dim=1) > 0, as_tuple=True)[0]
    torch.save({"rows": rows.cpu(), "values": grad.index_select(0, rows).cpu()}, path)


def rows(paths) -> int:
    """Per-element comparison of two dumped embedding gradients."""
    first, second = (torch.load(path) for path in paths)
    if not torch.equal(first["rows"], second["rows"]):
        shared = sorted(set(first["rows"].tolist()) & set(second["rows"].tolist()))
        print(f"row sets differ: {len(first['rows'])} vs {len(second['rows'])}, {len(shared)} shared")
        index = {int(r): i for i, r in enumerate(first["rows"].tolist())}
        other = {int(r): i for i, r in enumerate(second["rows"].tolist())}
        a = first["values"][[index[r] for r in shared]]
        b = second["values"][[other[r] for r in shared]]
    else:
        print(f"same {len(first['rows'])} rows received a contribution")
        a, b = first["values"], second["values"]
    diff = (a - b).abs()
    scale = a.abs().clamp_min(1e-30)
    relative = (diff / scale).flatten()
    row_diff = (a - b).norm(dim=1)
    row_norm = a.norm(dim=1)
    ulp = 2.0**-8
    print(
        f"elements {relative.numel()}, exactly equal {(diff == 0).sum().item()}, "
        f"past one bfloat16 step ({ulp:.3e} relative) {(relative > ulp).sum().item()}"
    )
    quantiles = torch.tensor([0.5, 0.9, 0.99, 1.0])
    values = torch.quantile(relative[relative > 0].float(), quantiles) if (relative > 0).any() else quantiles * 0
    print(
        "relative per element, nonzero only: "
        + ", ".join(f"p{int(q * 100)} {v:.3e}" for q, v in zip(quantiles.tolist(), values.tolist()))
    )
    worst = torch.topk(row_diff / row_norm.clamp_min(1e-30), min(8, row_diff.numel()))
    print("")
    print(f"{'row':>8} {'norm':>13} {'abs diff':>12} {'relative':>11}")
    for value, index in zip(worst.values.tolist(), worst.indices.tolist()):
        print(f"{int(first['rows'][index]):>8} {row_norm[index]:>13.6f} {row_diff[index]:>12.3e} {value:>11.3e}")
    print("")
    print(
        f"whole-tensor norms: {a.norm():.8f} against {b.norm():.8f}, "
        f"difference {abs(float(a.norm()) - float(b.norm())):.3e}"
    )
    return 0


def compare(paths) -> int:
    loaded = []
    for path in paths:
        blob = json.loads(Path(path).read_text())
        loaded.append((blob["arm"], blob["loss"], blob["norms"], blob.get("grad_dtype", [])))
    base_arm, base_loss, base, base_dtype = loaded[0]
    print(
        f"baseline {base_arm}: loss {base_loss:.6f}, {len(base)} tensors, "
        f"median norm {statistics.median(base.values()):.4f}, gradients in {'/'.join(base_dtype)}"
    )
    print("")
    print(
        f"{'arm':22} {'loss delta':>11} {'median ratio':>13} {'above 1.0':>11} {'over 1e-3':>10} "
        f"{'worst abs':>11} {'embed abs':>11}  grad dtype"
    )
    embed = next((k for k in base if k.endswith("embed_tokens.weight")), None)
    for arm, loss, norms, dtype in loaded[1:]:
        shared = sorted(set(base) & set(norms))
        if not shared:
            print(f"{arm:16} {'no shared tensors':>11}")
            continue
        ratios = sorted(norms[k] / base[k] for k in shared if base[k])
        diffs = [(abs(norms[k] - base[k]), k) for k in shared]
        worst, _worst_name = max(diffs)
        embed_abs = abs(norms[embed] - base[embed]) if embed in norms else float("nan")
        print(
            f"{arm:22} {abs(loss - base_loss):>11.3e} {ratios[len(ratios) // 2]:>13.6f} "
            f"{sum(1 for r in ratios if r > 1.0):>7}/{len(ratios):<3} "
            f"{sum(1 for d, _ in diffs if d > STATED_CRITERION_ABS):>10} {worst:>11.3e} "
            f"{embed_abs:>11.3e}  {'/'.join(dtype)}"
        )

    print("")
    print(f"{'arm':22} {'tensors past 1e-3, largest first':.<60}")
    for arm, _loss, norms, _dtype in loaded[1:]:
        shared = sorted(set(base) & set(norms))
        failing = sorted(((abs(norms[k] - base[k]), k) for k in shared), reverse=True)
        failing = [(d, k) for d, k in failing if d > STATED_CRITERION_ABS]
        if not failing:
            print(f"{arm:22} none")
            continue
        for d, name in failing:
            print(f"{arm:22} {name.split('model.')[-1]:40} {d:>11.3e} (base {base[name]:.6f})")
    return 0


def spread(paths) -> int:
    """Per-tensor range across repeats of one arm: how far a norm moves when nothing changes."""
    runs = [json.loads(Path(path).read_text())["norms"] for path in paths]
    names = sorted(set.intersection(*(set(r) for r in runs)))
    rows = []
    for name in names:
        values = [r[name] for r in runs]
        rows.append((max(values) - min(values), name, statistics.median(values)))
    rows.sort(reverse=True)
    over = sum(1 for r, _, _ in rows if r > STATED_CRITERION_ABS)
    print(
        f"{len(runs)} repeats of the same arm, {len(names)} tensors, "
        f"{over} whose range already exceeds {STATED_CRITERION_ABS:.0e}"
    )
    print("")
    print(f"{'tensor':44} {'range':>11} {'median norm':>13} {'relative':>10}")
    for rng, name, median in rows[:15]:
        print(f"{name.split('model.')[-1]:44} {rng:>11.3e} {median:>13.6f} {(rng / median if median else 0):>10.2%}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs="*")
    parser.add_argument("--spread", nargs="*")
    parser.add_argument("--rows", nargs=2)
    parser.add_argument(
        "--config", default="arctic_platform/correctness/configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config"
    )
    parser.add_argument("--spec", default=SPEC)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if args.rows:
        raise SystemExit(rows(args.rows))
    if args.spread:
        raise SystemExit(spread(args.spread))
    if args.compare:
        raise SystemExit(compare(args.compare))
    raise SystemExit(run_arm(args.arm, args.config, args.spec, args.out))
