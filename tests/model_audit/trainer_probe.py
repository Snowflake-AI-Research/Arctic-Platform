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
"""Trainer-side probe of the model audit. Launched by the tests under ``torchrun``; not collected by pytest.

Modes (rank 0 writes one ``torch.save`` dict to ``--out``):

- ``reference``: the transformers model of the tiny checkpoint in fp32 and in the launch dtype, one row per call.
  Per-token log-probs and gradients under checkpoint names. The fp32 run is the truth; the launch-dtype run measures
  how far plain dtype noise moves the same math (the noise floor).
- ``forward``: the production model, built from the profile's ``ModelSpec`` by ``build_model``. Packed and per-row
  log-probs, repeats for determinism, parameters without gradients, the exported weights for the round-trip check,
  and whether packs ending in a 1..8-token sample run at all.
- ``step``: the production model under a DeepSpeed engine with the profile's ``ds_config`` and parallelism. Packed
  log-probs, the engine-reduced gradients, and the weight and master-weight updates of one optimizer step, all
  HF-named (exported through the model's own weight-export contract); runs at any world size.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

import audit_lib  # noqa: E402


def _init_distributed() -> None:
    import deepspeed

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="nccl")


def _parallel_groups(parallelism: dict) -> dict:
    from arctic_platform.model.parallel.groups import expert_parallel_group
    from arctic_platform.model.parallel.groups import setup_parallel_groups

    sp = int(parallelism.get("sequence_parallel", 1))
    ep = int(parallelism.get("expert_parallel", 1))
    groups = setup_parallel_groups(sp, ep, world_size=dist.get_world_size(), gpus_per_node=torch.cuda.device_count())
    return {
        "ep_group": expert_parallel_group(ep) if ep > 1 else None,
        "sp_group": groups.sp_group,
        "sp_size": sp,
        "sp_rank": groups.sp_rank,
    }


def _build_production_model(profile, ckpt: str, parallelism: dict | None):
    from arctic_platform.model import ModelSpec
    from arctic_platform.model import build_model

    spec = ModelSpec(**profile.trainer_spec(model_path=ckpt, parallelism=parallelism))
    groups = _parallel_groups(spec.parallelism.model_dump())
    model = build_model(spec, parallel_groups={"ep_group": groups["ep_group"], "sp_group": groups["sp_group"]}).model
    device = torch.device("cuda", torch.cuda.current_device())
    if any(p.device.type != "cuda" for p in model.parameters()):
        model.to(device)
    return model, spec, groups


def _logprobs_from_output(output, labels: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(output):
        output = {"logits": output}
    get = output.get if hasattr(output, "get") else (lambda key, default=None: getattr(output, key, default))
    logprobs = get("logprobs", None)
    if logprobs is not None:
        return logprobs.float()
    logits = get("logits", None)
    assert logits is not None, f"model output has neither logprobs nor logits: {list(output.keys())}"
    safe = labels.clamp(min=0)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return lp.masked_fill(labels == -100, 0.0)


def _forward(model, ids, pos, labels, *, pass_labels: bool, groups: dict | None = None):
    """Per-token log-probs of the full packed row. With sequence parallelism each rank runs its contiguous slice."""
    device = torch.device("cuda", torch.cuda.current_device())
    sp = groups["sp_size"] if groups else 1
    if sp > 1:
        chunk = ids.shape[1] // sp
        sl = slice(groups["sp_rank"] * chunk, (groups["sp_rank"] + 1) * chunk)
        ids, pos, labels = ids[:, sl], pos[:, sl], labels[:, sl]
    kwargs = dict(input_ids=ids.to(device), position_ids=pos.to(device), use_cache=False)
    if pass_labels:
        kwargs["labels"] = labels.to(device)
    try:
        output = model(**kwargs)
    except TypeError as err:
        if "use_cache" not in str(err):
            raise
        kwargs.pop("use_cache")
        output = model(**kwargs)
    lp = _logprobs_from_output(output, labels.to(device))
    if sp > 1:
        local = lp.detach().contiguous()
        parts = [torch.empty_like(local) for _ in range(sp)]
        dist.all_gather(parts, local, group=groups["sp_group"])
        return lp, torch.cat(parts, dim=1)
    return lp, lp.detach()


def _loss(lp: torch.Tensor, labels: torch.Tensor, total_targets: int) -> torch.Tensor:
    mask = (labels.to(lp.device) != -100).float()
    return -(lp * mask).sum() / total_targets


def _head_contraction(
    model, hidden, labels, head_args, head_kwargs, token_chunk: int = 512
) -> dict[str, torch.Tensor]:
    """``sum_t dL/dh_t * h_t`` per channel at the output head's input: what a final norm's weight gradient sees.

    Elementwise, a biased head gradient can hide inside low-precision noise; contracted over tokens the bias adds up
    while the noise does not. Computed through the production head, an unfused head in the launch dtype, and fp64.
    """
    head = model.get_output_embeddings()
    weight = head.weight.detach()
    mask = labels != -100
    total = int(mask.sum())
    h = hidden.clone().requires_grad_(True)
    lp = _logprobs_from_output(head(h, *head_args, **head_kwargs), labels)
    (-(lp.float() * mask).sum() / total).backward()
    out = {"production": (h.grad.double() * hidden.double()).flatten(0, -2).sum(0).cpu()}
    head.zero_grad(set_to_none=True)
    flat_h, flat_labels = hidden.flatten(0, -2), labels.flatten()
    for tag, dtype in (("lowp", hidden.dtype), ("fp64", torch.float64)):
        acc = torch.zeros(hidden.shape[-1], dtype=torch.float64, device=hidden.device)
        w = weight.to(dtype)
        for start in range(0, flat_h.shape[0], token_chunk):
            x = flat_h[start : start + token_chunk].to(dtype).requires_grad_(True)
            y = flat_labels[start : start + token_chunk]
            logits = (x @ w.t()).to(torch.promote_types(dtype, torch.float32))
            tok = torch.log_softmax(logits, -1).gather(-1, y.clamp(min=0)[:, None]).squeeze(-1)
            (-(tok * (y != -100)).sum() / total).backward()
            acc += (x.grad.double() * x.detach().double()).sum(0)
        out[tag] = acc.cpu()
    return out


def _export_hf(model, ckpt: str) -> dict[str, torch.Tensor]:
    from arctic_platform.model import iter_model_weights

    out = {}
    for name, tensor in iter_model_weights(
        model, "hf", is_master=dist.get_rank() == 0, is_zero3=False, include_skipped=True, checkpoint_dir=ckpt
    ):
        out[name] = tensor.detach().float().cpu().clone()
    return out


@contextlib.contextmanager
def _params_hold(model, values: dict[int, torch.Tensor]):
    """Temporarily swap parameter data (e.g. for gradients) so the weight-export path maps them to HF names."""
    saved = {}
    for param in model.parameters():
        if id(param) in values:
            saved[id(param)] = param.data
            param.data = values[id(param)]
    try:
        yield
    finally:
        for param in model.parameters():
            if id(param) in saved:
                param.data = saved[id(param)]


def _checkpoint_named(model, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map transformers runtime names (fused experts, renamed modules) back to checkpoint names, as saving does."""
    from transformers.core_model_loading import revert_weight_conversion

    return {name: tensor.float() for name, tensor in revert_weight_conversion(model, dict(tensors)).items()}


_inputs = audit_lib.probe_inputs


def run_reference(profile, ckpt: str) -> dict:
    import transformers

    config, rows, pad_multiple = _inputs(profile, ckpt)
    ids, pos, labels = audit_lib.pack_rows(rows, pad_multiple)
    total = int((labels != -100).sum())
    arch = config["architectures"][0]
    cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
    result = {"ids": ids, "pos": pos, "labels": labels, "rows": rows}
    for tag, dtype in (("fp32", torch.float32), ("lowp", getattr(torch, profile.dtype))):
        model = cls.from_pretrained(ckpt, dtype=dtype, attn_implementation="eager").cuda()
        model.train()
        # One sequence per call: not every transformers model masks at position resets (GLM-5.3 sparse attention
        # does not), and a packed reference would then attend across rows.
        lp = torch.zeros(ids.shape, dtype=torch.float32)
        params = dict(model.named_parameters())
        seen = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
        # Sum over rows of |row gradient|: bf16 noise scales with this, not with the (possibly cancelling) total.
        mass = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
        for sl in audit_lib.row_slices(rows):
            row_lp, _ = _forward(model, ids[:, sl], pos[:, sl], labels[:, sl], pass_labels=False)
            _loss(row_lp, labels[:, sl], total).backward()
            lp[:, sl] = row_lp.detach().float().cpu()
            for n, p in params.items():
                if p.grad is not None:
                    mass[n] += (p.grad.float() - seen[n]).abs()
                    seen[n] = p.grad.float().clone()
        del seen
        result[f"{tag}_logprobs"] = lp
        result[f"{tag}_grad_mass"] = _checkpoint_named(model, {n: m.cpu() for n, m in mass.items()})
        del mass
        result[f"{tag}_grads"] = _checkpoint_named(
            model,
            {
                n: (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).float().cpu()
                for n, p in model.named_parameters()
            },
        )
        result[f"{tag}_param_names"] = sorted(result[f"{tag}_grads"])
        del model
        torch.cuda.empty_cache()
    return result


def run_forward(profile, ckpt: str) -> dict:
    from arctic_platform.model import canonical_parameter_name

    config, rows, pad_multiple = _inputs(profile, ckpt)
    pass_labels = bool(profile.raw.get("probe", {}).get("pass_labels", False))
    model, spec, groups = _build_production_model(profile, ckpt, None)
    model.train()
    ids, pos, labels = audit_lib.pack_rows(rows, pad_multiple)
    total = int((labels != -100).sum())
    result = {"loader": spec.loader, "spec": spec.model_dump(mode="json")}

    head_call = {}
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda m, args, kwargs: head_call.update(args=args, kwargs=kwargs), with_kwargs=True
    )
    lp, full = _forward(model, ids, pos, labels, pass_labels=pass_labels, groups=groups)
    hook.remove()
    _loss(lp, labels, total).backward()
    result["packed_logprobs"] = full.float().cpu()
    result["no_grad_params"] = sorted(
        canonical_parameter_name(n)
        for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not bool(p.grad.abs().sum() > 0))
    )
    result["nonfinite_grad_params"] = sorted(
        canonical_parameter_name(n)
        for n, p in model.named_parameters()
        if p.grad is not None and not bool(torch.isfinite(p.grad).all())
    )
    result["frozen_params"] = sorted(
        canonical_parameter_name(n) for n, p in model.named_parameters() if not p.requires_grad
    )
    model.zero_grad(set_to_none=True)
    hidden, *head_args = (a.detach() if torch.is_tensor(a) else a for a in head_call["args"])
    head_kwargs = {k: v.detach() if torch.is_tensor(v) else v for k, v in head_call["kwargs"].items()}
    result["head_contraction"] = _head_contraction(model, hidden, labels.to(hidden.device), head_args, head_kwargs)
    model.zero_grad(set_to_none=True)

    # Same autograd mode as the first pass: no_grad can select different kernels.
    result["repeat_logprobs"] = _forward(model, ids, pos, labels, pass_labels=pass_labels, groups=groups)[1].cpu()
    with torch.no_grad():
        result["no_grad_logprobs"] = _forward(model, ids, pos, labels, pass_labels=pass_labels, groups=groups)[1].cpu()
        reversed_rows = list(reversed(rows))
        r_ids, r_pos, r_labels = audit_lib.pack_rows(reversed_rows, pad_multiple)
        result["reversed_logprobs"] = _forward(model, r_ids, r_pos, r_labels, pass_labels=pass_labels, groups=groups)[
            1
        ].cpu()
        isolated = []
        for row in rows:
            i_ids, i_pos, i_labels = audit_lib.pack_rows([row], pad_multiple)
            isolated.append(
                _forward(model, i_ids, i_pos, i_labels, pass_labels=pass_labels, groups=groups)[1][0, : len(row)].cpu()
            )
        result["isolated_logprobs"] = isolated
        result["short_rows"] = {}
        for length in audit_lib.SHORT_ROW_LENGTHS:
            s_ids, s_pos, s_labels = audit_lib.pack_rows([rows[1], rows[0][:length]], pad_multiple)
            try:
                s_lp = _forward(model, s_ids, s_pos, s_labels, pass_labels=pass_labels, groups=groups)[1]
                finite = bool(torch.isfinite(s_lp[s_labels.to(s_lp.device) != -100]).all())
                result["short_rows"][length] = "ok" if finite else "non-finite log-probs"
            except Exception as err:  # noqa: BLE001
                result["short_rows"][length] = f"{type(err).__name__}: {str(err).strip().splitlines()[-1][:200]}"
        result["exported"] = _export_hf(model, ckpt)
        result["config_resave"] = _config_resave(ckpt)
        try:
            from arctic_platform.model import iter_model_weights

            result["vllm_export_names"] = sorted(
                name for name, _ in iter_model_weights(model, "vllm", is_master=True, is_zero3=False)
            )
        except (ValueError, NotImplementedError, RuntimeError) as err:
            result["vllm_export_error"] = f"{type(err).__name__}: {err}"
    return result


def _config_resave(ckpt: str) -> dict:
    """What the config class production resolves reads differently once the pinned transformers re-saves the config.

    Runs after the production build, so AutoConfig carries every registration the production loaders make.
    """
    import tempfile

    import transformers
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    model_type = json.loads(Path(ckpt, "config.json").read_text())["model_type"]
    native = getattr(transformers, CONFIG_MAPPING_NAMES.get(model_type, ""), None)
    if native is None:
        return {"skipped": f"transformers {transformers.__version__} has no built-in config for {model_type!r}"}
    with tempfile.TemporaryDirectory() as tmp:
        native.from_pretrained(ckpt).save_pretrained(tmp)
        shipped, resaved = AutoConfig.from_pretrained(ckpt), AutoConfig.from_pretrained(tmp)
    return {
        "model_type": model_type,
        "resolved_class": f"{type(shipped).__module__}.{type(shipped).__qualname__}",
        "changed": _config_changes(shipped.to_dict(), resaved.to_dict()),
    }


def _config_changes(before: dict, after: dict, prefix: str = "") -> dict:
    """Keys whose value differs, sub-configs compared key by key; a key unset before may come back as its default."""
    out = {}
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if key in ("transformers_version", "_name_or_path") or old is None:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            out.update(_config_changes(old, new, f"{prefix}{key}."))
        elif old != new:
            out[f"{prefix}{key}"] = (old, new)
    return out


def run_step(profile, ckpt: str, parallelism: dict, lr: float, engine_overrides: dict | None = None) -> dict:
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine
    from arctic_platform.common.optimizer import OptimizerSettings

    config, rows, pad_multiple = _inputs(profile, ckpt)
    pass_labels = bool(profile.raw.get("probe", {}).get("pass_labels", False))
    model, spec, groups = _build_production_model(profile, ckpt, parallelism)
    sp = groups["sp_size"]
    optimizer = OptimizerSettings(**profile.optimizer(lr=lr))
    engine_spec = TrainingEngineSpec(
        dtype=spec.dtype,
        world_size=dist.get_world_size(),
        sequence_parallel=sp,
        optimizer=optimizer,
        **{**profile.engine(), **(engine_overrides or {})},
    )
    from deepspeed.utils import groups as ds_groups

    ds_config = build_deepspeed_config(engine_spec)
    engine, _, choice = initialize_training_engine(
        model,
        ds_config,
        optimizer=optimizer,
        ep_size=spec.parallelism.expert_parallel,
        mpu=getattr(ds_groups, "mpu", None) if sp > 1 else None,
        model_path=ckpt,
        rank=dist.get_rank(),
    )
    engine.train()
    frozen_before = {n: p.detach().clone() for n, p in engine.module.named_parameters() if not p.requires_grad}
    before = _export_hf(engine.module, ckpt)
    ids, pos, labels = audit_lib.pack_rows(rows, pad_multiple * sp)
    total = int((labels != -100).sum())
    # Every data-parallel replica sees the same row, so the reduced gradient equals the single-replica gradient.
    lp, full = _forward(engine, ids, pos, labels, pass_labels=pass_labels, groups=groups)
    chunk = ids.shape[1] // sp
    local_labels = labels[:, groups["sp_rank"] * chunk : (groups["sp_rank"] + 1) * chunk] if sp > 1 else labels
    loss = _loss(lp, local_labels, total)
    engine.backward(loss)
    from deepspeed.utils import safe_get_full_grad

    # What the optimizer consumes, after the engine's DP/EP/SP reductions. Adam's first step is close to
    # sign(grad), so a wrong reduction scale is invisible in the update and has to be checked here.
    reduced = {}
    for param in engine.module.parameters():
        grad = safe_get_full_grad(param) if param.requires_grad else None
        reduced[id(param)] = (
            grad.detach().float().reshape(param.shape)
            if grad is not None
            else torch.full_like(param.data, float("nan"), dtype=torch.float32)
        )
    with _params_hold(engine.module, reduced):
        reduced_grads = _export_hf(engine.module, ckpt)
    # Exported names backed by buffers (e.g. a router's load-balancing bias) would carry their values, not a gradient.
    with _params_hold(engine.module, {i: torch.full_like(g, float("nan")) for i, g in reduced.items()}):
        buffer_backed = {n for n, t in _export_hf(engine.module, ckpt).items() if not bool(t.isnan().all())}
    reduced_grads = {n: g for n, g in reduced_grads.items() if n not in buffer_backed}
    norm = _reduced_grad_norm(engine.module, reduced)
    # ZeRO-1/2 keep the optimizer's flat partitions (and their gradients) in the master dtype.
    partitions = getattr(engine.optimizer, "single_partition_of_fp32_groups", None) or []
    if partitions:
        grad_dtype = partitions[0].dtype
    else:
        grad_dtype = (
            torch.bfloat16 if ds_config.get("bf16", {}).get("bf16_master_weights_and_grads") else torch.float32
        )
    engine.step()
    clip_norm = engine.get_global_grad_norm()
    after = _export_hf(engine.module, ckpt)
    from deepspeed.utils import safe_get_full_fp32_param

    masters = {}
    for param in engine.module.parameters():
        master = safe_get_full_fp32_param(param) if param.requires_grad else None
        if master is not None:
            masters[id(param)] = master.detach().float().reshape(param.shape)
    with _params_hold(engine.module, masters):
        master_after = _export_hf(engine.module, ckpt)
    result = {
        "packed_logprobs": full.float().cpu(),
        "parallelism": parallelism,
        "lr": lr,
        "optimizer_choice": choice,
        "frozen_params": sorted(frozen_before),
        "frozen_changed": sorted(
            n
            for n, p in engine.module.named_parameters()
            if n in frozen_before and not torch.equal(p, frozen_before[n])
        ),
    }
    if dist.get_rank() == 0:
        result["reduced_grads"] = reduced_grads
        result["update"] = {name: after[name] - before[name] for name in after if name in before}
        # The optimizer's own copy: with fp32 masters a step below the launch dtype's resolution survives here.
        result["master_update"] = {name: master_after[name] - before[name] for name in master_after if name in before}
        max_norm = float(engine_spec.gradient_clipping or 0.0)
        result["gradient_clipping"] = max_norm
        result["reduced_grad_norm"] = norm
        result["clip_norm"] = None if clip_norm is None else float(clip_norm)
        # Scaling a low-precision gradient by a 0-dim fp32 tensor rounds the scale to the gradient's dtype first.
        exact = _clip_coefficient(norm, max_norm)
        rounded = float(torch.tensor(exact).to(grad_dtype)) if torch.finfo(grad_dtype).bits < 32 else exact
        coefficients = {"exact": exact, "grad_dtype": rounded}
        agreements = {
            kind: _adamw_agreement(
                before,
                after,
                reduced_grads,
                profile.optimizer(lr=lr),
                getattr(torch, spec.dtype or profile.dtype),
                clip,
                grad_dtype,
            )
            for kind, clip in coefficients.items()
        }
        applied = max(agreements, key=lambda kind: (sum(agreements[kind].values()), kind == "exact"))
        result["clip_coefficients"] = coefficients
        result["clip_applied"] = applied
        result["adamw_agreement"] = agreements[applied]
    return result


def _reduced_grad_norm(model, reduced: dict[int, torch.Tensor]) -> float:
    """The global L2 norm of the reduced gradient, every trainable parameter counted once.

    Replicated parameters carry the same reduced gradient on every rank; an expert-parallel parameter (DeepSpeed's
    ``allreduce=False`` with a ``group_name``) holds this rank's experts, so its square sums over its EP group.
    """
    import deepspeed.utils.groups as ds_groups

    device = torch.device("cuda", torch.cuda.current_device())
    dense = torch.zeros((), dtype=torch.float64, device=device)
    experts: dict[str, torch.Tensor] = {}
    for param in model.parameters():
        grad = reduced.get(id(param))
        if not param.requires_grad or grad is None or bool(grad.isnan().any()):
            continue
        square = grad.to(device, torch.float64).pow(2).sum()
        if hasattr(param, "group_name") and getattr(param, "allreduce", True) is False:
            experts[param.group_name] = experts.get(param.group_name, torch.zeros_like(dense)) + square
        else:
            dense += square
    for name in sorted(experts):
        dist.all_reduce(experts[name], group=ds_groups._get_expert_parallel_group(name))
        dense += experts[name]
    return float(dense.sqrt())


def _clip_coefficient(norm: float, max_norm: float) -> float:
    """DeepSpeed's global-norm clipping: the gradient is divided by ``(norm + 1e-6) / max_norm`` when that exceeds 1."""
    if max_norm <= 0:
        return 1.0
    return min(1.0, max_norm / (norm + 1e-6))


def _adamw_agreement(
    before, after, grads, opt: dict, lowp: torch.dtype, clip: float, grad_dtype: torch.dtype
) -> dict[str, float]:
    """Per tensor, the fraction of elements whose stored update equals one AdamW step on the reduced gradient.

    First step with bias correction: ``m_hat = g`` and ``v_hat = g**2``, so the step is ``g / (|g| + eps)`` on the
    clipped gradient, stored in ``grad_dtype`` (clipping only matters where ``|g|`` is near ``eps``). The expectation
    is rounded to the stored dtype the same way the weights are.
    """
    out = {}
    for name, grad in grads.items():
        if name not in before or name not in after or bool(torch.isnan(grad).any()):
            continue
        w0 = before[name]
        grad = (grad * clip).to(grad_dtype).float()
        ideal = w0 - opt["lr"] * (grad / (grad.abs() + opt["eps"]) + opt["weight_decay"] * w0)
        expected = ideal.to(lowp).float() - w0
        out[name] = float((after[name] - w0 == expected).float().mean())
    return out


def run_export(profile, ckpt: str, export_dir: str) -> dict:
    """Write the production weight streams of ``ckpt`` as ``[(name, tensor)]`` files the sampler's receiver loads."""
    from arctic_platform.model import iter_model_weights
    from arctic_platform.model import supports_weight_format

    model, _, _ = _build_production_model(profile, ckpt, None)
    paths = {}
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for fmt in ("hf", "vllm"):
            if not supports_weight_format(model, fmt):
                continue
            kwargs = {"include_skipped": False, "checkpoint_dir": ckpt} if fmt == "hf" else {}
            weights = [
                (name, tensor.detach().cpu().clone())
                for name, tensor in iter_model_weights(model, fmt, is_master=True, is_zero3=False, **kwargs)
            ]
            path = Path(export_dir, f"{fmt}.pt")
            torch.save(weights, path)
            paths[fmt] = str(path)
    return {"paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--mode", choices=("reference", "forward", "step", "export"), required=True)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--parallelism", default="{}")
    parser.add_argument("--lr", type=float, default=None, help="step mode; defaults to the profile's audit_lr")
    parser.add_argument("--engine-overrides", default="{}", help="step mode; JSON TrainingEngineSpec fields")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _init_distributed()
    profile = audit_lib.load_profile(args.profile)
    torch.manual_seed(0)
    try:
        if args.mode == "reference":
            result = run_reference(profile, args.ckpt)
        elif args.mode == "forward":
            result = run_forward(profile, args.ckpt)
        elif args.mode == "export":
            result = run_export(profile, args.ckpt, args.export_dir)
        else:
            lr = profile.audit_lr if args.lr is None else args.lr
            overrides = json.loads(args.engine_overrides)
            result = run_step(profile, args.ckpt, json.loads(args.parallelism), lr, overrides)
        status = {"ok": True}
    except Exception as err:
        result = {}
        status = {"ok": False, "error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()}
        print(status["traceback"], file=sys.stderr, flush=True)
    if dist.get_rank() == 0:
        torch.save({**result, "status": status}, args.out)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
