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
"""Local reference of the model audit, on one GPU. Launched by the tests; not collected by pytest.

The model's transformers class (or the checkpoint's own inference code) on ``--ckpt`` in fp32 and in the launch
dtype, one row per call: per-token log-probs and gradients under checkpoint names. The fp32 run (eager attention) is
the truth; the launch-dtype run, with the launch's attention implementation where it loads here, measures how far
plain dtype noise moves the same math (the noise floor). Writes one ``torch.save`` dict to ``--out``.
"""

from __future__ import annotations

import argparse
import functools
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_lib  # noqa: E402


def _logprobs_from_output(output, labels: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(output):
        output = {"logits": output}
    get = output.get if hasattr(output, "get") else (lambda key, default=None: getattr(output, key, default))
    logits = get("logits", None)
    assert logits is not None, f"model output has no logits: {list(output.keys())}"
    safe = labels.clamp(min=0)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return lp.masked_fill(labels == -100, 0.0)


def _loss(lp: torch.Tensor, labels: torch.Tensor, total_targets: int) -> torch.Tensor:
    mask = (labels.to(lp.device) != -100).float()
    return -(lp * mask).sum() / total_targets


def _checkpoint_named(model, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map transformers runtime names (fused experts, renamed modules) back to checkpoint names, as saving does."""
    from transformers.core_model_loading import revert_weight_conversion

    return {name: tensor.float() for name, tensor in revert_weight_conversion(model, dict(tensors)).items()}


def _load(cls, ckpt: str, dtype: torch.dtype, attentions: list[str]):
    """The model with the first attention implementation that loads here, and its name."""
    for attn in attentions:
        try:
            return cls.from_pretrained(ckpt, dtype=dtype, attn_implementation=attn).cuda(), attn
        except (ImportError, ValueError) as err:
            print(f"attn_implementation={attn} unavailable: {err}", file=sys.stderr, flush=True)
    raise RuntimeError(f"none of {attentions} loads {ckpt}")


def run_reference(profile, ckpt: str) -> dict:
    import transformers

    config, rows, pad_multiple = audit_lib.probe_inputs(profile, ckpt)
    ids, pos, labels = audit_lib.pack_rows(rows, pad_multiple)
    total = int((labels != -100).sum())
    arch = config["architectures"][0]
    cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
    result = {"ids": ids, "pos": pos, "labels": labels, "rows": rows}
    # Fused bf16 attention kernels move gradients several times further from fp32 than eager bf16 does, so the floor
    # runs the launch's kernel; the fp32 truth stays eager.
    launch_attn = profile.launch_training().get("training_config", {}).get("attn_implementation")
    attentions = {"fp32": ["eager"], "lowp": [a for a in (launch_attn, "sdpa", "eager") if a]}
    for tag, dtype in (("fp32", torch.float32), ("lowp", getattr(torch, profile.dtype))):
        if profile.reference == "deepseek_inference":
            import deepseek_inference

            model = deepseek_inference.Reference(profile.checkpoint, ckpt, dtype, max(len(r) for r in rows))

            def row_logprobs(sl, model=model):
                return _logprobs_from_output(model.logits(ids[:, sl]), labels[:, sl].cuda())

            def published(grads):
                return grads

        else:
            model, result[f"{tag}_attn_implementation"] = _load(cls, ckpt, dtype, attentions[tag])
            model.train()

            def row_logprobs(sl, model=model):
                device = torch.device("cuda")
                output = model(input_ids=ids[:, sl].to(device), position_ids=pos[:, sl].to(device), use_cache=False)
                return _logprobs_from_output(output, labels[:, sl].to(device))

            published = functools.partial(_checkpoint_named, model)
        # One sequence per call: not every transformers model masks at position resets, and a packed reference would
        # then attend across rows.
        lp = torch.zeros(ids.shape, dtype=torch.float32)
        params = dict(model.named_parameters())
        seen = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
        # Sum over rows of |row gradient|: low-precision noise scales with this, not with the (possibly cancelling)
        # total.
        mass = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
        for sl in audit_lib.row_slices(rows):
            row_lp = row_logprobs(sl)
            _loss(row_lp, labels[:, sl], total).backward()
            lp[:, sl] = row_lp.detach().float().cpu()
            for n, p in params.items():
                if p.grad is not None:
                    mass[n] += (p.grad.float() - seen[n]).abs()
                    seen[n] = p.grad.float().clone()
        del seen
        result[f"{tag}_logprobs"] = lp
        result[f"{tag}_grad_mass"] = published({n: m.cpu() for n, m in mass.items()})
        del mass
        result[f"{tag}_grads"] = published(
            {
                n: (p.grad.detach() if p.grad is not None else torch.zeros_like(p)).float().cpu()
                for n, p in model.named_parameters()
            }
        )
        result[f"{tag}_param_names"] = sorted(result[f"{tag}_grads"])
        del model
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    profile = audit_lib.load_profile(args.profile)
    torch.manual_seed(0)
    try:
        result = run_reference(profile, args.ckpt)
        status = {"ok": True}
    except Exception as err:
        result = {}
        status = {"ok": False, "error": f"{type(err).__name__}: {err}", "traceback": traceback.format_exc()}
        print(status["traceback"], file=sys.stderr, flush=True)
    torch.save({**result, "status": status}, args.out)


if __name__ == "__main__":
    main()
