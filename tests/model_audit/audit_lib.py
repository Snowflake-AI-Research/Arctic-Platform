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
"""Model-agnostic pieces of the model audit: profiles, client job bodies, inputs, checkpoint readers, metrics.

Not collected by pytest. Imported by the tests and by the probe subprocesses, so it must not import CUDA-only or
client-only modules at import time.

A *profile* (``profiles/<name>.yaml``) is the only per-model input. It names the checkpoint, the model-catalog
configs the service launches it with, the parallel layouts to compare, the token positions where code paths switch,
and the known issues that turn a check into a strict xfail. Everything else is derived from the checkpoint's own
``config.json`` and from what the service returns.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

import yaml

AUDIT_DIR = Path(__file__).resolve().parent
PROFILES_DIR = AUDIT_DIR / "profiles"

_EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts")
# Config keys whose value is a token position where an attention or rotary code path changes.
_THRESHOLD_KEYS = ("sliding_window", "attention_chunk_size", "original_max_position_embeddings")

CHECK_IDS: dict[str, str] = {}


def register_check(check_id: str, claim: str) -> str:
    """Record a check id and the claim it tries to disprove; returns the id for use in a marker."""
    assert check_id not in CHECK_IDS, f"duplicate check id {check_id!r}"
    CHECK_IDS[check_id] = claim
    return check_id


def audit_root() -> Path:
    root = Path(os.environ.get("AP_MODEL_AUDIT_DIR") or Path(tempfile.gettempdir()) / "ap-model-audit")
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    path: Path
    raw: dict

    @property
    def checkpoint(self) -> str:
        return os.path.expandvars(self.raw["checkpoint"])

    @property
    def client(self) -> dict:
        return self.raw.get("client", {})

    @property
    def probe(self) -> dict:
        return self.raw.get("probe", {})

    @property
    def expect(self) -> dict:
        return self.raw.get("expect", {})

    @property
    def tolerances(self) -> dict:
        defaults = {
            "noise_factor": 4.0,
            "noise_slack": 2e-3,
            "adamw_mismatch_fraction": 1e-3,
            "max_lost_update_fraction": 0.5,
            "grad_norm_rel": 1e-4,
            "engine_noise": 0.0,
        }
        return {**defaults, **self.raw.get("tolerances", {})}

    @property
    def dtype(self) -> str:
        return self.launch_training().get("dtype", "bfloat16")

    @property
    def reference(self) -> str:
        """``transformers`` (the model's transformers class) or ``deepseek_inference`` (the checkpoint's own code)."""
        return self.raw.get("reference", "transformers")

    def known_issue(self, check_id: str) -> dict | None:
        return (self.raw.get("known_issues") or {}).get(check_id)

    def not_applicable(self, check_id: str) -> str | None:
        return (self.raw.get("not_applicable") or {}).get(check_id)

    def _catalog_sub_job(self, kind: str) -> dict:
        root = os.environ.get("AP_AUDIT_CATALOG")
        if not root:
            raise LookupError("AP_AUDIT_CATALOG (the model-catalog/models directory) is not set")
        path = Path(root, self.client["catalog"], self.client.get("variant", "standard"), f"{self.client[kind]}.json")
        sub_jobs = json.loads(path.read_text())["sub_job_configs"]
        assert len(sub_jobs) == 1, f"{path}: expected one sub-job config, found {len(sub_jobs)}"
        return _deep_merge(sub_jobs[0], self.client.get("overrides", {}).get(kind, {}))

    def launch_training(self) -> dict:
        """The training sub-job config the service launches this model with."""
        return self._catalog_sub_job("training") if self.client.get("training") else {}

    def launch_sampling(self) -> dict:
        return self._catalog_sub_job("sampling") if self.client.get("sampling") else {}

    def job_body(self, *, training: dict | None = None, sampling: dict | None = None) -> dict:
        """A create-job body: the launch sub-jobs with overrides; ``None`` leaves that sub-job out."""
        for kind, wanted in (("training", training), ("sampling", sampling)):
            if wanted is not None and not self.client.get(kind):
                raise LookupError(f"{self.name} names no {kind} config in client.{kind}")
        sub_jobs = []
        if training is not None:
            sub = _deep_merge(self.launch_training(), training)
            # Without these the step reports no per-parameter norms and the gradient checks have nothing to read.
            sub["training_config"]["debug"] = {"gradient_norms_per_param": True, "gradient_sample_max_numel": 0}
            sub_jobs.append(sub)
        if sampling is not None:
            sub_jobs.append(_deep_merge(self.launch_sampling(), sampling))
        return {"hardware": self.client.get("hardware", "H200"), "sub_job_configs": sub_jobs}

    def optimizer(self, lr: float | None = None) -> dict:
        """The launch optimizer; ``lr`` overrides the learning rate (the direction check uses a visible one)."""
        launch = self.launch_training().get("training_config", {}).get("optimizer", {})
        settings = {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            **{k: v for k, v in launch.items() if k != "name"},
        }
        if lr is not None:
            settings["lr"] = lr
        return settings

    @property
    def audit_lr(self) -> float:
        return float(self.probe.get("audit_lr", 1e-3))

    @property
    def layouts(self) -> list[dict]:
        """Parallel layouts: ``{id, training_config}`` overrides of the launch training config."""
        return list(self.client.get("layouts", []))

    @property
    def sync_formats(self) -> list[str]:
        return list(self.client.get("sync_formats", ["vllm", "hf"]))


def load_profile(name_or_path: str) -> Profile:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = PROFILES_DIR / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw.get("name") == path.stem, f"{path}: name must equal the file stem"
    return Profile(name=raw["name"], path=path, raw=raw)


def all_profiles() -> list[Profile]:
    return [load_profile(str(p)) for p in sorted(PROFILES_DIR.glob("*.yaml"))]


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value) if key in base else copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------------------------------


def _resolve_file(checkpoint: str, filename: str) -> str | None:
    local = Path(checkpoint, filename)
    if local.is_file():
        return str(local)
    if Path(checkpoint).is_dir():
        return None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(checkpoint, filename)
    except EntryNotFoundError:
        return None


def source_config(profile: Profile) -> dict:
    path = _resolve_file(profile.checkpoint, "config.json")
    assert path is not None, f"{profile.checkpoint} has no config.json"
    return json.loads(Path(path).read_text())


def local_checkpoint(profile: Profile) -> Path:
    """The checkpoint's config, tokenizer and weights on local disk (the HF cache for a hub id)."""
    if Path(profile.checkpoint).is_dir():
        return Path(profile.checkpoint)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(profile.checkpoint, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))


def checkpoint_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))


def text_config_of(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def read_checkpoint_tensors(path: Path) -> dict:
    from safetensors.torch import load_file

    tensors = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        tensors.update(load_file(str(file)))
    return tensors


def zero_checkpoint(path: Path) -> dict:
    """fp32 master weights and Adam moments of a DeepSpeed ZeRO-1/2 checkpoint, merged across ranks, by name."""
    import torch

    path = Path(path)
    shapes = torch.load(next(path.glob("*mp_rank_00_model_states.pt")), map_location="cpu", weights_only=False)
    shapes = shapes["param_shapes"]
    rank_files = sorted(
        path.glob("*zero_pp_rank_*_mp_rank_00_optim_states.pt"),
        key=lambda f: int(re.search(r"zero_pp_rank_(\d+)_", f.name).group(1)),
    )
    assert rank_files, f"{path} holds no ZeRO optimizer partitions"
    pieces: dict[str, dict[str, list]] = {"masters": {}, "exp_avg": {}, "exp_avg_sq": {}}
    hyper = None
    for file in rank_files:
        state = torch.load(file, map_location="cpu", weights_only=False)["optimizer_state_dict"]
        base = state["base_optimizer_state"]
        hyper = hyper or {k: v for k, v in base["param_groups"][0].items() if k != "params"}
        for group, mapping in enumerate(state["param_slice_mappings"]):
            flats = {
                "masters": state["single_partition_of_fp32_groups"][group],
                "exp_avg": base["state"][group]["exp_avg"],
                "exp_avg_sq": base["state"][group]["exp_avg_sq"],
            }
            for name, fragment in mapping.items():
                for kind, flat in flats.items():
                    pieces[kind].setdefault(name, []).append(flat[fragment.start : fragment.start + fragment.numel])
        del state
    names = {name: shape for group in shapes for name, shape in group.items()}
    out = {
        kind: {n: torch.cat(parts).view(names[n]) for n, parts in by_name.items()} for kind, by_name in pieces.items()
    }
    return {**out, "hyper": hyper}


# ---------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------


def thresholds(profile: Profile, config: dict) -> list[dict]:
    """Token positions where the model switches code paths: declared by the profile plus generic config keys."""
    text = text_config_of(config)
    found = [dict(t) for t in profile.probe.get("thresholds", [])]
    rope = text.get("rope_scaling") or text.get("rope_parameters") or {}
    for key in _THRESHOLD_KEYS:
        value = text.get(key, rope.get(key) if isinstance(rope, dict) else None)
        if isinstance(value, int) and value > 0:
            found.append({"name": key, "position": value, "reason": f"config.{key}"})
    unique = {t["position"]: t for t in found}
    return [unique[p] for p in sorted(unique)]


def position_cuts(profile: Profile) -> list[int]:
    return [t["position"] for t in thresholds(profile, source_config(profile))]


SHORT_ROW_LENGTHS = tuple(range(1, 9))


def row_lengths(profile: Profile, config: dict) -> list[int]:
    """Row lengths: short rows plus rows that straddle every threshold, capped at ``probe.max_row_len``."""
    if profile.probe.get("row_lengths"):
        return list(profile.probe["row_lengths"])
    cap = int(profile.probe.get("max_row_len", 2048))
    lengths = [7, 61, 130]
    for t in thresholds(profile, config):
        if t["position"] + 64 <= cap:
            lengths.append(t["position"] + 64)
    return lengths


def make_rows(vocab_size: int, lengths: list[int], seed: int = 0) -> list[list[int]]:
    import torch

    gen = torch.Generator().manual_seed(seed)
    low, high = min(100, vocab_size // 4), min(vocab_size, 32000)
    return [torch.randint(low, high, (n,), generator=gen).tolist() for n in lengths]


def pack_rows(rows: list[list[int]], pad_to_multiple: int = 1, pad_token_id: int = 0):
    """One packed row: ``input_ids``/``position_ids`` ``[1, T]`` and next-token ``labels`` (-100 at row ends)."""
    import torch

    ids, pos, labels = [], [], []
    for row in rows:
        ids += row
        pos += list(range(len(row)))
        labels += row[1:] + [-100]
    pad = (-len(ids)) % pad_to_multiple
    if pad:
        ids += [pad_token_id] * pad
        pos += [0] * pad
        labels += [-100] * pad
    as_tensor = lambda x: torch.tensor([x], dtype=torch.long)  # noqa: E731
    return as_tensor(ids), as_tensor(pos), as_tensor(labels)


def pack_logprobs(rows: list[list[int]], per_row, pad_to_multiple: int = 1):
    """Per-row log-probs laid out like ``pack_rows(rows)``, zero at row ends and padding."""
    import torch

    flat = []
    for row, lp in zip(rows, per_row):
        lp = torch.as_tensor(lp, dtype=torch.float32)[: len(row)].clone()
        lp[len(row) - 1 :] = 0.0
        flat.append(lp)
    packed = torch.cat(flat)
    pad = (-packed.numel()) % pad_to_multiple
    return torch.cat([packed, torch.zeros(pad)]).unsqueeze(0)


def probe_inputs(profile: Profile, ckpt: str) -> tuple[dict, list[list[int]], int]:
    """The model config, the audit rows and the packing multiple; identical in every probe."""
    config = json.loads(Path(ckpt, "config.json").read_text())
    rows = make_rows(int(text_config_of(config)["vocab_size"]), row_lengths(profile, config), seed=0)
    return config, rows, int(profile.probe.get("pad_to_multiple", 8))


def row_slices(rows: list[list[int]]) -> list[slice]:
    out, start = [], 0
    for row in rows:
        out.append(slice(start, start + len(row)))
        start += len(row)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# One optimizer step, as the service's exports record it
# ---------------------------------------------------------------------------------------------------------------------


def clip_coefficient(norm: float, max_norm: float) -> float:
    """DeepSpeed's global-norm clipping: the gradient is divided by ``(norm + 1e-6) / max_norm`` when that exceeds 1."""
    if max_norm <= 0:
        return 1.0
    return min(1.0, max_norm / (norm + 1e-6))


def adamw_agreement(before, after, grads, opt: dict, lowp, clip: float, grad_dtype) -> dict[str, float]:
    """Per tensor, the fraction of elements whose stored update equals one AdamW step on the gradient.

    First step with bias correction: ``m_hat = g`` and ``v_hat = g**2``, so the step is ``g / (|g| + eps)`` on the
    clipped gradient, stored in ``grad_dtype``. The expectation is rounded to the stored dtype the same way the
    weights are.
    """
    import torch

    out = {}
    for name, grad in grads.items():
        if name not in before or name not in after or bool(torch.isnan(grad).any()):
            continue
        w0 = before[name].float()
        grad = (grad * clip).to(grad_dtype).float()
        ideal = w0 - opt["lr"] * (grad / (grad.abs() + opt["eps"]) + opt["weight_decay"] * w0)
        expected = ideal.to(lowp).float() - w0
        out[name] = float((after[name].float() - w0 == expected).float().mean())
    return out


def client_step(profile: Profile, run: dict) -> dict:
    """What one service step did, from the step reply and the weights-only and resumable exports around it.

    ZeRO keeps Adam's first moment in fp32, and after one step it is ``(1 - beta1)`` times the clipped gradient the
    optimizer consumed; dividing by the clip coefficient of the norm the engine reported gives the reduced gradient.
    """
    import torch

    reply, exports = run["step"]["reply"], run["exports"]
    before = read_checkpoint_tensors(Path(exports["initial"]))
    after = read_checkpoint_tensors(Path(exports["after_step"]))
    zero = zero_checkpoint(Path(exports["resumable"]))
    hyper = zero["hyper"]
    consumed = {n: m / (1 - hyper["betas"][0]) for n, m in zero["exp_avg"].items()}
    norms = reply.get("gradient_norms_per_param") or {}
    max_norm = float(run["job"]["training"]["training_config"].get("gradient_clipping") or 0.0)
    clip_norm = reply.get("grad_norm")
    exact = clip_coefficient(float(clip_norm), max_norm) if clip_norm is not None else 1.0
    reduced_norm = math.sqrt(sum(float(v) ** 2 for v in norms.values())) if norms else None
    consumed_norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in consumed.values()))
    opt = {"lr": float(run["step"]["lr"]), "eps": hyper["eps"], "weight_decay": hyper["weight_decay"]}
    lowp = getattr(torch, profile.dtype)
    return {
        "lr": opt["lr"],
        "loss": run["step"].get("loss"),
        "packed_logprobs": run["forward"]["packed_logprobs"],
        "reduced_grads": {n: g / exact for n, g in consumed.items()},
        "gradient_norms_per_param": norms,
        "update": {n: after[n].float() - before[n].float() for n in after if n in before},
        "master_update": {n: w - before[n].float() for n, w in zero["masters"].items() if n in before},
        "frozen_params": sorted(set(before) - set(consumed)),
        "frozen_changed": sorted(
            n for n in set(before) - set(consumed) if n in after and not torch.equal(after[n], before[n])
        ),
        "gradient_clipping": max_norm,
        "clip_norm": None if clip_norm is None else float(clip_norm),
        "reduced_grad_norm": reduced_norm,
        "clip_coefficients": {"exact": exact, "grad_dtype": float(torch.tensor(exact).to(lowp))},
        "clip_applied_measured": consumed_norm / reduced_norm if reduced_norm else None,
        "adamw_agreement": adamw_agreement(before, after, consumed, opt, lowp, 1.0, torch.float32),
        "optimizer_hyper": hyper,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------


def k3(actual, expected) -> float:
    """k3 KL estimate between two per-token log-prob tensors (``actual`` playing the trainer)."""
    d = (actual.double() - expected.double()).flatten()
    d = d[~d.isnan()]
    return float((d.exp() - 1 - d).mean()) if d.numel() else 0.0


def logprob_errors(actual, expected, mask=None) -> dict:
    d = (actual.double() - expected.double()).abs()
    if mask is not None:
        d = d[mask]
    d = d[~d.isnan()]
    if d.numel() == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "n": 0}
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "n": int(d.numel())}


def bucketed_errors(actual, expected, positions, labels, cuts: list[int]) -> dict[str, dict]:
    """Errors per position bucket split at ``cuts``, so an error that starts past a threshold is not diluted."""
    valid = labels != -100
    edges = [0] + sorted(set(cuts)) + [math.inf]
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = valid & (positions >= lo) & (positions < hi)
        if bool(mask.any()):
            out[f"[{lo},{hi})"] = {**logprob_errors(actual, expected, mask), "k3": k3(actual[mask], expected[mask])}
    return out


def relative_l2(actual, expected) -> float:
    a, e = actual.double().flatten(), expected.double().flatten()
    denom = float(e.norm())
    return float((a - e).norm()) / denom if denom > 0 else float(a.norm())


def within_noise(error: float, floor: float, profile: Profile) -> bool:
    """``error`` is at most ``noise_factor`` times the reference's own bf16-vs-fp32 error, plus a small slack."""
    tol = profile.tolerances
    return error <= tol["noise_factor"] * floor + tol["noise_slack"]


def matches(pattern_entries: list[dict], name: str) -> dict | None:
    for entry in pattern_entries or []:
        if re.search(entry["pattern"], name):
            return entry
    return None


_EXPERT_INDEX = re.compile(r"(\.experts\.)\d+(\.)")
_EXPERT_SLICE = re.compile(r"\[\d+\]$")


def _expert_count(profile: Profile) -> int | None:
    text = text_config_of(source_config(profile))
    return next((int(text[k]) for k in _EXPERT_COUNT_KEYS if text.get(k)), None)


def _split_fused_experts(tensors: dict, num_experts: int | None) -> dict:
    """Fused routed-expert tensors (under ``.experts.``, leading dim the expert count) as one entry per expert."""
    if not num_experts:
        return tensors
    out = {}
    for name, tensor in tensors.items():
        fused = ".experts." in name and _EXPERT_INDEX.search(name) is None
        if fused and tensor.dim() >= 3 and tensor.shape[0] == num_experts:
            out.update({f"{name}[{e}]": tensor[e] for e in range(num_experts)})
        else:
            out[name] = tensor
    return out


def _expert_group(name: str) -> str | None:
    """The layer-and-projection key a routed expert's tensor shares with its siblings; None for other tensors."""
    group = _EXPERT_SLICE.sub("", _EXPERT_INDEX.sub(r"\1N\2", name))
    return group if group != name else None


def _lower_quartile(values: list[float]) -> float:
    return statistics.quantiles(values, n=4, method="inclusive")[0]


def assert_gradients_match(profile: Profile, actual: dict, reference: dict, record) -> None:
    """Every reference gradient has a production gradient of the same shape within the per-tensor noise floor."""
    import torch

    experts = _expert_count(profile)
    truth, floor = (_split_fused_experts(reference[k], experts) for k in ("fp32_grads", "lowp_grads"))
    mass = _split_fused_experts(reference.get("fp32_grad_mass", {}), experts)
    actual = _split_fused_experts(actual, experts)
    allowed = profile.expect.get("no_grad_params", [])
    missing = sorted(n for n in truth if n not in actual and matches(allowed, n) is None)
    # Low-precision noise per unit of summed |row gradient|. A tensor whose rows cancel (e.g. a 3-element gate
    # scale) can land on a lucky floor; the median rate across tensors bounds how lucky.
    rate = {
        n: float((floor[n] - truth[n]).norm() / mass[n].norm())
        for n in truth
        if n in mass and n in floor and bool(mass[n].abs().sum() > 0)
    }
    typical_rate = statistics.median(rate.values()) if rate else 0.0
    # Routed experts: a token whose top-k flips at a near-tie moves a whole token's gradient between experts, and
    # which tokens flip under low precision is luck. Each expert is held to the worst floor among its siblings in the
    # layer (the reference's own flips set it), and each layer's lower-quartile expert to the lower-quartile floor:
    # flips touch only some experts, while an error in the expert gradient math touches all of them. Fused expert
    # tensors are compared per expert slice.
    siblings: dict[str, list[float]] = {}
    for name, expected in truth.items():
        group = _expert_group(name)
        if group and name in floor and bool(expected.abs().sum() > 0):
            siblings.setdefault(group, []).append(relative_l2(floor[name], expected))
    expert_errors: dict[str, list[float]] = {}
    rows, bad = [], []
    for name, expected in truth.items():
        if name not in actual or matches(allowed, name):
            continue
        got = actual[name]
        if got.shape != expected.shape:
            bad.append((name, f"shape {tuple(got.shape)} != {tuple(expected.shape)}"))
        elif not bool(expected.abs().sum() > 0):
            if bool(torch.nan_to_num(got).abs().sum() > 0):
                bad.append((name, "gradient where the reference has none"))
        elif bool(torch.isnan(got).all()):
            bad.append((name, "no gradient"))
        else:
            err, noise = relative_l2(got, expected), relative_l2(floor[name], expected)
            if name in rate:
                noise = max(noise, typical_rate * float(mass[name].norm() / expected.norm()))
            group = _expert_group(name)
            if group and len(siblings.get(group, ())) >= 4:
                noise = max(noise, max(siblings[group]))
                expert_errors.setdefault(group, []).append(err)
            rows.append((err, noise, name))
            if not within_noise(err, noise, profile):
                bad.append((name, f"rel_l2={err:.4g} floor={noise:.4g}"))
    for group, errs in expert_errors.items():
        low_err, low_floor = _lower_quartile(errs), _lower_quartile(siblings[group])
        if not within_noise(low_err, low_floor, profile):
            bad.append((group, f"lower quartile of {len(errs)} experts rel_l2={low_err:.4g} floor={low_floor:.4g}"))
    rows.sort(reverse=True)
    record(worst=[(n, round(e, 5), round(f, 5)) for e, f, n in rows[:15]], missing=missing[:30], compared=len(rows))
    assert not missing, f"reference gradients with no production gradient: {missing[:20]}"
    assert not bad, f"gradients beyond the noise floor: {bad[:20]}"


def assert_clip_norm_is_reduced_norm(profile: Profile, step: dict, record) -> None:
    """The global gradient norm the engine clips by is the norm of the gradient it reduced."""
    engine, reduced = step.get("clip_norm"), step["reduced_grad_norm"]
    record(clip_norm=engine, reduced_grad_norm=reduced, gradient_clipping=step.get("gradient_clipping"))
    assert engine is not None, "the engine reported no global gradient norm"
    rel = abs(engine - reduced) / max(reduced, 1e-30)
    assert (
        rel <= profile.tolerances["grad_norm_rel"]
    ), f"the engine's global gradient norm {engine:.6g} differs from the reduced gradient's {reduced:.6g} by {rel:.3%}"


def assert_update_is_adamw(profile: Profile, step: dict, record) -> None:
    """The stored update of every tensor is one AdamW step on the engine's reduced gradient, element for element."""
    allowed = profile.expect.get("no_grad_params", [])
    agreement = {n: f for n, f in step["adamw_agreement"].items() if matches(allowed, n) is None}
    limit = 1.0 - profile.tolerances["adamw_mismatch_fraction"]
    worst = sorted((f, n) for n, f in agreement.items())
    record(lr=step["lr"], compared=len(agreement), worst=[(n, round(f, 6)) for f, n in worst[:15]])
    assert agreement, "the step reported no tensors to compare"
    bad = [(n, round(f, 5)) for f, n in worst if f < limit]
    assert (
        not bad
    ), f"tensors whose update is not AdamW of the reduced gradient (fraction of elements agreeing): {bad[:20]}"
