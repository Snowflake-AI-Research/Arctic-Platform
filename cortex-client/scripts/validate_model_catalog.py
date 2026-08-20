#!/usr/bin/env python3
"""Validate the Cortex Training model catalog and recommended profiles."""

from __future__ import annotations

import argparse
import ast
import copy
from datetime import date
import json
from pathlib import Path
from typing import Any

CLIENT_ROOT = Path(__file__).resolve().parents[1]


CAPABILITY_KEYS = ("inference", "training")
TRAINING_PROFILE_KEYS = ("sftLora", "sftFull", "rlLora", "rlFull")
CAPABILITY_PROFILE_TYPES = {
    "inference": ("inference", "none"),
    "sftLora": ("sft", "lora"),
    "sftFull": ("sft", "full"),
    "rlLora": ("rl", "lora"),
    "rlFull": ("rl", "full"),
}
PROFILE_BUILDERS = {"training_job", "sampling_job"}
SUPPORTED_ZERO_STAGES = {0, 1, 2}


class CatalogValidationError(ValueError):
    """Raised when the catalog violates its cross-file contract."""


def _load_builder_signatures(client_root: Path) -> dict[str, tuple[set[str], set[str]]]:
    source_path = client_root / "dss_client" / "neutrino_client.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise CatalogValidationError(f"cannot inspect {source_path}: {exc}") from exc

    signatures: dict[str, tuple[set[str], set[str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "SubJobConfig":
            continue
        for member in node.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if member.name not in PROFILE_BUILDERS:
                continue
            allowed = {argument.arg for argument in member.args.kwonlyargs}
            required = {
                argument.arg
                for argument, default in zip(
                    member.args.kwonlyargs, member.args.kw_defaults
                )
                if default is None
            }
            signatures[member.name] = (allowed, required)

    missing = PROFILE_BUILDERS - set(signatures)
    if missing:
        raise CatalogValidationError(
            f"SubJobConfig is missing expected builder methods: {sorted(missing)}"
        )
    return signatures


def _require_dict(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogValidationError(f"{location} must be an object")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise CatalogValidationError(f"{location} must be a non-empty string")
    return value


def _require_positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogValidationError(f"{location} must be a positive integer")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"cannot read {path}: {exc}") from exc
    return _require_dict(value, str(path))


def load_catalog(config_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models_doc = _load_json(config_dir / "models.json")
    profiles: list[dict[str, Any]] = []
    profile_dir = config_dir / "profiles"
    for path in sorted(profile_dir.glob("*.json")):
        profile_doc = _load_json(path)
        values = profile_doc.get("profiles")
        if not isinstance(values, list):
            raise CatalogValidationError(f"{path}.profiles must be an array")
        for index, profile in enumerate(values):
            profile_obj = _require_dict(profile, f"{path}.profiles[{index}]")
            profile_obj = copy.deepcopy(profile_obj)
            profile_obj["_sourceFile"] = str(path.relative_to(config_dir))
            profiles.append(profile_obj)
    if not profiles:
        raise CatalogValidationError(f"{profile_dir} must contain at least one profile")
    return models_doc, profiles


def _validate_profile_shape(profile: dict[str, Any], repo_root: Path) -> None:
    profile_id = _require_string(profile.get("id"), "profile.id")
    workflow = profile.get("workflow")
    tuning_method = profile.get("tuningMethod")
    if workflow not in {"inference", "sft", "rl"}:
        raise CatalogValidationError(
            f"profile {profile_id}: invalid workflow {workflow!r}"
        )
    if tuning_method not in {"none", "lora", "full"}:
        raise CatalogValidationError(
            f"profile {profile_id}: invalid tuningMethod {tuning_method!r}"
        )
    if workflow == "inference" and tuning_method != "none":
        raise CatalogValidationError(
            f"profile {profile_id}: inference must use tuningMethod 'none'"
        )
    if workflow != "inference" and tuning_method == "none":
        raise CatalogValidationError(
            f"profile {profile_id}: training workflow requires lora or full"
        )

    _require_string(profile.get("summary"), f"profile {profile_id}.summary")
    evidence_paths = profile.get("evidencePaths")
    if not isinstance(evidence_paths, list) or not evidence_paths:
        raise CatalogValidationError(
            f"profile {profile_id}.evidencePaths must be a non-empty array"
        )
    for index, relative_path in enumerate(evidence_paths):
        value = _require_string(
            relative_path, f"profile {profile_id}.evidencePaths[{index}]"
        )
        if not (repo_root / value).is_file():
            raise CatalogValidationError(
                f"profile {profile_id}: evidence path does not exist: {value}"
            )

    recipe_path = profile.get("recipePath")
    if recipe_path is not None:
        recipe = _require_string(recipe_path, f"profile {profile_id}.recipePath")
        if not (repo_root / recipe).is_file():
            raise CatalogValidationError(
                f"profile {profile_id}: recipe path does not exist: {recipe}"
            )

    sub_jobs = profile.get("subJobs")
    if not isinstance(sub_jobs, list) or not sub_jobs:
        raise CatalogValidationError(
            f"profile {profile_id}.subJobs must be a non-empty array"
        )
    builders: list[str] = []
    for index, sub_job in enumerate(sub_jobs):
        item = _require_dict(sub_job, f"profile {profile_id}.subJobs[{index}]")
        builder = item.get("builder")
        if builder not in PROFILE_BUILDERS:
            raise CatalogValidationError(
                f"profile {profile_id}: invalid builder {builder!r}"
            )
        args = _require_dict(
            item.get("args"), f"profile {profile_id}.subJobs[{index}].args"
        )
        if "model_name" in args:
            raise CatalogValidationError(
                f"profile {profile_id}: model_name is injected from the model and must not appear in args"
            )
        builders.append(builder)

    expected_builders = {
        "inference": ["sampling_job"],
        "sft": ["training_job"],
        "rl": ["training_job", "sampling_job"],
    }[workflow]
    if sorted(builders) != sorted(expected_builders):
        raise CatalogValidationError(
            f"profile {profile_id}: workflow {workflow} requires builders {expected_builders}"
        )

    has_peft = []
    for sub_job in sub_jobs:
        args = sub_job["args"]
        extra = args.get("extra_training") or args.get("extra_sampling") or {}
        has_peft.append("peft_config" in extra)
    if tuning_method == "lora" and not all(has_peft):
        raise CatalogValidationError(
            f"profile {profile_id}: LoRA profiles require peft_config on every sub-job"
        )
    if tuning_method == "full" and any(has_peft):
        raise CatalogValidationError(
            f"profile {profile_id}: full profiles must not define peft_config"
        )


def _validate_profile_reference(
    *,
    model_id: str,
    profile_key: str,
    recommendation: dict[str, Any],
    profile_by_id: dict[str, dict[str, Any]],
    builder_signatures: dict[str, tuple[set[str], set[str]]],
) -> str:
    max_context = _require_positive_int(
        recommendation.get("maxContextTokens"),
        f"model {model_id}.{profile_key}.maxContextTokens",
    )
    profile_id = _require_string(
        recommendation.get("recommendedProfileId"),
        f"model {model_id}.{profile_key}.recommendedProfileId",
    )
    validated = recommendation.get("lastValidated")
    if validated is not None:
        validated = _require_string(
            validated,
            f"model {model_id}.{profile_key}.lastValidated",
        )
        try:
            date.fromisoformat(validated)
        except ValueError as exc:
            raise CatalogValidationError(
                f"model {model_id}.{profile_key}.lastValidated must be an ISO date"
            ) from exc
    profile = profile_by_id.get(profile_id)
    if profile is None:
        raise CatalogValidationError(
            f"model {model_id}.{profile_key}: unknown profile {profile_id}"
        )
    expected_workflow, expected_tuning = CAPABILITY_PROFILE_TYPES[profile_key]
    if (profile["workflow"], profile["tuningMethod"]) != (
        expected_workflow,
        expected_tuning,
    ):
        raise CatalogValidationError(
            f"model {model_id}.{profile_key}: profile {profile_id} has incompatible "
            f"workflow/tuningMethod"
        )

    for sub_job in profile["subJobs"]:
        args = copy.deepcopy(sub_job["args"])
        builder = sub_job["builder"]
        profile_max_seq_len = _require_positive_int(
            args.get("max_seq_len"),
            f"profile {profile_id}.{builder}.max_seq_len",
        )
        if profile_max_seq_len > max_context:
            raise CatalogValidationError(
                f"model {model_id}.{profile_key}: profile {profile_id} max_seq_len "
                f"{profile_max_seq_len} exceeds max context {max_context}"
            )

        allowed, required = builder_signatures[builder]
        unexpected = sorted(set(args) - allowed)
        missing = sorted(required - set(args))
        if unexpected or missing:
            raise CatalogValidationError(
                f"model {model_id}.{profile_key}: profile {profile_id} does not match "
                f"SubJobConfig.{builder}; unexpected={unexpected}, missing={missing}"
            )
        n_gpus = _require_positive_int(
            args.get("n_gpus"),
            f"profile {profile_id}.{builder}.n_gpus",
        )
        if n_gpus % 8 != 0:
            raise CatalogValidationError(
                f"profile {profile_id}.{builder}.n_gpus must be a multiple of 8"
            )
        if builder == "sampling_job":
            extra_sampling = _require_dict(
                args.get("extra_sampling"),
                f"profile {profile_id}.sampling_job.extra_sampling",
            )
            vllm_config = _require_dict(
                extra_sampling.get("vllm_config"),
                f"profile {profile_id}.sampling_job.extra_sampling.vllm_config",
            )
            tensor_parallel_size = _require_positive_int(
                vllm_config.get("tensor_parallel_size"),
                f"profile {profile_id}.sampling_job.tensor_parallel_size",
            )
            if tensor_parallel_size > n_gpus or n_gpus % tensor_parallel_size != 0:
                raise CatalogValidationError(
                    f"profile {profile_id}.sampling_job.tensor_parallel_size "
                    f"must be a divisor of n_gpus ({n_gpus})"
                )
        if builder == "training_job":
            optimizer = args.get("optimizer")
            if not isinstance(optimizer, dict) or not optimizer:
                raise CatalogValidationError(
                    f"profile {profile_id}.training_job.optimizer must be a non-empty object"
                )
            _require_positive_int(
                args.get("train_batch_size"),
                f"profile {profile_id}.training_job.train_batch_size",
            )
            extra_training = _require_dict(
                args.get("extra_training"),
                f"profile {profile_id}.training_job.extra_training",
            )
            ds_config = _require_dict(
                extra_training.get("ds_config"),
                f"profile {profile_id}.training_job.extra_training.ds_config",
            )
            zero_optimization = _require_dict(
                ds_config.get("zero_optimization"),
                f"profile {profile_id}.training_job.extra_training.ds_config.zero_optimization",
            )
            zero_stage = zero_optimization.get("stage")
            if (
                isinstance(zero_stage, bool)
                or not isinstance(zero_stage, int)
                or zero_stage not in SUPPORTED_ZERO_STAGES
            ):
                raise CatalogValidationError(
                    f"profile {profile_id}.training_job ZeRO stage {zero_stage!r} "
                    f"is unsupported; supported stages are {sorted(SUPPORTED_ZERO_STAGES)}"
                )
    return profile_id


def validate_catalog(
    models_doc: dict[str, Any],
    profiles: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    builder_signatures = _load_builder_signatures(repo_root / "cortex-client")

    if models_doc.get("schemaVersion") != 1:
        raise CatalogValidationError("models.json.schemaVersion must equal 1")
    if models_doc.get("catalog") != "cortex-training":
        raise CatalogValidationError("models.json.catalog must equal 'cortex-training'")
    reviewed = _require_string(
        models_doc.get("lastReviewed"), "models.json.lastReviewed"
    )
    try:
        date.fromisoformat(reviewed)
    except ValueError as exc:
        raise CatalogValidationError(
            "models.json.lastReviewed must be an ISO date"
        ) from exc

    profile_by_id: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        _validate_profile_shape(profile, repo_root)
        profile_id = profile["id"]
        if profile_id in profile_by_id:
            raise CatalogValidationError(f"duplicate profile id: {profile_id}")
        profile_by_id[profile_id] = profile

    models = models_doc.get("models")
    if not isinstance(models, list) or not models:
        raise CatalogValidationError("models.json.models must be a non-empty array")

    model_ids: set[str] = set()
    referenced_profiles: set[str] = set()
    for model_index, model in enumerate(models):
        item = _require_dict(model, f"models[{model_index}]")
        _require_string(item.get("name"), f"models[{model_index}].name")
        model_id = _require_string(
            item.get("modelId"), f"models[{model_index}].modelId"
        )
        if model_id in model_ids:
            raise CatalogValidationError(f"duplicate model id: {model_id}")
        model_ids.add(model_id)

        capabilities = _require_dict(
            item.get("capabilities"), f"model {model_id}.capabilities"
        )
        if set(capabilities) != set(CAPABILITY_KEYS):
            raise CatalogValidationError(
                f"model {model_id}: capabilities must be exactly {list(CAPABILITY_KEYS)}"
            )

        inference = _require_dict(
            capabilities["inference"],
            f"model {model_id}.capabilities.inference",
        )
        inference_supported = inference.get("supported")
        if not isinstance(inference_supported, bool):
            raise CatalogValidationError(
                f"model {model_id}.inference.supported must be a boolean"
            )
        if inference_supported:
            referenced_profiles.add(
                _validate_profile_reference(
                    model_id=model_id,
                    profile_key="inference",
                    recommendation=inference,
                    profile_by_id=profile_by_id,
                    builder_signatures=builder_signatures,
                )
            )
        else:
            unexpected = {
                "maxContextTokens",
                "recommendedProfileId",
                "lastValidated",
            } & set(inference)
            if unexpected:
                raise CatalogValidationError(
                    f"model {model_id}.inference: unsupported capabilities cannot define "
                    f"{sorted(unexpected)}"
                )

        training = _require_dict(
            capabilities["training"],
            f"model {model_id}.capabilities.training",
        )
        training_supported = training.get("supported")
        if not isinstance(training_supported, bool):
            raise CatalogValidationError(
                f"model {model_id}.training.supported must be a boolean"
            )
        if not training_supported:
            if "profiles" in training:
                raise CatalogValidationError(
                    f"model {model_id}.training: unsupported capabilities cannot define profiles"
                )
            continue
        if not inference_supported:
            raise CatalogValidationError(
                f"model {model_id}: training support requires inference support"
            )

        training_profiles = _require_dict(
            training.get("profiles"),
            f"model {model_id}.training.profiles",
        )
        if set(training_profiles) != set(TRAINING_PROFILE_KEYS):
            raise CatalogValidationError(
                f"model {model_id}.training.profiles must be exactly "
                f"{list(TRAINING_PROFILE_KEYS)}"
            )
        for profile_key in TRAINING_PROFILE_KEYS:
            recommendation = _require_dict(
                training_profiles[profile_key],
                f"model {model_id}.training.profiles.{profile_key}",
            )
            referenced_profiles.add(
                _validate_profile_reference(
                    model_id=model_id,
                    profile_key=profile_key,
                    recommendation=recommendation,
                    profile_by_id=profile_by_id,
                    builder_signatures=builder_signatures,
                )
            )

    unreferenced = sorted(set(profile_by_id) - referenced_profiles)
    if unreferenced:
        raise CatalogValidationError(f"unreferenced profiles: {unreferenced}")

    return {
        "schemaVersion": models_doc["schemaVersion"],
        "lastReviewed": reviewed,
        "models": len(models),
        "profiles": len(profiles),
    }


def load_and_validate(config_dir: Path, repo_root: Path) -> dict[str, Any]:
    models_doc, profiles = load_catalog(config_dir)
    return validate_catalog(models_doc, profiles, repo_root)


def main() -> int:
    default_config = CLIENT_ROOT / "configs" / "cortex-training"
    default_repo = CLIENT_ROOT.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=default_config)
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    args = parser.parse_args()

    summary = load_and_validate(args.config_dir.resolve(), args.repo_root.resolve())
    print(
        "Cortex Training catalog valid: "
        f"{summary['models']} models, {summary['profiles']} profiles, "
        f"schema v{summary['schemaVersion']}, reviewed {summary['lastReviewed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
