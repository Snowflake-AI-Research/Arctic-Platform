from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


CLIENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLIENT_ROOT.parent
sys.path.insert(0, str(CLIENT_ROOT / "scripts"))

from validate_model_catalog import (  # noqa: E402
    CatalogValidationError,
    load_catalog,
    load_and_validate,
    validate_catalog,
)


CONFIG_DIR = CLIENT_ROOT / "configs" / "cortex-training"


def test_checked_in_catalog_is_valid():
    summary = load_and_validate(CONFIG_DIR, REPO_ROOT)

    assert summary == {
        "schemaVersion": 1,
        "lastReviewed": "2026-08-19",
        "models": 9,
        "profiles": 23,
    }


def test_duplicate_model_id_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"].append(copy.deepcopy(models_doc["models"][0]))

    with pytest.raises(CatalogValidationError, match="duplicate model id"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_missing_recommended_profile_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"][0]["capabilities"]["inference"]["recommendedProfileId"] = (
        "missing"
    )

    with pytest.raises(CatalogValidationError, match="unknown profile missing"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_profile_context_cannot_exceed_capability_limit():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "dense-sft-full-8gpu")
    profile["subJobs"][0]["args"]["max_seq_len"] = 16384

    with pytest.raises(CatalogValidationError, match="exceeds max context 8192"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_unsupported_capability_cannot_carry_configuration():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    capability = models_doc["models"][-1]["capabilities"]["training"]
    capability["profiles"] = copy.deepcopy(
        models_doc["models"][0]["capabilities"]["training"]["profiles"]
    )

    with pytest.raises(
        CatalogValidationError,
        match="unsupported capabilities cannot define profiles",
    ):
        validate_catalog(models_doc, profiles, REPO_ROOT)
