# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Arctic on-policy distillation client."""

from arctic_platform.opd.client import ArcticOPDClient
from arctic_platform.opd.client import DEFAULT_PROCESSING
from arctic_platform.opd.client import create_arctic_opd_client
from arctic_platform.opd.config import ArcticOPDClientConfig
from arctic_platform.opd.scoring import score_teacher

__all__ = [
    "ArcticOPDClient",
    "ArcticOPDClientConfig",
    "DEFAULT_PROCESSING",
    "create_arctic_opd_client",
    "score_teacher",
]
