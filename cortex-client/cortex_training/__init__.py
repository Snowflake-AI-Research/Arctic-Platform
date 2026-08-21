"""Additive public facade for the existing ``dss_client`` implementation."""

import dss_client as _dss_client
from dss_client import *  # noqa: F401,F403
from dss_client import __version__, wire
from dss_client import NeutrinoClient, NeutrinoTrainingEngine

CortexTrainingClient = NeutrinoClient
CortexTrainingEngine = NeutrinoTrainingEngine

__all__ = [
    *_dss_client.__all__,
    "CortexTrainingClient",
    "CortexTrainingEngine",
]
