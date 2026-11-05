"""Compatibility re-export of shared native-FP8 helpers.

Implementation: ``arctic_platform.model.implementations.fp8``. GLM-internal call sites can
keep this path; the DeepSpeed worker imports the shared module so it does not
depend on glm52.
"""

from arctic_platform.model.implementations.fp8 import *  # noqa: F403
from arctic_platform.model.implementations.fp8 import (  # noqa: F401
    _pad_tokens_for_deepgemm,
    _unpad_tokens_from_deepgemm,
)
