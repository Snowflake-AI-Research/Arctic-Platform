"""Vendored SonicMoE functional kernels (https://github.com/Dao-AILab/sonic-moe, Apache 2.0, see LICENSE).

Only the functional layer is vendored (``functional/``, ``enums.py``, ``utils.py``). Local change: the up- and
down-projection backwards skip the expert weight gradients when the weights are frozen.
"""

from .enums import ActivationType
from .functional import moe_general_routing_inputs

__all__ = ["ActivationType", "moe_general_routing_inputs"]
