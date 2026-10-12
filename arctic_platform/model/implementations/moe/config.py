from typing import Literal

EPCommBackend = Literal["deepep", "deepep_v2", "uccl"]
DISPATCH_EP_BACKENDS: tuple[EPCommBackend, ...] = ("deepep", "deepep_v2", "uccl")
