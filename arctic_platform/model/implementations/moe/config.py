from typing import Literal

EPCommBackend = Literal["deepep", "uccl", "nccl"]
DISPATCH_EP_BACKENDS: tuple[EPCommBackend, ...] = ("deepep", "uccl", "nccl")
