from typing import Literal

EPCommBackend = Literal["deepep", "uccl"]
DISPATCH_EP_BACKENDS: tuple[EPCommBackend, ...] = ("deepep", "uccl")
