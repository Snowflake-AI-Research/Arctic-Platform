from __future__ import annotations

import logging
from functools import wraps
from traceback import walk_tb

from arctic_platform.inference.utils import require_supported_vllm_version

logger = logging.getLogger(__name__)


def ensure_xgrammar_request_error_isolation() -> None:
    """Route xgrammar runtime failures to vLLM's per-request FINISHED_ERROR path.

    A failed mask stays open until accept_tokens rejects the sampled output;
    those unconstrained tokens can accompany the final error response.
    Speculative validation and rollback must also stop using a failed matcher.
    """
    require_supported_vllm_version("xgrammar request error isolation")

    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    if getattr(XgrammarGrammar.accept_tokens, "_arctic_request_error_isolation", False):
        return

    def isolate(method):
        @wraps(method)
        def wrapped(self, *args):
            if getattr(self, "_arctic_xgrammar_failed", False):
                return None
            try:
                return method(self, *args)
            except RuntimeError as error:
                # Do not hide Python subclasses or RuntimeErrors from Arctic/tensor code.
                if type(error) is not RuntimeError or not any(
                    frame.f_globals.get("__name__", "").startswith("xgrammar.")
                    for frame, _ in walk_tb(error.__traceback__)
                ):
                    raise
                logger.exception("xgrammar failed in %s; finishing only that request", method.__name__)
                self._arctic_xgrammar_failed = True
                return None

        return wrapped

    original_accept_tokens = isolate(XgrammarGrammar.accept_tokens)
    original_validate_tokens = isolate(XgrammarGrammar.validate_tokens)
    original_fill_bitmask = isolate(XgrammarGrammar.fill_bitmask)

    def accept_tokens(self, request_id, tokens):
        return original_accept_tokens(self, request_id, tokens) is True

    def validate_tokens(self, tokens):
        return original_validate_tokens(self, tokens) or []

    @wraps(original_fill_bitmask)
    def fill_bitmask(self, bitmask, idx):
        original_fill_bitmask(self, bitmask, idx)
        if getattr(self, "_arctic_xgrammar_failed", False):
            bitmask[idx].fill_(-1)

    accept_tokens._arctic_request_error_isolation = True
    XgrammarGrammar.accept_tokens = accept_tokens
    XgrammarGrammar.validate_tokens = validate_tokens
    XgrammarGrammar.rollback = isolate(XgrammarGrammar.rollback)
    XgrammarGrammar.fill_bitmask = fill_bitmask
