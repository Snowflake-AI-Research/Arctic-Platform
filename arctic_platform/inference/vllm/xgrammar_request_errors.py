from __future__ import annotations

import logging

from arctic_platform.inference.utils import require_supported_vllm_version

logger = logging.getLogger(__name__)


def ensure_xgrammar_request_error_isolation() -> None:
    """Fail the one request whose grammar raises inside xgrammar, not the engine.

    xgrammar raises RuntimeError when one request's parser state outgrows its
    int32-indexed arrays (e.g. a JSON schema string with minLength 20000 and no
    maxLength). vLLM's scheduler calls accept_tokens and fill_bitmask without a
    try, so that error kills the EngineCore. Here accept_tokens returns False,
    vLLM's existing path for rejected tokens: the request finishes with
    FINISHED_ERROR. A raise while filling the mask leaves that row unmasked and
    fails the request at its next accept_tokens.
    """
    require_supported_vllm_version("xgrammar request error isolation")

    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    if getattr(XgrammarGrammar.accept_tokens, "_arctic_request_error_isolation", False):
        return

    original_accept_tokens = XgrammarGrammar.accept_tokens
    original_fill_bitmask = XgrammarGrammar.fill_bitmask

    def accept_tokens(self, request_id, tokens):
        if getattr(self, "_arctic_xgrammar_failed", False):
            return False
        try:
            return original_accept_tokens(self, request_id, tokens)
        except RuntimeError:
            logger.exception("xgrammar failed for request %s; finishing only that request", request_id)
            self._arctic_xgrammar_failed = True
            return False

    def fill_bitmask(self, bitmask, idx):
        try:
            original_fill_bitmask(self, bitmask, idx)
        except RuntimeError:
            logger.exception("xgrammar failed filling a token mask; the request ends at its next accept_tokens")
            self._arctic_xgrammar_failed = True
            bitmask[idx].fill_(-1)

    accept_tokens._arctic_request_error_isolation = True
    XgrammarGrammar.accept_tokens = accept_tokens
    XgrammarGrammar.fill_bitmask = fill_bitmask
