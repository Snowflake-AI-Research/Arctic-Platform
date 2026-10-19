from __future__ import annotations

import logging

from arctic_platform.inference.utils import require_supported_vllm_version

logger = logging.getLogger(__name__)


def ensure_xgrammar_request_error_isolation() -> None:
    """Fail the one request whose grammar raises inside xgrammar, not the engine.

    xgrammar raises RuntimeError when one request's parser state outgrows its
    int32-indexed arrays (e.g. a JSON schema string with minLength 20000 and no
    maxLength). vLLM's scheduler calls accept_tokens without a try, so that
    error kills the EngineCore. Returning False takes vLLM's existing path for
    rejected tokens: the request finishes with FINISHED_ERROR.
    """
    require_supported_vllm_version("xgrammar request error isolation")

    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    if getattr(XgrammarGrammar.accept_tokens, "_arctic_request_error_isolation", False):
        return

    original_accept_tokens = XgrammarGrammar.accept_tokens

    def accept_tokens(self, request_id, tokens):
        try:
            return original_accept_tokens(self, request_id, tokens)
        except RuntimeError:
            logger.exception("xgrammar failed for request %s; finishing only that request", request_id)
            return False

    accept_tokens._arctic_request_error_isolation = True
    XgrammarGrammar.accept_tokens = accept_tokens
