from __future__ import annotations

from arctic_platform.inference.utils import require_supported_vllm_version


def ensure_xgrammar_stop_mask_fix() -> None:
    """Prevent vLLM from sampling stop tokens xgrammar would reject.

    xgrammar can expose a stop token in its bitmask before the grammar is
    complete, while its matcher rejects that same token after vLLM samples it.
    """
    require_supported_vllm_version("xgrammar stop-token mask fix")

    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    if getattr(XgrammarGrammar.fill_bitmask, "_arctic_stop_mask_fix", False):
        return

    original_fill_bitmask = XgrammarGrammar.fill_bitmask

    def fill_bitmask(self, bitmask, idx):
        original_fill_bitmask(self, bitmask, idx)

        for token_id in self.matcher.stop_token_ids:
            word_idx, bit_idx = divmod(token_id, 32)
            word = int(bitmask[idx, word_idx]) & 0xFFFFFFFF
            if not word & (1 << bit_idx):
                continue
            if self.matcher.accept_token(token_id):
                self.matcher.rollback(1)
                continue
            word &= ~(1 << bit_idx)
            if word >= 1 << 31:
                word -= 1 << 32
            bitmask[idx, word_idx] = word

    fill_bitmask._arctic_stop_mask_fix = True
    XgrammarGrammar.fill_bitmask = fill_bitmask
