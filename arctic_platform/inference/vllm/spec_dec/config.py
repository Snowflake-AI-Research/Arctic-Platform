# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PreTrainedConfig


class MLPSpeculatorConfig(PreTrainedConfig):
    """Configuration for Arctic checkpoints after vLLM removed MLP speculators."""

    model_type = "mlp_speculator"
    attribute_map = {"hidden_size": "emb_dim"}

    def __init__(
        self,
        vocab_size: int = 32000,
        emb_dim: int = 4096,
        inner_dim: int = 0,
        n_predict: int = 3,
        top_k_tokens_per_head: list[int] | None = None,
        n_candidates: int = 5,
        tie_weights: bool = False,
        scale_input: bool = False,
        base_model_arch: str = "",
        **kwargs,
    ):
        if top_k_tokens_per_head is None:
            top_k_tokens_per_head = [5, 4, 3]
        assert len(top_k_tokens_per_head) == n_predict
        self.vocab_size = vocab_size
        self.emb_dim = int(emb_dim)
        self.inner_dim = inner_dim
        self.n_predict = n_predict
        self.top_k_tokens_per_head = top_k_tokens_per_head
        self.n_candidates = n_candidates
        self.num_lookahead_tokens = n_predict
        self.tie_weights = tie_weights
        self.scale_input = scale_input
        self.base_model_arch = base_model_arch
        kwargs.setdefault("num_attention_heads", 1)
        super().__init__(**kwargs)
