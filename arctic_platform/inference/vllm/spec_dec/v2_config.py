# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
    SpeculatorConfig,
    register_speculative_method,
)


class ArcticSpeculatorConfig(DraftModelSpeculatorConfig):
    base_method = "draft_model"
    speculator_class = "arctic_platform.inference.vllm.spec_dec.arctic_proposer.ArcticProposer"
    uses_target_hidden_states = True
    shares_target_positions = False
    draft_has_own_vocab = True
    prefer_dense_token_history = True
    adaptive_verification_uses_draft_logits = False

    @classmethod
    def resolve_num_speculative_tokens(cls, spec):
        if spec.num_speculative_tokens is None:
            spec.num_speculative_tokens = spec.draft_model_config.hf_config.n_predict
        if spec.draft_sample_method != "greedy":
            raise ValueError("Arctic speculators support greedy drafting only")
        if spec.enable_adaptive_verification:
            raise ValueError("Arctic speculators do not yet provide adaptive-verification confidences")

    @property
    def max_model_len(self):
        return None

    @property
    def num_lookahead_tokens(self):
        return self.num_speculative_tokens


class ArcticSuffixSpeculatorConfig(SpeculatorConfig):
    base_method = "suffix"
    speculator_class = "arctic_platform.inference.vllm.spec_dec.arctic_proposer.SuffixProposer"
    prefer_dense_token_history = True
    adaptive_verification_uses_draft_logits = False

    @classmethod
    def configure(cls, spec):
        super().configure(spec)
        if spec.enable_adaptive_verification:
            raise ValueError("Arctic suffix decoding does not yet provide adaptive-verification confidences")


def register_arctic_speculators():
    for method in ("arctic", "mlp_speculator"):
        register_speculative_method(method, ArcticSpeculatorConfig)
    register_speculative_method("arctic_suffix", ArcticSuffixSpeculatorConfig)
