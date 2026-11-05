# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score centering on the GRPO/CISPO policy term (arXiv 2609.20807)."""

from __future__ import annotations

import torch

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close

V, K = 12, 4
CAP = {"use_cispo_loss": True, "is_weight_clip_max": 5.0}
BAND = {"use_cispo_loss": True, "eps_clip": 0.2}


def _grouped(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary log-probs at ``ids`` plus the log of the remaining mass, as the grouped LM head returns."""
    log_p = logits.log_softmax(-1)
    head = log_p.gather(-1, ids)
    tail = log_p.scatter(-1, ids, float("-inf")).logsumexp(-1, keepdim=True)
    return torch.cat([head, tail], -1)


def _sampler(trainer_logits: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A sampler whose top-K differs from the trainer and whose tail is the trainer's tail rescaled.

    That tail is the estimator's model, so its top-K correction equals exact full-vocabulary centering.
    """
    p = trainer_logits.softmax(-1)
    q_logits = trainer_logits + 0.8 * torch.randn(trainer_logits.shape, generator=generator)
    ids = q_logits.topk(K, -1).indices
    head = q_logits.softmax(-1).gather(-1, ids)
    # Make one head token rare under the sampler so its ratio p/q exceeds the CISPO cap.
    head[..., -1] = head[..., -1] * 1e-3
    head_mass = head.sum(-1, keepdim=True)
    tail_p = p.scatter(-1, ids, 0.0)
    q = tail_p * (1.0 - head_mass) / tail_p.sum(-1, keepdim=True)
    q = q.scatter(-1, ids, head)
    return q, ids


def _setup(seed: int = 0, rows: int = 2, width: int = 5):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((rows, width, V), generator=generator) * 2.0
    q, ids = _sampler(logits, generator)
    tokens = torch.multinomial(q.reshape(-1, V), 1, generator=generator).reshape(rows, width)
    loss_mask = torch.ones((rows, width), dtype=torch.bool)
    loss_mask[0, 0] = False
    context = {
        "old_log_probs_shifted": q.log().gather(-1, tokens[..., None])[..., 0],
        "advantages": torch.randn((rows, width), generator=generator),
        "loss_mask": loss_mask,
        "sampler_topk_ids_shifted": ids.to(torch.int32),
        "sampler_topk_log_probs_shifted": q.log().gather(-1, ids),
    }
    return logits, q, ids, tokens, context


def _loss_grad(name, logits, ids, tokens, context, config):
    z = logits.clone().requires_grad_(True)
    outputs = {
        "logprobs": z.log_softmax(-1).gather(-1, tokens[..., None])[..., 0],
        "group_log_probs": _grouped(z, ids),
    }
    loss, metrics = LOSS_FNS[name](outputs, context, {}, config, "cpu")
    loss.backward()
    return loss.detach(), z.grad, metrics


def _exact_centering_grad(logits, q, tokens, context, low, high):
    """Gradient of ``-A (w_y log p_y - Σ_v sg(q_v w_v) log p_v)`` over the whole vocabulary, token-mean."""
    z = logits.clone().requires_grad_(True)
    log_p = z.log_softmax(-1)
    w = (log_p.detach().exp() / q).clamp(low, high)
    sampled = log_p.gather(-1, tokens[..., None])[..., 0]
    w_y = w.gather(-1, tokens[..., None])[..., 0]
    center = ((q * w) * log_p).sum(-1)
    mask = context["loss_mask"]
    per_token = -context["advantages"] * (w_y * sampled - center)
    (torch.where(mask, per_token, 0.0).sum() / mask.sum()).backward()
    return z.grad


class TestScoreCentering(TestCasePlus):
    def test_off_by_default_and_a_no_op_when_the_sampler_is_the_trainer(self):
        logits, _, _, tokens, context = _setup()
        p = logits.softmax(-1)
        ids = logits.topk(K, -1).indices
        on_policy = dict(
            context,
            old_log_probs_shifted=p.log().gather(-1, tokens[..., None])[..., 0],
            sampler_topk_ids_shifted=ids.to(torch.int32),
            sampler_topk_log_probs_shifted=p.log().gather(-1, ids),
        )
        base_loss, base_grad, base_metrics = _loss_grad("grpo", logits, ids, tokens, on_policy, CAP)
        self.assertFalse(any(key.startswith("score_centering") for key in base_metrics))
        loss, grad, metrics = _loss_grad("grpo", logits, ids, tokens, on_policy, {**CAP, "score_centering": True})
        torch_assert_close(grad, base_grad, rtol=0, atol=1e-6)
        self.assertLess(metrics["score_centering_coef_abs_sum"], 1e-5)
        self.assertEqual(metrics["score_centering_token_count"], float(context["loss_mask"].sum()))

    def test_matches_exact_centering_under_both_cispo_weights(self):
        for config, (low, high) in ((CAP, (0.0, 5.0)), (BAND, (0.8, 1.2))):
            for name in ("grpo", "ap_grpo"):
                with self.subTest(config=config, name=name):
                    logits, q, ids, tokens, context = _setup(seed=3)
                    _, grad, metrics = _loss_grad(
                        name, logits, ids, tokens, context, {**config, "score_centering": True}
                    )
                    torch_assert_close(grad, _exact_centering_grad(logits, q, tokens, context, low, high))
                    _, base_grad, _ = _loss_grad(name, logits, ids, tokens, context, config)
                    self.assertGreater(float((grad - base_grad).abs().max()), 1e-3)
                    self.assertGreater(metrics["score_centering_coef_abs_sum"], 1e-3)

    def test_through_the_chunked_grouped_lm_head(self):
        generator = torch.Generator().manual_seed(7)
        rows, width, hidden = 2, 5, 6
        hidden_states = torch.randn((rows, width, hidden), generator=generator)
        weight = torch.randn((V, hidden), generator=generator)
        logits = hidden_states @ weight.T
        q, ids = _sampler(logits, generator)
        tokens = torch.multinomial(q.reshape(-1, V), 1, generator=generator).reshape(rows, width)
        context = {
            "old_log_probs_shifted": q.log().gather(-1, tokens[..., None])[..., 0],
            "advantages": torch.randn((rows, width), generator=generator),
            "loss_mask": torch.ones((rows, width), dtype=torch.bool),
            "sampler_topk_ids_shifted": ids.to(torch.int32),
            "sampler_topk_log_probs_shifted": q.log().gather(-1, ids),
        }
        h = hidden_states.clone().requires_grad_(True)
        w = weight.clone().requires_grad_(True)
        logprobs, grouped = chunked_lm_head_logprobs(
            h, w, tokens, token_chunk_size=3, vocab_chunk_size=5, fp32_lm_head=True, group_token_ids=ids
        )
        loss, _ = LOSS_FNS["grpo"](
            {"logprobs": logprobs, "group_log_probs": grouped}, context, {}, {**CAP, "score_centering": True}, "cpu"
        )
        loss.backward()
        reference = _exact_centering_grad(logits, q, tokens, context, 0.0, 5.0)
        torch_assert_close(h.grad, reference @ weight, rtol=1e-4, atol=1e-6)
        torch_assert_close(
            w.grad, reference.reshape(-1, V).T @ hidden_states.reshape(-1, hidden), rtol=1e-4, atol=1e-6
        )

    def test_model_forward_callback_requests_grouped_log_probs_only_when_on(self):
        _, _, ids, _, context = _setup()
        context = dict(context, input_ids=torch.zeros(ids.shape[:2], dtype=torch.long))
        loss = resolve_loss("grpo")
        for config, compute in (({**CAP, "score_centering": True}, False), (CAP, True)):
            kwargs, keys = {"dss_compute_logprobs": compute}, ["logprobs"]
            loss.model_forward_callback(kwargs, context, config, keys)
            self.assertNotIn("group_token_ids", kwargs)
            self.assertEqual(keys, ["logprobs"])
        kwargs, keys = {"dss_compute_logprobs": True}, ["logprobs"]
        loss.model_forward_callback(kwargs, context, {**CAP, "score_centering": True}, keys)
        self.assertEqual(kwargs["group_token_ids"].dtype, torch.long)
        torch_assert_close(kwargs["group_token_ids"], ids)
        self.assertEqual(keys, ["logprobs", "group_log_probs"])
        outputs = {"logits": 0, "group_log_probs": 1, "logprobs": 2}
        loss.output_callback(outputs)
        self.assertEqual(outputs, {"logprobs": 2})

    def test_packed_call_trains_like_the_padded_one(self):
        logits, _, ids, tokens, context = _setup(seed=5, rows=1, width=6)
        config = {**CAP, "score_centering": True}
        padded_loss, padded_grad, _ = _loss_grad("grpo", logits, ids, tokens, context, config)
        packed = dict(context, cu_seqlens=torch.tensor([0, 6], dtype=torch.int32))
        packed_loss, packed_grad, _ = _loss_grad("grpo", logits, ids, tokens, packed, config)
        torch_assert_close(packed_loss, padded_loss)
        torch_assert_close(packed_grad, padded_grad)

    def test_rejects_what_it_was_not_derived_for(self):
        logits, _, ids, tokens, context = _setup()
        on = {"score_centering": True}
        cases = (
            ({}, "use_cispo_loss=True"),
            ({**CAP, "importance_sampling_level": "sequence"}, "importance_sampling_level='token'"),
            ({**CAP, "m2_threshold": 0.04}, "M2PO"),
            ({**CAP, "ratio_mask_bounds_pos": [0.8, 1.25]}, "ratio masks"),
            ({**CAP, "score_centering": 1}, "must be a bool"),
        )
        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _loss_grad("grpo", logits, ids, tokens, context, {**on, **config})
        missing = {key: value for key, value in context.items() if key != "sampler_topk_log_probs_shifted"}
        with self.assertRaisesRegex(ValueError, "sampler_topk_log_probs_shifted"):
            _loss_grad("grpo", logits, ids, tokens, missing, {**CAP, **on})
        z = logits.clone().requires_grad_(True)
        outputs = {"logprobs": z.log_softmax(-1).gather(-1, tokens[..., None])[..., 0]}
        with self.assertRaisesRegex(ValueError, "group_log_probs"):
            LOSS_FNS["grpo"](outputs, context, {}, {**CAP, **on}, "cpu")
