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
"""Per-step router health statistics, read off merged callback metrics the way a caller sees them."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.callbacks import Reduce
from arctic_platform.common.callbacks import merge_metrics
from arctic_platform.model.implementations.debug.config import router_stats_strides
from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter
from arctic_platform.model.loaders.glm_moe_dsa import GlmMoeDsaDebugOptions
from arctic_platform.model.loaders.qwen3_5_moe import DebugModelOptions
from arctic_platform.model.router_stats import RouterStatsCallback

EXPERTS, TOP_K, DIM = 4, 2, 8


@pytest.fixture(autouse=True)
def cpu_histc(monkeypatch):
    """The router histograms integer expert ids, which ``torch.histc`` supports only on CUDA."""
    histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda ids, **kwargs: histc(ids.float(), **kwargs))


def _training_config(stride: int = 1, call_stride: int = 1) -> dict:
    return {"debug": {"router_stats_sample_stride": stride, "router_stats_call_stride": call_stride}}


def _hooked(stride: int, call_stride: int = 1):
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.add_module("router", TokenChoiceTopKRouter(DIM, EXPERTS, TOP_K, "softmax", route_norm=True, route_scale=1.0))
    trainer = SimpleNamespace(model=model)
    callback = RouterStatsCallback(_training_config(stride, call_stride))
    callback.post_init(trainer)
    return model, callback, trainer


def _reported(callback: RouterStatsCallback) -> dict:
    """Driver-merged shape after one rank logs local totals with ``Reduce.SUM``."""
    merged = merge_metrics([{"router_stats": (Reduce.SUM, callback._report())}])
    return merged["router_stats"]["router"]


def test_a_step_counts_each_routed_token_once_and_skips_replay_pad_rows():
    model, hooked, trainer = _hooked(stride=1)
    x = torch.randn(5, DIM)
    replayed = torch.tensor([[1, 2], [3, 0], [2, 1], [0, 0], [0, 0]])
    model.router(x, None, routed_experts=replayed)
    hooked.pre_fwd_bwd_microbatch(trainer, is_padding=True)
    model.router(x, None, routed_experts=replayed)

    stats = _reported(hooked)

    assert stats["tokens_per_expert"] == pytest.approx([1.0, 2.0, 2.0, 1.0])
    assert stats["sampled_tokens"] == stats["replayed_tokens"] == pytest.approx(3.0)
    own = torch.topk(torch.softmax(model.router.gate(x[:3]), -1), TOP_K).indices
    agreement = sum(len(set(o.tolist()) & set(r.tolist())) / TOP_K for o, r in zip(own, replayed[:3]))
    assert stats["replay_agreement_sum"] == pytest.approx(agreement)
    assert _reported(hooked)["tokens_per_expert"] == [0.0] * EXPERTS, "a report must reset the per-step totals"


def test_driver_sums_local_totals_across_ranks():
    model, hooked, _trainer = _hooked(stride=1)
    model.router(torch.randn(4, DIM))
    local = hooked._report()
    merged = merge_metrics(
        [
            {"router_stats": (Reduce.SUM, local)},
            {"router_stats": (Reduce.SUM, local)},
        ]
    )
    assert merged["router_stats"]["router"]["sampled_tokens"] == pytest.approx(2.0 * local["router"]["sampled_tokens"])


@pytest.mark.parametrize("padding_calls", [0, 1])
@pytest.mark.parametrize("tokens", [2, 10])
@pytest.mark.parametrize("call_stride", [1, 2])
def test_the_strided_sample_covers_every_token_once_over_stride_calls(padding_calls, tokens, call_stride):
    """Every ``call_stride``-th counted call is observed and its offset advances."""
    x = torch.randn(tokens, DIM)
    full_model, full, _full_trainer = _hooked(stride=1)
    full_model.router(x)
    strided_model, strided, strided_trainer = _hooked(stride=4, call_stride=call_stride)
    for _ in range(4 * call_stride):
        strided.pre_fwd_bwd_microbatch(strided_trainer, is_padding=False)
        strided_model.router(x)
        strided.pre_fwd_bwd_microbatch(strided_trainer, is_padding=True)
        for _ in range(padding_calls):
            strided_model.router(x)

    every, sampled = _reported(full), _reported(strided)

    assert sampled["sampled_tokens"] == every["sampled_tokens"] == pytest.approx(float(tokens))
    assert sampled["entropy_sum"] == pytest.approx(every["entropy_sum"], rel=1e-5)
    assert sum(sampled["tokens_per_expert"]) == 4 * sum(every["tokens_per_expert"]), "counts come from observed calls"
    assert sampled["replayed_tokens"] == 0.0, "a router that chose its own experts replayed nothing"


def test_a_saturated_sigmoid_router_still_reports_its_entropy():
    model = torch.nn.Module()
    model.add_module("router", TokenChoiceTopKRouter(DIM, EXPERTS, TOP_K, "sigmoid", route_norm=True, route_scale=1.0))
    torch.nn.init.constant_(model.router.gate.weight, -1000.0)
    trainer = SimpleNamespace(model=model)
    hooked = RouterStatsCallback(_training_config(stride=1))
    hooked.post_init(trainer)
    model.router(torch.ones(3, DIM))

    assert _reported(hooked)["entropy_sum"] == pytest.approx(3 * math.log(EXPERTS), rel=1e-4)


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("biased", [False, True])
def test_replaying_the_routers_own_near_tied_choices_agrees_fully(score_func, biased):
    model = torch.nn.Module()
    model.add_module("router", TokenChoiceTopKRouter(EXPERTS, EXPERTS, TOP_K, score_func, True, 1.0))
    with torch.no_grad():
        model.router.gate.weight.copy_(torch.eye(EXPERTS))
    x = torch.randn(32, EXPERTS, generator=torch.Generator().manual_seed(49)) * 1e-7
    bias = torch.tensor([0.0, 0.0, 0.0, -0.1]) if biased else None
    selected = model.router(x, bias)[1]
    trainer = SimpleNamespace(model=model)
    hooked = RouterStatsCallback(_training_config(stride=1))
    hooked.post_init(trainer)
    model.router(x, bias, routed_experts=selected)

    stats = _reported(hooked)
    assert stats["replay_agreement_sum"] == stats["replayed_tokens"] == len(x)


def test_an_activation_checkpoint_rerun_of_the_router_is_not_counted_again():
    model, hooked, _trainer = _hooked(stride=1)
    x = torch.randn(6, DIM, requires_grad=True)
    rerun = lambda t: model.router(t)[0] * t[:, :TOP_K].exp()  # noqa: E731
    torch.utils.checkpoint.checkpoint(rerun, x, use_reentrant=False).sum().backward()

    stats = _reported(hooked)
    assert sum(stats["tokens_per_expert"]) == stats["sampled_tokens"] * TOP_K == float(len(x)) * TOP_K


def test_enabled_tracks_router_stats_sample_stride():
    assert RouterStatsCallback.enabled(_training_config(stride=1))
    assert not RouterStatsCallback.enabled(_training_config(stride=0))


@pytest.mark.parametrize("hook", ["pre_fwd_bwd_microbatch", "pre_fwd_only_microbatch"])
def test_microbatch_hooks_forward_to_counting(hook):
    model, callback, trainer = _hooked(stride=1)
    x = torch.randn(5, DIM)
    getattr(callback, hook)(trainer, is_padding=False)
    model.router(x)
    getattr(callback, hook)(trainer, is_padding=True)
    model.router(x)

    callback.post_step(trainer)

    reduce, report = callback.pending["router_stats"]
    assert reduce is Reduce.SUM
    assert report["router"]["sampled_tokens"] == pytest.approx(float(len(x)))
    assert report["router"]["replayed_tokens"] == 0.0, "padding call must not double count"


def test_padding_does_not_suppress_the_next_router_call():
    model, callback, trainer = _hooked(stride=1)
    x = torch.randn(4, DIM)

    callback.pre_fwd_only_microbatch(trainer, is_padding=True)
    model.router(x)
    callback.post_fwd_only_microbatch(trainer, is_padding=True)
    model.router(x)

    callback.post_step(trainer)
    _reduce, report = callback.pending["router_stats"]
    assert report["router"]["sampled_tokens"] == pytest.approx(float(len(x)))


def test_on_error_drops_partial_totals():
    model, callback, trainer = _hooked(stride=1)
    x = torch.randn(4, DIM)
    callback.pre_fwd_bwd_microbatch(trainer, is_padding=False)
    model.router(x)

    callback.on_error(trainer, "fwd_bwd", RuntimeError("mid-fwd"))
    callback.post_step(trainer)

    _reduce, report = callback.pending["router_stats"]
    assert report["router"]["sampled_tokens"] == 0.0
    assert report["router"]["tokens_per_expert"] == [0.0] * EXPERTS


@pytest.mark.parametrize(
    "config",
    [
        {"debug": {"router_stats_sample_stride": True}},
        {"debug": {"router_stats_sample_stride": "3"}},
        {"debug": {"router_stats_call_stride": 2.7}},
    ],
)
def test_strides_reject_values_that_are_not_ints(config):
    with pytest.raises(TypeError):
        router_stats_strides(config)


def test_post_step_logs_even_an_all_zero_report():
    _model, callback, trainer = _hooked(stride=1)
    callback.post_step(trainer)
    assert "router_stats" in callback.pending, "report() returns per-router zeros, which is still truthy"


def test_public_facade_exports_the_callback():
    from arctic_platform.model import router_stats as facade
    from arctic_platform.model.implementations.moe import router_stats as impl

    assert facade.RouterStatsCallback is impl.RouterStatsCallback


def test_public_recompute_facade_shares_one_idempotent_detector():
    import torch.utils.checkpoint as ckpt

    from arctic_platform.model import recompute as facade
    from arctic_platform.model.implementations.gpu import router_replay_recompute as impl

    assert facade.in_recompute is impl._in_recompute
    assert facade.install_recompute_detector is impl._install_recompute_detector

    facade.install_recompute_detector()
    first_hook = ckpt._recomputation_hook
    first_local = ckpt._recompute_thread_local
    facade.install_recompute_detector()
    impl._install_recompute_detector()
    assert ckpt._recomputation_hook is first_hook
    assert ckpt._recompute_thread_local is first_local
    assert facade.in_recompute() is False


def test_loader_debug_options_accept_router_stats_strides():
    assert DebugModelOptions(router_stats_sample_stride=4, router_stats_call_stride=2).router_stats_sample_stride == 4
    assert GlmMoeDsaDebugOptions(router_stats_sample_stride=8).router_stats_call_stride == 1
    with pytest.raises(Exception):
        DebugModelOptions(router_stats_sample_stride=True)
    with pytest.raises(Exception):
        GlmMoeDsaDebugOptions(router_stats_call_stride=0)
