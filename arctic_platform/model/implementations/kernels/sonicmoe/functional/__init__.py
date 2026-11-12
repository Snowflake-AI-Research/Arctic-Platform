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

# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import itertools
import logging
import time
from functools import partial

import quack.autotuner
import quack.gemm_config as _gc
import torch
import torch.nn.functional as F
from quack.autotuner import AutotuneConfig
from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm
from quack.gemm_interface import gemm_act
from quack.gemm_interface import gemm_tuned

from ..enums import ActivationType
from ..enums import is_glu
from .backward import _down_projection_backward_act
from .backward import _token_broadcast_backward
from .backward import _topk_softmax_bwd
from .backward import _up_projection_backward_act
from .forward import _router_forward
from .forward import _topk_softmax_fwd
from .triton_kernels import TC_topk_router_metadata_triton
from .triton_kernels import general_routing_router_metadata_triton

# QuACK autotuning accelerators, applied at import. QuACK sweeps its whole config space
# per problem key, and MoE keys move every step (M = routed tokens), so we shrink the
# space and let nearby M share a cache entry. Search only; the math is unchanged.


def _fast_sm90_configs(epilogue: str | None = None, tune_coop: bool = True) -> list[GemmConfig]:
    """Reduced SM90 (Hopper) config space — drop-in for quack.gemm_config._get_sm90_configs.

    Drops tile_n=208 (gated rejects non-multiples of 32; gather_A rejects it on SM90)
    and the odd (128, 224) / (192, 128) tiles: 44 -> 28 configs, 18 -> 14 for "gated".
    """
    tile_n_vals = [128, 160, 192]
    tile_mn_vals_coop = [(256, tile_n) for tile_n in tile_n_vals] + [(128, 256)]
    tile_mn_vals_pingpong = [(128, tile_n) for tile_n in tile_n_vals]
    if epilogue in ["gated"]:
        tile_mn_vals_coop = [(m, n) for m, n in tile_mn_vals_coop if n % 32 == 0 and m != 192]
        tile_mn_vals_pingpong = [(m, n) for m, n in tile_mn_vals_pingpong if n % 32 == 0]
    tile_mn_vals = []
    if tune_coop:
        tile_mn_vals += [(m, n, False) for m, n in tile_mn_vals_coop]
    tile_mn_vals += [(m, n, True) for m, n in tile_mn_vals_pingpong]
    cluster = [(1, 2), (2, 1)]
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    return [
        GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=pingpong,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            swap_ab=swap_ab,
            device_capacity=9,
            is_dynamic_persistent=False,  # SM90 has no CLC-based dynamic persistent scheduler
            use_tma_gather=False,  # TMA gather not supported on SM90
        )
        for (tile_m, tile_n, pingpong), (cluster_m, cluster_n), swap_ab in itertools.product(
            tile_mn_vals, cluster, swap_ab_vals
        )
    ]


def _fast_sm100_configs(epilogue: str | None = None) -> list[GemmConfig]:
    """Reduced SM100 (Blackwell datacenter) config space — a drop-in for
    quack.gemm_config._get_sm100_configs.
    """
    tile_n_vals = [128, 160, 192, 256]
    tile_mn_cluster_vals = (
        [(128, tile_n, (1, 2)) for tile_n in tile_n_vals]
        + [(128, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(256, tile_n, (2, 1)) for tile_n in tile_n_vals]
        + [(256, 512, (2, 1))]
    )
    swap_ab_vals = [False, True]
    if epilogue in ["lse", "gated"]:
        swap_ab_vals = [False]
    GemmConfigCls = partial(GemmConfig, pingpong=False, device_capacity=10)  # no pingpong on SM100
    use_clc_vals = [True, False]
    use_tma_gather_vals = [True, False]
    return [
        GemmConfigCls(
            tile_m=m,
            tile_n=n,
            cluster_m=cm,
            cluster_n=cn,
            swap_ab=sab,
            max_swizzle_size=8,
            is_dynamic_persistent=use_clc,
            use_tma_gather=use_tma_gather,
        )
        for (m, n, (cm, cn)), sab, use_clc, use_tma_gather in itertools.product(
            tile_mn_cluster_vals, swap_ab_vals, use_clc_vals, use_tma_gather_vals
        )
    ]


# M-range quantization: what picks a config is the order of magnitude of M, not its
# exact value, so the cache is keyed on a bucket of M — one every 3 powers of two, i.e.
# each spanning 8x: M<=8, 8<M<=64, 64<M<=512, 512<M<=4096, ...
def _bucketize_M(m: int) -> int:
    """Every M shares one bucket: each GEMM structure (trailing dims, stride classes, dtypes) is tuned once.

    Training receive buffers hold 1e4-1e6 tokens and routed rows that move with routing every layer, step and
    DeepEP chunk. Any M bucket a rank had not hit before started a full autotuning sweep mid-training -- CUDA-graph
    timing plus ``empty_cache`` -- on that rank while the others waited at the next collective (Qwen3.8 steps at
    30-40 s instead of 12 s, still with one bucket above 4096 rows). The winning tile config does not depend on M
    at these sizes (local change).
    """
    return 1 << 30


# The experts' feature dims (hidden, intermediate, 2 x intermediate, experts), recorded by the callers.
_FEATURE_DIMS: set[int] = set()


def note_feature_dims(*dims: int) -> None:
    """Keep these dims exact in autotune keys; every other dim is a token or routed-row count."""
    _FEATURE_DIMS.update(dims)


def _make_bucketized_key(self, args, kwargs) -> tuple:
    """QuACK's autotune cache key, with every tensor's leading dim (M) bucketized, and every other dim that is not
    an expert feature dim too: those are token or routed-row counts, which the weight-gradient GEMMs (variable-length
    K) carry as trailing dims and which would otherwise start a sweep on every call (local change).

    The feature dims stay exact, so structurally different GEMMs cannot alias onto one key,
    and the stringified entries never collide with QuACK's own exact-shape keys.
    """
    all_args = {**dict(zip(self.arg_names, args)), **kwargs}
    _args = {k: v for k, v in all_args.items() if k in self.arg_names}
    key = [str(_args[k]) for k in self.keys if k in _args]
    for arg in _args.values():
        if isinstance(arg, torch.Tensor):
            shape = [d if d in _FEATURE_DIMS else 1 << 30 for d in arg.shape]
            if shape:
                shape[0] = _bucketize_M(shape[0])
            key.append(str(tuple(shape)))
            # only the {0, 1, other} classes of a stride matter; same folding QuACK does
            key.append(str([s if s in {0, 1} else 2 for s in arg.stride()]))
            key.append(str(arg.dtype))
    return tuple(key)


# A sweep's cost per config, bounded by GPU time instead of by launch count (local change). QuACK benches each config
# on the call's real operands with 3 priming + 1 probe + max(50, 200 ms / t) warmup + 200 timed launches: >= 254
# launches, so a sweep's length grows with the rows the calling rank received. An EP rank that receives most of the
# routed rows (1e6+ rows at ~20 ms per launch) then sweeps for minutes while its peers wait inside the next DeepEP
# kernel, whose barrier traps after ~100 s ("DeepEP timeout check failed", then "unspecified launch failure"). These
# budgets keep every config at a few launches once a launch is long, and at QuACK's counts for short launches.
_TUNE_WARMUP_MS = 25.0
_TUNE_TIMED_MS = 25.0
_TUNE_MAX_WARMUP_CALLS = 50  # QuACK's minimum warmup count; the sweep's 200 ms GPU warmup already ramped the clocks
_TUNE_MIN_TIMED_CALLS = 3
_TUNE_MAX_TIMED_CALLS = 200  # QuACK's timed count, kept for short launches


def _bench_launch_counts(est_ms: float, n_sets: int) -> tuple[int, int]:
    """(warmup launches, timed rounds over the ``n_sets`` rotated operand sets) for a config whose probe launch took
    ``est_ms``: about ``_TUNE_WARMUP_MS`` of warmup (1 to ``_TUNE_MAX_WARMUP_CALLS`` launches) and ``_TUNE_TIMED_MS``
    of timed work (``_TUNE_MIN_TIMED_CALLS`` to ``_TUNE_MAX_TIMED_CALLS`` launches, whole rounds)."""
    n_warmup = min(_TUNE_MAX_WARMUP_CALLS, max(1, int(_TUNE_WARMUP_MS / est_ms)))
    n_timed = min(_TUNE_MAX_TIMED_CALLS, max(_TUNE_MIN_TIMED_CALLS, int(_TUNE_TIMED_MS / est_ms)))
    return n_warmup, max(1, -(-n_timed // n_sets))


def _bench_cuda_graph_l2_rotate_bounded(fn, arg_sets, kwarg_sets, extra_kwargs, quantiles=None, **_):
    """``quack.bench.bench_utils._bench_cuda_graph_l2_rotate`` (L2-cold round-robin over the cloned operand sets, timed
    as one CUDA-graph replay) with launch counts from ``_bench_launch_counts``."""
    n_sets = len(arg_sets)
    fn(*arg_sets[0], **kwarg_sets[0], **extra_kwargs)  # first launch: module load (or CompilePending, to the caller)
    torch.cuda.synchronize()
    probe_start = torch.cuda.Event(enable_timing=True)
    probe_end = torch.cuda.Event(enable_timing=True)
    probe_start.record()
    fn(*arg_sets[0], **kwarg_sets[0], **extra_kwargs)
    probe_end.record()
    torch.cuda.synchronize()
    n_warmup, rounds = _bench_launch_counts(max(probe_start.elapsed_time(probe_end), 1e-3), n_sets)
    for i in range(n_warmup):
        fn(*arg_sets[i % n_sets], **kwarg_sets[i % n_sets], **extra_kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(rounds):
            for i in range(n_sets):
                fn(*arg_sets[i], **kwarg_sets[i], **extra_kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / (rounds * n_sets)
    return [ms for _ in quantiles] if quantiles else ms


def _pick_l2_rotate_count_bounded(args, kwargs, target_ratio: int = 3, min_buffers: int = 4, max_buffers: int = 16):
    """``quack.bench.bench_utils._pick_l2_rotate_count``, except that one operand set larger than ``target_ratio`` L2
    caches is not cloned ``min_buffers`` times (it already defeats L2 reuse: QuACK cloned a hot rank's GBs of rows
    four times), and the free-memory cap applies below ``min_buffers`` too."""
    if not torch.cuda.is_available():
        return min_buffers
    tensor_bytes = sum(a.numel() * a.element_size() for a in args if isinstance(a, torch.Tensor)) + sum(
        v.numel() * v.element_size() for v in kwargs.values() if isinstance(v, torch.Tensor)
    )
    if tensor_bytes == 0:
        return min_buffers
    l2_size = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    n_by_l2 = -(-target_ratio * l2_size // tensor_bytes)
    n_by_mem = int(torch.cuda.mem_get_info()[0] * 0.5) // tensor_bytes
    wanted = 1 if n_by_l2 <= 1 else max(min_buffers, n_by_l2)
    return max(1, min(max_buffers, n_by_mem, wanted))


_orig_autotuner_call = quack.autotuner.Autotuner.__call__


@torch.compiler.disable  # the tuner must never be traced
def _autotuner_call_with_M_buckets(self, *args, **kwargs):
    if len(self.configs) > 1:
        bucket_key = _make_bucketized_key(self, args, kwargs)
        if bucket_key in self.cache:
            # An earlier M in this bucket was tuned already: reuse its winner, skip the
            # sweep. Mirrors the cache-hit tail of the original __call__.
            config = self.cache[bucket_key]
            self.best_config = config
            self.nargs = dict(zip(self.arg_names, args))
            ret = self.fn(*args, **kwargs, **config.all_kwargs())
            self.nargs = None
            return ret

    # First call with this GEMM structure: let QuACK tune for real, under its own exact-shape key ...
    if len(self.configs) > 1:
        logging.getLogger(__name__).warning(
            "SonicMoE autotuning %s for %s", getattr(self.fn, "__name__", self.fn), bucket_key
        )
        start = time.perf_counter()
    ret = _orig_autotuner_call(self, *args, **kwargs)

    # ... then alias that winner onto the bucket key, for every later M in the bucket.
    if len(self.configs) > 1 and hasattr(self, "best_config"):
        self.cache[_make_bucketized_key(self, args, kwargs)] = self.best_config
        logging.getLogger(__name__).warning(
            "SonicMoE autotuned %s in %.1f s", getattr(self.fn, "__name__", self.fn), time.perf_counter() - start
        )

    return ret


quack.autotuner.Autotuner.__call__ = _autotuner_call_with_M_buckets
quack.autotuner._bench_cuda_graph_l2_rotate = _bench_cuda_graph_l2_rotate_bounded
quack.autotuner._pick_l2_rotate_count = _pick_l2_rotate_count_bounded
_gc._get_sm90_configs = _fast_sm90_configs
_gc._get_sm100_configs = _fast_sm100_configs

# gemm_tuned (plain GEMMs: down projection, dw1/dw2) captured its config list at
# decoration time, so it needs an explicit rebuild. The per-epilogue autotuners behind
# gemm_act / gemm_dact are built on first call and read the patched generators already.
gemm_tuned.configs = [AutotuneConfig(config=c) for c in _gc.get_all_configs()]


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, router_logits: torch.Tensor, E: int, K: int, is_softmax_over_topk: bool, norm_topk_probs: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        topk_router_score = torch.empty(T, K, dtype=torch.float32, device=router_logits.device)
        topk_router_indices = torch.empty(T, K, dtype=torch.int32, device=router_logits.device)

        _topk_softmax_fwd(
            router_logits,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=is_softmax_over_topk,
            norm_topk_probs=norm_topk_probs,
        )

        # Save router_logits for topk(softmax()) backward (recompute full softmax).
        # For softmax(topk()) it's unused but save unconditionally for simplicity.
        ctx.save_for_backward(topk_router_score, topk_router_indices, router_logits)
        ctx.E = E
        ctx.dtype = router_logits.dtype
        ctx.is_softmax_over_topk = is_softmax_over_topk
        ctx.norm_topk_probs = norm_topk_probs

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor):
        T, K = dtopk_score.size()
        E = ctx.E
        topk_router_score, topk_router_indices, router_logits = ctx.saved_tensors
        dlogits = torch.zeros(T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device)

        _topk_softmax_bwd(
            router_logits,
            dlogits,
            None,
            dtopk_score,
            topk_router_score,
            topk_router_indices,
            E,
            K,
            is_softmax_over_topk=ctx.is_softmax_over_topk,
            norm_topk_probs=ctx.norm_topk_probs,
        )

        return dlogits, None, None, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_each_token_has_variable_activated_experts: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
        concat_layout: bool = False,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        a = torch.empty(TK, I, dtype=x.dtype, device=x.device)
        h = (
            torch.empty(TK, (2 * I if is_glu_activation else I), dtype=x.dtype, device=x.device)
            if (not is_inference_mode_enabled)
            else None
        )

        gemm_act(
            x,
            w1.permute(2, 1, 0),
            activation=activation_type.value,
            cu_seqlens_m=expert_frequency_offset,
            A_idx=x_gather_idx,
            preact_out=h,
            postact_out=a,
            store_preact=(not is_inference_mode_enabled),
            bias=b1,
            concat_layout=(
                (("B", "bias") if b1 is not None else ("B",)) if concat_layout and is_glu_activation else None
            ),
        )

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I  # noqa: E741
        ctx.is_each_token_has_variable_activated_experts = is_each_token_has_variable_activated_experts
        ctx.is_glu_activation = is_glu_activation
        ctx.concat_layout = concat_layout and is_glu_activation

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ctx.mark_non_differentiable(a)
        ctx.set_materialize_grads(False)

        return a, h

    @staticmethod
    def backward(ctx, _: None, dh: torch.Tensor):
        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_each_token_has_variable_activated_experts = ctx.is_each_token_has_variable_activated_experts
        concat_layout = ctx.concat_layout

        (
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = ctx.saved_tensors

        dx_expanded = torch.empty(TK, H, dtype=dh.dtype, device=dh.device)
        # Frozen experts (e.g. LoRA on attention only) need no weight gradient: skip its buffer and GEMM.
        dw1 = torch.empty_like(w1) if ctx.needs_input_grad[1] else None
        db1 = None if b1 is None else torch.empty_like(b1)

        _up_projection_backward_act(
            w1=w1,
            dx_expanded=dx_expanded,
            dh=dh,
            db1=db1,
            expert_frequency_offset=expert_frequency_offset,
            is_glu_activation=is_glu_activation,
            concat_layout=concat_layout,
        )

        if dw1 is not None:
            gemm(
                x.T,
                dh,
                out=dw1.permute(2, 1, 0),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
                concat_layout=(("out",) if concat_layout else None),
            )

        dx_reduced = torch.empty(T, H, dtype=dh.dtype, device=dh.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_each_token_has_variable_activated_experts else K),
            H=H,
            is_varlen_K=is_each_token_has_variable_activated_experts,
        )

        return dx_reduced, dw1, db1, *[None] * 13


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        h: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
    ) -> torch.Tensor:
        TK = a.size(0)
        H, I, E = w2.shape

        y = torch.empty(TK, H, dtype=a.dtype, device=a.device)

        gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=expert_frequency_offset, bias=b2)

        o = torch.empty(T, H, device=a.device, dtype=a.dtype)
        topk_scores = topk_scores.view(-1)

        _router_forward(
            y=y,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type

        ctx.save_for_backward(
            h,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type

        (
            h,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
        ) = ctx.saved_tensors

        dw2 = torch.empty_like(w2) if ctx.needs_input_grad[2] else None
        db2 = None if b2 is None else torch.empty_like(b2)
        dh = torch.empty_like(h)

        I = w2.size(1)  # noqa: E741
        TK = x_gather_idx.size(0)

        a_prime = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        ds = torch.empty_like(topk_scores)

        _down_projection_backward_act(
            dout=dout,
            h=h,
            w2=w2,
            dh=dh,
            ds=ds,
            b2=b2,
            db2=db2,
            a_prime=a_prime,
            topk_scores=topk_scores,
            expert_frequency_offset=expert_frequency_offset,
            x_gather_idx=x_gather_idx,
            s_scatter_idx=s_scatter_idx,
            activation_type=activation_type.value,
        )

        if dw2 is not None:
            gemm(
                dout.T,
                a_prime,
                out=dw2.permute(2, 0, 1),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )

        # TC top-K routing
        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dh, dw2, db2, ds, *[None] * 10


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
    concat_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    E = router_w.size(0)
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
        router_logits, E, K, is_softmax_over_topk, norm_topk_probs
    )

    T, K = topk_indices.size()
    TK = T * K
    device = topk_indices.device

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx
    )

    if type(activation_type) is str:
        activation_type = ActivationType(activation_type)

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_each_token_has_variable_activated_expert
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        topk_scores,
        expert_frequency_offset,
        T,
        K,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        None,
        False,  # is_each_token_has_variable_activated_expert
        activation_type,
    )

    return o, router_logits, expert_frequency


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Weight format requirements:
# - w1_weight: Shape (2*I, H, E), stride order (2, 0, 1)
#     concat_layout=False (default): interleaved [gate_row0, up_row0, gate_row1, up_row1, ...]
#     concat_layout=True:            concatenated [gate_row0, ..., gate_row_{I-1}, up_row0, ..., up_row_{I-1}]
# - w2_weight: Shape (H, I, E), stride order (2, 0, 1)


# We assume token_indices is already SORTED ascendingly !!!
#   and len(token_indices) = len(expert_indices) = len(router_scores)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
    concat_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"

    T = x.size(0)
    TK = router_scores.size(0)
    E = w2.size(-1)
    device = router_scores.device

    if router_scores.dtype != torch.float32:
        router_scores = router_scores.float()

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    num_activated_expert_per_token_offset = torch.empty(T + 1, dtype=torch.int32, device=device)

    general_routing_router_metadata_triton(
        token_indices,
        expert_indices,
        T,
        E,
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )

    a, h = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,  # K, not needed
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_each_token_has_variable_activated_expert
        activation_type,
        is_inference_mode_enabled,
        concat_layout,
    )

    o = _DownProjection.apply(
        a,
        h,
        w2,
        b2,
        router_scores,
        expert_frequency_offset,
        T,
        None,  # K, not needed
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_each_token_has_variable_activated_expert
        activation_type,
    )

    return o, expert_frequency
