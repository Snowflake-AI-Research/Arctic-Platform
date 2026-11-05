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

"""Router replay must pin a token's expert set whatever the row count of the call.

``TokenChoiceTopKRouter`` reaches a selection in one of two ways. It derives one with ``topk`` over its own gate
output (``arctic_platform.model.implementations.qwen35.models.layers.moe``), or it is handed a selection to
reuse and gathers the scores at those indices (``:256``). The derived selection follows the row count of the call
and the replayed one does not, which is the difference this module pins.

The gate is a bfloat16 ``nn.Linear`` and the float32
cast at ``:245`` is applied to the product rather than to its operands, so the ordering ``topk`` sees at ``:259``
is settled in bfloat16. cuBLAS chooses a kernel for a bfloat16 ``[rows, K] x [K, N]`` product partly from ``rows``,
and two kernels group the contraction differently, so a score can move by a bit when nothing but the row count
changed. That bit is a rounding-scale change in a score and a categorical change in a selection: adjacent scores
reorder, the token is dispatched to a different set of weights, and its routed output moves by an amount that has
no relation to the size of the perturbation. A token whose selected set changes this way is called a flip below.

Replay exists so that a trainer can reuse the routing a sampler already decided. What it fixes is the selection
and not the gating. Measured at hidden 4096, 256 experts, top-k 8, bfloat16 on one H200 under
``CUBLAS_WORKSPACE_CONFIG=:16:8``: a replayed one-row call returns the packed call's expert set exactly, while the
gating scores it returns for those same experts differ from the packed call's by 0.0017e-01 against a top-score
scale of 1.356e-01. Reusing a selection closes the categorical exposure and leaves the numerical one open, so it
is not a substitute for making the gate's product independent of its row count.

The guarantee itself does not depend on the workspace configuration, which is why this module does not set one.
Where the derived selection stops following the row count does: it flips through 32 rows under ``:16:8`` and
through 128 rows under ``:4096:8``, measured against a 16384-row reference at this geometry. The replayed
selection is invariant at every row count under both.

Both assertions are integer comparisons, deliberately. A tolerance cannot express "the same experts", and a
tolerance-shaped assertion on the scores is exactly what lets a flip through: the score moves by less than any
bound a reviewer would object to, and the token still goes somewhere else.

The two tests are a pair. The invariance test is worth nothing unless a derived selection really does follow the
row count on the device running it, and the repro test is what establishes that. It fails if a one-row call
agrees with a packed call everywhere, which would mean the invariance below is a property of the hardware and not
of the replay path.
"""

from __future__ import annotations

import pytest

from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

# The width at which the gate's product is row-count dependent. The configuration class default of 2048 is not: a
# contraction of 2048 into 256 columns returns bit-identical logits at 1, 2, 4 and 32 rows for four seeds under
# both workspace configurations, so it cannot exercise this property at all.
HIDDEN = 4096

# Tokens scored per seed, and the height of the packed call every other row count is compared against.
TOKENS = 4096

# Row counts the replayed selection must survive. Where the derived selection stops being exposed depends on the
# workspace configuration -- past 32 rows under ``:16:8`` and past 128 under ``:4096:8`` -- so the range spans
# both sides of either boundary. The replayed selection is asserted across all of it, because the path it takes
# never consults the gate's ordering and so has no boundary.
ROW_COUNTS = (1, 2, 3, 4, 8, 16, 32, 64, 128)

# A flip is a near-tie event, so the repro test samples for one instead of constructing it, and the sample has to
# be deep because a seed is not reliably productive: of seeds 0 through 23 at this geometry, 11 reach a flip
# inside TOKENS one-row scorings and 13 reach none, for 12 flips in 98304 scorings altogether.
#
# The seeds below are those 11, ordered by how few scorings each needs -- 199, 431, 519, 1137, 1679, 1992, 2349,
# 2441, 2962, 3268 and 3423. The order is what makes the depth cheap rather than slow: the search stops at the
# first flip, so on a device that reproduces this ordering it spends 199 scorings, and the remaining seeds are
# only paid for when the leading ones come up empty. On a device whose kernel selection differs these are 11
# arbitrary draws rather than 11 known-good ones, which is the case the depth is for -- at the 11-in-24 rate
# measured here, three seeds would leave a 16e-02 chance of a spurious failure and eleven leave 1e-03.
#
# The rate is a property of the geometry and not of the router. The width, the expert count, the number of
# experts selected and the score dtype each move the near-tie population that a flip needs, so a change to any of
# them needs its own measurement rather than these constants.
SEED_ORDER = (9, 10, 4, 7, 22, 17, 21, 3, 15, 14, 6)

# The scale of the load-balancing bias buffer once it has taken a few updates. Its value is immaterial to the
# replayed path, which is part of what the invariance test pins: ``routed_experts`` is consulted before
# ``expert_bias`` is, so a bias cannot reorder a replayed selection.
EXPERT_BIAS_SCALE = 1e-3


def _build_router(torch, config):
    # Imported here rather than at module scope: the configuration class needs only transformers, while the layer
    # module pulls the GPU stack, and collection has to succeed on a machine that has neither a device nor those
    # packages so that the skip below is what is reported instead of a collection error.
    from arctic_platform.model.implementations.moe.layers.moe import TokenChoiceTopKRouter

    return TokenChoiceTopKRouter(
        dim=HIDDEN,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        score_func="sigmoid",
        route_norm=True,
        route_scale=1.0,
    ).to(device="cuda", dtype=torch.bfloat16)


def _draw(torch, config, router, seed):
    """Give the router a seeded gate and return the window to score and the load-balancing bias to score it with."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = (
        torch.randn(HIDDEN, config.num_experts, generator=generator, device="cuda", dtype=torch.float32)
        * config.initializer_range
    ).to(torch.bfloat16)
    with torch.no_grad():
        router.gate.weight.copy_(weight.t())
    window = torch.randn(TOKENS, HIDDEN, generator=generator, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(config.num_experts, generator=generator, device="cuda", dtype=torch.float32)
    return window, bias * EXPERT_BIAS_SCALE


def _score_in_blocks(torch, router, window, bias, rows, *, replay=None):
    """Score ``window`` in blocks of ``rows`` rows, as one call per block, and stitch the results back together.

    ``replay`` is a selection to hand the router back, sliced to each block, or ``None`` to let it derive its own.
    Returns the selected expert indices, the gating scores and the per-expert token counts summed over the blocks.
    """
    indices, scores, counts = [], [], None
    for start in range(0, window.shape[0], rows):
        block = window[start : start + rows]
        given = None if replay is None else replay[start : start + block.shape[0]]
        with torch.no_grad():
            block_scores, block_indices, block_counts = router(block, bias, routed_experts=given)
        indices.append(block_indices)
        scores.append(block_scores)
        counts = block_counts if counts is None else counts + block_counts
    return torch.cat(indices), torch.cat(scores), counts


def _expert_sets(indices):
    return [frozenset(row) for row in indices.tolist()]


def _cuda_or_skip():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device: the row-count dependence is a property of the GEMM kernel the device picks")
    return torch


@pytest.mark.integration
def test_a_derived_selection_follows_the_row_count():
    """A router deriving its own selection routes some token to a different expert set in a one-row call.

    This is the half of the pair that fails if the exposure is absent. Were a one-row call to agree with a packed
    call everywhere, the invariance test below would pass without asserting anything about replay, and this test
    is what says so. It reports the scorings it spent, so a rate that has drifted is visible as a budget running
    close before it is visible as a flake.
    """
    torch = _cuda_or_skip()
    config = Qwen3_5MoeConfig(hidden_size=HIDDEN)
    router = _build_router(torch, config)

    spent = 0
    found = None
    for seed in SEED_ORDER:
        window, bias = _draw(torch, config, router, seed)
        packed_indices, packed_scores, packed_counts = _score_in_blocks(torch, router, window, bias, TOKENS)
        repeat_indices, repeat_scores, repeat_counts = _score_in_blocks(torch, router, window, bias, TOKENS)

        # The floor for everything below. Without it a difference between two row counts could be read as a
        # repeat of the same call disagreeing with itself, and the comparison would carry no claim at all.
        assert torch.equal(packed_indices, repeat_indices), (
            f"seed {seed}: a {TOKENS}-row call repeated selected different experts than itself, so no comparison "
            "against it can be read as a row-count effect"
        )
        assert torch.equal(packed_scores, repeat_scores), (
            f"seed {seed}: a {TOKENS}-row call repeated returned different gating scores than itself, by "
            f"{float((packed_scores.double() - repeat_scores.double()).abs().max()):.3e}"
        )
        assert torch.equal(packed_counts, repeat_counts), f"seed {seed}: the per-expert counts are not repeatable"

        one_row_indices, _, _ = _score_in_blocks(torch, router, window, bias, 1)
        spent += TOKENS
        packed_sets = _expert_sets(packed_indices)
        flipped = [
            token for token, selected in enumerate(_expert_sets(one_row_indices)) if selected != packed_sets[token]
        ]
        if flipped:
            found = (seed, flipped)
            break

    assert found is not None, (
        f"no one-row call selected a different expert set than the {TOKENS}-row call in {spent} scorings across "
        f"seeds {SEED_ORDER}. Either the row-count dependence is gone on this device -- which would make the "
        "invariance test in this module vacuous, since there would be nothing for replay to protect against -- "
        "or the flip rate has fallen below what this budget can see."
    )
    seed, flipped = found
    print(
        f"DERIVED seed {seed}: {len(flipped)} of {TOKENS} tokens changed expert set between a one-row call and a "
        f"{TOKENS}-row call (first at token {flipped[0]}), scorings spent {spent}"
    )


@pytest.mark.integration
def test_a_replayed_selection_does_not_follow_the_row_count():
    """Handed a selection to reuse, the router returns it unchanged at every row count.

    The comparison is against the same tokens' selection inside a packed call, at the row counts a derived
    selection is exposed at and above them, and it is an integer comparison of the selected sets. The per-expert
    counts are compared beside it, because those are what size the expert dispatch and they are derived from the
    selection rather than carried with it, so a path that returned the right indices and the wrong histogram
    would still misroute.

    The gating scores are reported and not asserted. Replay does not make them invariant -- it settles which
    experts are read, not how precisely their scores were computed -- and asserting a bound on them here would
    either encode the current kernel's rounding or be wide enough to permit a different selection.
    """
    torch = _cuda_or_skip()
    config = Qwen3_5MoeConfig(hidden_size=HIDDEN)
    router = _build_router(torch, config)
    seed = SEED_ORDER[0]
    window, bias = _draw(torch, config, router, seed)

    packed_indices, packed_scores, packed_counts = _score_in_blocks(torch, router, window, bias, TOKENS)
    repeat_indices, repeat_scores, repeat_counts = _score_in_blocks(torch, router, window, bias, TOKENS)

    # The floor the row-count comparisons below are measured against. A repeat of the same call has to agree with
    # itself before a disagreement at another row count can be attributed to the row count.
    assert torch.equal(
        packed_indices, repeat_indices
    ), f"seed {seed}: a {TOKENS}-row call repeated selected different experts than itself"
    assert torch.equal(packed_scores, repeat_scores), (
        f"seed {seed}: a {TOKENS}-row call repeated returned different gating scores than itself, by "
        f"{float((packed_scores.double() - repeat_scores.double()).abs().max()):.3e}"
    )
    assert torch.equal(packed_counts, repeat_counts), f"seed {seed}: the per-expert counts are not repeatable"

    packed_sets = _expert_sets(packed_indices)
    replay = packed_indices.clone()

    failures: list[str] = []
    reported: list[str] = []
    for rows in ROW_COUNTS:
        indices, scores, counts = _score_in_blocks(torch, router, window, bias, rows, replay=replay)
        flipped = [token for token, selected in enumerate(_expert_sets(indices)) if selected != packed_sets[token]]
        if flipped:
            failures.append(
                f"rows={rows}: {len(flipped)} of {TOKENS} tokens were replayed into a different expert set than "
                f"the same token in a {TOKENS}-row call (first: {', '.join(str(t) for t in flipped[:8])}). A "
                "replayed token is processed by the weights the caller named, so a difference here is not "
                "bounded by any tolerance -- the token went somewhere the caller did not ask for."
            )
        if not torch.equal(counts, packed_counts):
            worst = int((counts - packed_counts).abs().max())
            failures.append(
                f"rows={rows}: the per-expert token counts disagree with the {TOKENS}-row call by up to {worst}, "
                "so the dispatch would be sized for a distribution the selection does not have"
            )
        reported.append(f"rows={rows} d={float((scores.double() - packed_scores.double()).abs().max()):.3e}")

    print(
        f"REPLAYED seed {seed} scale {float(packed_scores.abs().max()):.3e} gating-score deviation from the "
        f"{TOKENS}-row call: "
        + ", ".join(reported)
    )
    assert not failures, "a replayed expert selection changed with the row count of the call:\n  " + "\n  ".join(
        failures
    )
