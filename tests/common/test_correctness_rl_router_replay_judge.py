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

from arctic_platform.correctness.checks.rl_router_replay import judge

# Decoder layer 0 is dense: the sampler's capture is as wide as the decoder, the trainer has no router there.
DECODER_LAYERS = 3
MOE_LAYERS = (1, 2)
TOKENS = list(range(100, 110))


def _sample():
    routing = [[[(t * 7 + layer * 3 + k) % 16 for k in range(2)] for layer in range(DECODER_LAYERS)] for t in range(9)]
    marker = {"replay_id": "rr1:a", "capture_len": len(routing), "generation_len": 2}
    return {"tokens": TOKENS, "prompt_len": len(TOKENS) - 2, "marker": marker, "routing": routing}


def _rank(rank, sample, start, end, received, layers=MOE_LAYERS):
    capture = sample["routing"]
    positions = list(range(start, end))
    routed = [capture[p] if p < len(capture) else [[0, 0]] * DECODER_LAYERS for p in positions]
    router = [
        {"layer": layer, "phase": phase, "replayed": True, "indices": [row[layer] for row in routed]}
        for phase in ("forward", "recompute")
        for layer in layers
    ]
    call = {
        "padding": False,
        "input_ids": TOKENS[start:end],
        "position_ids": positions,
        "routed_experts": routed,
        "router": router,
    }
    return {"rank": rank, "received": {"rr1:a": received}, "calls": [call], "unattributed_router_calls": []}


def _names(samples, traces):
    failed, _ = judge(samples, traces, full_recompute=True)
    return [m.name for m in failed]


def test_sequence_parallel_ranks_receive_their_windows_and_dense_layers_have_no_router():
    sample = _sample()
    capture = sample["routing"]
    traces = [_rank(0, sample, 0, 5, capture[0:5]), _rank(1, sample, 5, 10, capture[5:9])]
    assert _names([sample], traces) == []


def test_unsharded_rank_receives_the_whole_capture():
    sample = _sample()
    assert _names([sample], [_rank(0, sample, 0, 10, sample["routing"])]) == []


def test_window_other_than_the_ranks_own_positions_fails():
    sample = _sample()
    capture = sample["routing"]
    traces = [_rank(0, sample, 0, 5, capture[0:5]), _rank(1, sample, 5, 10, capture[4:8])]
    assert _names([sample], traces) == ["rank 1: sample 0 routing received differs from generate"]


def test_whole_capture_on_a_sequence_parallel_rank_fails():
    sample = _sample()
    capture = sample["routing"]
    traces = [_rank(0, sample, 0, 5, capture), _rank(1, sample, 5, 10, capture[5:9])]
    assert _names([sample], traces) == ["rank 0: sample 0 routing received differs from generate"]


def test_moe_layer_without_invocation_on_one_rank_fails():
    sample = _sample()
    capture = sample["routing"]
    traces = [_rank(0, sample, 0, 5, capture[0:5]), _rank(1, sample, 5, 10, capture[5:9], layers=(1,))]
    assert _names([sample], traces) == [
        "rank 1 call 0: layer 2 has no forward router invocation",
        "rank 1 call 0: layer 2 has no recompute router invocation",
    ]


def test_no_router_in_any_call_fails():
    sample = _sample()
    capture = sample["routing"]
    traces = [_rank(0, sample, 0, 5, capture[0:5], layers=()), _rank(1, sample, 5, 10, capture[5:9], layers=())]
    assert _names([sample], traces) == ["no MoE router invocation in any traced call"]


# An 88-token row spans sixteen 6-token SP windows. The final window holds only right-padding, not sample tokens.
SP_SIZE = 16
SP_DECODER_LAYERS = 5
SP_MOE_LAYERS = (3, 4)
TOP_K = 8
PAD_ID = 0


def _sp_sample():
    tokens = [1000 + t for t in range(88)]
    routing = [
        [[(t * 5 + layer * 3 + k) % 32 for k in range(TOP_K)] for layer in range(SP_DECODER_LAYERS)] for t in range(87)
    ]
    marker = {"replay_id": "rr1:a", "capture_len": len(routing), "generation_len": 47}
    return {"tokens": tokens, "prompt_len": 41, "marker": marker, "routing": routing}


def _sp_rank(rank, sample, *, routed_experts=True):
    capture, row = sample["routing"], len(sample["tokens"])
    window = -(-row // SP_SIZE)
    packed = range(rank * window, (rank + 1) * window)
    held = [p for p in packed if p < row]
    zero = [[0] * TOP_K for _ in range(SP_DECODER_LAYERS)]
    routed = [capture[p] if p < len(capture) else zero for p in packed] if held and routed_experts else None
    router = [
        {
            "layer": layer,
            "phase": phase,
            "replayed": routed is not None,
            "indices": [r[layer] for r in routed] if routed is not None else [list(range(TOP_K))] * window,
        }
        for phase in ("forward", "recompute")
        for layer in SP_MOE_LAYERS
    ]
    call = {
        "padding": False,
        "input_ids": [sample["tokens"][p] if p < row else PAD_ID for p in packed],
        "position_ids": [p if p < row else 0 for p in packed],
        "routed_experts": routed,
        "router": router,
    }
    received = capture[min(held[0], len(capture)) : min(held[-1] + 1, len(capture))] if held else []
    return {"rank": rank, "received": {"rr1:a": received}, "calls": [call], "unattributed_router_calls": []}


def _sp_traces(sample):
    return [_sp_rank(rank, sample) for rank in range(SP_SIZE)]


def test_sequence_parallel_rank_holding_only_padding_is_not_compared():
    sample = _sp_sample()
    failed, metrics = judge([sample], _sp_traces(sample), full_recompute=True)
    assert [m.name for m in failed] == []
    counted = ("compared_positions", "forward_invocations", "recompute_invocations", "unattributed_tokens")
    assert [metrics[key] for key in counted] == [87.0, 30.0, 30.0, 8.0]


def test_padding_only_rank_receiving_routing_fails():
    sample = _sp_sample()
    traces = _sp_traces(sample)
    traces[15]["received"] = {"rr1:a": sample["routing"][84:87]}
    assert _names([sample], traces) == ["rank 15: sample 0 routing received differs from generate"]


def test_rank_holding_sample_positions_without_routed_experts_fails():
    sample = _sp_sample()
    traces = _sp_traces(sample)
    traces[14] = _sp_rank(14, sample, routed_experts=False)
    assert _names([sample], traces) == [
        "rank 14 call 0: no routed_experts handed to the model",
        "rank 14: sample 0 routing received differs from generate",
        "sample 0: captured positions compared in a model call",
    ]
