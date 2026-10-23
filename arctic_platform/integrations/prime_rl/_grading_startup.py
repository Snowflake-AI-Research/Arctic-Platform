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

"""Runtime monkey-patch that adds the SWE agentic *grading policy* to the official
PrimeIntellect ``verifiers`` package — with no fork of verifiers or prime-rl.

Why a startup module + ``.pth`` (mirrors prime-rl's own ``sitecustomize`` +
``zz_prime_rl_startup.pth``): rollouts and scoring run in EnvWorker subprocesses that
prime-rl spawns with ``mp.get_context("spawn")``, so they do **not** inherit an
in-process patch applied by the launcher. A ``.pth`` ``import`` line runs at *every*
interpreter startup in the venv — orchestrator and each EnvWorker alike — so the patch
is present wherever verifiers' rollout/scoring code runs.

This module is force-included at the wheel *root* (published as
``arctic_prime_rl_grading_startup``) and imported by ``zz_arctic_prime_rl_grading.pth``.
It therefore must stay self-contained: it imports only the *public* ``verifiers`` /
``renderers`` packages (the ones public prime-rl pins) and ``pydantic``, lazily inside
``apply`` — never ``arctic_platform`` (importing the integration package would trip its
dependency-group gate at interpreter startup) and never any internal fork. Every symbol
it touches (``AgentConfig``, ``response_from_generate``, ``RolloutSession``, ``Task``,
``seed``, ``ToolCallParseStatus``) is a stock public symbol; the patch only *adds* the
grading behavior on top of an unmodified verifiers install.

The policy is opt-in via ``AgentConfig.grading_policy`` (off by default = no behavior
change). When on, a rollout is terminated and its reward forced to 0 on any of:
  * a format-invalid turn — a tool call the renderer could not cleanly parse (any non-OK
    ``ToolCallParseStatus``: malformed structure, invalid JSON, unclosed block, missing
    name, or an unknown tool);
  * a per-turn output-budget overrun — a model call that finished on ``length``; or
  * a framework turn/token limit (``max_turns`` / token caps) or failed compaction ending
    the run.
Surfaced as the W&B metric ``grading_policy/disqualified`` (fraction of rollouts zeroed).

It reproduces the six-edit verifiers change from the outside: an ``AgentConfig`` field,
``parse_issues`` carried on each ``Response``, two response ``@stop`` predicates injected
into the rollout session, and a disqualification branch in ``Task.score``.
"""

from __future__ import annotations

import functools
import sys

# Stop conditions that disqualify a rollout under the grading policy (the framework
# turn/token limits + failed compaction + the two grading @stops below).
_GRADING_DISQUALIFYING_STOPS = frozenset(
    {
        "max_turns",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "compaction_failed",
        "grading_invalid_format",
        "grading_per_turn_budget",
    }
)

_APPLIED = False


def _grading_disqualified(trace) -> bool:
    """Whether the grading policy should zero this rollout: a disqualifying stop condition,
    or any truncation (a framework limit, or a ``length``-finished final turn)."""
    return trace.stop_condition in _GRADING_DISQUALIFYING_STOPS or trace.is_truncated


def grading_invalid_format(response) -> bool:
    """Response @stop: end the rollout on a format-invalid turn — any non-OK tool-call parse
    status the renderer flagged (carried on the response as ``parse_issues``)."""
    return bool(getattr(response, "parse_issues", None))


def grading_per_turn_budget(response) -> bool:
    """Response @stop: end the rollout when a turn exhausts its per-turn output budget (the
    model call finished on ``length``)."""
    return response.finish_reason == "length"


def apply() -> None:
    """Apply the grading-policy patches to verifiers. Idempotent and defensive: a failure
    (e.g. verifiers not installed in this process) is logged to stderr, never raised, so it
    cannot break interpreter startup for unrelated processes sharing the venv."""
    global _APPLIED
    if _APPLIED:
        return
    try:
        _apply()
    except Exception as exc:  # pragma: no cover - defensive startup guard
        print(f"[arctic prime-rl] grading-policy patch skipped: {exc!r}", file=sys.stderr)
    _APPLIED = True


def _apply() -> None:
    # Importing verifiers here (not at module top) keeps the startup hook cheap in processes
    # that never use verifiers, and lets the defensive guard in apply() swallow its absence.
    from pydantic.fields import FieldInfo

    # --- Edit 1: AgentConfig.grading_policy (a real field so the recipe's
    #     env.agent.grading_policy=true validates and reaches trace.agent.config). ---
    from verifiers.v1.configs.agent import AgentConfig

    if "grading_policy" not in AgentConfig.model_fields:
        AgentConfig.model_fields["grading_policy"] = FieldInfo(annotation=bool, default=False)
        AgentConfig.model_rebuild(force=True)

    # --- Edits 2 & 3: parse_issues. Wrap response_from_generate to compute every non-OK
    #     tool-call parse status from the raw result and stash it on the Response. Carried as a
    #     plain attribute via object.__setattr__ (no Response schema change); the @stop reads it
    #     with getattr(..., "parse_issues", None). ---
    from renderers.base import ToolCallParseStatus
    from verifiers.v1.clients import train as _train

    _orig_rfg = _train.response_from_generate
    if not getattr(_orig_rfg, "_arctic_grading_wrapped", False):

        @functools.wraps(_orig_rfg)
        def response_from_generate(result, *args, **kwargs):
            response = _orig_rfg(result, *args, **kwargs)
            issues = [
                str(status)
                for tc in (result.get("tool_calls") or [])
                if (status := getattr(tc, "status", None)) is not None
                and status != ToolCallParseStatus.OK
            ] or None
            try:
                object.__setattr__(response, "parse_issues", issues)
            except Exception:
                pass
            return response

        response_from_generate._arctic_grading_wrapped = True
        _train.response_from_generate = response_from_generate

    # --- Edits 4 & 5: inject the two response @stops. RolloutSession is the sole consumer of
    #     response_stops and receives the trace (-> trace.agent.config), so prepend the grading
    #     stops to its response_stops kwarg when the policy is on. They run before the task's own
    #     stops, so a format-invalid / over-budget turn ends the rollout (and score zeros it)
    #     before any task stop can conclude it otherwise. ---
    from verifiers.v1.session import RolloutSession

    _orig_rs_init = RolloutSession.__init__
    if not getattr(_orig_rs_init, "_arctic_grading_wrapped", False):

        @functools.wraps(_orig_rs_init)
        def __init__(self, *args, **kwargs):
            trace = kwargs.get("trace")
            config = getattr(getattr(trace, "agent", None), "config", None)
            if getattr(config, "grading_policy", False) and "response_stops" in kwargs:
                stops = list(kwargs.get("response_stops") or [])
                kwargs["response_stops"] = [
                    grading_invalid_format,
                    grading_per_turn_budget,
                    *stops,
                ]
            return _orig_rs_init(self, *args, **kwargs)

        __init__._arctic_grading_wrapped = True
        RolloutSession.__init__ = __init__

    # --- Edit 6: Task.score disqualification. Before scoring, if the policy is on and the
    #     rollout is disqualified, seed + record 0 for every reward/judge, record the metric, and
    #     skip the (now meaningless) reward hooks; otherwise score normally and record the clean
    #     side of the fraction. ---
    from verifiers.v1.task import Task
    from verifiers.v1.utils.decorators import seed

    _orig_score = Task.score
    if not getattr(_orig_score, "_arctic_grading_wrapped", False):

        @functools.wraps(_orig_score)
        async def score(self, trace, runtime=None):
            if getattr(self, "scoring_deferred", False):
                return await _orig_score(self, trace, runtime)
            config = getattr(trace.agent, "config", None)
            grading_policy = getattr(config, "grading_policy", False)
            if grading_policy and _grading_disqualified(trace):
                rewards = self.hooks("reward")
                judges = self.plugged_judges()
                seed(trace.rewards, (fn.__name__ for fn in rewards))
                seed(trace.rewards, (judge.reward_name for judge in judges))
                for fn in rewards:
                    trace.record_reward(fn.__name__, 0.0, getattr(fn, "_vf_weight", 1.0))
                for judge in judges:
                    trace.record_reward(judge.reward_name, 0.0, judge.config.weight)
                trace.record_metric("grading_policy/disqualified", 1.0)
                return
            await _orig_score(self, trace, runtime)
            if grading_policy:
                trace.record_metric("grading_policy/disqualified", 0.0)

        score._arctic_grading_wrapped = True
        Task.score = score
