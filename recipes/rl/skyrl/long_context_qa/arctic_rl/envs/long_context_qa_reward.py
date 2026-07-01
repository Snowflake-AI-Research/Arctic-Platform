# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reward scoring for long-context multi-hop QA (LoongRL-style).

Self-contained scorer vendored from
``Arctic-Platform/recipes/rl/verl/long_context_qa/reward.py`` so the SkyRL
recipe doesn't depend on that file (and matches the verl recipe's behavior
exactly).

Scoring mode is selected via the ``REWARD_CALC_TYPE`` env var:
  - ``pure_exact_match`` (default): substring match of ground truth inside ``\\boxed{...}``
  - ``format_exact_match``: exact match with format and overflow penalties
  - ``format_f1_score``: token-level F1 with format enforcement

``compute_score`` returns a dict ``{"score": <float>}`` so it matches the
shape used by ``bird_reward.compute_score`` in this directory — keeps the
``BaseTextEnv`` subclasses uniform.

Reference: https://github.com/rStar-RL/LoongRL
"""

import os
import re
import string
from collections import Counter
from typing import Any, Dict, Optional


def compute_score(
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth=None,
    extra_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, float]:
    """Reward entry point. Returns ``{"score": <float>}``."""
    solution_str = (solution_str or "").strip()
    mode = os.getenv("REWARD_CALC_TYPE", "pure_exact_match")
    if mode == "pure_exact_match":
        score = _pure_exact_match(solution_str, ground_truth)
    elif mode == "format_exact_match":
        score = _format_exact_match(solution_str, ground_truth)
    elif mode == "format_f1_score":
        score = _format_f1_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Unknown REWARD_CALC_TYPE: {mode}")
    return {"score": float(score)}


# ---------------------------------------------------------------------------
# Scoring modes
# ---------------------------------------------------------------------------


def _pure_exact_match(solution_str: str, ground_truth) -> float:
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    for truth in ground_truth:
        try:
            boxed = _last_boxed_only_string(solution_str)
            if boxed is None:
                continue
            pred = _remove_boxed(boxed)
            if _is_gt_in_pred(pred, truth):
                return 1.0
        except Exception as exc:  # noqa: BLE001 — match verl recipe behavior
            print(f"long_context_qa reward error: {exc}")
            return 0.0
    return 0.0


def _format_exact_match(solution_str: str, ground_truth) -> float:
    max_boxed = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    answer_overflow = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
    eot_overflow = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    best = 0.0
    for truth in ground_truth:
        try:
            if max_boxed > 0 and solution_str.count("\\boxed") > max_boxed:
                raise ValueError("too many \\boxed segments")
            boxed = _last_boxed_only_string(solution_str)
            if boxed is None:
                continue
            pred = _remove_boxed(boxed)
            if punish_braces > 0 and (pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1):
                raise ValueError(f"multiple braces in pred: {pred}")
            if not _is_gt_in_pred(pred, truth):
                continue
            score = 1.0
            if len(pred) - len(truth) > answer_overflow:
                score -= 1.0
            tail = len(solution_str) - (solution_str.rfind(pred) + len(pred))
            if tail > eot_overflow:
                score -= 1.0
            best = max(best, score)
        except Exception as exc:  # noqa: BLE001
            print(f"long_context_qa reward error: {exc}")
            return best
    return best


def _format_f1_score(solution_str: str, ground_truth) -> float:
    max_boxed = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    f1_threshold = float(os.getenv("F1_SCORE_THRESHOLD", 0.5))
    answer_overflow = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
    eot_overflow = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    best = 0.0
    for truth in ground_truth:
        try:
            if max_boxed > 0 and solution_str.count("\\boxed") > max_boxed:
                raise ValueError("too many \\boxed segments")
            boxed = _last_boxed_only_string(solution_str)
            if boxed is None:
                continue
            pred = _remove_boxed(boxed)
            if punish_braces > 0 and (pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1):
                raise ValueError(f"multiple braces in pred: {pred}")
            score = _qa_f1(pred, truth)
            if score <= f1_threshold:
                continue
            if len(pred) - len(truth) > answer_overflow:
                score = 0.0
            tail = len(solution_str) - (solution_str.rfind(pred) + len(pred))
            if tail > eot_overflow:
                score = 0.0
            best = max(best, score)
        except Exception as exc:  # noqa: BLE001
            print(f"long_context_qa reward error: {exc}")
            return best
    return best


# ---------------------------------------------------------------------------
# Helpers (SQuAD-style normalization + LaTeX \boxed{} extraction)
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    """SQuAD-style: lowercase, strip articles/punctuation/whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(prediction, ground_truth) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def _qa_f1(prediction: str, ground_truth: str) -> float:
    norm_pred = _normalize_answer(prediction)
    norm_gt = _normalize_answer(ground_truth)
    return _f1(norm_pred.split(), norm_gt.split())


def _last_boxed_only_string(s: str) -> Optional[str]:
    """Return the last ``\\boxed{...}`` (or ``\\fbox{...}``) substring."""
    idx = s.rfind("\\boxed")
    if "\\boxed " in s:
        return "\\boxed " + s.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right = None
    depth = 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    if right is None:
        return None
    return s[idx : right + 1]


def _remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def _normalize_text(text: str) -> str:
    text = re.sub(r"[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：“”《》？]", " ", text.lower())
    text = re.sub(r"import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_gt_in_pred(pred: Optional[str], ground_truth: Optional[str]) -> bool:
    if pred is None and ground_truth is None:
        return True
    if pred is None or ground_truth is None:
        return False
    try:
        a = _normalize_text(pred)
        b = _normalize_text(ground_truth)
        return b in a or a in b
    except Exception:  # noqa: BLE001
        return ground_truth in pred or pred in ground_truth
