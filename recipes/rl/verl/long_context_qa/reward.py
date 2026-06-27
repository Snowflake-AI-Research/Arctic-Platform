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

"""Reward scoring for long-context QA tasks (LoongRL-style).

This is a self-contained reward function shipped with the recipe so that it works against an unmodified upstream verl
checkout. It is wired into training via verl's ``custom_reward_function`` hook:

    custom_reward_function.path=<this file>
    custom_reward_function.name=compute_score

verl invokes ``compute_score`` per sample with keyword arguments (``data_source``, ``solution_str``, ``ground_truth``,
``extra_info``), matching the signature of verl's ``default_compute_score``.

Scoring mode is selected via the ``REWARD_CALC_TYPE`` env var:
  - pure_exact_match (default): substring match of ground truth in \\boxed{...}
  - format_exact_match: exact match with format and overflow penalties
  - format_f1_score: token-level F1 with format enforcement

Reference: https://github.com/rStar-RL/LoongRL
"""

import os
import re
import string
from collections import Counter


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kwargs):
    """verl custom reward entry point.

    Only ``solution_str`` and ``ground_truth`` are used; the remaining arguments are accepted for compatibility with
    verl's reward-manager call convention.
    """
    solution_str = (solution_str or "").strip()
    reward_calc_type = os.getenv("REWARD_CALC_TYPE", "pure_exact_match")
    if reward_calc_type == "pure_exact_match":
        return _pure_exact_match_in_string(solution_str, ground_truth)
    elif reward_calc_type == "format_exact_match":
        return _format_exact_match_in_string(solution_str, ground_truth)
    elif reward_calc_type == "format_f1_score":
        return _format_f1_score(solution_str, ground_truth)
    else:
        raise ValueError(f"Unknown reward_calc_type: {reward_calc_type}")


def _pure_exact_match_in_string(solution_str, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if is_gt_in_pred(pred, truth):
                    return 1.0
        except Exception as e:
            print(f"Error encountered: {e}")
            return 0

    return 0


def _format_exact_match_in_string(solution_str, ground_truth):
    max_boxed_limit = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_multiple_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if max_boxed_limit > 0 and solution_str.count("\\boxed") > max_boxed_limit:
                raise ValueError(f"Too many boxed parts: {solution_str.count('\\boxed')} > {max_boxed_limit}")

            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if punish_multiple_braces > 0:
                    if pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1:
                        raise ValueError(f"Multiple braces found in pred: {pred}")
                if is_gt_in_pred(pred, truth):
                    retval = 1.0
                    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
                    if len(pred) - len(truth) > answer_over_flow_limit:
                        retval -= 1.0
                    eot_over_flow_limit = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))
                    tail_len = len(solution_str) - (solution_str.rfind(pred) + len(pred))
                    if tail_len > eot_over_flow_limit:
                        retval -= 1.0
                    max_retval = max(max_retval, retval)

        except Exception as e:
            print(f"Error encountered: {e}")
            return max_retval

    return max_retval


def _format_f1_score(solution_str, ground_truth):
    max_boxed_limit = int(os.getenv("MAX_BOXED_LIMIT", 1))
    punish_multiple_braces = int(os.getenv("PUNISH_MULTIPLE_BRACES", 1))
    f1_score_threshold = float(os.getenv("F1_SCORE_THRESHOLD", 0.5))
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    max_retval = 0
    for truth in ground_truth:
        try:
            boxed_part = last_boxed_only_string(solution_str)
            retval = 0
            if max_boxed_limit > 0 and solution_str.count("\\boxed") > max_boxed_limit:
                raise ValueError(f"Too many boxed parts: {solution_str.count('\\boxed')} > {max_boxed_limit}")

            if boxed_part is not None:
                pred = remove_boxed(boxed_part)
                if punish_multiple_braces > 0:
                    if pred.count("{") > 1 or pred.count("}") > 1 or pred.count("\\") > 1:
                        raise ValueError(f"Multiple braces found in pred: {pred}")
                score = qa_f1_score(pred, truth)
                if score > f1_score_threshold:
                    retval = score
                    answer_over_flow_limit = int(os.getenv("ANSWER_OVER_FLOW_LIMIT", 128))
                    if len(pred) - len(truth) > answer_over_flow_limit:
                        retval = 0
                    eot_over_flow_limit = int(os.getenv("EOT_OVER_FLOW_LIMIT", 32))
                    tail_len = len(solution_str) - (solution_str.rfind(pred) + len(pred))
                    if tail_len > eot_over_flow_limit:
                        retval = 0
                    max_retval = max(max_retval, retval)

        except Exception as e:
            print(f"Error encountered: {e}")
            return max_retval

    return max_retval


def normalize_answer(s):
    """SQuAD-style normalization: lowercase, strip punctuation/articles/whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    return f1_score(normalized_prediction.split(), normalized_ground_truth.split())


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def normalize_text(text):
    text = re.sub(r"[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：“”《》？]", " ", text.lower())
    text = re.sub(r"import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_gt_in_pred(pred, ground_truth, verbose=False):
    if pred is None and ground_truth is None:
        print("WARNING: Both None")
        return True
    if pred is None or ground_truth is None:
        return False

    try:
        ss1 = normalize_text(pred)
        ss2 = normalize_text(ground_truth)
        if verbose:
            print(ss1, ss2)
        return ss2 in ss1 or ss1 in ss2
    except Exception:
        return ground_truth in pred or pred in ground_truth
