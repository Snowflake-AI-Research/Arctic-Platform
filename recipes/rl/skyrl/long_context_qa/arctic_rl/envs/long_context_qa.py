r"""Long-context multi-hop QA env for skyrl-gym.

Single-turn env: the model emits ``<think> ... </think> \boxed{<answer>}``,
and ``long_context_qa_reward.compute_score`` is applied to extract the boxed
answer and match it against the ground truth (see ``REWARD_CALC_TYPE`` for
matching modes).

Required ``extras`` (forwarded by ``PromptDataset`` from the parquet — both
verl-format and SkyRL-format rows work):

  - ``reward_model.ground_truth`` *or* ``reward_spec.ground_truth``: gold
    answer (string or list of strings)
  - ``extra_info``: passed through to ``compute_score`` (currently unused
    by the matcher, kept for parity with the BIRD env shape)
  - ``data_source``: passed through (verl-style routing; benign for LoongRL)
"""

from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from .long_context_qa_reward import compute_score


class LongContextQAEnv(BaseTextEnv):
    """Single-turn long-context multi-hop QA env."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        reward_block = extras.get("reward_model") or extras.get("reward_spec") or {}
        ground_truth = reward_block.get("ground_truth")
        if ground_truth is None:
            raise ValueError(
                "LongContextQAEnv requires `reward_model.ground_truth` (or "
                "`reward_spec.ground_truth`) in env_extras — gold answer is "
                "missing from this prompt's parquet row."
            )
        self.ground_truth = ground_truth
        self.extra_info: Dict[str, Any] = dict(extras.get("extra_info") or {})
        self.data_source: str = extras.get("data_source", "long_context_qa")

    def _get_reward(self, response: str) -> float:
        result = compute_score(
            data_source=self.data_source,
            solution_str=response,
            ground_truth=self.ground_truth,
            extra_info=self.extra_info,
        )
        return float(result["score"])

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # Single-turn: one response -> one reward -> done.
        return BaseTextEnvStepOutput(
            observations=[],
            reward=self._get_reward(action),
            done=True,
            metadata={},
        )
