"""Recipe-local Cortex adapter for the TRL GSM8K driver.

Lives next to ``run_qwen3_1.7b_gsm8k_grpo_cortex.py``, which puts this directory's
parent on ``sys.path``. It is not ``arctic_platform.integrations.trl``: that
package is the on-prem adapter. This one reshapes packed TRL rows for Cortex
and normalizes forward results into ``batch.logprobs``.
"""
