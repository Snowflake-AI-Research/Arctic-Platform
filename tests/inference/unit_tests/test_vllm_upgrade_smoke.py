"""Single-GPU generation parity and sleep/wake across plugin configurations."""

import json
import sys
from pathlib import Path

import pytest
import torch

from arctic_platform.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    require_torch_gpu,
    torch_assert_close,
)


SMOKE = """
import json
import sys
from vllm import LLM, SamplingParams

kwargs = {}
if sys.argv[1] == "suffix":
    kwargs["speculative_config"] = {
        "method": "suffix", "suffix_speculative_tokens": 3,
    }
llm = LLM(
    model="Qwen/Qwen3-0.6B", max_model_len=256, max_num_seqs=4,
    gpu_memory_utilization=0.2, enforce_eager=True, seed=0,
    enable_sleep_mode=True, **kwargs,
)
params = SamplingParams(temperature=0, max_tokens=16, logprobs=3)
def generate():
    results = llm.generate(["The capital of France is", "2 + 2 ="], params)
    return [
        {
            "tokens": list(result.outputs[0].token_ids),
            "logprobs": [
                scores[token].logprob
                for token, scores in zip(
                    result.outputs[0].token_ids, result.outputs[0].logprobs
                )
            ],
        }
        for result in results
    ]
before = generate()
llm.sleep(level=1)
llm.wake_up()
assert generate() == before
with open(sys.argv[2], "w") as file:
    json.dump(before, file)
print("generation and sleep/wake passed")
"""


@pytest.mark.gpu
@require_torch_gpu
class TestVllmUpgradeSmoke(TestCasePlus):
    def test_plugin_parity_and_sleep_wake(self):
        output_dir = Path(self.get_auto_remove_tmp_dir())
        outputs = {}
        for mode in ("upstream", "cortex", "suffix"):
            env = self.get_env()
            env["VLLM_PLUGINS"] = "" if mode == "upstream" else "arctic_inference"
            env["ARCTIC_INFERENCE_ENABLED"] = "1" if mode == "suffix" else "0"
            output = output_dir / f"{mode}.json"
            execute_subprocess_async(
                [sys.executable, "-c", SMOKE, mode, str(output)],
                env=env, timeout=300,
            )
            outputs[mode] = json.loads(output.read_text())

        self.assertEqual(outputs["upstream"], outputs["cortex"])
        for baseline, suffix in zip(outputs["upstream"], outputs["suffix"]):
            self.assertEqual(baseline["tokens"], suffix["tokens"])
            # Speculation batches BF16 matmuls differently from single-token decode.
            torch_assert_close(
                torch.tensor(baseline["logprobs"]),
                torch.tensor(suffix["logprobs"]), rtol=0, atol=2e-6,
            )
