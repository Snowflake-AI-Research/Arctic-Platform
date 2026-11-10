"""Offline numerical checks for the Arctic v2 runner and native interfaces."""

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file

from arctic_platform.testing_utils import (
    TestCasePlus, execute_subprocess_async, require_torch_gpu, set_seed, torch_assert_close,
)


@require_torch_gpu
@pytest.mark.vllm
class TestV2RunnerIntegration(TestCasePlus):
    def setUp(self):
        super().setUp()
        self.output_dir = Path(self.get_auto_remove_tmp_dir())
        self.model_dir = self.output_dir / "model"
        set_seed(17)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=130, hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=256, bos_token_id=1, eos_token_id=129,
            tie_word_embeddings=False,
        ))
        model.save_pretrained(self.model_dir)
        from arctic_platform.inference.common.swiftkv import LlamaSwiftKVConfig
        swift_dir = self.output_dir / "swiftkv"
        config = LlamaSwiftKVConfig(**model.config.to_dict(), num_key_value_layers=1)
        config.architectures = ["LlamaSwiftKVForCausalLM"]
        config.save_pretrained(swift_dir)
        weights = {}
        for name, tensor in model.state_dict().items():
            if name.startswith("model.layers.1.self_attn."):
                for projection in ("q_proj", "k_proj", "v_proj"):
                    name = name.replace(f".{projection}.", f".{projection}_swiftkv.")
            weights[name] = tensor.contiguous()
        weights["model.norm_swiftkv.weight"] = weights["model.norm.weight"].clone()
        save_file(weights, swift_dir / "model.safetensors")

    def run_mode(self, name, options, enabled=True, plugins="arctic_inference"):
        env = self.get_env()
        env = {key: value for key, value in env.items()
            if not (key.startswith("VLLM_") and ("_PORT" in key or "_SERVICE_" in key))}
        env.update(VLLM_PLUGINS=plugins, ARCTIC_INFERENCE_ENABLED=str(int(enabled)),
                   HF_HUB_OFFLINE="1", NVIDIA_TF32_OVERRIDE="0", TRITON_F32_DEFAULT="ieee")
        output = self.output_dir / f"{name}.json"
        execute_subprocess_async(
            [sys.executable, str(Path(__file__).with_name("v2_smoke_worker.py")),
             str(self.model_dir), str(output), json.dumps(options)], env=env, timeout=600,
        )
        return json.loads(output.read_text())

    def assert_parity(self, baseline, actual, atol=1e-5):
        for scenario, rows in baseline.items():
            self.assertEqual(len(rows), len(actual[scenario]))
            for reference, result in zip(rows, actual[scenario]):
                self.assertEqual(reference["tokens"], result["tokens"])
                for key in ("logprobs", "prompt_logprobs"):
                    self.assertEqual(len(reference[key]), len(result[key]))
                    # FP32 kernels vary with batch shape and reduction order.
                    torch_assert_close(
                        torch.tensor([v for v in reference[key] if v is not None]),
                        torch.tensor([v for v in result[key] if v is not None]), rtol=0, atol=atol,
                        msg=lambda detail: f"{scenario}: {key}\n{detail}",
                    )

    def test_suffix_graphs_and_cortex_parity(self):
        baseline = self.run_mode("upstream", {"dtype": "float32"}, enabled=False, plugins="")
        self.assert_parity(baseline, self.run_mode("cortex", {"dtype": "float32"}, enabled=False))
        self.assert_parity(baseline, self.run_mode("suffix", {
            "dtype": "float32", "graphs": True, "engine": {"speculative_config": {"method": "suffix", "suffix_speculative_tokens": 3}},
        }))

    def test_neural_hybrid_graphs_and_level_two_sleep(self):
        engine = {"enable_prefix_caching": False}
        for dtype in ("float32", "float16"):
            with self.subTest(dtype=dtype):
                baseline = self.run_mode(f"baseline_{dtype}", {"dtype": dtype, "engine": engine})
                # FP16 exercises the compiled drafter kernels; batch shapes can
                # change rounding in the target attention and projection kernels.
                self.assert_parity(baseline, self.run_mode(f"hybrid_{dtype}", {
                    "dtype": dtype, "neural": True, "hybrid": True, "graphs": True, "sleep_levels": [1, 2],
                    "engine": engine,
                }), atol=1e-5 if dtype == "float32" else 1e-3)

    def test_swiftkv_row_selection_and_graphs(self):
        baseline = self.run_mode("swiftkv_all_rows", {"swiftkv": True, "all_swiftkv_rows": True})
        self.assert_parity(baseline, self.run_mode("swiftkv_eager", {"swiftkv": True}), atol=1e-3)
        self.assert_parity(baseline, self.run_mode("swiftkv_graphs", {"swiftkv": True, "graphs": True}), atol=1e-3)

    def test_swiftkv_flashinfer_selection(self):
        options = {"swiftkv": True, "engine": {"attention_config": {"backend": "FLASHINFER"}}}
        baseline = self.run_mode("flashinfer_all_rows", {**options, "all_swiftkv_rows": True})
        self.assert_parity(baseline, self.run_mode("flashinfer_selected", options), atol=1e-3)

    def test_swiftkv_ulysses_shift(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("SwiftKV Ulysses requires two GPUs")
        baseline = self.run_mode("swiftkv", {"swiftkv": True, "dtype": "float32"})
        self.assert_parity(baseline, self.run_mode("swiftkv_shift", {
            "swiftkv": True, "dtype": "float32", "graphs": True, "engine": {
                "ulysses_sequence_parallel_size": 2,
                "enable_shift_parallel": True, "shift_parallel_threshold": 8,
            },
        }))

    def test_neural_ulysses_shift_async(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Neural Ulysses requires two GPUs")
        options = {"dtype": "float32", "engine": {"enable_prefix_caching": False, "async_scheduling": True}}
        baseline = self.run_mode("neural_sp_reference", options)
        self.assert_parity(baseline, self.run_mode("neural_sp", {
            "dtype": "float32", "neural": True, "hybrid": True, "graphs": True,
            "engine": {**options["engine"], "ulysses_sequence_parallel_size": 2,
                "enable_shift_parallel": True, "shift_parallel_threshold": 8},
        }))

    def test_forest_shared_prefix_parity(self):
        # FP16 bounds the rounding from separately materialized prefix/suffix outputs.
        baseline = self.run_mode("baseline", {"long_prefix": True, "dtype": "float16"})
        self.assert_parity(baseline, self.run_mode("forest", {"forest": True, "dtype": "float16"}), atol=1e-3)

    def test_neural_model_length_boundary(self):
        options = {"dtype": "float32", "prefix_repeats": 28, "max_tokens": 32,
                   "engine": {"enable_prefix_caching": False}}
        baseline = self.run_mode("boundary_reference", options)
        self.assert_parity(baseline, self.run_mode("boundary_arctic", {**options, "neural": True}))

    def test_lstm_graphs_and_level_two_sleep(self):
        options = {"dtype": "float16", "engine": {"enable_prefix_caching": False}}
        baseline = self.run_mode("lstm_reference", options)
        self.assert_parity(baseline, self.run_mode("lstm", {
            **options, "neural": True, "lstm": True, "graphs": True, "sleep_levels": [2],
        }), atol=1e-3)

    def test_sp_tp_shift_graphs(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("SP=2/TP=2 requires four GPUs")
        baseline = self.run_mode("baseline", {"dtype": "float32", "engine": {"async_scheduling": False}})
        self.assert_parity(baseline, self.run_mode("shift", {"dtype": "float32", "graphs": True, "engine": {
            "async_scheduling": False, "ulysses_sequence_parallel_size": 2, "tensor_parallel_size": 2,
            "enable_shift_parallel": True, "shift_parallel_threshold": 8,
        }}))
