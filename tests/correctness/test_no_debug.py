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

"""CPU admission controls and seven-check CLI registration without model downloads or a target service."""

import json
import re
import subprocess
import sys
import textwrap

from arctic_platform.testing_utils import TestCasePlus

_BLOCK_DEBUG = """
import importlib.abc
import importlib
import sys

class BlockDebug(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "arctic_platform.debug" or fullname.startswith("arctic_platform.debug."):
            raise ModuleNotFoundError("Added debug modules are unavailable", name=fullname)

sys.meta_path.insert(0, BlockDebug())
try:
    importlib.import_module("arctic_platform.debug")
except ModuleNotFoundError as exc:
    assert exc.name == "arctic_platform.debug"
else:
    raise AssertionError("debug import blocker is not active")
"""


class TestNoDebugCorrectness(TestCasePlus):
    def _run_isolated(self, program, *arguments, block_debug=True):
        env = self.get_env()
        env.update(
            CUDA_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        return subprocess.run(
            [
                sys.executable,
                "-c",
                (_BLOCK_DEBUG if block_debug else "import sys\n") + textwrap.dedent(program),
                *arguments,
            ],
            cwd=self.repo_root_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    def _cli(self, *arguments):
        return self._run_isolated(
            """
            import runpy
            sys.argv = ["arctic_platform.correctness", *sys.argv[1:]]
            try:
                runpy.run_module("arctic_platform.correctness", run_name="__main__")
            except SystemExit as exc:
                assert exc.code in (None, 0), exc.code
            """,
            *arguments,
            block_debug=False,
        )

    def test_list_registers_all_seven_checks(self):
        config_root = self.get_auto_remove_tmp_dir()
        result = self._cli("list", "--config-dir", str(config_root))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        tests = result.stdout.split("Tests:", 1)[1]
        identifiers = re.findall(r"^([a-z][a-z0-9-]+)\s+", tests, re.MULTILINE)
        self.assertEqual(len(identifiers), 7)
        self.assertCountEqual(
            identifiers,
            [
                "single-step-grads",
                "single-step-optimizer",
                "checkpoint-resume-loss",
                "inference-checkpoint-loss",
                "e2e-sft-train-validate",
                "rl-weight-sync",
                "rl-router-replay",
            ],
        )

    def test_onboard_help_loads_full_suite(self):
        result = self._cli("onboard", "--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--test-id", result.stdout)
        self.assertIn("single-step-grads", result.stdout)
        self.assertIn("single-step-optimizer", result.stdout)
        for identifier in [
            "single-step-grads",
            "single-step-optimizer",
            "checkpoint-resume-loss",
            "inference-checkpoint-loss",
            "e2e-sft-train-validate",
            "rl-weight-sync",
            "rl-router-replay",
        ]:
            self.assertIn(identifier, re.sub(r"-\s+", "-", result.stdout))

    def test_run_help_loads_full_suite(self):
        result = self._cli("run", "--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--test", result.stdout)
        self.assertIn("--tokens", result.stdout)
        self.assertIn("--reference-records-dir", result.stdout)

    def test_onboarding_regression_tokens_survive_cli_and_api_without_debug(self):
        result = self._run_isolated(
            """
            from arctic_platform.correctness import api
            from arctic_platform.correctness.harness.execution import _validate_regression_tokens

            for test_id in ("single-step-grads", "single-step-optimizer"):
                request, options = api.parse_run_args([
                    "--config", "synthetic.config", "--test", test_id, "--tokens", "128",
                ])
                assert request.tests == (test_id,)
                assert options.tokens == 128
                assert _validate_regression_tokens(api._namespace(request, options)) == 128
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_onboarding_cpu_admission_rejects_before_model_or_target_work(self):
        directory = self.get_auto_remove_tmp_dir()
        config = directory / "tiny.json"
        config.write_text(
            json.dumps(
                {
                    "model_name": "unused/no-download-model",
                    "dtype": "bfloat16",
                    "max_seq_len": 32,
                    "training_gpus": 1,
                    "sampling_gpus": 0,
                    "log_prob_gpus": 0,
                    "training": {
                        "ds_worker_config": {
                            "sp_size": 1,
                            "attn_implementation": "sdpa",
                            "max_tokens_per_mb": 32,
                        },
                        "ds_config": {"data_types": {"grad_accum_dtype": "fp32"}},
                    },
                    "sampling": {},
                    "backend": {"type": "onprem"},
                }
            )
        )
        result = self._run_isolated(
            """
            import argparse
            import json
            from pathlib import Path
            import torch
            import huggingface_hub
            import transformers
            from arctic_platform.correctness.harness.config import load_config
            from arctic_platform.correctness.onboarding import onboard

            def forbidden(*args, **kwargs):
                raise AssertionError("CPU admission reached model/download work")

            assert not torch.cuda.is_available()
            huggingface_hub.hf_hub_download = forbidden
            huggingface_hub.snapshot_download = forbidden
            for factory in (transformers.AutoConfig, transformers.AutoModelForCausalLM,
                            transformers.AutoTokenizer):
                factory.from_pretrained = forbidden
            onboard._model_shape = forbidden
            onboard._parent = forbidden
            root = Path(sys.argv[1]).parent
            parser = argparse.ArgumentParser()
            onboard.add_arguments(parser)
            args = parser.parse_args([
                "--config", sys.argv[1], "--test-id", "single-step-grads",
                "--base-url", "http://127.0.0.1:1", "--cache-root", str(root / "cache"),
                "--output-dir", str(root / "work"), "--test-spec", str(root / "spec.json"),
                "--reference-records-dir", str(root / "records"),
            ])
            config = load_config(args.config)
            try:
                onboard._onboarding_preflight(args, config, ["single-step-grads"])
            except ValueError as exc:
                assert "CUDA is unavailable" in str(exc), str(exc)
            else:
                raise AssertionError("CPU onboarding was admitted")
            assert not (root / "cache").exists()
            assert not (root / "spec.json").exists()
            print("CPU_ADMISSION_REJECTED")
            """,
            str(config),
            block_debug=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CPU_ADMISSION_REJECTED", result.stdout)

    def test_participation_observes_backward_and_restores_hooks_without_debug(self):
        result = self._run_isolated(
            """
            from types import SimpleNamespace
            import torch
            from arctic_platform.correctness.harness.optimizer import WorkerParticipation
            from arctic_platform.correctness.harness.optimizer import merge_worker_receipts

            model = torch.nn.Linear(2, 1, bias=False)
            worker = SimpleNamespace(
                _engine=SimpleNamespace(module=model, global_steps=0), rank=0, job_id="cpu-child"
            )
            capture = WorkerParticipation(worker)
            try:
                model(torch.ones(1, 2)).sum().backward()
                worker._engine.global_steps = 1
                receipt = capture.finish()
                assert receipt["after"]["participation"]["parameters"]["weight"]["state"] == "active"
                merged = merge_worker_receipts([receipt], job_id="cpu-child", world_size=1)
                assert merged["training_child_id"] == "cpu-child"
                assert merged["parameters"]["weight"]["state"] == "active"
            finally:
                capture.close()
            assert not model.weight._backward_hooks
            assert not model._forward_hooks
            print("PARTICIPATION_OBSERVED")
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PARTICIPATION_OBSERVED", result.stdout)

    def test_participation_rejects_incomplete_rank_evidence_without_debug(self):
        result = self._run_isolated(
            """
            from arctic_platform.correctness.harness.optimizer import merge_worker_receipts

            for receipts in ([], [{"rank": 0, "job_id": "cpu-child"}]):
                try:
                    merge_worker_receipts(receipts, job_id="cpu-child", world_size=2)
                except ValueError as exc:
                    assert "every actual training worker" in str(exc)
                else:
                    raise AssertionError("incomplete rank evidence was accepted")
            print("INCOMPLETE_RANKS_REJECTED")
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("INCOMPLETE_RANKS_REJECTED", result.stdout)
