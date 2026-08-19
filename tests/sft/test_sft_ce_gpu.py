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
"""GPU tests for memory-efficient ``sft_ce`` (H8).

These exercise the real server-side CE path on CUDA. They skip when no GPU is
visible (``@require_torch_gpu``) — run them via autorun on the GPU box.

HTTP e2e: client process uses ``CUDA_VISIBLE_DEVICES=`` empty; the launched server
gets ``--server-cuda-visible-devices``. Do **not** force the torch CE fallback
here: whatever kernel the server has (flash-attn or torch) is what production uses.
"""

from __future__ import annotations

import sys

import pytest
import torch

from arctic_platform.sft.processor import sft_ce_sum_from_hidden
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import make_tied_lm_head_model
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close

# Worker-owned 8-port block; root conftest claims ``base`` for MASTER_PORT.
_PORT_BASE = get_unique_port_number()

# fp32 CE paths that implement the same math must match to float noise only
# (rtol=0). Do not widen atol to paper over a wrong kernel / shift / mask.
_FP32_ATOL = 1e-5


def _parse_step_metrics(out: str) -> tuple[list[float], list[float]]:
    """Parse demo lines ``step i/n loss=… grad_norm=…`` into parallel lists."""
    losses: list[float] = []
    grad_norms: list[float] = []
    for line in out.splitlines():
        if "loss=" not in line or "step " not in line:
            continue
        try:
            loss_s = line.split("loss=")[1].split()[0].rstrip(",")
            losses.append(float(loss_s))
        except (IndexError, ValueError):
            continue
        if "grad_norm=" in line:
            try:
                gn_s = line.split("grad_norm=")[1].split()[0].rstrip(",")
                grad_norms.append(float(gn_s))
            except (IndexError, ValueError):
                pass
    return losses, grad_norms


class TestParseStepMetrics(TestCasePlus):
    """CPU characterization of the e2e's stdout parser (no GPU needed)."""

    def test_loss_and_grad_norm_paired(self):
        out = "step 1/2 loss=1.5 grad_norm=3.0\nstep 2/2 loss=0.25 grad_norm=2.0"
        self.assertEqual(_parse_step_metrics(out), ([1.5, 0.25], [3.0, 2.0]))

    def test_trailing_commas_stripped(self):
        losses, grad_norms = _parse_step_metrics("step 1/1 loss=1.5, grad_norm=2.0,")
        self.assertEqual((losses, grad_norms), ([1.5], [2.0]))

    def test_grad_norm_optional(self):
        # A step line without grad_norm contributes a loss but no grad_norm, so
        # the lists diverge in length — the e2e's separate len==2 checks catch that.
        losses, grad_norms = _parse_step_metrics("step 1/2 loss=1.5\nstep 2/2 loss=0.25 grad_norm=2.0")
        self.assertEqual((losses, grad_norms), ([1.5, 0.25], [2.0]))

    def test_non_step_and_malformed_lines_ignored(self):
        out = "loading loss=9.9\nstep 1/1 loss=abc grad_norm=2.0\nblah"
        # "loading …" lacks "step "; the malformed-loss step line is dropped whole.
        self.assertEqual(_parse_step_metrics(out), ([], []))


@require_torch_gpu
class TestSftCeSumFromHiddenParityGPU(TestCasePlus):
    """Numeric + gradient parity for ``compute`` / ``memory`` vs full-logits CE."""

    def _reference(self, model, hidden, labels):
        V = model.config.vocab_size
        sh = model.lm_head(hidden)[:, :-1, :].float().reshape(-1, V)
        sl = labels[:, 1:].reshape(-1)
        ce = torch.nn.functional.cross_entropy(sh, sl, ignore_index=-100, reduction="sum")
        good = int((sl != -100).sum().item())
        return ce, good

    def test_value_parity_with_masking(self):
        set_seed(0)
        B, S, H, V = 2, 7, 8, 32
        model = make_tied_lm_head_model(H, V)
        hidden = torch.randn(B, S, H, device="cuda")
        labels = torch.randint(0, V, (B, S), device="cuda")
        labels[:, :2] = -100
        labels[1, :] = -100
        ref_ce, ref_good = self._reference(model, hidden, labels)
        for mode in ("compute", "memory"):
            ce, good = sft_ce_sum_from_hidden(model, hidden, labels, mode=mode)
            self.assertEqual(good, ref_good)
            torch_assert_close(ce, ref_ce, rtol=0, atol=_FP32_ATOL, msg=f"{mode}")

    def test_grad_parity_wrt_hidden(self):
        set_seed(1)
        B, S, H, V = 2, 6, 8, 24
        model = make_tied_lm_head_model(H, V)
        base = torch.randn(B, S, H, device="cuda")
        labels = torch.randint(0, V, (B, S), device="cuda")
        labels[:, 0] = -100

        h_full = base.clone().requires_grad_(True)
        ref_ce, _ = self._reference(model, h_full, labels)
        ref_ce.backward()
        g_full = h_full.grad.clone()

        for mode in ("compute", "memory"):
            h = base.clone().requires_grad_(True)
            model.lm_head.zero_grad()
            ce, _ = sft_ce_sum_from_hidden(model, h, labels, mode=mode)
            ce.backward()
            torch_assert_close(h.grad, g_full, rtol=0, atol=_FP32_ATOL, msg=f"{mode} grad")


@require_torch_gpu
@pytest.mark.gpu_serial
class TestSftCeHttpE2EModesGPU(TestCasePlus):
    """End-to-end: CPU-blanked client + local GPU server, all three sft_ce modes.

    Matches production topology: client ``CUDA_VISIBLE_DEVICES=`` empty, server
    child sees GPUs via ``server_cuda_visible_devices``. Per-step ``loss`` and
    ``grad_norm`` must match exactly across ``none`` / ``compute`` / ``memory``
    on the same batch (same math, same seeds — not "close enough").
    """

    def test_modes_match_loss_and_grad_norm_curves(self):
        ckpt = self.get_auto_remove_tmp_dir()
        losses_by_mode: dict[str, list[float]] = {}
        grad_norms_by_mode: dict[str, list[float]] = {}

        for mode in ("none", "compute", "memory"):
            env = self.get_env()
            env["CUDA_VISIBLE_DEVICES"] = ""  # client blank
            env["WANDB_DISABLED"] = "true"
            env.setdefault("HF_HOME", "/data-fast/huggingface")
            # HTTP port from this worker's block; DeepSpeed rendezvous uses a second probe.
            http_port = reserve_free_port(_PORT_BASE + 1, span=3)
            env["MASTER_PORT"] = str(reserve_free_port(_PORT_BASE + 4, span=4))
            cmd = [
                sys.executable,
                "-m",
                "arctic_platform.sft.examples.run_sft_http_demo",
                "--launch-local-server",
                "--server-cuda-visible-devices",
                "0",
                "--training-gpus",
                "1",
                "--steps",
                "2",
                "--loss-fn",
                "sft_ce",
                "--logits-optimization",
                mode,
                "--port",
                str(http_port),
                "--checkpoint-dir",
                str(ckpt / mode),
            ]
            result = execute_subprocess_async(cmd, env=env, timeout=600)
            out = "\n".join(result.stdout + result.stderr)
            step_losses, step_grad_norms = _parse_step_metrics(out)
            self.assertEqual(
                len(step_losses),
                2,
                f"mode={mode}: expected 2 step losses, got {step_losses}\n{out[-2000:]}",
            )
            self.assertEqual(
                len(step_grad_norms),
                2,
                f"mode={mode}: expected 2 step grad_norms, got {step_grad_norms}\n{out[-2000:]}",
            )
            losses_by_mode[mode] = step_losses
            grad_norms_by_mode[mode] = step_grad_norms

        # Exact match on the printed curves (same batch / seeds / math).
        self.assertEqual(losses_by_mode["none"], losses_by_mode["compute"])
        self.assertEqual(losses_by_mode["none"], losses_by_mode["memory"])
        self.assertEqual(grad_norms_by_mode["none"], grad_norms_by_mode["compute"])
        self.assertEqual(grad_norms_by_mode["none"], grad_norms_by_mode["memory"])
