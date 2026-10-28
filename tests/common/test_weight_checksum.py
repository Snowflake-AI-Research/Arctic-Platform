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

"""The weight-sync transport checksum: trainer-side sent sums, colocated aggregation, and the comparison."""

from __future__ import annotations

import asyncio
import base64
import math
import os
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.common.utils.weight_sync import sync_weights_cpu_file
from arctic_platform.common.utils.weight_sync import sync_weights_cuda_ipc
from arctic_platform.common.utils.weight_sync import sync_weights_cuda_ipc_low_mem
from arctic_platform.common.weight_checksum import WEIGHT_CHECKSUM_RTOL
from arctic_platform.common.weight_checksum import compare_weight_checksums
from arctic_platform.common.weight_checksum import tensor_l2_sq
from arctic_platform.testing_utils import TestCasePlus

# 1 + 4, 9 + 16, and bf16 0.25 + 2.25: every square and sum is exact in float64.
_PARAM_L2_SQ = {"a": 5.0, "b": 25.0, "c": 2.5}
_TOTAL = 32.5


class _Remote:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _ipc_payload(names: list[str]) -> dict:
    return {
        "names": names,
        "dtype_names": ["float32"] * len(names),
        "shapes": [[1]] * len(names),
        "ipc_handles_pickled": base64.b64encode(pickle.dumps([{} for _ in names])).decode("utf-8"),
        "num_params": len(names),
    }


class _Trainer:
    def __init__(self, rank: int, sent: dict[str, float | None]):
        names = list(sent)
        self.gather_cuda_ipc_handles = _Remote(lambda: {**_ipc_payload(names), "sent_l2_sq": sum(sent.values())})
        self.get_parameter_names = _Remote(lambda: names)
        self.get_cuda_ipc_handle = _Remote(lambda name: {**_ipc_payload([name]), "sent_l2_sq": sent[name]})
        self.release_ipc_handles = _Remote(lambda: True)
        written = {"num_params": len(names), "sent_l2_sq": sum(sent.values())} if rank == 0 else {"num_params": 0}
        self.gather_and_save_state_dict = _Remote(lambda path: written)


class _Sampler:
    """One replica whose TP ranks each received ``received[rank][name]``. ``None`` is a chunk without a value."""

    def __init__(self, received: list[dict[str, float | None]]):
        def _ranks(names):
            return [None if None in (r[n] for n in names) else sum(r[n] for n in names) for r in received]

        self.load_weights_cuda_ipc = _Remote(lambda payload: {"rank_recv_l2_sq": _ranks(payload["names"])})
        self.load_weights_cuda_ipc_chunk = _Remote(
            lambda payload, validate: {"rank_recv_l2_sq": _ranks(payload["names"])}
        )
        self.load_weights_from_shm_path = _Remote(lambda path: {"rank_recv_l2_sq": _ranks(list(_PARAM_L2_SQ))})


class _Pool:
    _config = object()

    def __init__(self, workers):
        self._workers = workers
        self.num_replicas = len(workers)


def _run(sync, *args):
    with patch("arctic_platform.common.utils.weight_sync.ray.get", side_effect=lambda value: value):
        return asyncio.run(sync(*args))


class TestCompareWeightChecksums(TestCasePlus):
    def test_tensor_l2_sq_is_float64_and_chunk_invariant(self):
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        self.assertEqual(tensor_l2_sq(value).dtype, torch.float64)
        self.assertEqual(tensor_l2_sq(value).item(), 30.0)
        self.assertEqual(tensor_l2_sq(value.t(), chunk_numel=3).item(), 30.0)

    def test_match(self):
        result = compare_weight_checksums({"training rank 0": _TOTAL}, {"r0": _TOTAL, "r1": _TOTAL})
        self.assertEqual(
            result,
            {
                "sent_l2_sq": _TOTAL,
                "recv_l2_sq": {"r0": _TOTAL, "r1": _TOTAL},
                "checksum_rtol": WEIGHT_CHECKSUM_RTOL,
                "checksum_match": True,
            },
        )

    def test_tolerance_boundary(self):
        """2**30 + 1 is 9.3e-10 relative to 2**30, inside 1e-9; 2**30 + 2 (1.9e-9) is not."""
        sent = {"training rank 0": 2.0**30}
        self.assertTrue(compare_weight_checksums(sent, {"r0": 2.0**30 + 1})["checksum_match"])
        self.assertFalse(compare_weight_checksums(sent, {"r0": 2.0**30 + 2})["checksum_match"])

    def test_mismatch_names_the_receiver(self):
        result = compare_weight_checksums({"training rank 0": _TOTAL}, {"r0": _TOTAL, "r1": 27.5})
        self.assertFalse(result["checksum_match"])
        self.assertEqual(result["checksum_error"], "received {'r1': 27.5}, sent 32.5")

    def test_missing_values_fail_instead_of_passing(self):
        result = compare_weight_checksums({"training rank 0": None}, {"r0": _TOTAL, "r1": None})
        self.assertFalse(result["checksum_match"])
        self.assertEqual(
            result["checksum_error"],
            "no sent checksum from ['training rank 0']; no received checksum from ['r1']",
        )
        self.assertEqual(
            compare_weight_checksums({}, {})["checksum_error"],
            "no sent checksum from any training rank; no received checksum from any sampling rank",
        )

    def test_non_finite_checksums_do_not_match(self):
        result = compare_weight_checksums({"t0": math.inf}, {"r0": math.inf})

        self.assertFalse(result["checksum_match"])
        self.assertIn("non-finite sent checksum", result["checksum_error"])

    def test_large_values_stay_finite(self):
        total = tensor_l2_sq(torch.tensor([2.0**64])).item()

        self.assertEqual(total, 2.0**128)

    def test_training_ranks_must_agree(self):
        result = compare_weight_checksums({"training rank 0": _TOTAL, "training rank 1": 30.0}, {"r0": _TOTAL})
        self.assertFalse(result["checksum_match"])
        self.assertIn("training ranks sent different checksums", result["checksum_error"])


class TestColocatedSyncChecksum(TestCasePlus):
    def test_cuda_ipc_compares_every_rank(self):
        trainers = [_Trainer(0, _PARAM_L2_SQ), _Trainer(1, _PARAM_L2_SQ)]
        pool = _Pool([_Sampler([_PARAM_L2_SQ, _PARAM_L2_SQ])])
        lp_pool = _Pool([_Sampler([_PARAM_L2_SQ])])
        result = _run(sync_weights_cuda_ipc, trainers, pool, lp_pool)
        self.assertEqual(
            result,
            {
                "status": "ok",
                "sent_l2_sq": _TOTAL,
                "recv_l2_sq": {
                    "sampling replica 0 rank 0": _TOTAL,
                    "sampling replica 0 rank 1": _TOTAL,
                    "log_prob replica 0 rank 0": _TOTAL,
                },
                "checksum_rtol": WEIGHT_CHECKSUM_RTOL,
                "checksum_match": True,
            },
        )

    def test_low_memory_sums_every_chunk(self):
        trainers = [_Trainer(0, _PARAM_L2_SQ), _Trainer(1, _PARAM_L2_SQ)]
        result = _run(sync_weights_cuda_ipc_low_mem, trainers, _Pool([_Sampler([_PARAM_L2_SQ, _PARAM_L2_SQ])]))
        self.assertEqual(result["sent_l2_sq"], _TOTAL)
        self.assertEqual(
            result["recv_l2_sq"], {"sampling replica 0 rank 0": _TOTAL, "sampling replica 0 rank 1": _TOTAL}
        )
        self.assertTrue(result["checksum_match"])

    def test_low_memory_catches_one_corrupt_chunk_on_one_rank(self):
        corrupt = {**_PARAM_L2_SQ, "b": 24.0}
        trainers = [_Trainer(0, _PARAM_L2_SQ)]
        result = _run(sync_weights_cuda_ipc_low_mem, trainers, _Pool([_Sampler([_PARAM_L2_SQ, corrupt])]))
        self.assertFalse(result["checksum_match"])
        self.assertEqual(result["checksum_error"], "received {'sampling replica 0 rank 1': 31.5}, sent 32.5")

    def test_low_memory_chunk_without_a_value_leaves_the_total_unavailable(self):
        partial = {**_PARAM_L2_SQ, "b": None}
        trainers = [_Trainer(0, _PARAM_L2_SQ)]
        result = _run(sync_weights_cuda_ipc_low_mem, trainers, _Pool([_Sampler([partial])]))
        self.assertEqual(result["recv_l2_sq"], {"sampling replica 0 rank 0": None})
        self.assertFalse(result["checksum_match"])

    def test_cpu_file_uses_rank_zero_the_only_writer(self):
        trainers = [_Trainer(0, _PARAM_L2_SQ), _Trainer(1, _PARAM_L2_SQ)]
        path = os.path.join(self.get_auto_remove_tmp_dir(), "weights.pt")
        result = _run(sync_weights_cpu_file, path, trainers, _Pool([_Sampler([_PARAM_L2_SQ])]))
        self.assertEqual(result["sent_l2_sq"], _TOTAL)
        self.assertEqual(result["recv_l2_sq"], {"sampling replica 0 rank 0": _TOTAL})
        self.assertTrue(result["checksum_match"])


class _Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.b = nn.Parameter(torch.tensor([[3.0], [4.0]]))
        self.c = nn.Parameter(torch.tensor([0.5, 1.5], dtype=torch.bfloat16))


def _trainer(rank: int = 0):
    # `DeepSpeedWorker` is a Ray actor class; this is the plain class behind it.
    worker = DeepSpeedWorker.__ray_metadata__.modified_class(rank, 2, 29500)
    worker.engine = SimpleNamespace(module=_Module())
    worker._device = "cpu"
    return worker


class TestTrainerSentChecksum(TestCasePlus):
    def test_send_weights_reports_what_it_handed_the_sender(self):
        worker = _trainer()
        shipped = []
        worker._weight_sender = SimpleNamespace(send=lambda weights: shipped.extend(weights) or {"status": "done"})
        result = worker.send_weights()
        self.assertEqual(result, {"status": "done", "sent_l2_sq": _TOTAL})
        self.assertEqual([name for name, _ in shipped], ["a", "b", "c"])

    def test_weight_norm_squares_the_same_way(self):
        self.assertEqual(_trainer().weight_norm(), {"norm": _TOTAL**0.5, "sq_sum": _TOTAL, "num_params": 3})

    def test_only_rank_zero_reports_the_file_it_wrote(self):
        path = os.path.join(self.get_auto_remove_tmp_dir(), "weights.pt")
        self.assertEqual(_trainer(0).gather_and_save_state_dict(path)["sent_l2_sq"], _TOTAL)
        self.assertNotIn("sent_l2_sq", _trainer(1).gather_and_save_state_dict(path))
