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

"""CPU controls for owned startup diagnostics; no workers or model downloads are started."""

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.correctness.harness import cortex
from arctic_platform.correctness.harness import execution
from arctic_platform.correctness.harness.cortex import CorrectnessCortexTransport
from arctic_platform.correctness.harness.cortex import UnsafeCleanupError
from arctic_platform.correctness.onboarding import onboard

BATCH_ERROR = (
    "AssertionError: Check batch related parameters. train_batch_size is not equal to "
    "micro_batch_per_gpu * gradient_acc_step * world_size 8 != 1 * 1 * 2.0"
)


@pytest.mark.parametrize("encoding", ["mapping", "json", "http-repr", "human-log", "inline-msg"])
def test_local_cortex_creation_failure_reports_owned_deepspeed_cause(tmp_path, encoding):
    child = "owned-native-child"
    path = tmp_path / "training-zone-control.log"
    remote_traceback = (
        "Traceback (most recent call last):\n"
        '  File "deepspeed/runtime/config.py", line 1000, in _batch_assertion\n'
        "    assert train_batch == micro_batch * grad_acc * self.world_size\n"
        + BATCH_ERROR
    )
    if encoding == "human-log":
        text = f"12:00:00 ERROR [zone] Failed to create job {child}\n{remote_traceback}\n12:00:01 INFO done\n"
    elif encoding == "inline-msg":
        text = json.dumps({"msg": f"Failed to create job {child}\n{remote_traceback}"}) + "\n"
    else:
        text = json.dumps({"msg": f"Failed to create job {child}", "exc": remote_traceback}) + "\n"
    path.write_text(text)
    detail = {
        "message": f"job {child} failed",
        "zone_id": path.stem,
        "zone_log_path": str(path),
        "init_debug": None,
    }
    reason = (
        json.dumps({"detail": detail})
        if encoding == "json"
        else f"sub-job training failed: 500: {detail!r}" if encoding == "http-repr" else detail
    )
    result = cortex._local_creation_failure(reason, {"training": child})
    assert result.startswith(f"Remote server error: {BATCH_ERROR}\n")
    assert f"Server log: {path}" in result and remote_traceback in result
    assert "INFO done" not in result


@pytest.mark.parametrize(
    "message",
    [
        "Failed to create job owned-native-child-extra\n",
        "Failed to create job owned-native-child \n",
        "Failed to create job unrelated-child\n",
        "prefix Failed to create job owned-native-child\n",
        "Failed to create job owned-native-child: ",
    ],
)
def test_local_cortex_creation_failure_rejects_inline_message_near_prefixes(message):
    record = json.dumps({"msg": message + BATCH_ERROR})
    assert cortex._creation_failure_traceback(record, "owned-native-child") is None


@pytest.mark.parametrize(
    "broken",
    [
        "different-child",
        "different-message-child",
        "different-zone",
        "missing",
        "directory",
        "symlink",
        "unreadable",
        "malformed-reason",
        "oversized-reason",
        "no-traceback",
    ],
)
def test_local_cortex_creation_failure_never_attributes_unrelated_or_unreadable_log(tmp_path, monkeypatch, broken):
    child = "owned-native-child"
    path = tmp_path / "training-zone-control.log"
    detail = {"message": f"job {child} failed", "zone_id": path.stem, "zone_log_path": str(path)}
    if broken == "directory":
        path.mkdir()
    elif broken != "missing":
        path.write_text(
            json.dumps(
                {
                    "msg": f"Failed to create job {('unrelated-child' if broken == 'different-child' else child)}",
                    "exc": "no exception in this record" if broken == "no-traceback" else BATCH_ERROR,
                }
            )
            + "\n"
        )
    if broken == "different-message-child":
        detail["message"] = "job unrelated-child failed"
    elif broken == "different-zone":
        detail["zone_id"] = "training-zone-unrelated"
    elif broken == "symlink":
        link = tmp_path / "training-zone-link.log"
        link.symlink_to(path)
        detail.update(zone_id=link.stem, zone_log_path=str(link))
    elif broken == "unreadable":
        original_open = os.open

        def opened(value, *args, **kwargs):
            if Path(value) == path:
                raise PermissionError("unreadable control")
            return original_open(value, *args, **kwargs)

        monkeypatch.setattr(cortex.os, "open", opened)
    reason = (
        "sub-job failed: {malformed}"
        if broken == "malformed-reason"
        else "{" + "x" * 65536 if broken == "oversized-reason" else detail
    )
    assert cortex._local_creation_failure(reason, {"training": child}) is None


def test_local_cortex_creation_failure_selects_exact_json_record_and_bounds_log_read(tmp_path):
    path = tmp_path / "training-zone-control.log"
    child = "owned-native-child"
    path.write_text(
        "x" * (1 << 20)
        + "\n"
        + json.dumps({"msg": f"Failed to create job {child}", "exc": BATCH_ERROR})
        + "\n"
        + json.dumps({"msg": "Failed to create job unrelated-child", "exc": "ValueError: stale unrelated failure"})
        + "\n"
    )
    result = cortex._local_creation_failure(
        {"message": f"job {child} failed", "zone_id": path.stem, "zone_log_path": str(path)},
        {"training": child},
    )
    assert BATCH_ERROR in result
    assert "stale unrelated failure" not in result and len(result) < 2000


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("recoverable", [False, True])
def test_local_cortex_creation_failure_real_poll_preserves_cause_and_banner(
    tmp_path, monkeypatch, capsys, local, recoverable
):
    child = "owned-native-child"
    path = tmp_path / "training-zone-control.log"
    path.write_text(json.dumps({"msg": f"Failed to create job {child}", "exc": BATCH_ERROR}))
    transport = object.__new__(CorrectnessCortexTransport)
    transport.config = CortexConfig(base_url="http://127.0.0.1:8003" if local else "http://remote.test")
    transport.poll_interval = 0.1
    transport.job_id = "owned-parent"
    transport._native_children = {"training": child}
    detail = {"message": f"job {child} failed", "zone_id": path.stem, "zone_log_path": str(path)}
    if not recoverable:
        detail["zone_log_path"] = str(tmp_path / "missing.log")
    monkeypatch.setattr(CortexTransport, "_job", lambda self: {"status": "failed", "reason": detail})
    with pytest.raises(RuntimeError) as caught:
        transport._wait_running()
    reason = str(caught.value)
    assert "terminal state 'failed'" in reason
    if local and recoverable:
        assert reason.startswith(f"Remote server error: {BATCH_ERROR}\n")
        assert isinstance(caught.value.__cause__, RuntimeError)
    else:
        assert "Remote server error:" not in reason
    onboard._onboarding_failure(reason)
    assert reason.splitlines()[0] in capsys.readouterr().err


@pytest.mark.parametrize("unsafe_cleanup", [False, True])
def test_onboarding_target_creation_error_leads_final_banner_and_preserves_unsafe_stop(
    tmp_path, monkeypatch, capsys, unsafe_cleanup
):
    args = execution.parse_args(["onboard", "--config", str(tmp_path / "model.json"), "--test", "single-step-grads"])
    args.output_dir = tmp_path
    args.checkout_root = tmp_path
    cause = f"Remote server error: {BATCH_ERROR}"
    reason = cause + "\nServer log: /tmp/raydss/training-zone-control.log"
    if unsafe_cleanup:
        reason = "UnsafeCleanupError: Original workload failed: RuntimeError: " + reason

    def child(command, **kwargs):
        out = args.output_dir / "validation" / "4-layers"
        (out / "report.json").write_text(
            json.dumps(
                {
                    "totals": {"error": 1},
                    "results": [{"test_id": args.test_id, "arm": "gas1", "outcome": "error", "reason": reason}],
                }
            )
        )
        (out / "batch.json").write_text(json.dumps({"stopped_unsafe": unsafe_cleanup}))
        return SimpleNamespace(stdout=io.StringIO(reason + "\n"), wait=lambda: 1)

    monkeypatch.setattr(onboard.subprocess, "Popen", child)
    error_type = UnsafeCleanupError if unsafe_cleanup else RuntimeError
    with pytest.raises(error_type, match="8 != 1") as caught:
        validation = onboard._run_final_regression(args, {"layers": 4})
        onboard._require_validation_pass(validation, "final 4-layer regression")
    assert str(caught.value).splitlines()[0] == f"single-step-grads / gas1: {reason.splitlines()[0]}"
    assert "Server log:" in str(caught.value) and "Complete log:" in str(caught.value)
    if unsafe_cleanup:
        assert "Further onboarding stopped" in str(caught.value)
        assert not (tmp_path / "validation" / "4-layers" / "validation.json").exists()
    onboard._onboarding_failure(str(caught.value))
    printed = capsys.readouterr().err
    assert f"!!!!! ONBOARDING FAILED : {str(caught.value).splitlines()[0]} !!!!!" in printed
    assert "8 != 1 * 1 * 2.0" in printed
