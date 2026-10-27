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

"""Stop a command's owned gateway when its lifetime pipe closes unexpectedly."""

from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path


def cleanup_owned_gateway(record_path: Path, token: str) -> None:
    from arctic_platform.correctness.harness import dss_driver

    with (record_path.parent / "owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Do not touch its registry; still cover an old launch that was never recorded.
            _cleanup_unrecorded_gateway(token)
            return
        try:
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if record.get("token") == token:
                    dss_driver._recover_owned_gateway(record_path)
        finally:
            _cleanup_unrecorded_gateway(token)


def _cleanup_unrecorded_gateway(token: str) -> None:
    from arctic_platform.correctness.harness import dss_driver

    # The owner may die between launching the gateway and publishing its ownership record.
    marker = f"DSS_CORRECTNESS_GATEWAY_OWNER={token}".encode()
    for directory in dss_driver._GATEWAY_PROC_ROOT.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            identity = dss_driver._gateway_process_identity(int(directory.name))
            if identity is None or identity["uid"] != os.getuid():
                continue
            if marker not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            if os.getpgid(identity["pid"]) != identity["pid"]:
                continue
            dss_driver._stop_stale_gateway(identity, "gateway.log")
            dss_driver._reap_owned_gateway_descendants(identity, token)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def main() -> None:
    descriptor, record_path, token = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    with os.fdopen(descriptor, "rb") as lifetime:
        print("ready", flush=True)
        os.dup2(2, 1)
        if lifetime.readline() == b"done\n":
            return

    # Import runtime dependencies only after owner death; startup needs only the standard library.
    cleanup_owned_gateway(record_path, token)


if __name__ == "__main__":
    main()
