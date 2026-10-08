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

"""The correctness gateway starts after references and can use a multi-node allocation."""

from arctic_platform.correctness.harness import dss_driver


def test_available_gpus_reads_a_hostfile_without_a_prestarted_gateway(tmp_path, monkeypatch) -> None:
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("node-a slots=8\nnode-b slots=8\n")
    monkeypatch.delenv("DSS_GATEWAY_URL", raising=False)
    monkeypatch.setenv("DSS_GATEWAY_HOSTFILE", str(hostfile))

    assert dss_driver.available_gpus() == 16
