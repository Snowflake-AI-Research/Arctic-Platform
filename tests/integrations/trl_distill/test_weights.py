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

from __future__ import annotations

from arctic_platform.integrations.trl_distill import ArcticOPDWeightTransfer


class FakeOPD:
    def __init__(self):
        self.synced = 0
        self.reset = 0

    def sync_weights(self, cuda_ipc=None, low_memory=None):
        self.synced += 1
        return {"ok": True}

    def reset_student_prefix_cache(self, drain=True, timeout_s=60.0, retry_interval_s=0.1):
        self.reset += 1
        return {"ok": True}


def test_weight_transfer_syncs_student_only():
    client = FakeOPD()
    transfer = ArcticOPDWeightTransfer(client)
    transfer.init_weight_transfer()
    assert transfer.sync(model=object()) == {"ok": True}
    assert transfer.send_weights(iter(())) == {"ok": True}
    transfer.pause()
    transfer.resume()
    transfer.destroy()
    assert client.synced == 2
    assert client.reset == 2
