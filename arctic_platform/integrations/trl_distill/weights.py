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

"""TRL ``WeightTransferProtocol``: ignore local params, Arctic syncs remotely."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


class ArcticOPDWeightTransfer:
    """``weight_transfer=`` backend. Does not stream ``named_parameters()`` over NCCL."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def init_weight_transfer(self) -> None:
        return None

    def pause(self) -> None:
        return None

    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]] | None = None) -> dict:
        if iterator is not None:
            for _name, _tensor in iterator:
                pass
        return self.client.sync_weights()

    def resume(self) -> None:
        return None

    def destroy(self) -> None:
        return None

    def sync(self, model: Any = None) -> dict:
        del model
        return self.send_weights(None)
