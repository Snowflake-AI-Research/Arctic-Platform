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

"""Ray binds the inference-checkpoint worker calls the server actually issues."""

from __future__ import annotations

import inspect

from ray._common import signature as ray_sig
from ray.actor import is_class_method
from ray.actor import is_static_method

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker


def _parameters(name: str):
    cls = DeepSpeedWorker.__ray_metadata__.modified_class
    method = inspect.unwrap(getattr(cls, name))
    keep_first = is_class_method(method) or is_static_method(cls, name)
    return ray_sig.extract_signature(method, ignore_first=not keep_first)


def _bind(name: str, *args, **kwargs):
    ray_sig.validate_args(_parameters(name), args, kwargs)


def test_inference_checkpoint_worker_calls_match_ray_signatures():
    """The calls issued after DeepSpeed workers load weights bind under Ray's validator.

    ``Signature.bind`` reports a mismatch as ``too many positional arguments``. A keyword-only
    ``_ray_trace_ctx`` slot cannot absorb an extra positional, so each call below is the form the
    server uses.
    """
    _bind("__init__", 0, 1, 2)
    _bind("initialize", "10.0.0.1", {"job": True})
    _bind("get_ip")
    _bind("forward_backward", batch={"batch": {}})
    _bind("compute_log_probs", {"batch": {}})
    _bind("forward_no_grad", {"batch": {}})
    _bind("step", learning_rate=2e-05)
    _bind("save_checkpoint", path="/tmp/ckpt", export_hf=True)
    _bind("load_checkpoint", path="/tmp/ckpt")
    _bind("prune_checkpoint_dirs", "/tmp/ckpt", 1)
