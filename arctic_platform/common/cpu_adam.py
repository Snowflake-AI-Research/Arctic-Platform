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

"""Non-fused AdamW arithmetic for DeepSpeed's CPU-offloaded optimizer state."""

from __future__ import annotations

import torch
from torch.optim.adamw import adamw


class _NonfusedAdamWOps:
    def __init__(self, native_ops):
        self.native_ops = native_ops

    def destroy_adam(self, optimizer_id):
        return self.native_ops.destroy_adam(optimizer_id)

    @torch.no_grad()
    def adam_update(
        self,
        optimizer_id,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction,
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
    ):
        if not bias_correction:
            raise ValueError("Non-fused CPU AdamW requires bias correction")
        # CPUAdam already increments its integer counter; PyTorch increments this
        # temporary scalar itself. Keep the checkpoint's native counter unchanged.
        state_step = torch.tensor(step - 1, dtype=torch.float32)
        adamw(
            [parameter],
            [gradient.to(exp_avg.dtype)],
            [exp_avg],
            [exp_avg_sq],
            [],
            [state_step],
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            maximize=False,
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )

    def adam_rollback(self, *args, **kwargs):
        raise NotImplementedError("Non-fused CPU AdamW does not support kernel rollback")


class _CpuAdamCheckpointState:
    """Preserve moment precision across PyTorch's parameter-dtype casting."""

    def __init__(self):
        self.moments = []

    def before_load(self, optimizer, state_dict):
        self.moments.clear()
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(optimizer.param_groups):
            return
        if any(
            len(saved["params"]) != len(current["params"])
            for saved, current in zip(saved_groups, optimizer.param_groups)
        ):
            return
        for saved, current in zip(saved_groups, optimizer.param_groups):
            for index, parameter in zip(saved["params"], current["params"]):
                state = state_dict["state"].get(index, {})
                dtype = torch.float32 if optimizer.fp32_optimizer_states else parameter.dtype
                if dtype == parameter.dtype:
                    continue
                moments = {
                    name: state[name].to(device=parameter.device, dtype=dtype, copy=True)
                    for name in ("exp_avg", "exp_avg_sq")
                    if name in state
                }
                if moments:
                    self.moments.append((parameter, moments))

    def after_load(self, optimizer):
        for parameter, moments in self.moments:
            optimizer.state[parameter].update(moments)
        self.moments.clear()


def configure_nonfused_cpu_adam(optimizer):
    """Keep CPUAdam's native validation, state allocation and checkpoint format.

    ZeRO's BF16 masters require an exact DeepSpeedCPUAdam instance. Replace only
    this instance's update kernel, not the optimizer class or ZeRO's validators.
    """
    if not optimizer.adam_w_mode:
        raise ValueError("Non-fused CPU arithmetic requires AdamW mode")
    optimizer.ds_opt_adam = _NonfusedAdamWOps(optimizer.ds_opt_adam)
    checkpoint_state = _CpuAdamCheckpointState()
    optimizer.register_load_state_dict_pre_hook(checkpoint_state.before_load)
    optimizer.register_load_state_dict_post_hook(checkpoint_state.after_load)
    return optimizer
