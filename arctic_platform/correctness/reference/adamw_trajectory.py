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

"""One AdamW optimizer held across a whole run, over an FP32 master copy written back every step.

``optimizer_step.run_adamw_step`` applies one independent step and then clears the optimizer state, which
is what a single-step comparison wants and what a trajectory cannot use. Adam's first and second moments
and its step counter are exactly the state that makes step *k* depend on the ``k - 1`` steps before it, so
a loop over that function would repeat step one a hundred times.

Two things differ from the single step, and both are properties of there being a next step:

- the moments persist, so the bias correction advances and the update is not the first one again;
- the master copy is written back into the model's parameters, because the next forward reads the model.
  The single step does not write back: it records the update and nothing runs after it.

The parameter grouping, the optimizer-name validation and the master precision come from
``optimizer_step`` rather than being restated, so a grouping change cannot move only one of the two.
"""

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

from .optimizer_step import assert_adamw_semantics
from .optimizer_step import build_adamw
from .optimizer_step import master_dtype


class AdamWTrajectory:
    """An AdamW step that can be taken repeatedly against one persistent optimizer state."""

    def __init__(
        self,
        model,
        optimizer_config: dict,
        *,
        learning_rate: float,
        gradient_clipping: float | None,
        optimizer_dtype: str,
        optimizer_chunk_elements: int | None = None,
    ) -> None:
        assert_adamw_semantics(optimizer_config)
        if optimizer_chunk_elements is not None and (
            isinstance(optimizer_chunk_elements, bool)
            or not isinstance(optimizer_chunk_elements, int)
            or optimizer_chunk_elements <= 0
        ):
            raise ValueError("optimizer_chunk_elements must be a positive integer")
        self._model = model
        self._optimizer_config = optimizer_config
        self._learning_rate = float(learning_rate)
        self._gradient_clipping = gradient_clipping
        self._dtype = master_dtype(optimizer_dtype)
        self._optimizer_chunk_elements = optimizer_chunk_elements
        self._sources: Optional[List[Tuple[str, "torch.nn.Parameter"]]] = None
        self._masters: Dict[str, "torch.nn.Parameter"] = {}
        self._optimizer = None
        self.steps = 0

    @property
    def trainable_names(self) -> List[str]:
        """The parameters this optimizer owns, empty until the first step has decided them."""
        return [name for name, _ in self._sources or []]

    def _adopt(self, gradients: Dict[str, "torch.Tensor"]) -> None:
        """Take the FP32 master copy, once, from the parameters the first request produced a gradient for.

        Deferred to the first step for the same reason the single step selects the same way: a parameter
        that requires a gradient and never receives one is not part of the optimization, and which those
        are is a property of the model and the request rather than of the config.
        """
        import torch

        named = [
            (name, parameter)
            for name, parameter in self._model.named_parameters()
            if parameter.requires_grad and name in gradients
        ]
        if not named:
            raise ValueError("optimizer trajectory found no trainable parameters with gradients")
        self._sources = named
        self._masters = {
            name: torch.nn.Parameter(
                parameter.detach()
                .to(self._dtype)
                .clone(
                    memory_format=(
                        torch.contiguous_format
                        if self._optimizer_chunk_elements is not None
                        else torch.preserve_format
                    )
                ),
                requires_grad=True,
            )
            for name, parameter in named
        }
        self._optimizer = build_adamw(
            [(name, self._masters[name]) for name, _ in named],
            self._optimizer_config,
            learning_rate=self._learning_rate,
        )

    @torch.no_grad()
    def _bounded_step(self, gradients: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        from torch.optim._functional import adamw

        # Preserve the reference's per-parameter norm reduction and global clipping. Only one
        # parameter's gradient is on the device during this pass; the update streams smaller chunks.
        norms = []
        for name, _ in self._sources:
            master = self._masters[name]
            gradient = gradients[name].detach().to(device=master.device, dtype=self._dtype)
            norms.append(torch.linalg.vector_norm(gradient))
            del gradient
        norm = torch.linalg.vector_norm(torch.stack(norms))
        coefficient = None
        if self._gradient_clipping is not None and float(self._gradient_clipping) > 0:
            coefficient = (float(self._gradient_clipping) / (norm + 1e-6)).clamp(max=1.0)

        for group in self._optimizer.param_groups:
            for master in group["params"]:
                state = self._optimizer.state[master]
                if not state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(master)
                    state["exp_avg_sq"] = torch.zeros_like(master)

        groups = {id(master): group for group in self._optimizer.param_groups for master in group["params"]}
        for name, _ in self._sources:
            master = self._masters[name]
            group = groups[id(master)]
            state = self._optimizer.state[master]
            parameter = master.view(-1)
            first_moment = state["exp_avg"].view(-1)
            second_moment = state["exp_avg_sq"].view(-1)
            source_gradient = gradients[name].detach().reshape(-1)
            for start in range(0, master.numel(), self._optimizer_chunk_elements):
                stop = start + self._optimizer_chunk_elements
                gradient = source_gradient[start:stop].to(device=master.device, dtype=self._dtype).clone()
                if coefficient is not None:
                    gradient.mul_(coefficient.to(device=master.device))
                # Functional AdamW increments its scalar counter. Each chunk starts from the same
                # previous parameter step, so chunking cannot advance bias correction more than once.
                chunk_step = state["step"].clone()
                adamw(
                    [parameter[start:stop]],
                    [gradient],
                    [first_moment[start:stop]],
                    [second_moment[start:stop]],
                    [],
                    [chunk_step],
                    foreach=False,
                    fused=False,
                    amsgrad=False,
                    beta1=group["betas"][0],
                    beta2=group["betas"][1],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                )
                del gradient
            state["step"].add_(1)
        return norm

    def step(self, gradients: Dict[str, "torch.Tensor"]) -> float:
        """Apply one step and write the result back into the model. Returns the pre-clip gradient norm."""
        import torch

        if self._optimizer is None:
            self._adopt(gradients)
        missing = [name for name, _ in self._sources if name not in gradients]
        if missing:
            raise ValueError(
                f"step {self.steps + 1} received no gradient for {len(missing)} parameter(s) the "
                f"optimizer owns, the first being {missing[0]!r}; the trainable set changed mid-run"
            )
        for name, source in self._sources:
            if gradients[name].shape != source.shape:
                raise ValueError(
                    f"step {self.steps + 1} gradient {name!r} has shape {tuple(gradients[name].shape)}, "
                    f"expected {tuple(source.shape)}"
                )
        if self._optimizer_chunk_elements is not None:
            norm = self._bounded_step(gradients)
        else:
            for name, _ in self._sources:
                master = self._masters[name]
                master.grad = gradients[name].detach().to(device=master.device, dtype=self._dtype).clone()

            norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(self._masters[name].grad) for name, _ in self._sources])
            )
            if self._gradient_clipping is not None and float(self._gradient_clipping) > 0:
                torch.nn.utils.clip_grad_norm_(list(self._masters.values()), float(self._gradient_clipping))
            self._optimizer.step()

        # The next forward reads the model, not the master copy, so the step is not finished until the
        # parameters carry it. Narrowing back to the parameter dtype here is what the engine's bf16
        # optimizer does at the same point.
        with torch.no_grad():
            for name, source in self._sources:
                source.copy_(self._masters[name].detach().to(dtype=source.dtype))
        for name, _ in self._sources:
            self._masters[name].grad = None
        self.steps += 1
        return float(norm.detach().cpu())

    def state_norms(self) -> Dict[str, Dict[str, float]]:
        """Each owned parameter's moment norms and step count, for a diagnostic that reads the state.

        The step count is the evidence that the state persisted: a loop that rebuilt its optimizer would
        report one at every step, and its updates would be first-step updates forever.
        """
        import torch

        out: Dict[str, Dict[str, float]] = {}
        for name, _ in self._sources or []:
            state = self._optimizer.state[self._masters[name]]
            if not state:
                continue
            out[name] = {
                "exp_avg": float(torch.linalg.vector_norm(state["exp_avg"]).detach().cpu()),
                "exp_avg_sq": float(torch.linalg.vector_norm(state["exp_avg_sq"]).detach().cpu()),
                "step": float(state["step"]),
            }
        return out
