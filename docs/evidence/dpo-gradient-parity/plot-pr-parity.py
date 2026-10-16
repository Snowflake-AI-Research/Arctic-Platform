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

"""Render the completed parity comparison for reviewers; no experiment execution."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "pr-parity-chart-data.json").read_text())
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "font.size": 11, "text.color": "#18324a", "axes.labelcolor": "#18324a"}
)
fig = plt.figure(figsize=(12.4, 6.7), facecolor="white")
fig.text(0.065, 0.94, "Does the client/server split preserve DPO gradients?", fontsize=20, weight="bold")
fig.text(
    0.065,
    0.89,
    "One completed configuration · 1 H200 GPU · Qwen3-0.6B · rank-4 LoRA · BF16 · 2 training steps",
    fontsize=11,
    color="#566777",
)
grid = fig.add_gridspec(1, 2, left=0.08, right=0.96, bottom=0.25, top=0.78, width_ratios=[1.05, 1.0], wspace=0.28)
scatter = fig.add_subplot(grid[0])
colors = ["#237fa1", "#d78939"]
extent = 0.0
for i, step in enumerate(data["steps"]):
    sample = step["sample"]
    x, y = np.asarray(sample["reference"]), np.asarray(sample["candidate"])
    extent = max(extent, float(np.abs(x).max()), float(np.abs(y).max()))
    scatter.scatter(x, y, s=10, alpha=0.45, color=colors[i], label=f"Step {step['step']}", rasterized=True)
limit = extent * 1.12
scatter.plot(
    [-limit, limit], [-limit, limit], linestyle="--", linewidth=1.1, color="#627585", label="Equal gradients", zorder=0
)
scatter.set_xlim(-limit, limit)
scatter.set_ylim(-limit, limit)
scatter.set_aspect("equal", adjustable="box")
scatter.set_title("Parameter gradients agree", loc="left", pad=13, weight="bold", fontsize=13)
scatter.set_xlabel("Full DPO loss → model backward", labelpad=9)
scatter.set_ylabel("Client derivatives → server backward", labelpad=9)
scatter.spines[["top", "right"]].set_visible(False)
scatter.spines[["left", "bottom"]].set_color("#c1cbd3")
scatter.tick_params(colors="#566777", labelsize=9)
scatter.grid(alpha=0.14)
scatter.set_axisbelow(True)
scatter.legend(loc="upper left", frameon=False, fontsize=9)

table = fig.add_subplot(grid[1])
table.set_axis_off()
table.text(0, 1.02, "Every captured value matches exactly", fontsize=13, weight="bold")
table.text(0, 0.94, "112 trainable tensors checked in each step", fontsize=10, color="#566777")
for x, step in zip((0.60, 0.87), data["steps"]):
    table.text(x, 0.80, f"Step {step['step']}", ha="center", fontsize=11, weight="bold")
    table.text(x, 0.74, f"{step['pairs']} pairs", ha="center", fontsize=9, color="#566777")
for y, label, quantities in (
    (0.59, "Parameter gradients", ("gradient",)),
    (0.40, "Parameter updates", ("delta",)),
    (0.21, "Adam moments", ("exp_avg", "exp_avg_sq")),
):
    table.text(0, y, label, va="center", fontsize=11)
    for x, step in zip((0.60, 0.87), data["steps"]):
        assert all(step["quantities"][q]["bitwise_equal"] for q in quantities)
        table.add_patch(
            FancyBboxPatch(
                (x - 0.10, y - 0.065),
                0.20,
                0.13,
                boxstyle="round,pad=0.005,rounding_size=0.025",
                facecolor="#e6f3ed",
                edgecolor="none",
            )
        )
        table.text(x, y, "Exact", ha="center", va="center", fontsize=11, color="#18734e", weight="bold")
table.text(0, 0.055, "Maximum absolute difference: 0", fontsize=10, color="#18734e")

fig.text(
    0.065,
    0.115,
    "Both methods use identical weights, score arithmetic and backward accumulation order.",
    fontsize=10,
    color="#566777",
)
total = sum(step["quantities"]["gradient"]["elements"] for step in data["steps"])
fig.text(
    0.065,
    0.075,
    f"Dots show sampled gradients; all {total:,} gradient values were checked across the two steps.",
    fontsize=9,
    color="#566777",
)
fig.text(
    0.065,
    0.037,
    "Scope: one-GPU packed comparison on fixed synthetic pairs. Four-GPU weight updates were not compared.",
    fontsize=9,
    color="#566777",
)
for extension in ("png", "svg", "pdf"):
    fig.savefig(ROOT / f"dpo-gradient-parity.{extension}", dpi=230, facecolor="white")
