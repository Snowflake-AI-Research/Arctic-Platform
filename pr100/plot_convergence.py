"""Draw the two runs side by side: the descent, and the control that reverses it."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
main = json.loads((HERE / "convergence.json").read_text())
ctrl = json.loads((HERE / "convergence_control.json").read_text())

fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.3))

steps = [h["step"] for h in main["history"]]
nll = [h["nll"] for h in main["history"]]
left.plot(steps, nll, marker="o", ms=4, color="#1f77b4")
left.set_yscale("log")
left.set_title("Client-side loss over Cortex\noverfitting one batch")
left.set_xlabel("optimizer step")
left.set_ylabel("client-side NLL (log scale)")
left.grid(alpha=0.3, which="both")
left.annotate(
    f"{nll[0]:.2f} → {nll[-1]:.2e}\ndown on {sum(1 for a, b in zip(nll, nll[1:]) if b < a)}/{len(nll) - 1} steps",
    xy=(0.42, 0.72),
    xycoords="axes fraction",
    fontsize=9,
    bbox=dict(boxstyle="round", fc="#eef4fb", ec="#9dbfe0"),
)

csteps = [h["step"] for h in ctrl["history"]]
cnll = [h["nll"] for h in ctrl["history"]]
right.plot(csteps, cnll, marker="o", ms=4, color="#d62728")
right.set_yscale("log")
right.set_title("Falsification control\nsame loop, advantages negated")
right.set_xlabel("optimizer step")
right.set_ylabel("client-side NLL (log scale)")
right.grid(alpha=0.3, which="both")
right.axvline(8, ls="--", lw=1, color="#888")
right.annotate(
    "Adam momentum from the\nprevious run reverses here",
    xy=(8, max(cnll) * 0.02),
    xytext=(9.5, max(cnll) * 0.0015),
    fontsize=8,
    color="#555",
    arrowprops=dict(arrowstyle="->", color="#888", lw=0.9),
)

fig.suptitle(
    f"TRL client-side loss → Arctic-Platform transport → Cortex (Qwen3-8B, lr={main['lr']}, "
    f"{main['supervised_positions']} supervised tokens)",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / "convergence.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
