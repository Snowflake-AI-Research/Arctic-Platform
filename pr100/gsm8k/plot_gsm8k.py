"""Reward curve for the GSM8K run, plus the ratio that says whether to believe it."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "gsm8k_converge.json"
data = json.loads(src.read_text())
hist = [h for h in data["log_history"] if "reward" in h]
steps = [h["step"] for h in hist]
reward = [h["reward"] for h in hist]
ratio = [h.get("ratio") for h in hist]


def running(xs, k=10):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        window = xs[lo : i + 1]
        out.append(sum(window) / len(window))
    return out


fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.3))

left.plot(steps, reward, color="#bbb", lw=1, label="per step")
left.plot(steps, running(reward), color="#2ca02c", lw=2, label="running mean (10)")
left.set_xlabel("optimizer step")
left.set_ylabel("accuracy_reward")
left.set_title("GSM8K GRPO over Cortex, client-side loss\nQwen3-1.7B, grpo encoding")
left.grid(alpha=0.3)
left.legend(fontsize=8)
k = min(30, len(reward) // 3)
first, last = sum(reward[:k]) / k, sum(reward[-k:]) / k
left.annotate(
    f"first {k} steps {first:.3f}  →  last {k} {last:.3f}   ({last - first:+.3f})",
    xy=(0.03, 0.04),
    xycoords="axes fraction",
    fontsize=9,
    bbox=dict(boxstyle="round", fc="#eefbf0", ec="#9dd0a8"),
)

clean = [(s, r) for s, r in zip(steps, ratio) if r is not None]
right.plot([s for s, _ in clean], [r for _, r in clean], color="#1f77b4", lw=1)
right.axhline(1.0, ls="--", lw=1, color="#888")
right.set_xlabel("optimizer step")
right.set_ylabel("importance ratio")
right.set_title("Sanity check: on-policy ratio\n(off-by-one log-prob frame would blow this up)")
right.grid(alpha=0.3)
if clean:
    lo = min(r for _, r in clean)
    hi = max(r for _, r in clean)
    right.annotate(
        f"range {lo:.4f} – {hi:.4f}",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="#eef4fb", ec="#9dbfe0"),
    )

fig.suptitle(
    f"Tunji's gsm8k recipe with the backend swapped to Cortex "
    f"({data['args']['num_prompts']} prompts, n={data['args']['num_generations']}, lr={data['args']['lr']})",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "gsm8k_convergence.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}  ({len(hist)} logged steps)")
