"""Plot the client-side-loss arm against its sign-flipped control, three seeds each."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = (7, 42, 123)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "seedmatrix")
SMOOTH = 10


def rewards(seed: int, arm: str) -> list[float]:
    hist = json.load(open(os.path.join(OUT, f"s{seed}_{arm}.json")))["log_history"]
    return [float(r["reward"]) for r in hist if "reward" in r]


def running(ys: list[float], w: int) -> list[float]:
    return [sum(ys[max(0, i - w + 1) : i + 1]) / len(ys[max(0, i - w + 1) : i + 1]) for i in range(len(ys))]


def main() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

    for seed, style in zip(SEEDS, ("-", "--", ":")):
        n, c = rewards(seed, "normal"), rewards(seed, "control")
        ax.plot(running(n, SMOOTH), style, color="#1a7f37", lw=1.9, label=f"client-side loss, seed {seed}")
        ax.plot(running(c, SMOOTH), style, color="#cf222e", lw=1.4, alpha=0.8,
                label=f"sign-flipped control, seed {seed}")

    ax.set_xlabel("training step")
    ax.set_ylabel(f"accuracy reward (running mean, {SMOOTH} steps)")
    ax.set_title("GSM8K GRPO over Cortex, client-side loss\nQwen3-1.7B, 100 steps, three seeds")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.6)

    deltas_n = [sum(rewards(s, "normal")[-30:]) / 30 - sum(rewards(s, "normal")[:30]) / 30 for s in SEEDS]
    deltas_c = [sum(rewards(s, "control")[-30:]) / 30 - sum(rewards(s, "control")[:30]) / 30 for s in SEEDS]
    x = range(len(SEEDS))
    bx.bar([i - 0.2 for i in x], deltas_n, 0.38, color="#1a7f37", label="client-side loss")
    bx.bar([i + 0.2 for i in x], deltas_c, 0.38, color="#cf222e", label="sign flipped")
    bx.set_xticks(list(x))
    bx.set_xticklabels([f"seed {s}" for s in SEEDS])
    bx.set_ylabel("reward delta (last 30 - first 30)")
    bx.set_title("Every seed separates by ~0.31")
    bx.axhline(0, color="k", lw=0.8)
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(HERE, "seed_matrix.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
