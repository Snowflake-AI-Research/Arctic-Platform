"""Pull a reference W&B run's config and metric history for side-by-side comparison.

Reads the API key from WANDB_API_KEY, falling back to the workspace .env, so the
key never has to be passed on a command line (argv is world-readable via /proc).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://snowflake.wandb.io"


def load_key() -> str:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("WANDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(
        "No WANDB_API_KEY found.\n"
        f"  Add it to {env_file} as WANDB_API_KEY=<key> (file is mode 600), or\n"
        "  export WANDB_API_KEY=<key> in this shell.\n"
        "  Get the key from https://snowflake.wandb.io/authorize"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="yak")
    ap.add_argument("--project", default="snow_swe_coco")
    ap.add_argument("--run", required=True, help="run id, not the display name")
    ap.add_argument("--base-url", default=os.environ.get("WANDB_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--outdir", default=str(REPO / "runs" / "reference-wandb"))
    args = ap.parse_args()

    os.environ["WANDB_API_KEY"] = load_key()
    os.environ["WANDB_BASE_URL"] = args.base_url

    import wandb

    api = wandb.Api(overrides={"base_url": args.base_url})
    path = f"{args.entity}/{args.project}/{args.run}"
    run = api.run(path)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"name:    {run.name}")
    print(f"state:   {run.state}")
    print(f"created: {run.created_at}")
    print(f"url:     {run.url}")

    (outdir / "config.json").write_text(json.dumps(dict(run.config), indent=2, default=str))
    (outdir / "summary.json").write_text(json.dumps(dict(run.summary), indent=2, default=str))

    # scan_history streams every logged step; the default history() samples to 500
    # rows, which would silently smooth over exactly the early-step spikes we care
    # about when comparing reward collapse.
    rows = list(run.scan_history())
    if not rows:
        print("no history rows")
        return 0

    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    csv_path = outdir / "history.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} steps, {len(keys)} metrics -> {csv_path}")
    print("\nmetric keys:")
    for k in sorted(keys):
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
