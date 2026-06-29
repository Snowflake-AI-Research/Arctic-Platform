# Text2SQL with Arctic RL

GRPO training for **Qwen3-32B** on the BIRD SQL benchmark, served by [Arctic RL](../../../../arctic_platform/rl/)
with the [ZoRRo](../../../../arctic_platform/rl/zorro_train/) trainer. Mirrors the hyperparameters of the stock-verl
baseline at `verl_opensource/examples/bird_sql/run_qwen3_32b_bird_grpo.sh` so the two backends can be compared
apples-to-apples on wall-clock speed. Pure GRPO, without a frozen reference model (no KL anchoring).

* **Model:** Qwen/Qwen3-32B
* **Topology:** 4 nodes × 8 H200 GPUs (32 GPUs total), `colocate=True` (training + sampling share each GPU bundle),
  DeepSpeed ZeRO stage-3 with CPU optimizer offload, vLLM rollout (TP=2). Without KL there is no frozen reference
  model, so the ref log-prob pool is disabled (`log_prob_gpus=0`); under ZoRRo log-probs are recomputed through the
  training engine itself.
* **Data:** [BIRD-SQL](https://bird-bench.github.io/) only — train on BIRD `train.json`, validate on BIRD `dev.json`.
  Spider / GretelAI are not used in this recipe.
* **Reward:** SQLite execution-match against the gold SQL (`bird_reward.py`)

## 1. Ray and multi-node hostfile

When using a multi-node training environment Ray and DeepSpeed need a special file called `hostfile` (comes from MPI)
that they use to find the participating nodes and the number of gpus on each node.

Most likely your CSP already provides one for you, if that's the case please note its path on the filesystem.

If you don't have one you need to create it. It looks like this:

```
10.1.1.1 slots=8
10.1.1.2 slots=8
10.1.1.3 slots=8
10.1.1.4 slots=8
```
the first column is the IPs of the participating nodes, the second column is the number of gpus on each node.

Export its path - the install step (and the launcher) use it to fan out across all nodes:
```
export JOB_HOSTFILE=/path/of/hostfile
```
(or you can change the `HOSTFILE` setting on top of both scripts)

The Ray cluster itself is started later, in the Train step, once the environment is installed on every node.

For a single-node run you can skip the hostfile — the launcher falls back to `NNODES=1` and uses the 8 GPUs of the
local node.

## 2. Install packages

Conda environments are **node-local**: each node has its own `~/miniconda3` even when your code and data sit on
shared storage, so the environment must be created and populated on **every** node in `$JOB_HOSTFILE` (from
[step (1)](#1-ray-and-multi-node-hostfile)). Use a fresh, recipe-specific env name (don't reuse a shared/dev env)
so the install is actually exercised on all nodes.

We fan out with `ds_ssh` (the DeepSpeed multi-node helper) and install with `uv pip install --python
<env>/bin/python`, i.e. we address the env by **absolute path** rather than relying on `conda activate`. A
non-interactive `ds_ssh`/pdsh shell does not reliably keep `conda activate` in effect, so activation-based installs
can silently land in the base env; absolute paths avoid that (this is the same reason `restart_multi_ray.sh` starts
Ray via the env's absolute `ray` binary).

Pick the env name and resolve the env's `bin/` once (the path is identical on every node):
```bash
export CONDA_ENV=txt2sql
CONDA_BASE=$(conda info --base)
ENV=$CONDA_BASE/envs/$CONDA_ENV/bin        # the env's python / uv / ds_ssh / ray live here, on each node
```

Bootstrap the env on the launching node first — this is what gives you `uv` and `ds_ssh`:
```bash
conda create -y -n $CONDA_ENV python=3.12
$ENV/python -m pip install -q uv
$ENV/uv pip install --python $ENV/python deepspeed       # provides $ENV/ds_ssh
```

Clone this repo (it carries `requirements.txt` and the launcher scripts) and the verl fork onto storage visible to
all nodes:
```bash
git clone https://github.com/Snowflake-AI-Research/Arctic-Platform
git clone -b arctic_rl_share_v0.7.1 --single-branch https://github.com/Snowflake-AI-Research/verl
cd Arctic-Platform/recipes/rl/verl/txt2sql
```

Create the env on the remaining nodes (idempotent — the launching node already has it), then install the pinned
dependencies on all nodes. cuda-12.9 is assumed — if you use a different version change the `torch` index URL below
and the `cuda-bindings` pin in `requirements.txt`. `arctic-inference` patches vllm-0.18.0, so that exact version is
pinned in `requirements.txt`; `overrides.txt` forces the few transitive deps (flashinfer / numpy / transformers /
datasets) this recipe is validated against, which vLLM 0.18.0's metadata otherwise pins higher (without it the
single resolve is unsatisfiable).
```bash
# create the env on every node (guarded: `conda create -y` on an existing env would wipe and recreate it, which
# would clobber the uv/ds_ssh you just bootstrapped on the launching node — the `[ -x ... ]` makes it a no-op there)
$ENV/ds_ssh -f $JOB_HOSTFILE "[ -x $ENV/python ] || $CONDA_BASE/bin/conda create -y -n $CONDA_ENV python=3.12"
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/python -m pip install -q uv"

# torch (CUDA 12.9) first, then the rest of the pinned packages
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu129 -U"
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python -r $PWD/requirements.txt --override $PWD/overrides.txt"
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python -U pip wheel packaging setuptools"
```

To install flash attention, you can build it from source (may take a long time to build):
```bash
$ENV/ds_ssh -f $JOB_HOSTFILE "$ENV/uv pip install --python $ENV/python flash-attn --no-build-isolation"
```
or you can install directly from a wheel, find the automatic instructions
[here](https://windreamer.github.io/flash-attention3-wheels/) or download directly from
https://github.com/Dao-AILab/flash-attention/releases.

Install verl (Snowflake fork) editable on all nodes (the source is shared, but the editable install must register it
in each node's env):
```bash
cd ../../../../../verl
grep -v flash-attn requirements.txt > requirements-no-fa.txt
$ENV/ds_ssh -f $JOB_HOSTFILE "cd $PWD && $ENV/uv pip install --python $ENV/python -r requirements-no-fa.txt && $ENV/uv pip install --python $ENV/python -e ."
cd -
```

The BIRD reward (`bird_reward.py`) executes the generated SQL with Python's standard library (`sqlite3`,
`concurrent.futures`), so no extra packages are needed beyond `requirements.txt`.

> For a **single-node** run you don't need `ds_ssh`: create the env, bootstrap `uv`, and run the same
> `$ENV/uv pip install --python $ENV/python ...` commands directly on the local node.

## 3. Get the raw BIRD data

[BIRD-SQL](https://bird-bench.github.io/) ships `train.json`, `dev.json`, and per-database SQLite files plus per-table
`database_description/*.csv` files (column-level semantic descriptions and value semantics). The official
`bird-bench.oss-cn-beijing.aliyuncs.com` endpoint is slow/unreliable from many regions, so we pull the same content
(train 9428 rows, dev 1534 rows, plus the database archives) from the
[`Sudnya/bird-sql`](https://huggingface.co/datasets/Sudnya/bird-sql) HuggingFace mirror. This only needs
`huggingface_hub` + `pandas`, both already in `requirements.txt`.

Fetch onto shared storage (visible to all nodes — `bird_reward.py` opens the SQLite files at training time on every
node). The `train_databases.zip` archive is ~20 GB, so this can take a while:
```bash
export BIRD_DIR=/data/bird
$ENV/python - <<'PY'
import os, glob, shutil, tempfile, zipfile
import pandas as pd
from huggingface_hub import hf_hub_download
import json

repo, bird = "Sudnya/bird-sql", os.environ["BIRD_DIR"]

def dump_questions(member, out_json):
    df = pd.read_parquet(hf_hub_download(repo, member, repo_type="dataset"))
    rows = [{"db_id": r["db_id"], "question": r["question"],
             "evidence": ("" if pd.isna(r.get("evidence")) else r["evidence"]), "SQL": r["SQL"]}
            for _, r in df.iterrows()]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(rows, open(out_json, "w"))
    print(f"wrote {len(rows)} rows -> {out_json}")

def fetch_dbs(member, dest_parent, dbname):
    z = hf_hub_download(repo, member, repo_type="dataset")
    tmp = tempfile.mkdtemp(dir=dest_parent)
    with zipfile.ZipFile(z) as zf: zf.extractall(tmp)
    root = os.path.dirname(os.path.dirname(glob.glob(f"{tmp}/**/*.sqlite", recursive=True)[0]))
    os.replace(root, os.path.join(dest_parent, dbname))
    print(f"ready -> {os.path.join(dest_parent, dbname)}")

dump_questions("data/train-00000-of-00001.parquet", f"{bird}/train/train.json")
dump_questions("data/validation-00000-of-00001.parquet", f"{bird}/dev/dev.json")
fetch_dbs("databases/train_databases.zip", f"{bird}/train", "train_databases")
fetch_dbs("databases/dev_databases.zip", f"{bird}/dev", "dev_databases")
PY
```

You should end up with:

```
/data/bird/
├── train/
│   ├── train.json
│   └── train_databases/<db_id>/<db_id>.sqlite     (+ database_description/*.csv)
└── dev/
    ├── dev.json
    └── dev_databases/<db_id>/<db_id>.sqlite       (+ database_description/*.csv)
```

## 4. Preprocess to verl parquets

`preprocess_bird.py` turns raw BIRD into verl-compatible parquets with **heavily augmented** training prompts and
**clean** validation prompts.

### Train: extended prompts

For each BIRD `train.json` row we build a prompt in the `arctic_text_to_sql_r1` format (system prompt + user message
asking the model to think inside `<think>` tags and answer inside `<answer>` tags with a fenced `sql` block). The user
message contains the database schema as `CREATE TABLE` DDL extracted from the SQLite file, augmented with **all** of
the following — this is the "extended" prompt the recipe is tuned for:

1. **Foreign-key summary line** at the top of each table block, capturing both `FOREIGN KEY (...) REFERENCES ...`
   declarations and inline column-level `REFERENCES` syntax:
   ```
   -- Foreign keys: movie_id -> movies.movie_id; user_id -> users.user_id
   ```
2. **Sample-rows block** as a markdown-style pipe table at the top of each `CREATE TABLE` (10 rows by default, reusing
   the same `SELECT ... LIMIT` used for per-column examples — no extra DB queries).
3. **Per-column example values** as inline `-- example: [...]` comments (10 distinct values per column by default), so
   the model sees realistic data shapes (date formats, ID styles, units, …).
4. **BIRD `database_description/*.csv` enrichment** appended to the same column comments: `name`, `desc`, and
   `values` semantics for columns that BIRD documents.
5. **Evidence text** from the BIRD row appended to the natural-language question when present (BIRD's "evidence"
   hints).

A Qwen3-tokenizer length filter then drops samples whose tokenized prompt exceeds the cap (default 32 768 tokens with
`Qwen/Qwen3-1.7B`). BIRD's outlier databases `works_cycles` and `movie_3` produce >80 K-token prompts at full
augmentation, so 32 K is the natural break for Qwen3. Set `--max_tokens 0` to disable filtering.

In a typical preprocess run this yields **~8.6 k train rows** out of the ~9.4 k raw BIRD train rows.

### Val: clean prompts

For BIRD `dev.json` we **deliberately disable** the augmentations to avoid leaking auxiliary context into validation:

* No `database_description` enrichment (`use_descriptions=False`)
* No top-of-block sample-rows table (`sample_rows=0`)
* No FK summary line (`include_fk_summary=False`)
* Only 3 inline `-- example: [...]` values per column (vs. 10 in train)
* No token-length filter

This produces **~1.5 k val rows** matching the raw-BIRD evaluation contract.

### Run it

From the recipe directory (`Arctic-Platform/recipes/rl/verl/txt2sql`, cloned in step 2), using the environment
installed in step 2 (`pandas` / `datasets` / `transformers` come from `requirements.txt`):

```bash
$ENV/python preprocess_bird.py \
    --bird_dir /data/bird \
    --output_dir /data/snowflakesql/txt2sql \
    --max_tokens 32768 \
    --tokenizer Qwen/Qwen3-1.7B \
    --num_examples 10 \
    --sample_rows 10
```

Output:

```
/data/snowflakesql/txt2sql/
├── train.parquet     # BIRD train, augmented, token-filtered  (~8.6k rows)
└── val.parquet       # BIRD dev, clean, no token filter        (~1.5k rows)
```

> Each row is in verl's standard schema: `data_source`, `prompt` (system + user messages), `ability` (`"sql"`),
> `reward_model` (`{style: "rule", ground_truth: <gold_sql>}`), and `extra_info` (`db_id`, `db_path`, `question`,
> `split`, `index`). The `extra_info.db_path` is what `bird_reward.py` opens at training time to execute the
> predicted SQL against, so the SQLite files must still exist at that absolute path on every node when training runs.

`preprocess_bird.py` also has flags for ablating the augmentations (`--no_descriptions`, `--no_fk_summary`,
`--num_examples 0`, `--sample_rows 0`) and for adding Spider / GretelAI sources (`--sources bird spider gretelai`).
Those are not used by this recipe, which trains on BIRD only.

## 5. Train

First start the Ray cluster across all nodes (now that the environment is installed everywhere, and `$JOB_HOSTFILE`
is exported from step 1):
```bash
bash ./restart_multi_ray.sh
```

The recipe `run_qwen3_32b_bird_grpo_arl_zorro_yes.sh` runs GRPO without a KL penalty (matches the verl baseline 1:1).
It derives the node count from the `hostfile` set up in step 1 (`NGPU_PER_JOB = 8 × NNODES`); the documented topology
is a **4-node × 8-GPU** Ray cluster (32 GPUs total).

Next edit the environment variables in `run_qwen3_32b_bird_grpo_arl_zorro_yes.sh` to match your setup. In particular:
- `HF_HOME` - where your HF hub cache is (you can unset it as well)
- `VLLM_CACHE_ROOT` - some path where vllm could cache its work

```bash
bash run_qwen3_32b_bird_grpo_arl_zorro_yes.sh \
    data.train_files=/data/snowflakesql/txt2sql/train.parquet \
    data.val_files=/data/snowflakesql/txt2sql/val.parquet
```

The script defaults `DATA_DIR` to `/data/snowflakesql/txt2sql` (the preprocessing output from step 4), so if you kept
that path you can launch with no overrides:

```bash
bash run_qwen3_32b_bird_grpo_arl_zorro_yes.sh
```

Otherwise override the data paths inline (as above), set `DATA_DIR=...` in the environment, or edit `DATA_DIR` at the
top of the script.

The SQL reward is scored by `bird_reward.py` (shipped with this recipe and auto-wired in the launcher via
`custom_reward_function`): it executes the model's predicted SQL against the row's SQLite database and compares the
result set to the gold query's. Upstream `verl` has no built-in scorer for this `data_source`, so the recipe supplies
its own — no extra setup needed.

## Files

| File | What it is |
| --- | --- |
| `run_qwen3_32b_bird_grpo_arl_zorro_yes.sh` | GRPO + Arctic/ZoRRo recipe (no KL) |
| `restart_multi_ray.sh` | Multi-node Ray launcher (reads the `hostfile`) |
| `requirements.txt` | Pinned Python dependencies (installed with `--override overrides.txt`) |
| `overrides.txt` | uv override-pins for vLLM 0.18.0's transitive deps |
| `bird_reward.py` | SQLite-based exec-match reward (referenced via `custom_reward_function.path`) |
| `preprocess_bird.py` | Raw BIRD JSON + SQLite → augmented verl parquets |
