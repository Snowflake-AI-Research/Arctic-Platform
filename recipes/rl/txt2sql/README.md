# Text2SQL with Arctic RL

GRPO training for **Qwen3-32B** on the BIRD SQL benchmark, served by
[Arctic RL](../../) with the [Zorro](../../zorro) trainer. Mirrors the
hyperparameters of the stock-verl baseline at
`verl_opensource/examples/bird_sql/run_qwen3_32b_bird_grpo.sh` so the two
backends can be compared apples-to-apples on wall-clock speed.

* **Model:** Qwen/Qwen3-32B
* **Topology:** 4 nodes × 8 H200 GPUs (32 GPUs), `colocate=True` with
  **3-way colocation** (training + sampling + ref log-prob share each GPU
  bundle). Non-KL runs set `log_prob_gpus=0` (Zorro recomputes actor
  log-probs on the training engine); KL runs set `log_prob_gpus=32` on
  the same bundles — no 50/50 GPU split needed.
* **Data:** [BIRD-SQL](https://bird-bench.github.io/) only — train on
  BIRD `train.json`, validate on BIRD `dev.json`. Spider / GretelAI are
  not used in this recipe.
* **Reward:** SQLite execution-match against the gold SQL
  (`bird_reward.py`)

## 1. Get the raw BIRD data

[BIRD-SQL](https://bird-bench.github.io/) ships `train.json`, `dev.json`,
and per-database SQLite files plus per-table `database_description/*.csv`
files (column-level semantic descriptions and value semantics).

```bash
mkdir -p /data/bird && cd /data/bird

# train: ~9.4k rows, ~95 SQLite databases
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip                  # creates train/{train.json,train_databases/...}

# dev: ~1.5k rows, 11 SQLite databases (used for validation)
wget https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip                    # creates dev/{dev.json,dev_databases/...}
```

After unzipping you should have:

```
/data/bird/
├── train/
│   ├── train.json
│   └── train_databases/<db_id>/<db_id>.sqlite     (+ database_description/*.csv)
└── dev/
    ├── dev.json
    └── dev_databases/<db_id>/<db_id>.sqlite       (+ database_description/*.csv)
```

## 2. Preprocess to verl parquets

`preprocess_bird.py` turns raw BIRD into verl-compatible parquets with
**heavily augmented** training prompts and **clean** validation prompts.

### Train: extended prompts

For each BIRD `train.json` row we build a prompt in the
`arctic_text_to_sql_r1` format (system prompt + user message asking the
model to think inside `<think>` tags and answer inside `<answer>` tags
with a fenced `sql` block). The user message contains the database
schema as `CREATE TABLE` DDL extracted from the SQLite file, augmented
with **all** of the following — this is the "extended" prompt the
recipe is tuned for:

1. **Foreign-key summary line** at the top of each table block,
   capturing both `FOREIGN KEY (...) REFERENCES ...` declarations and
   inline column-level `REFERENCES` syntax:
   ```
   -- Foreign keys: movie_id -> movies.movie_id; user_id -> users.user_id
   ```
2. **Sample-rows block** as a markdown-style pipe table at the top of
   each `CREATE TABLE` (10 rows by default, reusing the same
   `SELECT ... LIMIT` used for per-column examples — no extra DB
   queries).
3. **Per-column example values** as inline `-- example: [...]`
   comments (10 distinct values per column by default), so the model
   sees realistic data shapes (date formats, ID styles, units, …).
4. **BIRD `database_description/*.csv` enrichment** appended to the
   same column comments: `name`, `desc`, and `values` semantics for
   columns that BIRD documents.
5. **Evidence text** from the BIRD row appended to the natural-language
   question when present (BIRD's "evidence" hints).

A Qwen3-tokenizer length filter then drops samples whose tokenized
prompt exceeds the cap (default 32 768 tokens with
`Qwen/Qwen3-1.7B`). BIRD's outlier databases `works_cycles` and
`movie_3` produce >80 K-token prompts at full augmentation, so 32 K is
the natural break for Qwen3. Set `--max_tokens 0` to disable filtering.

In a typical preprocess run this yields **~8.6 k train rows** out of
the ~9.4 k raw BIRD train rows.

### Val: clean prompts

For BIRD `dev.json` we **deliberately disable** the augmentations to
avoid leaking auxiliary context into validation:

* No `database_description` enrichment (`use_descriptions=False`)
* No top-of-block sample-rows table (`sample_rows=0`)
* No FK summary line (`include_fk_summary=False`)
* Only 3 inline `-- example: [...]` values per column (vs. 10 in train)
* No token-length filter

This produces **~1.5 k val rows** matching the raw-BIRD evaluation
contract.

### Run it

```bash
cd recipes/rl/txt2sql

pip install pandas datasets transformers numpy
pip install func_timeout              # required by bird_reward.py at training time

python preprocess_bird.py \
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

> Each row is in verl's standard schema: `data_source`, `prompt`
> (system + user messages), `ability` (`"sql"`), `reward_model`
> (`{style: "rule", ground_truth: <gold_sql>}`), and `extra_info`
> (`db_id`, `db_path`, `question`, `split`, `index`).
> The `extra_info.db_path` is what `bird_reward.py` opens at training
> time to execute the predicted SQL against, so the SQLite files must
> still exist at that absolute path on every node when training runs.

`preprocess_bird.py` also has flags for ablating the augmentations
(`--no_descriptions`, `--no_fk_summary`, `--num_examples 0`,
`--sample_rows 0`) and for adding Spider / GretelAI sources
(`--sources bird spider gretelai`). Those are not used by this recipe,
which trains on BIRD only.

## 3. Train

The base recipe `run_qwen3_32b_bird_grpo_arl_zorro_yes.sh` runs without
a KL penalty (matches the verl baseline 1:1). The `_kl` variant enables
the low-variance KL loss against a frozen reference model and sets
`log_prob_gpus=32` for 3-way colocation on the same 32-GPU placement
group (no separate ref pool).

The script assumes a **4-node × 8-GPU** Ray cluster (32 GPUs total).
Override data paths or other settings via Hydra on the command line.

```bash
# No-KL run (matches verl baseline 1:1)
bash run_qwen3_32b_bird_grpo_arl_zorro_yes.sh \
    data.train_files=/data/snowflakesql/txt2sql/train.parquet \
    data.val_files=/data/snowflakesql/txt2sql/val.parquet

# KL-anchored run
bash run_qwen3_32b_bird_grpo_arl_zorro_yes_kl.sh \
    data.train_files=/data/snowflakesql/txt2sql/train.parquet \
    data.val_files=/data/snowflakesql/txt2sql/val.parquet
```

Override the data paths inline (as above) or edit `DATA_DIR` at the top
of `run_qwen3_32b_bird_grpo_arl_zorro_yes.sh` if you want a permanent
default.

## Files

| File | What it is |
| --- | --- |
| `run_qwen3_32b_bird_grpo_arl_zorro_yes.sh` | Base GRPO + Arctic/Zorro recipe (no KL) |
| `run_qwen3_32b_bird_grpo_arl_zorro_yes_kl.sh` | KL-enabled recipe (`use_kl_loss=True`, 3-way colocate with `log_prob_gpus=32`) |
| `bird_reward.py` | SQLite-based exec-match reward (referenced via `custom_reward_function.path`) |
| `preprocess_bird.py` | Raw BIRD JSON + SQLite → augmented verl parquets |
