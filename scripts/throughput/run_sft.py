#!/usr/bin/env python3
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

"""Drive one SFT job on a DSS gateway through the cortex-training client and time every step.

Rows (--mode), every token supervised:
  text        real text: the corpus (--corpus, JSONL of {"text": ...}) through the model's tokenizer, each text
              ended by EOS, cut into full-length rows (repeated if it is short)
  throughput  uniform random token ids (MoE: every token fans out to as many EP ranks as it can)
  repeat      max_seq_len - 1 copies of token 1 and one random token (MoE: every token routes to the same experts)

A step is one forward_backward and one optimizer_step; its seconds are their wall time as the client sees it. --out
gets every step's record (seconds, tokens, loss, grad norm, peak GPU memory) and a summary over the steps after
--warmup.
"""

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gateway", default="http://localhost:8003")
    p.add_argument("--cortex", required=True, help="cortex-training checkout: src/ (the client) and recipes/")
    p.add_argument("--job-config", required=True, help="catalog training file (create-job body)")
    p.add_argument("--model-name", required=True, help="checkpoint dir, sent as the job's model_name")
    p.add_argument("--mode", choices=["text", "throughput", "repeat"], required=True)
    p.add_argument("--corpus", help="JSONL, one {'text': ...} object per line (--mode text)")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2, help="steps the summary leaves out")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=None, help="rows per step (default: the file's train_batch_size)")
    p.add_argument("--lr", type=float, default=None, help="default: the file's optimizer.lr")
    p.add_argument("--set", action="append", default=[], help="training_config override, key.path=json")
    p.add_argument("--seed", type=int, default=0, help="random token stream (throughput, repeat)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.mode == "text" and args.corpus is None:
        p.error("--mode text needs --corpus (README.md: the corpus)")
    return args


def text_rows(tok, corpus, n, seq_len, row):
    ids = []
    with open(corpus) as f:
        for line in f:
            ids.extend(tok(json.loads(line)["text"], add_special_tokens=False).input_ids + [tok.eos_token_id])
            if len(ids) > n * seq_len:
                break
    while len(ids) < n * seq_len:
        ids.extend(ids[: n * seq_len - len(ids)])
    return [row(ids[i * seq_len : (i + 1) * seq_len]) for i in range(n)]


def main():
    args = parse_args()
    cortex = Path(args.cortex)
    sys.path[:0] = [str(cortex / "src"), str(cortex)]
    from cortex_training.client import CortexTrainingClient
    from transformers import AutoTokenizer

    from recipes.utils import TrainSequence
    from recipes.utils import collate
    from recipes.utils import forward_backward
    from recipes.utils import load_job_body
    from recipes.utils import optimizer_step
    from recipes.utils import running_job

    body = load_job_body(args.job_config, search_dirs=[cortex / "recipes" / "sft" / "configs"])
    sub = next(s for s in body["sub_job_configs"] if s["job_type"] == "training")
    sub["model_name"] = args.model_name
    tc = sub["training_config"]
    for item in args.set:
        key, _, value = item.partition("=")
        node = tc
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = json.loads(value)
    if args.batch_size:
        tc["train_batch_size"] = args.batch_size
    tc["max_seq_len"] = max(int(tc["max_seq_len"]), args.seq_len)
    lr = args.lr if args.lr is not None else float(tc["optimizer"]["lr"])
    batch = int(tc["train_batch_size"])
    print("job training_config:", json.dumps(tc)[:1500], flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def row(ids):
        # loss_mask[i] weights the prediction of token i + 1: every token but the first is a target.
        return TrainSequence(input_ids=ids, loss_mask=[1.0] * (len(ids) - 1) + [0.0])

    rng = random.Random(args.seed)
    if args.mode == "text":
        corpus = text_rows(tok, args.corpus, args.steps * batch, args.seq_len, row)
        digest = hashlib.sha1(str([s.input_ids[:64] for s in corpus[:8]]).encode()).hexdigest()[:12]
        print(f"text corpus: {len(corpus)} sequences of {args.seq_len}, fingerprint {digest}", flush=True)

    # Poll tightly: the client's default 0.5 s interval is a per-request latency the measurement would count as
    # training time. The first step compiles kernels for minutes, and a read timed out at the client's 600 s default
    # is retried as a duplicate forward_backward, so wait as long as a step can take.
    client = CortexTrainingClient(
        base_url=args.gateway,
        database="DB",
        schema="PUBLIC",
        poll_timeout=7200.0,
        poll_interval=0.01,
        poll_backoff_multiplier=1.0,
        poll_max_interval=0.01,
        request_timeout=(30.0, 7200.0),
    )
    processing = dict(loss_fn="causal_cross_entropy", post=["compute_logprobs"], config={})
    records = []
    t_create = time.time()
    with running_job(client, body) as job_id:
        print(f"job {job_id} ready after {time.time() - t_create:.1f}s", flush=True)
        for step in range(args.steps):
            if args.mode == "text":
                seqs = corpus[step * batch : (step + 1) * batch]
            elif args.mode == "repeat":
                seqs = [row([1] * (args.seq_len - 1) + [rng.randrange(1000, 100000)]) for _ in range(batch)]
            else:
                seqs = [row([rng.randrange(1000, 100000) for _ in range(args.seq_len)]) for _ in range(batch)]
            kwargs, context = collate(seqs, pad_token_id=pad, max_seq_len=tc["max_seq_len"])
            t0 = time.time()
            fb = forward_backward(client, job_id, kwargs, context=context, processing=processing)
            t1 = time.time()
            st = optimizer_step(client, job_id, learning_rate=lr)
            dt = time.time() - t0
            tokens = sum(len(s.input_ids) for s in seqs)
            rec = dict(
                step=step,
                loss=float(fb["avg_loss"]),
                seconds=dt,
                fwd_bwd_s=t1 - t0,
                opt_s=dt - (t1 - t0),
                tokens=tokens,
                tokens_per_s=tokens / dt,
                grad_norm=(st.get("metrics") or {}).get("grad_norm", st.get("grad_norm")),
            )
            if st.get("peak_memory"):  # training_config.step_peak_memory_log: the job-wide max over ranks
                rec["peak_memory_gib"] = {k: round(v / 2**30, 2) for k, v in st["peak_memory"].items()}
            records.append(rec)
            print(json.dumps(rec), flush=True)
    warm = records[args.warmup :] or records
    summary = dict(
        model=args.model_name,
        mode=args.mode,
        seq_len=args.seq_len,
        batch=batch,
        steps=len(records),
        mean_tokens_per_s=sum(r["tokens"] for r in warm) / sum(r["seconds"] for r in warm),
        median_step_s=statistics.median(r["seconds"] for r in warm),
        first_loss=records[0]["loss"],
        last_loss=records[-1]["loss"],
        training_config=tc,
        records=records,
    )
    Path(args.out).write_text(json.dumps(summary, indent=2))
    brief = {k: v for k, v in summary.items() if k not in ("records", "training_config")}
    print("SUMMARY", json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
