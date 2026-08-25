# Arctic Platform SFT

Supervised fine-tuning (SFT) on Arctic Platform: a **CPU-only client** drives
a remote DeepSpeed training server over HTTP (default) or Ray. The client owns
the data loop; the server owns model weights, forward/backward, and the
optimizer.

Docs index: [index.md](index.md) · Shared server: [common.md](common.md) · RL: [rl.md](rl.md)

```
┌──────────────────────────────────────────────┐
│  Client process (CUDA_VISIBLE_DEVICES= ok)   │
│  tokenize / collate → ArcticSFTClient        │
│                     fwd_bwd / step / save    │
└──────────────────────┬───────────────────────┘
                       │ HTTP or Ray
                       ▼
┌──────────────────────────────────────────────┐
│  Arctic Platform server (GPUs)               │
│  DeepSpeed workers → run_sft_pipeline        │
│                     → sft / sft_ce loss      │
└──────────────────────────────────────────────┘
```

SFT is training-only (no sampling / log-prob / vLLM). Shared server
infrastructure lives under `arctic_platform.common`; SFT-specific code under
`arctic_platform.sft`.

## Quick start

Colocated HTTP server, CPU-blanked client:

```bash
CUDA_VISIBLE_DEVICES= python -m arctic_platform.sft.examples.run_sft_http_demo \
  --launch-local-server --server-cuda-visible-devices 0,1 --training-gpus 2
```

Or drive the client yourself:

```python
from arctic_platform.client import OnPremConfig, TrainingConfig
from arctic_platform.sft import ArcticSFTClient, ArcticSFTClientConfig

config = ArcticSFTClientConfig(
    model_name="NousResearch/Llama-3.2-1B",
    training_gpus=2,
    backend=OnPremConfig(
        host="localhost",
        port=8765,
        launch_local_server=True,
        server_cuda_visible_devices="0,1",
    ),
    training=TrainingConfig(
        checkpoint_path="/data-fast/my-run/ckpt",  # required for new jobs
        ds_config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0.0},
            },
        },
    ),
)

client = ArcticSFTClient(config)
try:
    for _ in range(steps):
        out = client.train_step(wire_batch)  # fwd_bwd + step; metrics is the union
        print(out["metrics"]["loss"], out["metrics"].get("grad_norm"))
    client.save_checkpoint()
finally:
    client.shutdown()
```

Connect to an already-running server by setting `launch_local_server=False` on
the `OnPremConfig` (and optionally `training_job_id=` on the client config to
reconnect). Start the server with:

```bash
python -m arctic_platform.common.http_server \
  --host 0.0.0.0 --port 8765 \
  --training-gpus 2 --sampling-gpus 0 --log-prob-gpus 0
```

## Package layout

| Path | Role |
|------|------|
| `arctic_platform/client/requests.py` | Every op -> `Request` builder (the op vocabulary) |
| `arctic_platform/client/base.py` | `_ArcticClientCore` + the `ArcticClient` / `AsyncArcticClient` op surfaces |
| `arctic_platform/client/sft.py` | `ArcticSFTClient`, `ArcticSFTClientConfig`, `train_step` / `merge_sft_step_metrics` |
| `arctic_platform/client/config.py` | `ArcticClientConfig` (shared by SFT and RL) |
| `arctic_platform/sft/processor.py` | `run_sft_pipeline`, `sft` / `sft_ce` losses |
| `arctic_platform/sft/examples/` | HTTP/Ray demos |
| `arctic_platform/common/` | DeepSpeed worker, HTTP/Ray servers, utils, loss registry |

## Client API

`ArcticSFTClient` subclasses `ArcticClient`, so it carries the full shared (blocking) op surface. The SFT-specific part is the default loss contract on the two forward ops, plus `train_step` which returns one merged `metrics` dict.

Use `ArcticSFTClientConfig` rather than the shared `ArcticClientConfig`: it adds
no fields, only validators requiring `training_gpus > 0` and a
`training.checkpoint_path` (both waived when reconnecting via `training_job_id`),
so a bad config fails before any job or GPU is claimed. The shared config stays
permissive because RL needs both exemptions.

| Method | Meaning |
|--------|---------|
| `fwd_bwd(batch, processing=None)` | Forward + loss + backward. Defaults `processing` to `{"loss_fn": "sft"}`. Returns `metrics` from the loss pipeline (at least token-mean `loss`). |
| `fwd_no_grad(batch, processing=None)` | Forward + loss, no backward (eval). Same `metrics` shape as `fwd_bwd`. Narrower than the base, which also takes `reference_model` for the log-prob engine an SFT run never allocates. |
| `step(learning_rate=None)` | One optimizer update. LR is normally server-authoritative (set at engine init from `ds_config` + scheduler); an unset value is omitted from the wire. Returns optimizer `metrics` (at least `grad_norm`). |
| `train_step(batch, processing=None)` | `fwd_bwd` + `step` with one merged `metrics` dict (same contract as RL `update_actor`: step first, then fwd_bwd keys win). |
| `save_checkpoint(path=..., step=..., export_hf=...)` | Save (inherited unchanged from `ArcticClient`, so pass these by keyword). `path` overrides the job's `checkpoint_path` when given. |
| `load_checkpoint(path=None, step=None)` | Restore weights/optimizer/LR/step. |
| `generate(prompts, ...)` / `sync_weights()` | Sampling ops; require `sampling_gpus > 0`. |
| `reconnect_config()` | Config that reattaches to these jobs (job ids set). |
| `shutdown()` | Tear down transport / local server. |

`training.checkpoint_path` is **required** to start a new training job (the
server asserts on it at init and at save). Reconnects via `training_job_id`
inherit the existing job's path.

## Wire batch

Every `fwd_bwd` / `fwd_no_grad` sends this shape. Tensors stay on **CPU** —
the client must not touch CUDA.

```python
{
    "batch": [                              # GAS microbatches (list; preferred)
        {
            "input_ids": LongTensor[B, S_i],
            "attention_mask": LongTensor[B, S_i],  # optional when sample_packing
            "labels": LongTensor[B, S_i],          # -100 = ignore (prompt / pad)
            "position_ids": LongTensor[B, S_i],    # required for sample packing
        },
        # ... len(list) == gradient_accumulation_steps
    ],
    "meta": {
        "pad_token_id": int,
        "gas_microbatches": True,
        "sample_packing": bool,             # optional
    },
    "processing": {"loss_fn": "sft"},       # or "sft_ce"
}
```

A legacy single concatenated tensor-dict is still accepted (demos); the Arctic
Axolotl trainer always sends the list form so the server need not re-pad to a
common length and `split_dict` the GAS axis back apart.

Constraints:

- Each microbatch's `B` must be ≥ `training_gpus` and divisible by it — the
  server DP-shards **each** microbatch independently.
- `train_micro_batch_size_per_gpu` in `ds_config` is the **per-rank** size
  after that shard (`B / training_gpus`).
- `len(batch)` must equal the server's `gradient_accumulation_steps`.
- Missing `labels` raises a clear `ValueError`. All-masked labels
  (`labels == -100`) contribute `(loss.sum=0, loss.tokens=0)` so they cancel
  cleanly in the global token-mean.

### Sample packing

When Axolotl `sample_packing: true` (FA2), each dataloader row is a
concatenated multi-document sequence with **`position_ids` that reset to 0
at each document boundary**. The Arctic trainer forwards those
`position_ids` and sets `meta["sample_packing"]=True`. It does **not**
synthesize an all-ones `attention_mask` (that would force dense
cross-segment attention).

On the server, `run_sft_pipeline` passes `position_ids` into the HF causal LM.
With `flash_attention_2` and `batch_size==1` per DeepSpeed microbatch, HF
derives varlen `cu_seqlens` from the position-id resets
(`_is_packed_sequence`) — attention does not run as a single dense rectangle
across packed documents.

## Loss functions: `sft` vs `sft_ce`

Both compute the same causal-LM cross-entropy over unmasked tokens
(`labels != -100`). The difference is **who computes it** and **how the mean
is normalized**.

### `sft` — trust HuggingFace's loss (default)

1. Forwards `labels` into the HF causal LM.
2. Reads `outputs.loss` (HF's internal shift + CE with `ignore_index=-100`).
3. That scalar is already a **per-shard token-mean**.
4. Metrics reconstruct `loss.sum = loss × n_valid` so DP / gas aggregation can
   still form a global token-mean for *logging*.

Simpler; matches HF Trainer's default objective. This is the path used for
bit-exact parity against native Axolotl SFT.

### `sft_ce` — recompute CE from logits

1. Does **not** pass `labels` to the model (skips HF's CE).
2. Takes raw `logits`, shifts them, runs
   `F.cross_entropy(..., reduction="sum")` in fp32.
3. Normalization is under our control:
   - If `meta["global_num_tokens"]` (+ `dp_size`) is set → a true **global
     token-mean** across DP ranks / microbatches, scaled so DeepSpeed's
     gradient all-reduce (which *averages*) still yields the right grads.
   - Otherwise → falls back to a per-shard token-mean (same as `sft`).

The DeepSpeed worker injects `global_num_tokens` / `dp_size` automatically
for `sft_ce` (cross-rank all-reduce over valid targets) before the loss runs.

Useful when ranks see unequal valid-token counts and you need an exact global
mean, or when you don't want to trust HF's scalar.

### Comparison

| | `sft` | `sft_ce` |
|---|---|---|
| Source of loss | HF `outputs.loss` | Explicit CE on logits |
| Default normalization | Per-shard token-mean | Global token-mean (worker-injected meta); else per-shard |
| Extra cost | None | Softmax / CE over vocab |
| Select via | `processing={"loss_fn": "sft"}` | `processing={"loss_fn": "sft_ce"}` |

On a single GPU (or equal token counts per rank) they agree numerically up to
fp details. They diverge when DP ranks see unequal valid-token counts and you
need a true global mean — that's what `sft_ce` + the worker's
`global_num_tokens` injection is for.

Both emit paired `loss.sum` / `loss.tokens` metrics; the HTTP/Ray servers collapse those into a single global token-mean `metrics["loss"]` for the client. `step()` adds optimizer metrics (`grad_norm`); `train_step` / `merge_sft_step_metrics` combine both into one dict (RL `update_actor` does the same).

### `sft_ce` memory strategy (`logits_optimization`)

`sft_ce` normally materializes the full `[B, S, V]` logits and runs an fp32 CE
over them. For large vocab / long sequences that tensor dominates activation
memory. The `logits_optimization` knob (forwarded on the wire as
`processing.config.logits_optimization`, default `none`) trades it off using the
shared primitives in `arctic_platform/common/utils/tiled_logits.py` — the same
code the RL/ZoRRO logprob path uses:

| Mode | Full `[B,S,V]` logits? | How |
|------|------------------------|-----|
| `none` | Yes | Classic `sft_ce_loss` on `outputs.logits` |
| `compute` | Once | Full logits, but the softmax follow-up runs in token chunks bounded by `logits_optimization_peak_mem_size_in_gib` |
| `memory` | Never | Hidden states are tiled through the LM head under `no_grad` and replayed in backward (`TiledLogProbEntropy`) |

For `compute` / `memory` the server forwards with `output_hidden_states=True`
and `logits_to_keep=1` (HF projects only the last token), then computes the CE
straight from `hidden_states[-1]` (post-final-norm). The next-token shift and
`-100` masking happen in `sft_ce_sum_from_hidden`, and the same
`global_num_tokens` / `dp_size` scaling as `none` is applied, so the loss and
gradients match the full-logits path. `memory` requires the DeepSpeed engine
(tied lm-head / embedding grad bookkeeping) — always true on the server.

Only relevant to `sft_ce`; the default `sft` loss uses HF's own fused CE and
ignores this knob.

## Config reference (`ArcticClientConfig`)

SFT and RL share one config. Engine knobs are nested rather than flattened:
connection settings live on `backend`, DeepSpeed settings on `training`, and
vLLM settings on `sampling`.

| Field | Default | Notes |
|-------|---------|-------|
| `model_name` | **required** | HF model id |
| `training_gpus` | `0` | Server training GPUs (or set `training_job_id` to reconnect) |
| `max_seq_len` | `8192` | Max sequence length (training + sampling) |
| `backend.protocol` | `"http"` | `"http"` or `"ray"` for `OnPremConfig` |
| `backend.host` / `backend.port` | `localhost` / `8000` | Server address |
| `backend.colocate` | `false` | Share GPUs between training and sampling |
| `backend.launch_local_server` | `false` | Spawn local HTTP server from the client |
| `backend.server_cuda_visible_devices` | `null` | GPU list for that subprocess (e.g. `"0,1"`) |
| `backend.startup_timeout` | `600` | Seconds to wait for a launched server |
| `training.checkpoint_path` | **required** for new jobs | Server-side checkpoint dir |
| `training.ds_config` | `null` | DeepSpeed config (optimizer, scheduler, micro-batch, ZeRO, bf16, …) |
| `training.ds_worker_config` | `null` | e.g. `attn_implementation`, `enable_gradient_checkpointing` |
| `training.peft` | `null` | LoRA adapter config. **Cortex only** — rejected at config construction against an `OnPremConfig` backend, which trains dense |
| `training.cuda_ipc` / `training.low_memory` | `false` / `false` | Colocated weight-sync strategy (only with `sampling_gpus > 0`), set on the training job at init; `sync_weights(cuda_ipc=…, low_memory=…)` overrides one call |
| `sampling_gpus` / `sampling.vllm` | `0` / `{}` | Optional vLLM sampling job for `generate` / `sync_weights` |
| `training_job_id` / `sampling_job_id` | `null` | Reattach to existing jobs |
| `job_ready_timeout` / `request_timeout` | 1800 / 1800 | Seconds |

The LR scheduler's total optimizer-step count is DeepSpeed's `total_num_steps`
inside `ds_config`. Set it to the real step budget, not epoch count.

Because the config is shared, `ArcticSFTClient` also accepts a `CortexConfig`
backend and a log-prob job; neither is used by a plain SFT run.

## CPU-only client requirement

The client process must be runnable with `CUDA_VISIBLE_DEVICES=` (empty). All
GPU work happens on the server. When colocating via
`backend.launch_local_server=True`, pass `backend.server_cuda_visible_devices`
so the server child still sees GPUs.

## Framework integrations

- **Axolotl** — plugin under `axolotl.integrations.arctic_platform.sft`. See the
  workspace-level [USAGE.md](../../USAGE.md) (how to enable) and
  [INTEGRATION.md](../../INTEGRATION.md) (how it is wired).

## Examples

```bash
# HTTP smoke (colocated)
CUDA_VISIBLE_DEVICES= python -m arctic_platform.sft.examples.run_sft_http_demo \
  --launch-local-server --server-cuda-visible-devices 0,1 --training-gpus 2

# Unified example (http or ray)
CUDA_VISIBLE_DEVICES= python -m arctic_platform.sft.examples.sft_example \
  --backend onprem-http --server-cuda-visible-devices 0,1 --training-gpus 2
```
