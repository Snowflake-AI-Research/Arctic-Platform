# Tinker integration

Serve [Tinker](https://tinker-docs.thinkingmachines.ai/)'s HTTP API over Arctic
backends, so an **unmodified** [`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook)
recipe trains on Cortex Training. The recipe believes it is talking to Tinker;
`TINKER_BASE_URL` is the only thing that changes.

Verified against `tinker==0.25.0` / `tinker-cookbook==0.5.5`.

## Layout

| File | Role |
|---|---|
| [`router.py`](./router.py) | Tinker's protocol as a FastAPI router. Backend-agnostic: `init_tinker_state` injects five handlers and the router knows nothing else. Owns sessions, futures, the datum/batch conversion and the loss-name mapping. |
| [`proto_wire.py`](./proto_wire.py) | The protobuf codec the modern SDK requires (§5). Imports `tinker.proto.tinker_public_pb2` lazily, so the router still works without the SDK installed. |
| [`cortex.py`](./cortex.py) | Binds the five verbs to `AsyncArcticRLClient`. Owns the envelope dialect and the row-alignment round trip (§6). |
| [`serve.py`](./serve.py) | Provisions the Cortex job from flags and serves the app. Releases the job on shutdown unless `--job-id` attached it to someone else's. |

The router being a pure adapter is what keeps this small: the Tinker surface
reaches Cortex without the on-prem server, DeepSpeed, or Ray being involved at
all. Provisioning is deliberately outside Tinker's protocol — there is no Tinker
verb for "give me two GPUs with ZeRO-2 and FA3", so `serve.py` takes that from
flags and binds the Tinker surface onto the result.

## 1. Install

```bash
pip install "arctic_platform[cortex]"
pip install "tinker==0.25.0" "tinker-cookbook==0.5.5"
```

The server needs `tinker` only for its generated protobuf schema (§5). It does
not need a GPU — Cortex owns those.

## 2. Set Cortex env

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<your PAT>
```

`CortexConfig` is a `pydantic-settings` model, so these populate it directly.
Pass `--config conn.json` instead to read the connection from a file.

## 3. Run the server

```bash
python -m arctic_platform.integrations.tinker.serve \
    --model Qwen/Qwen3-0.6B \
    --training-gpus 1 --sampling-gpus 1 \
    --max-prompt-length 1024 --max-response-length 512 \
    --zero-stage 2 --port 8112
```

Expect ~3–4 min before it answers: the Cortex job has to be placed. Startup is
complete when `get_server_capabilities` returns:

```bash
curl -s http://127.0.0.1:8112/api/v1/get_server_capabilities
# {"supported_models":[{"model_name":"Qwen/Qwen3-0.6B"}]}
```

`--job-id <id>` attaches to an already-running job instead of creating one,
and then leaves it running on shutdown. Without it, stopping the server
releases the GPUs.

## 4. Point a recipe at it

```bash
TINKER_API_KEY=tml-dummy python -m tinker_cookbook.recipes.math_rl.train \
    base_url=http://127.0.0.1:8112 \
    model_name=Qwen/Qwen3-0.6B \
    renderer_name=qwen3_disable_thinking \
    lora_rank=0 \
    env=gsm8k group_size=8 groups_per_batch=8 \
    max_tokens=384 temperature=1.0 learning_rate=2e-6
```

Three of those arguments are not free choices, and each is a constraint from
§7 rather than a tuning preference:

* **`TINKER_API_KEY` must start with `tml-`.** The SDK validates the prefix
  client-side before any request. The value is otherwise unused here.
* **`renderer_name` must be explicit** for models below 4B.
  `get_recommended_renderer_name` looks the model up in a hardcoded table that
  starts at 4B, so a 0.6B raises `KeyError` before any request is sent.
  `math_rl` accepts the name directly, which short-circuits the table.
* **`lora_rank=0`** — Cortex rejects `rank > 0` with a 400, so this path is
  full fine-tuning only. That has a consequence for the learning rate, below.

## 5. The protobuf wire

Modern SDKs post `forward_backward` as `Content-Type: application/x-protobuf`
and **reject a JSON reply** for `ForwardBackwardOutput` and `SampleResponse`:

```
Server returned a JSON payload for {model_cls}, which this SDK version only
supports as proto. The server predates proto response serialization for this type.
```

So the wire is not negotiable, and a JSON-only server fails at the first
training step regardless of how correct its numbers are. Details worth knowing
if you touch `proto_wire.py`:

* Sample *requests* are still JSON. Only the two response types above, and the
  `forward_backward` request, are proto.
* `forward` is folded into the `forward_backward` endpoint upstream, carried by
  a `forward_only` flag on the request.
* The `Tensor` oneof is named `encoding` (`dense` / `sparse_csr`), not
  `payload`.
* `EncodedTextChunk.tokens` is **int32** while request tensors are float32 or
  int64. `BatchedTensor.offsets` are int64 **byte** offsets with n+1 entries.
* `proto_compress_fwdbwd` (zstd) is server-advertised and defaults off, so it
  can be left off. The older `proto_write_fwdbwd` flag is no longer consulted.
* `encode_forward_backward_output` packs only the fields present on *every*
  datum, because a field missing from one datum would shift every later datum's
  slice.

[`test_proto_wire.py`](../../../tests/integrations/tinker/test_proto_wire.py)
uses the SDK's own `request_conv` / `response_conv` converters as the oracle in
both directions. That matters: comparing the router against hand-copied
schema types cannot catch upstream drift, which is exactly how the JSON-only
assumption survived as long as it did.

## 6. Two conventions that silently produce wrong numbers

Both of these fail by returning a plausible-looking number rather than by
raising, which is why they get their own section.

### Row alignment

The router lays a row out as `[pad… prompt][response pad…]` — real tokens
contiguous in the middle. Cortex's packer requires them in the *leading*
columns. Aligning is not enough on its own: the log-probs come back in the
aligned frame while the router's row slices index the original one, so
`cortex.py` inverts the permutation on the way back. Aligning without inverting
shifts every row's log-probs by its own prompt padding — no error, just a wrong
number.
[`test_skipping_the_inverse_would_shift_rows`](../../../tests/integrations/tinker/test_cortex_binder.py)
is the discriminative test.

### The final target token

Tinker's datum convention is `model_input = full[:-1]`, `target_tokens =
full[1:]`, with per-token `advantages` / `mask` / `logprobs` sliced `[1:]`. So
`logprobs[k] = log p(full[k+1] | full[:k+1])` — the predicting-position frame,
which matches Cortex's `compute_logprobs` exactly, with no shift needed.

But `target_tokens[-1]` is a token that does **not** appear in `model_input`.
Without appending it, the last scored position has nothing to predict: its
log-prob is meaningless, and because the advantage sitting there still
multiplies it, so is its gradient. The router appends it (kept out of
`response_mask` and `advantages`, so it is scored *against*, never scored),
which is the same thing the cookbook's own `rl/metrics.py` does with
`model_input.append_int(target_tokens[-1])`.

This was a real bug, and it showed up only as a small persistent divergence in
`kl_sample_train_v1` — 0.35–0.49, against 0.001–0.005 after the fix. Watch that
metric; it exists precisely to catch this class of error.

## 7. Cortex constraints

| Constraint | Consequence |
|---|---|
| Loss registry has `causal_cross_entropy`, `grpo`, `grpo_echo_v1` | The router asks for `verl_grpo`, which does not exist there. `cortex.py` pins `grpo`, whose PPO shape is what Tinker's `ppo` and `importance_sampling` both lower to. `cross_entropy` is gated off in v1. |
| Post-processors are `identity` and `compute_logprobs` only | No `apply_temperature`, so **`temperature=1.0` only**. No `compute_entropy_and_logprobs` either; the zone refuses the request before any model call. |
| `generate` takes no `n` | N samples means sending the prompt N times. Per-position log-probs come back as dicts keyed by token-id *string*, so the sampled token is looked up by id, never positionally — these become `old_log_probs`, and a misaligned list would bias the importance ratio without looking wrong. Unknown sampling params are fatal. |
| LoRA `rank > 0` returns 400 | Full fine-tuning only (§4), which changes the safe learning rate (§8). |
| The image ships FA3 only | `attn_implementation=flash_attention_3`; FA2 dies at model load. |
| DeepSpeed requires `train_batch == micro × accum × dp` | `serve.py` derives it. `offload_optimizer` is omitted entirely rather than set to `{"device": "none"}` — the latter is still enough for DeepSpeed to instantiate CPUAdam, which then asserts its params are on cuda. |
| A job's weights are **persistent, with no reset verb** | Every run against the same job inherits the previous run's weights. Recycle the job between runs or a baseline number is meaningless (§8). |

`forward` returns log-probs as a top-level tensor while `forward-backward`
returns them as nested lists under `post_process_outputs`. Both are rectangular
and padded to full width, so only the lookup differs. The published Cortex API
spec is **stale** on this; the shapes above are what a live job actually
returns, pinned in `TestCortexResponseShapes`.

`_require_logprobs` raises rather than defaulting when it cannot find them. The
router's fallback would be an empty `loss_fn_outputs` entry, which the cookbook
hits as a bare `KeyError` several frames away.

## 8. What a healthy run looks like, and how to wreck one

`kl_sample_train_v1` is the metric to watch. It is the divergence between what
the sampler produced and what the trainer thinks it produced, so it catches
frame and alignment errors that nothing else will. **It should sit in
0.0005–0.005.** Anything above ~0.05 is a plumbing bug, not a hyperparameter.

Two properties of this backend are worth knowing before reading any curve:

* **`approx_kl` reads exactly 0.0 and clipping is inert, by construction.**
  Cortex re-derives π_old from the live forward rather than from a rollout-time
  snapshot, and the cookbook issues one `forward_backward` and one optimizer
  step per batch, so the policy cannot move within a step and π_old ≡ π_new
  holds. Confirmed live: `importance_weight` 1.0, `clip_ratio` 0.0, and
  `avg_loss` equal to −mean(advantage). This is single-update on-policy GRPO; a
  recipe relying on clipping to take several updates per batch cannot be
  reproduced here.
* **The cookbook's default `learning_rate=1e-5` assumes `lora_rank=32`.** Since
  Cortex forces full fine-tuning, that default is far too hot for a 0.6B.

### A healthy run

One full epoch of GSM8K — 935 steps, `done_frac` 1.0, the §4 command verbatim
on Qwen3-0.6B. Windowed means:

| steps | 0–115 | 116–231 | 232–347 | 348–463 | 464–579 | 580–695 | 696–811 | 812–927 |
|---|---|---|---|---|---|---|---|---|
| `correct` | 0.631 | 0.699 | 0.725 | 0.738 | 0.708 | 0.726 | 0.718 | 0.740 |
| `ac_tokens_per_turn` | 221 | 190 | 174 | 188 | 190 | 196 | 215 | 245 |
| `kl_sample_train_v1` | 0.00110 | 0.00086 | 0.00084 | 0.00158 | 0.00160 | 0.00160 | 0.00259 | 0.00353 |
| `entropy` | 0.339 | 0.349 | 0.296 | 0.284 | 0.246 | 0.224 | 0.174 | 0.155 |

**63% → 74% correct**, first quarter 0.665 against last quarter 0.725. Most of
the gain lands in the first ~230 steps and then plateaus, which is a converged
run at this configuration rather than one stopped while still climbing.

Two things to read alongside the headline number. Responses get *shorter* while
accuracy rises through step ~347 (221 → 174 tokens) — the direction matters,
because a model gaming the reward gets longer, not shorter, which is exactly
what the `1e-5` run below does. And from ~step 460 on, entropy keeps falling
(0.25 → 0.155) while token count and `kl_sample_train_v1` both creep up
(0.0016 → 0.0035). The policy is sharpening and starting to drift; accuracy is
flat by then, so there is nothing to gain past ~step 500 at this learning rate.
KL stays inside the healthy band the whole way, peaking at 0.032 on a single
step against the ~0.05 threshold.

Throughput is roughly 5–6 steps/min at this shape, so the epoch is about 90
minutes, plus ~4 min of Cortex provisioning.

### An unhealthy one

The default learning rate is the easiest way to get a run that is plumbed
correctly and still learns nothing, so it is worth showing. Same recipe and
same shape, at the cookbook's default `1e-5` on full fine-tuning:

| steps | 0–115 | 116–231 | 232–347 | 348–463 | 464–579 | 580–695 | 696–811 | 812–927 |
|---|---|---|---|---|---|---|---|---|
| `correct` | 0.424 | 0.098 | 0.013 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `ac_tokens_per_turn` | 344 | 384 | 384 | 384 | 384 | 384 | 384 | 384 |
| `kl_sample_train_v1` | 0.00092 | 0.00146 | 0.00075 | 0.00052 | 0.00051 | 0.00050 | 0.00049 | 0.00048 |

Responses grow until every rollout hits the 384-token cap and gets truncated
before it can state an answer; `correct` reaches exactly 0.000 by step ~348 and
never recovers for the remaining 590 steps. Correctness anti-correlates with
response length at **−0.635** across the run, against −0.321 for the healthy
one.

The instructive part is that `kl_sample_train_v1` *falls* to 0.0005 and stays
there. The transport keeps reporting faithfully while the policy destroys
itself, so a low KL confirms the integration is correct and says nothing about
whether the run is any good. These are independent checks: read KL for
plumbing, `correct` and `ac_tokens_per_turn` for learning.

Two smaller traps in the same family:

* `max_tokens` defaults to **5** in `math_rl`, tuned for `env=arithmetic`.
  GSM8K at that setting is unsolvable by construction and reward pins at the
  format penalty with zero variance. Since the cookbook filters
  constant-reward groups, GRPO then has no gradient signal at all.
* `env=arithmetic` is the wrong smoke test for learning: Qwen3-0.6B scores
  ~95% on it at step 0 and saturates at 100%. It is a good correctness check
  and a useless demonstration of a curve.

## Troubleshooting

| what you see | what it means |
|---|---|
| `Server returned a JSON payload for … only supports as proto` | The response went out as JSON. See §5 — `retrieve_future` must encode proto when the client asks for it. |
| `RequestValidationError: Input should be a valid dictionary`, with binary in `input` | A proto `forward_backward` request was parsed as JSON. The endpoint reads the raw body and dispatches on content type. |
| `The api_key must start with the 'tml-' prefix` | Client-side SDK validation. Use `TINKER_API_KEY=tml-dummy`. |
| `KeyError: 'Qwen3-0.6B'` from `get_recommended_renderer_name` | The cookbook's hardcoded model table starts at 4B. Pass `renderer_name` explicitly (§4). |
| `cortex … returned no per-token log-probs` | `_require_logprobs` could not find them in `post_process_outputs`, `batch`, or the top level. The listed response keys tell you which shape actually came back. |
| `packing requires left-aligned rows` | A payload reached Cortex with padding at the head of a row, i.e. a path that bypassed the alignment in §6. |
| `KeyError: 'mask'` building a `tinker.Datum` | `_KEY_TO_TYPE` has no `mask`; pass `TensorData.from_torch(...)`. |
| `training_config.train_batch_size must be > 0` | DeepSpeed's invariant is unsatisfiable. Set `--micro-batch-size` / `--gradient-accumulation-steps` (§7). |
| Job sits in `PLACING` for 10–25 min | Shared-cluster GPU capacity, not a hang. `PLACING` is easy to miss when filtering for busy jobs, which makes GPUs look free when they are not. 1+1 GPUs places in ~3–4 min; 4+1 can take far longer. |
| `429 … per-account GPU cap reached` | A previous job still holds GPUs. Releasing is not instant; leave a moment between a cancel and the next launch. |
| Reward looks plausible but `kl_sample_train_v1` > 0.05 | A frame or alignment bug, not a hyperparameter. Start with §6. |

## Tests

```bash
pytest tests/integrations/tinker
```

103 tests, no GPU and no Cortex account required. `test_proto_wire.py` skips
without the `tinker` SDK installed.
