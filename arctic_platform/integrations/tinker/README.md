# Tinker integration

This integration serves Tinker's HTTP API over Cortex Training. Compatible
`tinker-cookbook` recipes run unchanged except for configuration such as
`TINKER_BASE_URL`, model, and training parameters.

Validated with `tinker==0.25.0` and `tinker-cookbook==0.5.5`.

## Install

```bash
pip install "arctic_platform[tinker]"
pip install "tinker==0.25.0" "tinker-cookbook[math-rl]==0.5.5"
```

The server is CPU-only. Cortex runs the training and sampling workers.

## Configure Cortex

```bash
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
export ARCTIC_CORTEX_DATABASE=<db>
export ARCTIC_CORTEX_SCHEMA=<schema>
export ARCTIC_CORTEX_PAT=<your PAT>
```

The account needs:

- an existing database and schema
- a Snowflake PAT
- quota for both the training and sampling sub-jobs

A job can remain in `PLACING` while it waits for GPU capacity.

## Start the server

```bash
python -m arctic_platform.integrations.tinker.serve \
    --model Qwen/Qwen3-0.6B \
    --training-gpus 1 \
    --sampling-gpus 1 \
    --max-prompt-length 1024 \
    --max-response-length 512 \
    --zero-stage 2 \
    --port 8112
```

Check readiness:

```bash
curl -s http://127.0.0.1:8112/api/v1/get_server_capabilities
```

Use `--job-id <id>` to attach to an existing Cortex job. The server releases
jobs it creates when it shuts down; attached jobs remain running.

On-policy distillation samples a teacher with
`create_sampling_client(base_model=...)` and scores the student's rollouts with
`compute_logprobs`. Start the server with the teacher as well:

```bash
python -m arctic_platform.integrations.tinker.serve \
    --model Qwen/Qwen3.5-9B-Base \
    --teacher-model Qwen/Qwen3.5-9B \
    --teacher-sampling-gpus 2 \
    ...
```

The teacher runs from its base weights as a sampling-only Cortex job of its
own, created and released with the server. A sampler for a model that is
neither the trained model nor the teacher returns 400.

## Run a cookbook recipe

```bash
TINKER_API_KEY=tml-dummy python -m tinker_cookbook.recipes.math_rl.train \
    base_url=http://127.0.0.1:8112 \
    model_name=Qwen/Qwen3-0.6B \
    renderer_name=qwen3_disable_thinking \
    lora_rank=0 \
    env=gsm8k \
    group_size=8 \
    groups_per_batch=8 \
    max_tokens=384 \
    temperature=1.0 \
    learning_rate=2e-6 \
    save_every=0 \
    eval_every=0
```

Required settings:

- `TINKER_API_KEY` must start with `tml-`; the value is otherwise unused.
- Set `renderer_name` explicitly for models absent from the cookbook's
  recommendation table.
- Use `lora_rank=0`.
- Use `temperature=1.0`.
- Keep `max_tokens < --max-response-length`.
- Ensure rendered prompts fit `--max-prompt-length`.
- Use `save_every=0` unless acknowledgment-only saves are acceptable.

Training rows are never truncated. A prompt or response longer than its limit
is accepted as long as the whole datum fits
`--max-prompt-length + --max-response-length`; a longer datum returns 400.

The cookbook's default learning rate is intended for LoRA. Full fine-tuning
may require a lower learning rate.

## Supported scope

Supported:

- text-only full fine-tuning
- `ppo`, `importance_sampling`, and `cross_entropy`, with the importance
  ratio taken against the sampler's log-probs; `ppo` accepts
  `clip_low_threshold` and `clip_high_threshold` in `loss_fn_config`
- sampling, forward-backward, optimizer step, and sampler weight sync
- client-defined custom losses through `forward_backward_custom`
- on-policy distillation from one teacher, including `compute_logprobs`

Current limitations:

| Limitation | Behavior |
|---|---|
| LoRA | `lora_rank > 0` returns 400. |
| Temperature | Sampling temperatures other than `1.0` return 400. |
| Checkpoints | `save_weights` returns an acknowledgment path; load and resume are not implemented. |
| Sequence limits | A datum longer than `--max-prompt-length + --max-response-length` returns 400. |
| Loss config | `loss_fn_config` keys other than PPO's two clip thresholds return 400. |
| Forward | `forward` (log-probs without gradients) is routed to Cortex, which currently fails it with `KeyError: 'pad_token_id'` in its training pipeline. The cookbook's NLL evaluator uses it, so run `chat_sl` with `eval_every=0`. |
| Teacher | One teacher, from its base weights. A teacher checkpoint (the distillation recipe's `teacher_checkpoint`) and `topk_prompt_logprobs` return 400. |
| Base-model samplers | `create_sampling_client(base_model=...)` for the trained model reads its untrained weights, which the sampler holds only until the first weight sync; after that it returns 409. |
| Optimizer overrides | Only the learning rate is applied at step time. |
| Multimodal input | Only encoded text tokens are passed to Cortex. |
| Authentication | The local Tinker server does not authenticate requests. |

Recipes that require audio, images, LoRA, checkpoint resume, reference-model
workers, external tools, or external graders are not covered by this
integration.

## Protocol notes

Current Tinker SDKs use protobuf for:

- `forward_backward` requests
- `ForwardBackwardOutput` responses
- `SampleResponse` responses

Sample requests remain JSON. The codec uses the SDK's generated
`tinker_public_pb2` schema.

The adapter also handles three tensor conventions:

- Tinker rows contain left prompt padding; Cortex expects valid tokens in
  leading columns. The Cortex binder aligns rows before the request and restores
  the original order in returned log-probs.
- The final `target_tokens` item is appended as a scoring token. It is excluded
  from the response mask and advantages.
- Cortex refuses a batch with fewer rows than training GPUs. Short batches are
  padded with copies of a row that carry no loss, and the copies are dropped
  from the returned log-probs.

## Custom losses

`forward_backward_custom` computes a loss in the SDK and sends
`dC/dlogprobs` as per-token weights. The adapter maps this to Cortex `grpo` with:

- `advantages = -weights`
- no old log-probs, making the importance ratio `1`
- `batch_num_tokens = 1`

This preserves the client loss gradient. Server-side loss and entropy metrics
on this path do not represent the client's loss or model entropy.

## Validation

Live validation covered `math_rl`, `chat_sl`, and `rl_loop` with importance
sampling, PPO, and cross-entropy. Detailed results are recorded in the PR.

For RL runs, `kl_sample_train_v1` checks agreement between sampler and trainer
log-probs. Its level depends on the model. With Qwen3.5-9B and full
fine-tuning it stayed between `0.03` and `0.045` on GSM8K. On MATH it rose from
`0.02` to a peak of `0.084` while accuracy climbed fastest, then settled near
`0.04`. A value far above the model's usual level from the first step, or one
that keeps rising, points to a request-shape or alignment problem.

## Tests

```bash
pip install -e ".[testing]"
pytest -q tests/integrations/tinker
```

The tests cover API endpoints, request schemas, protobuf conversion, datum
packing, Cortex lowering, row alignment, custom-loss gradients, and error
handling. They do not require a Cortex account or GPU.

## Troubleshooting

| Error | Action |
|---|---|
| `The api_key must start with the 'tml-' prefix` | Set `TINKER_API_KEY=tml-dummy`. |
| `KeyError` from `get_recommended_renderer_name` | Pass `renderer_name` explicitly. |
| `Server returned a JSON payload ... only supports as proto` | Confirm the protobuf response path is installed and active. |
| `cortex ... returned no per-token log-probs` | Check the Cortex response shape and configured post-processor. |
| `packing requires left-aligned rows` | Confirm requests pass through the Cortex binder. |
| Job remains in `PLACING` | Check account GPU capacity and quota. |
| `429 ... GPU cap reached` | Release an existing job and retry after cancellation completes. |

## Files

| File | Role |
|---|---|
| `router.py` | Tinker API routes, sessions, futures, and datum conversion |
| `proto_wire.py` | Tinker protobuf request and response codec |
| `cortex.py` | Cortex request lowering and tensor alignment |
| `serve.py` | Cortex job provisioning and ASGI server |
