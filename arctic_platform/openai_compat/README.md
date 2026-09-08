# `arctic_platform.openai_compat`

An OpenAI-compatible `/v1` endpoint over an Arctic sampling job. Point any
OpenAI client at it — the `openai` SDK, LiteLLM, an eval harness, curl — and
change nothing but `base_url`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=CORTEX_PAT)
client.chat.completions.create(model="Qwen/Qwen3-8B", messages=[{"role": "user", "content": "hi"}])
```

## Serving

Attach to a sampling job that already exists and serve until interrupted:

```bash
python -m arctic_platform.openai_compat --config client.json --port 8000
```

`--config` is an `ArcticClientConfig` with `sampling_job_id` set — this serves
an endpoint, it does not create one. The sampling job **stays up when the
gateway exits**; tear it down with the Cortex CLI when you're actually done
with it.

From inside a driver that already holds a client:

```python
from arctic_platform.openai_compat import OpenAIGateway

with OpenAIGateway(client=client, tokenizer=tok, model_name=name, max_model_len=8192) as gateway:
    run_my_harness(base_url=gateway.base_url)
```

### Reaching it from somewhere else

The default bind is `127.0.0.1`, which is not reachable from inside a
container. Agent harnesses that run a CLI inside a sandbox and point it at
`OPENAI_BASE_URL` need a routable address, so bind one — and because that also
makes the endpoint reachable by anything else that can route to the host,
`--api-key` becomes mandatory:

```bash
python -m arctic_platform.openai_compat --config client.json --host 0.0.0.0 --api-key "$TOKEN"
```

Binding off-loopback without a key is refused rather than served wide open.

## Compatibility

Non-streaming. `stream=true` is refused with a 400 rather than faked by
replaying a finished generation as SSE deltas.

The governing rule: **a parameter that changes what the model should produce is
either implemented or rejected, never silently ignored.** Accepting one and
dropping it returns a plausible-looking 200 that is simply wrong, which is the
most expensive failure mode to diagnose.

### Supported

| Parameter | Notes |
| --- | --- |
| `model` | Any value is accepted (one model is served) and echoed back. |
| `messages` | Strings, content-part arrays, and `null` content all normalized. |
| `max_tokens` / `max_completion_tokens` | Omitted means *the rest of the context window*, per OpenAI — not the engine's 16-token default. |
| `temperature`, `top_p`, `n`, `stop`, `seed` | |
| `presence_penalty`, `frequency_penalty` | |
| `logprobs`, `top_logprobs` | Token strings and bytes are decoded, not stubbed. |
| `tools`, `tool_choice` | `auto` and `none`. Rendered into the chat template; calls parsed back into `tool_calls`. |
| `prompt` (completions) | String, token-id array, or an array of either. |
| `top_k`, `min_p`, `repetition_penalty` | vLLM extensions. |
| `chat_template_kwargs` | e.g. `{"enable_thinking": true}`. |

### Accepted and ignored

`user`, `store`, `metadata`, `service_tier`, `prompt_cache_key`,
`safety_identifier`, `stream_options`, `parallel_tool_calls` — none of these
can change the sampled text.

### Rejected with a 400

| Parameter | Why |
| --- | --- |
| `stream=true` | This endpoint is non-streaming. |
| `response_format` | Constrained decoding is not wired through the sampling job, so the reply would be unconstrained text. |
| `logit_bias` | Not forwarded to the sampler. |
| `tool_choice` (specific tool / `required`) | Forcing a tool needs constrained decoding. |
| `functions`, `function_call` | Deprecated by OpenAI; use `tools`. |
| `reasoning_effort` | Use `chat_template_kwargs` if the model's template supports it. |
| `audio`, `modalities`, `prediction`, `web_search_options` | Text-only models. |
| `echo`, `suffix`, `best_of` | Not supported by the sampling path. |
| Image / audio / file content parts | Text-only models. |
| Anything unrecognized | Fails loudly instead of being dropped. |

Errors use OpenAI's `{"error": {...}}` envelope, so the SDK raises
`BadRequestError`, `AuthenticationError`, `RateLimitError`, or `NotFoundError`
with a message that names the offending parameter. Backend capacity errors are
relayed as `429` with `Retry-After`, which stock client retry policies absorb.

### Extensions

Responses carry `prompt_token_ids` at the top level and `token_ids` per choice,
matching vLLM's OpenAI-server extension. RL harnesses read these to turn an
eval transcript into trainable rollouts without a second pass; clients that
don't know the fields ignore them.

## Concurrency

Requests are served concurrently. A blocking client is driven through
`asyncio.to_thread` rather than awaited on the server's event loop, which would
stall every other in-flight request behind whichever one is talking to the job
and silently serialize a parallel workload. `--max-concurrency` bounds how many
requests reach the job at once; the rest queue as coroutines.

## Layering

| Module | Role |
| --- | --- |
| `translation.py` | Pure OpenAI ⇄ sampling-job functions. No I/O, no framework, no client. |
| `schemas.py` | Request models plus the supported / ignored / rejected policy. |
| `errors.py` | OpenAI's error envelope. |
| `backend.py` | The `SamplingBackend` seam and the client adapters. |
| `router.py` | The `/v1` routes. |
| `server.py` | `build_app`, `OpenAIGateway`, and the CLI. |

Tests live in `tests/openai_compat/`. The end-to-end suite drives a real
uvicorn server with the real `openai` SDK, so every response is validated
against OpenAI's own Pydantic models rather than against our idea of them.
