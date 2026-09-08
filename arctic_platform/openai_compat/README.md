# `arctic_platform.openai_compat`

An OpenAI-compatible `/v1` endpoint over an Arctic sampling job. Point any
OpenAI client at it — the `openai` SDK, LiteLLM, an eval harness, curl — and
change nothing but `base_url`.

```bash
python -m arctic_platform.openai_compat --config client.json --port 8000
```

```python
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=CORTEX_PAT)
client.chat.completions.create(model="Qwen/Qwen3-8B", messages=[{"role": "user", "content": "hi"}])
```

`--config` is an `ArcticClientConfig` with `sampling_job_id` set — this serves
an endpoint, it does not create one. The job **stays up when the gateway
exits**; tear it down with the Cortex CLI.

The default bind is `127.0.0.1`, which is unreachable from inside a container.
Binding wider is allowed but requires a key, since it also makes the endpoint
reachable by anything else that can route to the host:

```bash
python -m arctic_platform.openai_compat --config client.json --host 0.0.0.0 --api-key "$TOKEN"
```

## Two things that bite in practice

**Pick a simple served model name.** Some clients validate the model string
before it ever reaches us. LiteLLM's `hosted_vllm` provider — the path Harbor's
LiteLLM backend takes — requires exactly one `/` and a name matching
`[A-Za-z0-9._-]{1,64}`, so `hosted_vllm/Qwen/Qwen3-8B` is rejected for having
two slashes. Serve an alias instead, and the endpoint will advertise and echo
it:

```bash
python -m arctic_platform.openai_compat --config client.json --served-model-name qwen3-8b
```

That provider also requires a `model_info` block (`max_input_tokens`,
`max_output_tokens`, `input_cost_per_token`, `output_cost_per_token`); the costs
can be `0.0`.

**Token-id prompts only work against Cortex.** `/v1/completions` accepts a
token-id array, but the on-prem sampling server validates `prompts` as
`list[str]` and rejects them. Chat completions are unaffected — they always send
a rendered string.

## Compatibility

Non-streaming: `stream=true` is refused rather than faked.

The rule: **a parameter that changes what the model should produce is either
implemented or rejected, never silently ignored.** Accepting one and dropping it
returns a plausible-looking 200 that is simply wrong.

**Supported** — `model` (echoed back), `messages` (strings, content-part arrays,
and `null` content all normalized), `max_tokens` / `max_completion_tokens`
(omitted means *the rest of the context window*, per OpenAI, not the engine's
16-token default), `temperature`, `top_p`, `n`, `stop`, `seed`,
`presence_penalty`, `frequency_penalty`, `logprobs`, `top_logprobs`, `tools` and
`tool_choice` (`auto`/`none`), plus the vLLM extensions `top_k`, `min_p`,
`repetition_penalty`, `chat_template_kwargs`. `/v1/completions` takes a string,
a token-id array, or an array of either.

**Accepted and ignored**, because they can't change the sampled text — `user`,
`store`, `metadata`, `service_tier`, `prompt_cache_key`, `safety_identifier`,
`stream_options`, `parallel_tool_calls`.

**Rejected with a 400 naming the parameter** — `stream=true`,
`response_format` and `logit_bias` (not wired through the sampling job), a
forced `tool_choice` (needs constrained decoding), `functions` /
`function_call` (deprecated), `reasoning_effort`, `audio` / `modalities` /
`prediction` / `web_search_options`, `echo` / `suffix` / `best_of`, non-text
content parts, and anything unrecognized.

Errors use OpenAI's `{"error": {...}}` envelope, so the SDK raises a typed
exception naming the offending parameter. Capacity errors relay as `429` with
`Retry-After`. Responses also carry `prompt_token_ids` and per-choice
`token_ids` (vLLM's extension) so RL harnesses can build rollouts from an eval
transcript; clients that don't know the fields ignore them.

## Layout

`translation.py` is pure — request models, the parameter policy, and
OpenAI ⇄ sampling-job functions, with no I/O or framework. `server.py` has the
routes, the app, and the CLI. `client` is anything with
`generate(prompts, sampling_params)`, sync or async, so the contract is testable
against a stub. Tests: `tests/openai_compat/`.
