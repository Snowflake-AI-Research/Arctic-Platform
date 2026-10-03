# Prepared Generation Streaming

Prepared-generation streaming delivers model output incrementally instead of
waiting for a complete answer. Callers can generate from prepared text or token
IDs, request multiple choices, and cancel individual requests without unloading
the model.

`Driver.stream_generate` returns an async iterator immediately and registers a
model-scoped request ID before remote dispatch. Generation starts on first read.
Use unique request IDs and close early exits explicitly:

```python
from uuid import uuid4
from arctic_platform.inference.server.streaming import StreamLimits

async with driver.stream_generate(
    model_id="loaded-model",
    request_id=uuid4().hex,
    prompt=prepared_token_ids,
    sampling_params={"n": 2, "max_tokens": 64, "temperature": 0.7, "top_p": 0.9},
    limits=StreamLimits(timeout_s=300, stall_timeout_s=30),
) as stream:
    async for event in stream:
        consume(event)
```

`await driver.abort(model_id, request_id)` is explicit request-scoped cancellation.
Possible statuses: `aborted`, `not_started`, `already_terminal`, `not_found`,
`cleanup_unconfirmed`. Cleanup uncertainty quarantines the selected worker from
new scheduled work. Operator recovery may be required. Engine abort acknowledgement
is not a guarantee about physical GPU memory; verify that on the target runtime.
Quarantine blocks both legacy and streaming dispatch and is not cleared by
ordinary availability restoration after weight sync. Replacing the worker clears
the scheduler quarantine; an unhealthy worker also rejects direct generation.
If worker completion precedes delivery of `completed` to the caller, cancellation
of the still-registered public request returns `already_terminal`, not `not_found`.

## Contract

Version 1 events are dictionaries with `version`, zero-based contiguous `sequence`,
`request_id`, and `type`. `delta` carries `choice_index`, incremental `text`, and
incremental `token_ids`. When `logprobs` was requested, a delta also carries
`logprobs`, one entry per token ID in order:
`{"token_id", "token", "logprob", "top": [{"token_id", "token", "logprob"}]}`.
`top` holds the requested number of most likely tokens by rank; the chosen
token keeps its own entry even when it is not among them. A `-inf` logprob is
sent as `-9999.0`. Without `logprobs` the key is absent. `choice_finished` carries the index and `finish_reason`
(`stop` or `length`). `usage` carries `prompt_tokens`, `completion_tokens`, and
`total_tokens`. Prompt usage is counted once, completion usage across all choices.
`completed` follows final usage, after the engine output iterator is closed
successfully. A `terminal_error` contains a sanitized `code`, never successful
completion. The pinned vLLM context-window validation error, identified by its
structured `input_tokens` parameter with a narrow legacy-message fallback, maps
to `context_length_exceeded`; other unexpected engine exceptions map to
`engine_error`. Transport failures and local cancellation can raise instead of
delivering an event. EOF without completed is an error.

Inputs are one prepared text prompt, token-ID list or chat prompt (below). Supported sampling parameters:
temperature, top_p, frequency_penalty, presence_penalty, max_tokens, stop, n,
optional seed, logit_bias, structured_output, thinking_token_budget and logprobs. Unknown options are rejected. Defaults: temperature=1, top_p=1,
frequency_penalty=0, presence_penalty=0, max_tokens=4096, n=1. Penalties must
be in [-2,2]. Limits: n<=8,
max_tokens<=131072, 1 MiB text input or 131072 input token IDs. After vLLM
tokenizes a text prompt, the stream rejects `prompt_tokens + max_tokens` above
the loaded model's context limit. Stop accepts one string or up to four nonempty
strings, each <=4096 UTF-8 bytes. Stop holdback/detokenization remain
engine-owned. `logit_bias` maps at most 300 token IDs (integers or
decimal strings) to biases in [-100,100]. vLLM checks the IDs against the loaded
vocabulary; out-of-vocabulary IDs, or logit_bias on a speculative-decoding
deployment, end the stream with `invalid_sampling_params`. `structured_output` is
`{"json": <JSON schema object>}` (serialized schema <=64 KiB) or
`{"json_object": true}`; the worker turns it into vLLM structured outputs. A
schema that no vLLM structured-output backend accepts ends the stream with
`invalid_structured_output`. `thinking_token_budget` is an integer in
[1, max_tokens]; vLLM rejects it with `invalid_sampling_params` unless the
model was loaded with a reasoning parser. `logprobs` is an integer in [0, 20],
the number of alternatives reported per token. `"sampling_params" in
STREAM_CAPABILITIES` tells callers these four parameters are accepted. Active LoRA selection is forwarded. No HTTP, SSE, or
training-specific prompt mutation occurs here. For nonstream responses DSS can
collect the same events into a complete response.

## Chat Prompts

`prompt` may also be a `ChatPrompt` (`arctic_platform.inference.server.chat`):
OpenAI-style `messages`, optional `tools`, `tool_choice`, `parallel_tool_calls`
and `reasoning_effort`. `"chat_prompt" in STREAM_CAPABILITIES` tells callers the
installed version supports it. The worker renders it with vLLM's own chat front
end on the loaded engine, so the model's template applies (including DeepSeek-V4
and gpt-oss Harmony), and splits output with vLLM's reasoning and tool parsers.
Which parsers apply is engine configuration: `reasoning_parser`,
`tool_call_parser` (popped by the worker, like vllm serve's flag) and, for
DeepSeek-V4, `tokenizer_mode`. `reasoning_effort` is passed to the template as
is; callers map OpenAI values to the model family's own.

- Before rendering, any string in messages, tools or a named `tool_choice` that
  contains one of the tokenizer's special or added tokens fails the stream with
  `invalid_message_content` and `param` (for example `messages[1]`). Input the
  template or vLLM rejects fails with `invalid_chat_request`, with `param` when
  vLLM names one. Neither carries message text.
- If `max_tokens` is omitted, the budget is `min(context left after the
  rendered prompt, 4096)`. A prompt that leaves no room fails with
  `context_length_exceeded` and `context_limit_source="prompt"`.
- Instead of `delta`, a chat stream emits `content_delta` (`text`),
  `reasoning_delta` (`token_count` only; reasoning text is never emitted) and
  `tool_call_delta` (`index`, `arguments`, plus `id` and `name` on a call's first
  event). Markup the parser is still matching emits nothing until it resolves.
  `parallel_tool_calls=false` keeps only the first call.
- `choice_finished` reports `tool_calls` when a choice that called a tool stops.
  `usage` adds `reasoning_tokens`, counted by the reasoning parser across choices.
- Undelivered events merge only with the same kind of the same choice; tool-call
  arguments merge only within one call.
- With `logprobs`, each `content_delta` carries the `token_ids` and `logprobs`
  of the engine output that produced it; reasoning and tool-call tokens carry
  none, as OpenAI reports logprobs for the answer only.
- `structured_output` cannot be combined with a chat prompt whose tools need a
  grammar of their own (`invalid_chat_request`, `param="structured_output"`):
  vLLM applies one grammar per request. Without `max_tokens`,
  `thinking_token_budget` is checked against the budget after rendering.

## Flow Control and Lifecycle

Native Ray async generators are eager. Each delivered event therefore requires a
sequence acknowledgement before the worker yields another; ClientStream performs
this automatically on the next read. No ObjectRefGenerator crosses an actor boundary.
The engine pump never waits for the client; it drains into a bounded queue and
aborts on overflow rather than silently dropping output or pausing the shared engine.

Defaults: 128 queued events, 1 MiB queued serialized payload, 256 KiB per event,
and 128 sessions per worker. Streaming and legacy
`generate()` requests share each worker's `active_requests` / `concurrency_limit`
budget. Each dispatched stream consumes one shared slot and is additionally counted
against the per-worker `MAX_WORKER_STREAMS` cap (128); this is an extra ceiling,
not a reserved capacity pool. With the default worker concurrency of 128, all 128
slots can serve streaming requests. A lower configured worker concurrency still
limits dispatch. There is no separate scheduler-wide streaming admission cap:
additional streams wait for worker capacity, like legacy requests. For example,
two replicas at the default concurrency can run 256 streams, with further
requests waiting until a slot opens. The waiting request count is not bounded;
deployments that need a bounded backlog must apply a separate admission policy.
Per-stream buffer limits, deadlines, and stalled-consumer cleanup still apply.
Neither traffic class has reserved headroom or a fairness guarantee: heavy traffic
of either kind can delay the other. For example, with a concurrency limit of 4,
three active streams leave at most one slot for legacy generation. A full worker
causes queued streams to wait; a definite registration rejection is not an engine
cleanup failure and does not quarantine the replica. Byte limits measure
JSON payload, not Python heap/Ray/vLLM RSS. Event count, payload size, token/input
caps and concurrency bound retained application objects; exact engine/object-store
memory requires runtime verification. Limits can be reduced with StreamLimits;
hard ceilings are checked. Stalled consumers abort after 30 seconds; requests expire
after 300 seconds including queueing. Cleanup uses a separate finite timeout.

The caller enforces its deadline with a local monotonic clock, including time
waiting for dispatch. Registration sends the remaining duration, not a wall-clock
timestamp; the worker starts its own monotonic deadline when registration runs.
No synchronized host clocks are required. Transport/actor-mailbox delay is still
covered by the caller's deadline and cancellation; the worker's independent timer
starts on receipt and cannot account for that delay if the caller disappears.
This internal registration argument changed from an absolute timestamp to a
duration; deploy matching caller/worker library versions.

Each session has one shared engine-abort task for producer failures and explicit
cancellation. Its result is reused rather than retrying an abort during teardown.
Unconfirmed engine or iterator cleanup keeps the worker unhealthy and cancellation
returns `cleanup_unconfirmed`, causing the scheduler to quarantine that worker.
Cancelling an iterator-close waiter cancels and joins its child cleanup task within
a finite cleanup interval. Interrupted or timed-out iterator teardown is not
confirmed by a successful engine abort; it blocks subsequent lifecycle mutation.

Random internal attempt IDs isolate engine cancellation. Recently completed public
IDs are remembered in a 100,000-entry, one-hour TTL cache with oldest-first eviction.
The cache never blocks new request admission when full. Duplicate detection is
best effort after eviction or pool recreation: callers must use globally unique
IDs and must not reuse IDs for delayed DSS retries. No transparent retry after output.

Weight sync, engine pause and sleep abort streams before mutation. New worker
stream registration is blocked during those operations. Pool shutdown aborts
registered streams and proceeds to worker termination even if graceful stream
cleanup fails, then reports the failure. Scale-down aborts only streams assigned
to the removed worker, leaving retained replicas' requests running. If graceful
scale-down cleanup fails, the selected worker is still removed and a force-kill
is attempted before the cleanup failure is propagated. A kill failure is logged;
physical termination still requires runtime verification.
Individual and bulk cancellation release worker sessions and cancel watchdogs,
even if no reader was ever created. Bulk cancellation attempts every session
before reporting cleanup failure, including on an already-unhealthy worker.
Existing completed-result APIs remain unchanged. Tests are in `tests/streaming/`.

Sleep and every weight-update strategy abort all scheduler-owned streams, including
unread and queued requests, before draining legacy work or mutating the engine.
This also applies to broadcast, LoRA, and speculative-model updates. Additive
scale-up does not close streaming admission on existing healthy replicas while
the new replica initializes.

If stream cleanup fails before sleep or any weight update begins,
the operation fails without mutating the engine. Scheduling is restored for
healthy replicas while workers with unconfirmed cleanup remain quarantined.
Existing pauses, including a sleeping pool, and worker unavailability are
preserved across all weight-sync entry points. This rollback only
covers preparation failures; it does not resume workers after an interrupted
engine mutation whose outcome is uncertain.

Driver-wide shutdown attempts every model pool even when an earlier pool fails.
Each attempted pool is removed from the Driver in `finally`; model-specific
shutdown also removes an already-torn-down pool on failure. One failure is
re-raised directly; multiple failures are propagated as an exception group after
all pools have been attempted. Failed model-specific shutdown does not trigger
automatic capacity rebalancing.
On Python 3.10, the server extra installs the `exceptiongroup` backport for the
same multi-failure shutdown contract. Python 3.11 and later use the built-in type.
