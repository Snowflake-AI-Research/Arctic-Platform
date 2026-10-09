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
A request whose cleanup was unconfirmed keeps answering `cleanup_unconfirmed`,
after retirement and on every repeated abort, rather than `already_terminal`.

## Contract

Version 1 events are dictionaries with `version`, zero-based contiguous `sequence`,
`request_id`, and `type`. `delta` carries `choice_index`, incremental `text`, and
incremental `token_ids`. When `logprobs` was requested, a delta also carries
`logprobs`, one entry per token ID in order:
`{"token_id", "token", "logprob", "top": [{"token_id", "token", "logprob"}]}`.
`top` holds the requested number of most likely tokens by rank; the chosen
token keeps its own entry even when it is not among them. A `-inf` logprob is
sent as `-9999.0`. Without `logprobs` the key is absent. `choice_finished`
carries the index and `finish_reason`: `stop`, `length`, or (chat prompts
only) `tool_calls`. `usage` carries
`prompt_tokens`, `completion_tokens`, and `total_tokens`. Prompt usage is
counted once, completion usage across all choices.
`completed` follows final usage, after the engine output iterator is closed
successfully. A `terminal_error` contains a sanitized `code`, never successful
completion. Only the chat input errors `invalid_message_content` and
`invalid_chat_request` may add `param`, the name of the offending request field
(never its content). The pinned vLLM context-window validation error, identified by its
structured `input_tokens` parameter with a narrow legacy-message fallback, maps
to `context_length_exceeded`; other unexpected engine exceptions map to
`engine_error`. If the stream's cleanup after that error is unconfirmed, the
delivered `code` is `cleanup_unconfirmed` instead and `context_limit_source` and
`param` are omitted, because the engine may still be running the request. Transport failures
and local cancellation can raise instead of delivering an event. EOF without completed is an error.

Inputs are one prepared text prompt, token-ID list or chat prompt (below). Supported sampling parameters:
temperature, top_p, frequency_penalty, presence_penalty, max_tokens, stop, n,
optional seed, logit_bias, structured_outputs, thinking_token_budget and
logprobs. Unknown options are rejected. Defaults: temperature=1, top_p=1,
frequency_penalty=0, presence_penalty=0, max_tokens=4096, n=1. Penalties must
be in [-2,2]. Limits: n<=8,
max_tokens<=131072, 1 MiB text input or 131072 input token IDs. After vLLM
tokenizes a text prompt, the stream rejects an explicit `max_tokens` when
`prompt_tokens + max_tokens` is above the loaded model's context limit. The
default `max_tokens` is never rejected this way: generation stops at the context
limit with finish reason `length`. Stop accepts one string or up to four nonempty
strings, each <=4096 UTF-8 bytes. Stop holdback/detokenization remain
engine-owned. `logit_bias` maps at most 300 token IDs (integers or
decimal strings) to biases in [-100,100]. vLLM checks the IDs against the loaded
vocabulary; out-of-vocabulary IDs, or logit_bias on a speculative-decoding
deployment, end the stream with `invalid_sampling_params`. `structured_outputs`
is `{"json": <JSON schema object>}` (serialized schema <=64 KiB, objects and
arrays nested <=64 levels) or
`{"json_object": true}`; the worker turns it into vLLM's
`StructuredOutputsParams`. A schema that no vLLM structured-output backend
accepts ends the stream with `invalid_structured_output`.
`thinking_token_budget` is an integer in [1, max_tokens], where an omitted
max_tokens counts as 4096; vLLM rejects it with `invalid_sampling_params` unless
the model was loaded with a reasoning parser. `logprobs` is an integer in
[0, 20], the number of alternatives reported per token; a value above the loaded
model's `max_logprobs` ends the stream with `invalid_sampling_params`.
Active LoRA selection is forwarded. No HTTP, SSE, or
training-specific prompt mutation occurs here. For nonstream responses DSS can
collect the same events into a complete response.

## Chat Prompts

`prompt` may also be a `ChatPrompt` (`arctic_platform.inference.server.chat`):
OpenAI-style `messages`, optional `tools`, `tool_choice`, `parallel_tool_calls`
and `reasoning_effort`. The worker renders it with vLLM's own chat front
end on the loaded engine, so the model's template applies (including DeepSeek-V4
and gpt-oss Harmony), and splits output with vLLM's reasoning and tool parsers.

Which parsers apply comes from `CHAT_MODELS` in `chat.py`, keyed by the
architecture vLLM resolves for the checkpoint, so every checkpoint of a listed
architecture gets chat mode without engine kwargs:

| Architectures | Family | Reasoning / tool parser | Thinking off | `reasoning_effort` sent to the template | Verified |
|---|---|---|---|---|---|
| `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` | Qwen3 | `qwen3` / `hermes` | yes | as is (ignored) | GPU (Qwen3-0.6B) |
| `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration`, `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration` | Qwen3.5, 3.6, 3.8 | `qwen3` / `qwen3_coder` | yes | `minimal` becomes `low`, `high` and `max` become `xhigh` (Qwen3.8 takes low, medium, xhigh) | not yet |
| `GlmMoeDsaForCausalLM` | GLM-5, 5.1, 5.2 | `glm47` / `glm47` | yes | `minimal`, `low`, `medium` become `high`; `xhigh` becomes `max` | not yet |
| `Glm4MoeForCausalLM` | GLM-4.5, 4.6, 4.7 | `glm47` / `glm47` | yes | as is (ignored) | not yet |
| `Glm5NextForCausalLM`, `Glm5NextForConditionalGeneration` | GLM-5.3-Flash | `glm47` / `glm47` | no | `minimal` becomes `low`, `medium` becomes `high`, `xhigh` becomes `max` | not yet |
| `DeepseekV4ForCausalLM`, `DeepseekV4ForConditionalGeneration` | DeepSeek-V4 | `deepseek_v4` / `deepseek_v4` | yes | as is (vLLM's tokenizer maps it) | not yet |
| `GptOssForCausalLM` | gpt-oss | `openai_gptoss` / `openai` | no | `minimal` becomes `low`, `xhigh` and `max` become `high` | not yet |
| `MiniMaxM2ForCausalLM` | MiniMax-M2, M2.1 | `minimax_m2` / `minimax_m2` | no | as is (ignored) | not yet |
| `AfmoeForCausalLM` | Arcee Trinity (Large-Preview; see below) | none / `hermes` | yes (never thinks) | as is (ignored) | not yet |
| `NemotronHForCausalLM` | Nemotron 3 Nano, Super | `nemotron_v3` / `qwen3_coder` | yes | as is (ignored) | not yet |

"Verified" is how far each family has been run: GPU is `test_gpu_chat.py`
(Qwen3-0.6B); QA6 is a DSS run on QA6, which no family has had yet (the DSS
QA6 matrix runs before the pin bump). Entries not yet verified are best-effort
from vLLM's parsers and the families' templates.

A reasoning parser of "none" means the family does not reason: everything it
writes is content, logprobs are allowed, and `reasoning_tokens` is 0. Trinity's
checkpoints differ under one architecture, and the table follows
Trinity-Large-Preview. Trinity-Large-Thinking opens `<think>` and writes XML
tool calls, so it needs `chat_reasoning_parser=deepseek_r1` and
`tool_call_parser=qwen3_coder`; Trinity-Mini opens `<think>` with hermes calls
and needs `chat_reasoning_parser=deepseek_r1`. A reasoner on a model that never
closes `</think>` would return its whole answer as reasoning.

That is 18 architectures in 10 families. "Thinking off" means
`reasoning_effort="none"` turns thinking off: vLLM hands the template
`enable_thinking=False`. Where it can't, `none` fails the stream with
`invalid_chat_request` and `param="reasoning_effort"` rather than thinking
anyway. Every other value Arctic accepts reaches the template as a level it
takes (`test_chat_models.py` checks each family). The table suits checkpoints that keep
their family's chat template; one with a different template (an instruct-only
or thinking-only variant, a coder model) can set the `chat_reasoning_parser` and
`tool_call_parser` engine kwargs, which override the table and are popped by the
worker like vllm serve's flags. They also enable chat on an unlisted
architecture. Without either, a chat stream on an unlisted architecture fails
with `chat_unsupported`, logged once per worker. `tokenizer_mode` is left to
vLLM, which picks DeepSeek-V4's by architecture.

`Driver.get_chat_support(model_id)` (and `ReplicaPool.get_chat_support()`)
reports `{"chat_prompt": bool, "thinking_optional": bool}` for a loaded model:
whether its streams take a `ChatPrompt`, and whether thinking can be turned off.
It builds the worker's chat front end, so `chat_prompt` is false when that
fails, and when vLLM finds no chat template for the model (in the tokenizer,
the processor or vLLM's fallbacks; gpt-oss and DeepSeek-V4 render without
one). Streams on such a model fail with `chat_unsupported`.

The chat reasoning parser applies to chat streams only. The job's
`reasoning_parser` also changes `/generate` (it prefills `<think>` for
`enable_thinking`, splits reasoning out of the result and feeds action-mask
replay), so it is not set for chat; `/generate` keeps no parser. Chat grammars
(`tool_choice` `required` or named) must still wait for the end of reasoning,
and vLLM does that only with an engine-wide structured-output reasoner, so the
worker sets one when the engine has none and passes `reasoning_ended=True` with
every `/generate` request and plain stream. vLLM then constrains those from the
first token, exactly as with no reasoner. An engine that already has a reasoner
keeps it, and chat parses with it: the job's `reasoning_parser`,
`structured_outputs_config.reasoning_parser`, or the model's default (gpt-oss
gets `openai_gptoss`). An explicit `chat_reasoning_parser` that differs from it
fails engine start, since there is one per engine.

A job `reasoning_parser` whose think tokens the tokenizer lacks is dropped with
a warning (vLLM fails engine creation otherwise). The table's chat reasoner is
checked against the tokenizer the same way and left out with a warning, so chat
on such a checkpoint streams reasoning as content; an explicit
`chat_reasoning_parser` that fails this check fails engine start.

- Before rendering, any string in messages, tools or a named `tool_choice` that
  contains one of the tokenizer's special or added tokens fails the stream with
  `invalid_message_content` and `param` (for example `messages[1]`). Input the
  template or vLLM rejects fails with `invalid_chat_request`, with `param` when
  vLLM names one. Neither carries message text. A worker that cannot render
  chat at all (for example, its engine has no tokenizer) or a model with no chat
  template fails with `chat_unsupported`, logged once per worker.
- If `max_tokens` is omitted, the budget is `min(context left after the
  rendered prompt, 4096)`. A prompt that leaves no room fails with
  `context_length_exceeded` and `context_limit_source="prompt"`.
- Instead of `delta`, a chat stream emits `content_delta` (`text`),
  `reasoning_delta` (`token_count` only, never the reasoning text) and
  `tool_call_delta` (`index`, `arguments`, plus `id` and `name` on a call's first
  event). Markup the parser is still matching emits nothing until it resolves.
  `parallel_tool_calls=false` keeps only the first call. Tool calls are parsed
  only when the prompt has `tools`; without them, tool-call markup the model
  writes is content (gpt-oss drops a Harmony tool message instead).
- The engine detokenizes as the parsers ask: `skip_special_tokens` and
  `spaces_between_special_tokens` come from the request after the parsers'
  `adjust_request`, as in vllm serve, so tool and reasoning markup made of
  special tokens reaches the parser. A parser failure mid-stream ends it with
  `engine_error` and is logged by type and stack only.
- `choice_finished` reports `tool_calls` when a choice that called a tool stops,
  except under a named `tool_choice`, which reports `stop` as OpenAI does.
  `usage` adds `reasoning_tokens`, counted by the reasoning parser across choices.
  Report that count; the sum of `reasoning_delta` `token_count`s can differ,
  since it leaves out markup the parser consumed without emitting anything.
  gpt-oss is the exception: vLLM's gpt-oss parser does not count, so
  `reasoning_tokens` is that sum, and a delta that ends reasoning counts whole,
  since that parser cannot split one.
- Undelivered events merge only with the same kind of the same choice; tool-call
  arguments merge only within one call.
- `logprobs` are refused when the model will reason: a reasoning parser is
  active and the rendered prompt does not already end reasoning (for example
  Qwen3 with thinking on), or the model is gpt-oss, which always reasons. The
  stream fails before any token with `invalid_chat_request` and
  `param="logprobs"`, because the engine delta that ends reasoning can carry
  reasoning and answer tokens together, and their logprobs would expose the
  reasoning. OpenAI's reasoning models do not take logprobs either. With
  thinking off, every generated token is answer or tool-call text. Without a
  reasoning parser nothing is split out, so a model that reasons anyway streams
  its reasoning as content.
- With `logprobs`, each `content_delta` carries the `token_ids` and `logprobs`
  of the engine output that produced it; tool-call tokens carry none, as
  OpenAI reports logprobs for the answer only. Logprobs follow engine token
  deltas, so when the parser holds text back and releases it later, or splits
  one delta into content and a tool call, they may not line up one-to-one with
  the content text.
- `structured_outputs` cannot be combined with a chat prompt whose tools need a
  grammar of their own (`invalid_chat_request`, `param="structured_outputs"`):
  vLLM applies one grammar per request. Without `max_tokens`,
  `thinking_token_budget` is checked against the budget after rendering.

### Adding a model family

1. Add its architectures to `CHAT_MODELS` in `chat.py`: the reasoning and tool
   parsers vLLM's recipe for it pairs (both must be registered in the pinned
   vLLM; `reasoning_parser=None` for a family that does not reason), whether its template turns thinking off with `enable_thinking`, and a
   `reasoning_efforts` map if its template takes only some levels. A family
   Arctic trains but cannot chat with goes in `TRAINED_WITHOUT_CHAT` in
   `test_chat_models.py`, with the reason.
2. Add it to the tests in `test_chat_models.py` (its parsers, and the levels
   its template takes in `TEMPLATE_EFFORTS`) and run the CPU suite.
3. Run `test_gpu_chat.py` with a checkpoint of the family, then a QA6 run
   through DSS.
4. Release Arctic, then bump DSS's Arctic pin.

## Flow Control and Lifecycle

Native Ray async generators are eager, so the worker waits for an
acknowledgement before it hands over more events. Each round trip delivers a
batch: the next event plus everything already buffered behind it. The reader
acknowledges the batch once, by its last sequence number; ClientStream does this
automatically before fetching the next batch. A reader that keeps up gets
one-event batches, as before. No ObjectRefGenerator crosses an actor boundary.
ClientStream reads still return one event at a time. `read_buffered(limit)`
returns up to `limit` more events from the current batch without a round trip,
for callers that relay events onward in groups.

The engine pump never waits for the client; it drains into a bounded queue and
aborts on overflow rather than silently dropping output or pausing the shared engine.
While the reader is behind, a new delta joins its choice's newest undelivered
delta (text, `token_ids` and `logprobs` concatenated), so one delivered delta
can carry several engine steps. A slow reader then needs one queue slot per
choice instead of one per token. Deltas never merge across choices, past a
later event of the same choice, or past `usage`; finish events never merge, so
the end of a stream needs up to 2n + 2 slots. A merge that would exceed the
per-event byte limit starts a new event instead.

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

Logprobs make each token's payload much larger: with `logprobs=20` a token
serializes to about 1.5 KB, roughly 20 times a one-token delta without them, so
a reader that falls about 700 tokens behind overflows the default buffer.
Callers that request logprobs should raise `max_buffer_bytes` (ceiling 16 MiB,
about 11,000 undelivered tokens at `logprobs=20`).

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
Existing completed-result APIs remain unchanged. Tests are in `tests/inference/streaming/`.

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
