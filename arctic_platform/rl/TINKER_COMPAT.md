# Tinker HTTP layer for Arctic-Platform (v1)

Expose Arctic's RL server over the
[Tinker](https://github.com/thinking-machines-lab/tinker) HTTP
protocol so the upstream `tinker` Python client can drive Arctic.
[SkyRL-tx](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl) is a
reference open-source implementation of the same protocol.

Scope: **RL only**, colocated (`colocate=True`, CUDA-IPC weight
sync), single global training run, no auth. **Full-weight DeepSpeed
training via the SkyRL-tx `LoraConfig(rank=0)` = FFT convention** —
`rank>0` returns HTTP 400 in v1.

## Wire protocol summary (pinned against upstream `tinker` SDK)

All routes are `POST /api/v1/<verb>` with a strict Pydantic body,
except `client/config` (some newer SDKs GET it). Every
long-running verb (`forward`, `forward_backward`, `optim_step`,
`save_weights_for_sampler`, `asample`, `create_model`) returns a
**future**:

```json
{"request_id": "42", "status": "pending", "type": "future"}
```

Clients then poll `POST /api/v1/retrieve_future {request_id}`. In v1
we execute inline in the request handler and stash the result in an
in-memory `dict[request_id] -> response`, so retrieve_future returns
the completed result on first call.

## Metric naming

Tinker's `combine_fwd_bwd_output_results` requires every metric key
to encode its cross-actor reduction as `name:reduction` (`:mean`,
`:sum`, `:min`, `:max`, `:slack`, `:hash_unordered`, `:unique`).
Arctic handlers emit plain keys (`loss`, `grad_norm`, `kl`, ...), so
the Tinker router post-processes them via
`arctic_metrics_to_tinker(...)`: it drops non-numeric values and
appends `:mean` to any key that lacks a reduction suffix. `:mean` is
the safe default because it weights by per-actor sample count
(`len(loss_fn_outputs)`).

## Verb mapping

| Tinker route | Upstream body | Arctic handler | Notes |
| :--- | :--- | :--- | :--- |
| `/api/v1/create_session` | `CreateSessionRequest {tags, user_metadata, sdk_version}` | in-memory bookkeeping | returns `{session_id}` |
| `/api/v1/session_heartbeat` | `SessionHeartbeatRequest` | no-op | returns `{}` |
| `/api/v1/client/config` | `ClientConfigRequest {sdk_version}` | static response | force `proto_write_fwdbwd=false` (JSON path only in v1) |
| `/api/v1/auth/token` | `{}` | echo `{jwt: "tml-dummy"}` | no auth in v1 |
| `/api/v1/telemetry` | `TelemetrySendRequest` | no-op ack | drop events on the floor |
| `/api/v1/get_server_capabilities` | `{}` | static | one entry: the server's `base_model` |
| `/api/v1/create_model` | `CreateModelRequest {session_id, model_seq_id, base_model, lora_config, user_metadata}` | full-weight if `lora_config.rank==0`, else 400 | returns `{model_id, base_model, lora_config, status, request_id}` (future) |
| `/api/v1/get_info` | `GetInfoRequest {model_id, type}` | reads in-memory model | returns `{model_id, status, model_data}` |
| `/api/v1/create_sampling_session` | `CreateSamplingSessionRequest {session_id, sampling_session_seq_id, base_model?, model_path?}` | in-memory bookkeeping | returns `{sampling_session_id}` bound to a weight generation |
| `/api/v1/forward` | `ForwardRequest {forward_input, model_id, seq_id?}` | `fwd_bwd(..., forward_only=True)` | returns future; result carries per-token logprobs |
| `/api/v1/forward_backward` | `ForwardBackwardRequest {forward_backward_input, model_id, seq_id?}` | `fwd_bwd(...)` with `loss_fn_config` threaded into `actor_config` | returns future; result = `ForwardBackwardOutput` |
| `/api/v1/optim_step` | `OptimStepRequest {adam_params, model_id, seq_id?}` | `step(optim_overrides=...)` | returns future; result = `OptimStepResponse{metrics}` |
| `/api/v1/save_weights_for_sampler` | `SaveWeightsForSamplerRequest {model_id, path?, sampling_session_seq_id?, seq_id?, ttl_seconds?}` | `sync_weights(cuda_ipc=True)` + bump weight-gen | returns future; result = `SaveWeightsForSamplerResponse{path, sampling_session_id}` |
| `/api/v1/asample` | `SampleRequest {prompt, sampling_params, num_samples, base_model?, model_path?, sampling_session_id?, seq_id?, prompt_logprobs?, topk_prompt_logprobs?}` | `generate(...)` | returns future; result = `SampleResponse{sequences}` |
| `/api/v1/retrieve_future` | `FutureRetrieveRequest {request_id, allow_metadata_only}` | pop from in-memory store | polymorphic body: `TryAgainResponse` if not ready, else the terminal response |

Not implemented in v1 (return 501): `load_weights`, `save_weights`,
`unload_model`, `weights_info`, `training_runs/*`. All are E1
extensions.

## Loss functions (v1)

`ForwardBackwardInput.loss_fn` is a
`Literal["cross_entropy","importance_sampling","ppo","cispo","dro"]`.
`loss_fn_config: Optional[Dict[str, float]]` is a free-form dict; the
built-in Tinker loss functions read specific keys from it.

| `loss_fn` | v1 status | Maps to |
| :--- | :--- | :--- |
| `ppo` | supported | Arctic PPO; `loss_fn_config` keys → Arctic `actor_config` clip fields (Q2/Q3) |
| `importance_sampling` | supported | Arctic PPO with both clip bounds disabled (unbounded IS ratio) |
| `cispo`, `dro` | HTTP 400 | not in Arctic — separate PR |
| `cross_entropy` | HTTP 400 | RL-only v1; SFT/CE out of scope. Blocks `forward_backward_custom`. |
| `forward_backward_custom` | — | Client-side SDK sugar; not a wire verb. Unblocks automatically once `cross_entropy` + `forward_only` land (E3). |

## Architecture

```
                   ┌──────────────────┐
   Tinker client   │                  │   Native client (verl, dev scripts)
   (HTTP + tinker  │                  │   (HTTP or Ray)
    Python SDK)    │                  │
        │          │                  │           │
        ▼          │                  │           ▼
  POST /api/v1/*   │                  │  POST /fwd-bwd, /step, ...
        │                                          │
        ▼                                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │                FastAPI app (one process)                │
  │  ┌────────────────────────┐  ┌─────────────────────────┐│
  │  │  tinker_server.py      │  │  http_server.py         ││
  │  │  (NEW router)          │──▶  (native routes, unchg) ││
  │  │  · Datum → batch       │  │  · /fwd-bwd             ││
  │  │  · AdamParams → overr. │  │  · /step (+ overrides)  ││
  │  │  · SamplingParams → vL │  │  · /generate            ││
  │  │  · weight-gen counter  │  │  · /sync-weights        ││
  │  │  · future store        │  │                         ││
  │  └────────────────────────┘  └───────────┬─────────────┘│
  └──────────────────────────────────────────┼──────────────┘
                                             ▼
                            ┌────────────────────────────────┐
                            │  Ray placement group (colocated)│
                            │   ├─ DeepSpeed training workers │
                            │   ├─ vLLM engine (CUDA IPC ⇄ DS)│
                            │   └─ log-prob workers           │
                            └────────────────────────────────┘
```

Tinker router lowers into existing Arctic HTTP handlers via in-process
function calls, not a second HTTP hop.

## Futures model

Tinker's wire is future-based. v1 executes inline and stashes results
so `retrieve_future` returns the finished result on the first poll:

```python
# tinker_server.py
_FUTURE_STORE: dict[str, dict] = {}
_FUTURE_COUNTER = itertools.count()

def _new_request_id() -> str:
    return str(next(_FUTURE_COUNTER))

async def _submit(runner: Callable[[], Awaitable[dict]]) -> UntypedAPIFuture:
    request_id = _new_request_id()
    result = await runner()  # inline for v1; async task in E-async
    _FUTURE_STORE[request_id] = result
    return {"request_id": request_id, "status": "completed", "type": "future"}
```

`E-async` (below) upgrades `_submit` to `asyncio.create_task` +
`TryAgainResponse` polling.

## Sync vs async RL semantics

Tinker's `SamplingClient` is immutable and picklable, bound to a
`sampling_session_id` that identifies a weight snapshot. Cookbook
async-RL loops mint a new snapshot per training step and pass the
client to headless rollout workers; `importance_sampling` handles
off-policy correction between the sampler's stale weights and the
in-flight training policy.

Arctic's colocated CUDA-IPC path holds **one** weight version on the
sampler at a time — no on-GPU snapshot store. v1 adopts
**strict-monotonic** semantics:

```
sampling_session_id = f"ss@{gen}"    # gen = monotonic weight-sync counter
POST /api/v1/asample:
    if int(gen) <  app.state.tinker_weight_gen:  → TryAgainResponse (client will retry, then hit 409 on retrieve_future)
    if int(gen) == app.state.tinker_weight_gen:  → serve from current vLLM weights
```

This matches Tinker's synchronous RL recipes (train → save → sample →
train). True async-RL requires concurrent LoRA snapshots on vLLM and
is captured as **E1** below.

## RL step data flow

```
Client                Tinker route              Arctic handler       Compute
──────                ────────────              ──────────────       ───────
create_session       ─▶ /api/v1/create_session  ─▶ in-memory       ─▶ —
create_model         ─▶ /api/v1/create_model    ─▶ in-memory        ─▶ —
   (base_model,        (rank==0 → FFT)             (model_id)
    LoraConfig(0))

sample(prompt, n=16) ─▶ /api/v1/asample        ─▶ /generate         ─▶ vLLM
retrieve_future      ─▶                        ◀──  {results}       ◀── (rollouts,
                                                                          logprobs)

  [client computes rewards + advantages]

forward_backward     ─▶ /api/v1/forward_backward ─▶ /fwd-bwd        ─▶ DeepSpeed
  (data, "ppo",         (datum → batch,             ({"loss_fn":         (existing PPO
   loss_fn_config)       loss_fn_config              "ppo", ...},         path; clip
                         → actor_config)             actor_config          ratios from
                                                     updated)              loss_fn_config)
retrieve_future     ◀── {loss_fn_outputs, metrics}                   ◀──

optim_step           ─▶ /api/v1/optim_step    ─▶ /step             ─▶ DeepSpeed
  (AdamParams(lr=…))   (adam → overrides)      (overrides applied      optimizer
                                                to param_groups)
retrieve_future     ◀── {metrics}                                    ◀──

save_weights_for_    ─▶ /api/v1/save_weights_ ─▶ /sync-weights     ─▶ CUDA IPC
  sampler               for_sampler             (cuda_ipc=True,       (DS → vLLM)
                        (bump weight_gen)        colocate=True)
retrieve_future     ◀── {path, sampling_session_id}                  ◀──

[loop]

──────────────────────────────────────────────────────────────────────────
Per step: 4 wire calls (asample, forward_backward, optim_step,
save_weights_for_sampler) + their retrieve_future polls. Wire cost is
one HTTP round-trip more per verb than Arctic native today; those
polls are ~free in v1 (inline completion) and become real in E-async.
```

## Concrete changes

Six files.

### 1. `arctic_platform/rl/deepspeed_worker.py` (modify, ~20 LoC)

Add optional per-call optimizer overrides to `step()`. Legacy path
(no overrides) is unchanged.

```python
def step(self, optim_overrides: dict | None = None) -> dict:
    if optim_overrides:
        for pg in self.engine.optimizer.param_groups:
            for k in ("lr", "betas", "eps", "weight_decay"):
                if k in optim_overrides:
                    pg[k] = optim_overrides[k]
    self.engine.step()
    grad_norm = self.engine.get_global_grad_norm()
    ...
```

### 2. `arctic_platform/rl/http_server.py::/step` (modify, ~5 LoC)

```python
class StepRequest(BaseModel):
    optim_overrides: dict[str, Any] | None = None

@app.post("/step")
async def step(job_id: int, request: StepRequest = Body(default_factory=StepRequest)):
    _verify_job(job_id, "training")
    results = await asyncio.gather(
        *[w.step.remote(request.optim_overrides) for w in app.state.training_workers]
    )
    ...
```

### 3. `arctic_platform/rl/ray_server.py::step` (modify, ~5 LoC)

Same passthrough for the Ray transport.

### 4. `arctic_platform/rl/tinker_server.py` (new, ~500 LoC)

Single new file. Pydantic models mirroring the upstream `tinker`
SDK's wire schema, adapters inline, and a FastAPI `APIRouter`.

Layout:

```python
# --- Pydantic wire models (redefined locally, not re-exported from tinker) ---
# We redefine each request/response as a Pydantic BaseModel so we don't force
# the tinker package as a server-side dependency. Schemas track upstream
# exactly (regressed by tests/tinker_layer/test_wire_schema.py).

class TensorData(BaseModel):
    data: list[float] | list[int]
    dtype: Literal["float32", "int64"]
    shape: list[int] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None

class EncodedTextChunk(BaseModel):
    type: Literal["encoded_text"] = "encoded_text"
    tokens: list[int]

class ModelInput(BaseModel):
    chunks: list[EncodedTextChunk]   # v1: text only

class Datum(BaseModel):
    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData]

class ForwardBackwardInput(BaseModel):
    data: list[Datum]
    loss_fn: Literal["ppo", "importance_sampling", "cispo", "dro", "cross_entropy"]
    loss_fn_config: dict[str, float] | None = None

class ForwardBackwardRequest(BaseModel):
    forward_backward_input: ForwardBackwardInput
    model_id: str
    seq_id: int | None = None

class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    loss_fn_outputs: list[dict[str, TensorData]]
    metrics: dict[str, float] = {}

class AdamParams(BaseModel):
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

class OptimStepRequest(BaseModel):
    adam_params: AdamParams
    model_id: str
    seq_id: int | None = None

class OptimStepResponse(BaseModel):
    metrics: dict[str, float] | None = None

class LoraConfig(BaseModel):
    rank: int
    seed: int | None = None
    train_unembed: bool = True
    train_mlp: bool = True
    train_attn: bool = True

class CreateModelRequest(BaseModel):
    session_id: str
    model_seq_id: int
    base_model: str
    user_metadata: dict[str, Any] | None = None
    lora_config: LoraConfig | None = None

class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str | None = None
    sampling_session_seq_id: int | None = None
    seq_id: int | None = None
    ttl_seconds: int | None = None

class SaveWeightsForSamplerResponse(BaseModel):
    path: str
    sampling_session_id: str | None = None

class SamplingParams(BaseModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: str | Sequence[str] | Sequence[int] | None = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0

class SampleRequest(BaseModel):
    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int = 1
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0

class StopReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"

class SampledSequence(BaseModel):
    tokens: list[int]
    logprobs: list[float] | None = None
    stop_reason: StopReason

class SampleResponse(BaseModel):
    sequences: list[SampledSequence]
    prompt_logprobs: list[float] | None = None

class UntypedAPIFuture(BaseModel):
    request_id: str
    model_id: str | None = None
    type: Literal["future"] = "future"

class TryAgainResponse(BaseModel):
    type: Literal["try_again"] = "try_again"

# ... plus small models for create_session / heartbeat / client_config /
#     auth_token / telemetry / get_info / get_server_capabilities.
```

#### Adapters (inline)

```python
def datum_list_to_arctic_batch(
    data: list[Datum],
    loss_fn: str,
    loss_fn_config: dict[str, float] | None,
    max_prompt_length: int,      # CONFIG max, not batch-local (ZoRRo invariant)
    max_response_length: int,
    pad_token_id: int,
    forward_only: bool = False,
) -> dict:
    """Pack Datum list into Arctic's fwd_bwd batch dict.

    ModelInput → concatenated tokens; loss_fn_inputs["target_tokens"|"weights"|
    "advantages"|"logprobs"|"clip_low_threshold"|"clip_high_threshold"] →
    Arctic batch fields.

    Returns { "args": ..., "kwargs": {"input_ids", "attention_mask",
    "position_ids"}, "context": {advantages, old_log_probs, prompt_mask, ...},
    "meta_data": {actor_config, "forward_only": bool},
    "processing": {"loss_fn": <mapped>} }.

    ZoRRo invariant: pad to (max_prompt_length + max_response_length), never
    batch-local max.
    """

def adam_params_to_optim_overrides(p: AdamParams) -> dict:
    return {"lr": p.learning_rate, "betas": (p.beta1, p.beta2),
            "eps": p.eps, "weight_decay": p.weight_decay}

def sampling_params_tinker_to_vllm(p: SamplingParams, num_samples: int) -> dict:
    return {"n": num_samples, "temperature": p.temperature, "top_p": p.top_p,
            "top_k": p.top_k, "max_tokens": p.max_tokens, "logprobs": 1,
            "stop": p.stop, "seed": p.seed}
```

#### Routes (sketch)

```python
router = APIRouter(prefix="/api/v1")

@router.post("/create_session")
async def create_session(req: CreateSessionRequest):
    sid = f"sess-{uuid.uuid4().hex[:8]}"
    app.state.tinker_sessions[sid] = {"created_at": time.time()}
    return {"session_id": sid, "type": "create_session"}

@router.post("/session_heartbeat")
async def session_heartbeat(req: SessionHeartbeatRequest):
    return {}

@router.post("/client/config")
async def client_config(req: ClientConfigRequest):
    return {"proto_write_fwdbwd": False, "proto_compress_fwdbwd": False,
            "fwd_via_fwdbwd": False, **_static_client_flags()}

@router.post("/auth/token")
async def auth_token():
    return {"jwt": "tml-dummy"}

@router.post("/telemetry")
async def telemetry(req: dict):
    return {"received": True}

@router.post("/get_server_capabilities")
async def get_server_capabilities():
    return {"supported_models": [{"model_name": app.state.base_model}]}

_V1_SUPPORTED_LOSSES = {"ppo", "importance_sampling"}
_V1_UNSUPPORTED_LOSSES = {"cispo", "dro", "cross_entropy"}

@router.post("/create_model")
async def create_model(req: CreateModelRequest):
    if req.lora_config is not None and req.lora_config.rank != 0:
        raise HTTPException(400,
            "Arctic v1 supports only full-weight training; pass "
            "LoraConfig(rank=0) to opt into the SkyRL-tx FFT convention.")
    if req.base_model != app.state.base_model:
        raise HTTPException(400, f"server started with base_model={app.state.base_model!r}")
    model_id = "main"
    app.state.tinker_models[model_id] = {"base_model": req.base_model,
                                         "lora_config": req.lora_config}
    return await _submit_future(lambda: {
        "model_id": model_id, "base_model": req.base_model,
        "lora_config": req.lora_config, "status": "created",
        "type": "create_model",
    })

@router.post("/forward_backward")
async def forward_backward(req: ForwardBackwardRequest):
    fbi = req.forward_backward_input
    if fbi.loss_fn in _V1_UNSUPPORTED_LOSSES:
        raise HTTPException(400,
            f"loss_fn={fbi.loss_fn!r} not supported in v1; supported: {sorted(_V1_SUPPORTED_LOSSES)}")
    if fbi.loss_fn not in _V1_SUPPORTED_LOSSES:
        raise HTTPException(400, f"unknown loss_fn={fbi.loss_fn!r}")
    batch = datum_list_to_arctic_batch(
        fbi.data, fbi.loss_fn, fbi.loss_fn_config,
        app.state.max_prompt_length, app.state.max_response_length,
        app.state.pad_token_id, forward_only=False)
    async def runner():
        r = await _fwd_bwd_handler(app.state.training_job_id, batch)
        return _pack_fwd_bwd_output(r).model_dump(mode="json")
    return await _submit_future(runner, model_id=req.model_id)

@router.post("/forward")
async def forward(req: ForwardRequest):
    fi = req.forward_input
    batch = datum_list_to_arctic_batch(
        fi.data, "ppo", None,
        app.state.max_prompt_length, app.state.max_response_length,
        app.state.pad_token_id, forward_only=True)
    async def runner():
        r = await _fwd_bwd_handler(app.state.training_job_id, batch)
        return _pack_fwd_bwd_output(r).model_dump(mode="json")
    return await _submit_future(runner, model_id=req.model_id)

@router.post("/optim_step")
async def optim_step(req: OptimStepRequest):
    overrides = adam_params_to_optim_overrides(req.adam_params)
    async def runner():
        r = await _step_handler(app.state.training_job_id,
                                StepRequest(optim_overrides=overrides))
        return OptimStepResponse(metrics=r.get("metrics", {})).model_dump(mode="json")
    return await _submit_future(runner, model_id=req.model_id)

@router.post("/save_weights_for_sampler")
async def save_weights_for_sampler(req: SaveWeightsForSamplerRequest):
    async def runner():
        app.state.tinker_weight_gen += 1
        gen = app.state.tinker_weight_gen
        await _sync_weights_handler(SyncWeightsRequest(
            training_job_id=app.state.training_job_id,
            sampling_job_id=app.state.sampling_job_id,
            colocate=True, cuda_ipc=True))
        return {"path": f"tinker://main/sampler_weights/{gen}",
                "sampling_session_id": f"ss@{gen}",
                "type": "save_weights_for_sampler"}
    return await _submit_future(runner, model_id=req.model_id)

@router.post("/create_sampling_session")
async def create_sampling_session(req: CreateSamplingSessionRequest):
    gen = app.state.tinker_weight_gen
    return {"sampling_session_id": f"ss@{gen}",
            "type": "create_sampling_session"}

@router.post("/asample")
async def asample(req: SampleRequest):
    ss_id = req.sampling_session_id
    gen = None
    if ss_id and ss_id.startswith("ss@"):
        gen = int(ss_id.split("@", 1)[1])
    async def runner():
        if gen is not None and gen < app.state.tinker_weight_gen:
            raise HTTPException(409, f"stale sampling_session_id: {ss_id}")
        vllm_params = sampling_params_tinker_to_vllm(req.sampling_params, req.num_samples)
        r = await _generate_handler(
            app.state.sampling_job_id,
            GenerateRequest(prompts=[_decode(req.prompt)], sampling_params=vllm_params))
        return _pack_sample_response(r).model_dump(mode="json")
    return await _submit_future(runner)

@router.post("/retrieve_future")
async def retrieve_future(req: FutureRetrieveRequest):
    result = app.state.tinker_futures.pop(req.request_id, None)
    if result is None:
        return {"type": "try_again"}
    return result

@router.post("/get_info")
async def get_info(req: GetInfoRequest):
    m = app.state.tinker_models.get(req.model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    return {"model_id": req.model_id, "status": "created",
            "model_data": {"base_model": m["base_model"],
                           "lora_config": m.get("lora_config"),
                           "model_name": m["base_model"]}}
```

### 5. `arctic_platform/rl/http_server.py` (modify, ~5 LoC)

Mount the router and initialize per-app state.

```python
from arctic_platform.rl.tinker_server import router as tinker_router
app.include_router(tinker_router)
app.state.tinker_weight_gen = 0
app.state.tinker_sessions = {}
app.state.tinker_models = {}
app.state.tinker_futures = {}
# app.state.{base_model, max_prompt_length, max_response_length, pad_token_id,
#           training_job_id, sampling_job_id} already set at startup
```

### 6. `pyproject.toml` (modify)

```toml
[project.optional-dependencies]
tinker = ["tinker>=X.Y.Z"]  # only needed to run the E2E conformance test;
                             # server runtime has no dependency on the SDK
```

## Tests

```
tests/tinker_layer/
  test_adapters.py         # datum→batch shape + ZoRRo invariant;
                           # adam/sp round-trip; loss_fn_config →
                           # actor_config; forward_only=True path
  test_wire_schema.py      # our Pydantic models parse valid upstream
                           # bodies and reject malformed ones
  test_tinker_server.py    # httpx.AsyncClient in-process round-trip
                           # for every route; mocked Arctic backend;
                           # loss_fn allow/deny; rank!=0 → 400;
                           # stale sampling_session_id → 409;
                           # retrieve_future try_again → complete;
                           # future_store cleanup
  test_rl_loop.py          # end-to-end 10-step RL loop with a mocked
                           # Arctic backend + real upstream
                           # tinker.ServiceClient over httpx: proves
                           # wire compatibility.
```

`test_rl_loop.py` is the acceptance test — points the upstream
`tinker.ServiceClient` at our local `httpx.AsyncClient`, runs a
minimal 10-step RL loop against a mocked backend that returns
deterministic tokens/logprobs, asserts that every wire verb is hit,
and that the sampling client's `sample()` future resolves correctly.

## Task ordering

```
    ┌──────────────────────────────────┐
    │ Task 0: schema spike (DONE inline)│
    │  · route prefix /api/v1/          │
    │  · futures + retrieve_future      │
    │  · rank==0 = FFT (SkyRL-tx)       │
    │  · loss_fn_config keys per SDK    │
    └──────────────┬───────────────────┘
                   │
    ┌──────────────▼──────────────┐   ┌────────────────────────────────┐
    │ Task 1: optim overrides     │   │ Task 2: forward_only=True path │
    │  · files 1, 2, 3            │   │  · Arctic fwd_bwd skip .backward│
    │                             │   │  · return logprobs in outputs   │
    └──────────────┬──────────────┘   └────────────────┬───────────────┘
                   │                                   │
    ┌──────────────▼──────────────┐   ┌────────────────▼───────────────┐
    │ Task 3: datum→batch adapter │   │ Task 4: sampling adapters      │
    │  · loss_fn_config passthrough│  │  · SamplingParams / ModelInput │
    │  · ZoRRo-safe padding maxes  │  │                                │
    └──────────────┬──────────────┘   └────────────────┬───────────────┘
                   │                                   │
                   └────────────┬──────────────────────┘
                                ▼
             ┌─────────────────────────────────┐
             │ Task 5: FastAPI router          │
             │  · Pydantic wire models         │
             │  · in-memory future store       │
             │  · weight-gen counter + 409     │
             │  · loss allow/deny gate         │
             │  · LoraConfig(rank!=0) 400      │
             └────────────────┬────────────────┘
                              ▼
             ┌─────────────────────────────────┐
             │ Task 6: conformance test        │
             │  · upstream tinker.ServiceClient│
             │  · 10-step RL loop w/ PPO       │
             └─────────────────────────────────┘
```

## Open questions (schema spike — mostly resolved above)

Items below were resolved by reading the upstream `tinker` SDK
source (`/api/v1/*` routes, `types/*.py`) plus SkyRL-tx
(`skyrl/tinker/api.py`). Remaining Q2/Q3 are Arctic-side field
names.

**Q1 (resolved). `create_lora_training_client` vs full-weight
Arctic.** Adopt SkyRL-tx's FFT convention: `LoraConfig(rank=0)` = full
fine-tuning, `rank>0` = HTTP 400 in v1. Precedent:
`skyrl/train/config/config.py::145`
(`assert self.lora.rank == 0` gates full-weight sync) and
`skyrl/backends/skyrl_train_backend.py::452`
(`"FFT (rank=0) keeps the original single-tenant gate"`).

**Q2. `loss_fn_config` key names.** Docs say "See the loss function
source." Best guess from upstream signatures: `epsilon_low`,
`epsilon_high`, `kl_coef`, `entropy_coef`. Pin during Task 3
implementation by reading `tinker.public.loss_fns` module.

**Q3. Arctic PPO clip field names.** Symmetric spike on Arctic's
side — confirm `actor_config` uses `clip_ratio_low` / `clip_ratio_high`
(or single `clip_ratio` + separate bound) before wiring adapter.

**Q4 (resolved). Auth.** The SDK creates `ServiceClient()` without
requiring env vars; it hits `/api/v1/auth/token` and expects
`{jwt: <string>}` back. v1 returns `{jwt: "tml-dummy"}` unconditionally.

**Q5 (resolved). `SamplingParams` / `SampleResponse` wire shape.**
Pinned in `tinker.types._pydantic_types.sampling_params` and
`tinker.types.sample_response`. `SampleResponse` has `sequences: list[SampledSequence]`;
each `SampledSequence` has `tokens`, `logprobs`, `stop_reason`.

**Q6 (resolved). `ModelInput` shape.** `chunks: list[ModelInputChunk]`
where each chunk is `EncodedTextChunk{type:"encoded_text", tokens: list[int]}`
(v1 supports text only — images/asset-pointers/dmel → 400).

## Extensions (post-v1)

### E-async. Real async future execution

v1 executes inline. E-async wraps `_submit_future` with
`asyncio.create_task` and `TryAgainResponse` polling from
`retrieve_future`. This lets the client pipeline forward_backward +
save_weights_for_sampler + asample without serializing behind each
inline call.

### E1. Async-RL snapshot store

Enables Tinker's documented async-RL pattern: multiple live
`SamplingClient` snapshots serving in parallel with the training loop.

Requires:
- **Snapshot storage** — LoRA adapters on disk / blob, or full-weight
  `DeepSpeed.save_checkpoint` per snapshot.
- **vLLM adapter reload** — sampler picks the adapter/weights tagged
  by `sampling_session_id` per request (via vLLM's `LoRARequest`).
- **Off-policy correction** — client uses `importance_sampling` /
  `ppo` with `loss_fn_inputs.logprobs` set to the sampler's logprobs
  at rollout time (already the RL Datum shape).

v1 rejects async use with HTTP 409 on stale `sampling_session_id`;
E1 flips the semantics from strict-monotonic to versioned-snapshot.

### E2. Server-side rollout retention

`asample` returns a `rollout_id` and caches tokens/logprobs
server-side. `forward_backward(rollout_id, advantages_only=…)` accepts
just the id + advantages. Non-standard extension via optional field.

Wire savings: 1 rollout crosses the wire per step instead of 2. Big
on 8B+ scale where rollouts are 10-50 MB each way.

### E3. SFT + `forward_backward_custom`

Wire up `cross_entropy` as a named loss (plus `forward_only`
end-to-end). Once both land, the upstream `tinker` SDK's
client-side `forward_backward_custom` works against Arctic with no
new server endpoint.

### E4. Non-colocated snapshot backend

NCCL broadcast or blob store for cross-node weight sync. Config knob:
`snapshot_backend=cuda_ipc | nccl | blob`.

### E5. Auth

Real bearer-token middleware; drop the `tml-dummy` special-case.

### E6. Fused RL-step endpoint

`POST /api/v1/rl_step {prompts, reward_fn, adam_params}` — one
round-trip per step. Breaks pure Tinker shape but cuts most of the
wire hops.

## Files touched at v1 completion

```
arctic_platform/rl/
  TINKER_COMPAT.md         # this doc
  __init__.py              # modified: lazy imports so tinker_server can be
                           #           imported without pulling in torch/deepspeed
  deepspeed_worker.py      # modified: step() gains optim_overrides
  http_server.py           # modified: /step accepts overrides; mount router
  ray_server.py            # modified: step() gains overrides
  utils/server_models.py   # modified: StepRequest pydantic model (optim_overrides)
  tinker_server.py         # new: router + adapters + wire models + future store
tests/tinker_layer/        # renamed from tests/tinker/ to avoid shadowing
                           # the upstream ``tinker`` package on import path
  __init__.py              # new
  conftest.py              # new: mock_backend + app/client fixtures
  pytest.ini               # new: asyncio auto mode
  test_adapters.py         # new: 22 tests
  test_wire_schema.py      # new: 21 tests
  test_tinker_server.py    # new: 27 tests (in-process httpx round-trip)
  test_rl_loop.py          # new: 5 tests (real ``tinker.ServiceClient``
                           #      against uvicorn-hosted app)
```

## Test results

Environment: `/data-fast/karthik/conda_envs/tinker_dev` (Python 3.12.13,
tinker 0.x, pytest 9.1.1, pytest-asyncio 1.4.0, fastapi + httpx + uvicorn,
no torch / no vLLM / no DeepSpeed). Runs on any CPU-only box.

```
$ python -m pytest tests/tinker_layer/ -v
============================= 75 passed in 2.03s ==============================
```

Per-module breakdown:

| Module | Tests | Coverage |
| :--- | :---: | :--- |
| `test_wire_schema.py` | 21 | request/response pydantic models parse valid upstream bodies and reject malformed ones; `TensorData` sparse encoding round-trips; `ModelInput` rejects non-`encoded_text` chunks |
| `test_adapters.py` | 22 | `datum_list_to_arctic_batch` shape + ZoRRo padding invariant, `attention_mask`, `loss_mask`, `weights`, `advantages`, `logprobs` propagation; `forward_only` flag; `loss_fn_config` → `actor_config` mapping (`ppo` clip low/high, `importance_sampling` disables clip, `kl_coef`, `entropy_coef`); `AdamParams`/`SamplingParams` round-trips (`stop` as str / list[str] / list[int]) |
| `test_tinker_server.py` | 27 | one test per HTTP verb via `httpx.AsyncClient` in-process: bootstrap (`create_session`, `session_heartbeat`, `client/config`, `auth/token`, `telemetry`, `get_server_capabilities`); model lifecycle (`create_model` FFT accepted, `rank>0` → 400, wrong `base_model` → 400, `get_info` happy + 404); training (`forward_backward` happy + `importance_sampling` + unsupported-loss 400 matrix, `forward` returns logprobs, `optim_step` threads AdamParams into `step(optim_overrides=...)`); weight sync (`save_weights_for_sampler` bumps gen, `create_sampling_session` reflects current gen, `asample` current-gen serves + stale 409 + no-session serves); futures (unknown request_id → try_again, pop semantics); misconfig (unwired app → 500) |
| `test_rl_loop.py` | 5 | acceptance test using the real upstream `tinker.ServiceClient` against a `uvicorn`-hosted app: `ServiceClient` bootstrap; `create_lora_training_client(rank=0)` succeeds; `rank>0` raises; a single RL step (sample → forward_backward → optim_step); a full 10-step RL loop (`save_weights_and_get_sampling_client` → sample → forward_backward → optim_step) — verifies wire compatibility with the upstream SDK end-to-end |

Notes captured while landing this suite (all resolved in code, called out
here so a reviewer can spot-check the interop assumptions):

1. **Metric names carry `:reduction` suffix.** Tinker's
   `combine_fwd_bwd_output_results` at
   `tinker/lib/chunked_fwdbwd_helpers.py` splits every metric key on `:`
   to look up the reduction function. Plain `"loss"` from Arctic must be
   annotated as `"loss:mean"`. See `arctic_metrics_to_tinker` in
   `tinker_server.py`.
2. **`ForwardBackwardOutput.loss_fn_outputs` doubles as the reduction
   weight.** The same helper uses `len(loss_fn_outputs)` as the
   per-actor weight for `:mean` and `:slack`. If the list is empty
   `np.average` raises `ZeroDivisionError`. Arctic returns one aggregated
   batch, so the adapter emits one empty `LossFnOutput` per input
   `Datum` to keep weights well-defined.
3. **`TelemetryResponse` expects `status: Literal["accepted"]`.** Not
   `received: bool`. Pinned via the SDK's `TelemetryResponse` pydantic
   model.
4. **Tests directory renamed `tests/tinker/` → `tests/tinker_layer/`.**
   Otherwise the local package shadows the upstream `tinker` SDK on the
   `sys.path` in `test_rl_loop.py` and `from tinker import types` fails
   with `ImportError: cannot import name 'types' from 'tinker'`.
5. **`arctic_platform.rl.__init__` uses PEP 562 lazy imports.**
   `tinker_server` must be importable in a CPU-only conda env for tests;
   eager imports of `.deepspeed_worker` etc. pull in torch and break
   collection.
