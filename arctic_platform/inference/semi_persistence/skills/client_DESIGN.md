# OrchestratorClient -- Design

`client.OrchestratorClient` is a class-method HTTP client that sits in
front of `orch_server` and presents a job-keyed view of the
orchestrator's single-model registry.  This document explains the
deliberately-overloaded calling shapes that make the API ergonomic for
both scripts and interactive use, and the persistence layer that keeps
those job bindings alive across client restarts.

For the orchestrator's own state machine and wire protocol, see
[`orchestrator_DESIGN.md`](orchestrator_DESIGN.md).

## Two-layer model

```
caller-chosen names                     server-assigned names
-------------------                     ---------------------
   job_id  ──────►  _jobs (in-mem)  ──────►  model_id
   "job 1"           {"job 1": "model_1"}     "model_1"
                                              │
                                              ▼
                                         Orchestrator
                                          (image cache,
                                           state ladder,
                                           slots)
```

- **Job ids** are the caller's vocabulary.  They live only in the
  client process (`_jobs` dict).  A workload uses whatever names it
  finds natural -- `"job 1"`, `"alice's exploration"`, `"baseline"`.
- **Model ids** are the orchestrator's vocabulary.  They are either
  auto-assigned (`model_1`, `model_2`, ...) or rediscovered from the
  server's registry when an existing model already matches the
  requested `vllm_config`.

The client's job is to translate.  Two jobs with the same
`vllm_config` deduplicate to a single model on the server -- this is
how a caller fans out work over a logical "job 1 vs. job 2" while
sharing a single backing model.

## Calling shapes

The per-job API (`generate`, `wait`, `remove`, `pause`, `resume`,
`move`) takes `job_id` as its first positional arg.  Several methods
overload that slot to keep common operations terse.  The convention
is **type-based dispatch**: `str` means a job id, `int` means
something semantically appropriate to that method, `None`/omitted
means "fan out across every registered job".

### `register(job_id, vllm_config: dict | str)`

Two forms, dispatched on the *type* of the second arg:

| Call | Meaning |
|---|---|
| `register("job 1", {...})` | dict is a `vllm_config`; dedup against server registry, otherwise cold-start a new `model_N`. |
| `register("job 1", "model_3")` | str is an explicit existing `model_id`; pure direct association.  Raises if the model isn't on the server. |

Dedup is the headline feature of the dict form: if the orchestrator
already has a model with the same `vllm_config`, no new server
registration is issued and the job binds to the existing model.  Two
jobs registered with the same dict end up sharing a single backing
process.

### `generate(job_id, prompts, sampling_params)`

The most-overloaded surface.  Five distinct call shapes:

| Call | job(s) hit | prompts | sampling |
|---|---|---|---|
| `generate()` | all registered | `"Hello World!"` | `max_tokens=1000` |
| `generate("job 1")` | job 1 | `"Hello World!"` | `max_tokens=1000` |
| `generate("job 1", "foo", 50)` | job 1 | `"foo"` | `max_tokens=50` |
| `generate(N)` | all registered | `"Hello World!"` | `max_tokens=N` |
| `generate(None, "foo", 50)` | all registered | `"foo"` | `max_tokens=50` |

The `int` shorthand (`generate(N)`) is dispatch-on-type: when the
first arg is an `int` it is interpreted as `sampling_params` and
`job_id` is implicitly `None` (fan out).  Mixing the int shorthand
with explicit `prompts` / `sampling_params` arguments raises
`TypeError` -- the alternative would silently override the int the
caller wrote.

`prompts` and `sampling_params` retain the orchestrator's existing
shorthand: a bare string becomes `[string]`, an `int` for
`sampling_params` becomes `{"max_tokens": N, "ignore_eos": True}`.

### `pause(job_id)` and `resume(job_id)`

Three shapes each, symmetric:

| Call | Meaning |
|---|---|
| `pause("job 1")` / `resume("job 1")` | single job. |
| `pause()` / `resume()` | fan out across every registered job. |
| `pause(N)` | `int` shorthand: pause every job whose backing model is currently resident on GPU *N*, **recording the pairing**. |
| `resume(N)` | `int` shorthand: resume every job that a previous `pause(N)` recorded against GPU *N*. |

The two shorthands are tied together by a small piece of client-side
state -- a `_paused_gpu: dict[job_id, gpu]` map -- rather than by
re-querying the server on every call:

```
pause(1)    ── GET /state ──► {model_1, model_3}  on GPU 1
            ── for j1 (model_1), j3 (model_3): /pause + record gpu=1
            _paused_gpu = {"j1": 1, "j3": 1}

resume(1)   (no /state)
            ── for jids with _paused_gpu == 1: /resume + drop record
            _paused_gpu = {}
```

Why client-side state instead of asking the server "what's paused on
GPU N?":

- **Exact undo of the previous pause.**  `resume(N)` resumes
  *exactly the set that `pause(N)` paused* -- not "everything that
  happens to be paused on GPU N right now".  Avoids resuming a job
  that was paused independently (e.g. via `pause("j5")`) or
  re-pausing entries left over from a partial server crash.
- **No extra round-trip.**  `resume(N)` is a pure local lookup,
  followed by N `/resume` posts.  `pause(N)` still needs one
  `GET /state` to learn current GPU placement.
- **Predictability.**  The client state is the source of truth for
  "things I paused via `pause(N)`".  The server's view (gpu +
  paused fields) can drift independently and that's fine.

`pause(N)` only records pairings for jobs it actually pauses (i.e.
those whose backing model has `gpu == N`).  Each `resume(N)` clears
the matching records as it issues the resumes, so back-to-back
`resume(N)` calls are no-ops.  `remove(job_id)` and `remove()` also
drop their `_paused_gpu` entries to keep the map honest.

Single-job `pause("job 1")` does not record anything -- the user
already has the job id, so undoing it is just `resume("job 1")`.

The plain `pause()` / `resume()` no-arg fan-outs also do not consult
or update `_paused_gpu`; they are a flat per-job iteration.

### `wait` and `remove`

The simplest cases.  `job_id` is optional (`None` / omitted = fan
out); no other type-based dispatch.

| Call | Meaning |
|---|---|
| `wait()` | client-side barrier across every registered job. |
| `wait("job 1")` | single job. |
| `remove()` | drop every binding (and rewrite the bound session file as empty, in one write). |
| `remove("job 1")` | drop one binding. |

`remove` is a pure local operation on the jobs directory -- the
orchestrator-side model is never touched.  To delete a model from the
server, talk to the orchestrator's `/remove` endpoint directly.

### `move` -- the explicit fan-out exception

`move` is the only method that keeps a separate `*_all` variant.

```
client.move("job 1", "sleep")      # one job
client.move_all("sleep")           # every job
```

The reason is purely ergonomic: `move(job_id, target)` has *two*
required positional args, so the obvious "`job_id=None` means all"
trick would force the all-form to use a keyword arg
(`move(target="sleep")`), which reads worse than the existing
positional `move_all("sleep")`.  Keeping `move_all` as a one-line
separate method preserves `move("job", "target")` for the common
single-job case.

### Client-only methods

`init`, `jobs`, `requests`, `paused`, `model_of`, `status`, and
`status_raw` don't follow the per-job pattern -- they manage the
connection or report it.

The reporting methods are layered:

| Call | Sections | Use |
|---|---|---|
| `jobs()` | JOBS | "what's bound to what, and where". |
| `requests()` | REQUESTS | "what's in flight on the server". |
| `paused()` | PAUSED + STALE PAUSE RECORDS | "what's paused right now and how to resume each". |
| `status()` | JOBS + REQUESTS + PAUSE RECORDS | combined view, single `GET /state` so the sections agree. |
| `status_raw()` | (raw JSON dump of `/state`) | escape hatch for wire-level debugging. |

`status()` calls one `GET /state` and renders both tables from the
same snapshot; the focused methods each do their own GET when
called standalone.  `paused()` joins the server's `entry["paused"]`
flag with the client's `_paused_gpu` records so each row also tells
you whether the pause is undoable via `resume(N)` (record present)
or only via `resume("job_id")` (no record).  The PAUSE RECORDS
footer (contents of `_paused_gpu`) appears in `status()` and is
omitted when empty.

### Why these specific overloads?

The shapes are not random.  Each one captures a real interactive
workflow:

- `generate(N)` -- "spray a quick benchmark at every job".  The
  prompt content rarely matters; the token count does.
- `pause(N)` -- "I need GPU *N* free for something else".  Think of
  it as the symmetric to `sub(N)` (drain a GPU): you want to quiet
  every resident, not look up which jobs you happen to have on it.
- `register("job", "model_id")` (str second arg) -- "rebind to a
  model that already exists, don't re-create".  Useful after
  `init()` shows pre-existing server-side models.
- The ubiquitous `f()` no-arg form -- the interactive "do this
  everywhere" verb.

Symmetric ops we deliberately did *not* overload:

- `generate("foo")` (string-as-prompt for fan-out) -- ambiguous with
  job ids and never asked for.

When in doubt, the keyword form (`generate(prompts="foo")`,
`generate(sampling_params=42)`) always works and never overloads.

## Session persistence

The job directory lives only in the client process.  The
orchestrator persists *models* (via its image cache), but it knows
nothing about caller-chosen `job_id`s.  Without a session file, a
client restart loses the `job_id -> model_id` map even though every
backing model is still alive on the server.

`init(session=path)` closes that gap by mirroring the in-memory
directory to a JSON file.

```
process A              file (jobs.json)             server
─────────              ────────────────             ──────
init(path)          ─► load (empty)
register("j1", cfg) ─► write {"j1": cfg}        ─► register model_1
register("j2", cfg) ─► write {"j1":cfg,"j2":cfg}   (dedup; share model_1)
remove("j1")        ─► write {"j2": cfg}
                                                    ... (process A dies) ...
process B
─────────
init(path)          ◄──────────────────────────  /state lists model_1
                                                  _model_counter ← 1
                    ◄─ read {"j2": cfg}
  for j2: register("j2", cfg)
                                                ─► dedup hit; bind j2 ─► model_1
generate("j2", ...)                             ─► /generate model_1
```

Three behavioural pieces make this work:

1. **Replay via `register`.**  The session file doesn't store
   `model_id`s; it stores whatever the caller originally passed to
   `register` (a `vllm_config` dict, or an explicit `model_id`
   string).  On reload `init` just calls `register(job_id, spec)`
   for each entry, which goes through the normal dedup path and
   rebinds the job to whichever server-side model now matches.
   This survives server reboots that re-number models, and mixed
   scenarios where some models survive and some don't.
2. **Atomic mirroring.**  Every `register` / `remove` writes the
   updated map via `tmp + os.replace`.  Parent directories are
   created on demand.  The whole-directory operations (`remove()`)
   collapse to a single rewrite, not N.  During `init`'s replay
   the per-`register` saves are no-ops -- the path is published
   only after the loop, then a single consolidated snapshot is
   written, so the file is untouched if any replayed register
   raises mid-way.
3. **Counter advance on `init`.**  The auto-id counter
   (`_model_counter`) is reset by `init`, then bumped past the
   highest existing `model_N` on the server.  Without this step a
   restarted client would re-issue `model_1` for a freshly
   registered job; the server's `register` is idempotent on
   `model_id` (silently no-ops on collision), so the client would
   silently mis-bind to an unrelated existing model.

Calling `init(...)` without `session` (or with a different
`session`) detaches from any previously-bound file -- the new
in-memory directory comes from the new file (if any) or starts
empty.  There's no separate "unbind without re-init" call; if you
want to stop mirroring mid-run, re-`init`.

### Concurrency contract

**One file per client.**  No cross-process locking is performed; two
clients pointing at the same path silently stomp each other.  Files
are intended to belong to a single user / process at a time.  If you
need a shared directory across processes, layer your own locking on
top, or give each process its own session file.

## State translation

The client passes wire-protocol state names through to the server
verbatim.  `move(job_id, "checkpoint")` posts `target=checkpoint`;
`status()` prints whatever state strings the server reports
(`"saved"`, `"checkpoint"`, `"sleep"`, `"up"`, ...).  No client-side
state vocabulary.

## Transport

`urllib.request` + `json` -- no third-party dependencies.  Every call
goes through `_post` or `_get`; `_timeout_s` (default 600 s) applies
uniformly so long-blocking endpoints like `/wait` work out of the
box.  Errors from the server (`HTTPError` body parsed as JSON) are
re-raised as `RuntimeError` with the server's error message
attached.
