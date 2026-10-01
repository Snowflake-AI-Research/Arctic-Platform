"""HTTP client with a job directory layered over a remote orchestrator.

Workloads register *jobs* (logical names they choose) instead of model
ids.  Internally the client maps each job to a model on the
orchestrator, deduplicating on ``vllm_config`` so two jobs with the
same config share one underlying model::

    from client import OrchestratorClient as client

    client.init()  # localhost:8157; lists available models
    cfg = {"model": "Qwen/Qwen3-32B", "gpu_memory_utilization": 0.8}
    client.register("job 1", cfg)         # auto-registers model_1
    client.register("job 2", cfg)         # same vllm_config -> binds to model_1
    client.register("job 3", "model_1")   # explicit bind to existing model_1
    client.wait()                         # no job_id -> barrier across all
    client.move("job 1", "sleep", target_gpu=0)
    client.generate("job 1", "What is gravity?", 100)
    client.wait()
    client.remove("job 1")   # pure unbind; model stays alive on server
    client.remove()          # drop the rest; orchestrator still has model_1

The job directory is process-local by default.  Pass a session file
to :meth:`init` to mirror it to disk so the directory survives
client restarts (the orchestrator already persists models by image
cache, so on reconnect each job rebinds via vllm_config dedup)::

    client.init("/data-fast/jobs/alice.json")
    client.register("job 1", cfg)         # mirrored to alice.json
    # ... process dies ...
    client.init("/data-fast/jobs/alice.json")  # job 1 replayed + rebound
    client.generate("job 1", "...", 100)

The server is :mod:`orch_server`; the wire protocol is the JSON HTTP API
exposed by :class:`state_server.StateHandler`.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from abstract import OrchestratorClientBase

_UNSET = object()


class OrchestratorClient(OrchestratorClientBase):
    """Class-method API that talks to a remote orchestrator over HTTP.

    Layers a *jobs directory* (job_id -> model_id) over the orchestrator's
    single-model API.  Job ids are caller-chosen; model ids are either
    auto-assigned (``model_{counter}``) or reused from the orchestrator
    registry when an existing model already matches the requested
    ``vllm_config``.  Uses only the standard library (``urllib.request``
    + ``json``) -- no third-party dependencies.
    """

    _base: str = ""
    _timeout_s: float = 600.0
    _jobs: dict[str, str] = {}
    _model_counter: int = 0
    _session_path: str | None = None
    _session_specs: dict[str, dict | str] = {}
    # Records the GPU id passed to ``pause(N)`` for each job it paused.
    # Populated by the int-shorthand of :meth:`pause`, consumed (and
    # cleared per-entry) by the int-shorthand of :meth:`resume`.
    _paused_gpu: dict[str, int] = {}

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    DEFAULT_BASE_URL = "http://localhost:8157"

    @classmethod
    def init(cls, session: str | None = None, *,
             base_url: str = DEFAULT_BASE_URL,
             timeout_s: float = 600.0) -> None:
        """Bind to a server, optionally bind a session file, list models.

        Single entry point for an interactive session::

            from client import OrchestratorClient as client
            client.init()                              # localhost, no session
            client.init("alice.jsonl")                 # bind session, default URL
            client.init(base_url="http://other:9000")  # explicit URL, no session
            client.init("alice.jsonl",                 # both
                        base_url="http://other:9000")

        Behaviour:

        1. Binds to *base_url* (default ``http://localhost:8157`` --
           matches :mod:`orch_server`'s default port);
           *timeout_s* applies to every subsequent
           HTTP call, including long-blocking ones like :meth:`wait`.
           Bump it if your server runs huge cold-starts under
           contention.
        2. Resets the local jobs directory.  Existing server-side
           models are *not* touched -- subsequent :meth:`register`
           calls will rediscover and reuse them by ``vllm_config``
           match (or by explicit ``model_id``).
        3. Sanity-pings ``GET /state`` and prints a one-line-per-model
           summary of every registered model, with shape::

               {
                   "state": "saved" | "checkpoint" | "sleep" | "up" | ...,
                   "vllm_config": {...},
               }

           Every entry corresponds to a model with an image on disk in
           the orchestrator's image cache and is a valid target for
           direct association via ``register(job_id, model_id)``.
        4. If *session* is provided, the JSON file at that path is
           the on-disk mirror of the in-memory job directory.  Any
           existing entries are replayed via :meth:`register` (so
           dedup against current server-side models applies and each
           job rebinds to whichever model ``vllm_config``-matches),
           and every subsequent :meth:`register` / :meth:`remove`
           atomically rewrites the file via ``write tmp +
           os.replace``.  The file is created on demand; parent
           directories are created on demand.

           **One owner per file.**  No cross-process locking is
           performed; concurrent writers will silently stomp each
           other.

           Calling ``init(...)`` without *session* (or with a
           different *session*) detaches from any previously-bound
           file -- bindings come from the new file (if any) or
           start empty.

        Returns ``None``; the data is intentionally only printed.
        """
        cls._base = base_url.rstrip("/")
        cls._timeout_s = timeout_s
        cls._jobs = {}
        cls._model_counter = 0
        cls._session_path = None
        cls._session_specs = {}
        cls._paused_gpu = {}

        _, body = cls._get("/state")
        snapshot = body.get("models", {})

        # Advance the auto-id counter past any pre-existing ``model_N`` on
        # the server so a fresh process can register new models without
        # colliding with ids the server already owns.  Server-side
        # ``register`` is idempotent on ``model_id`` (silently no-ops on
        # collision), which would otherwise cause the next register here
        # to silently mis-bind to an unrelated existing model.
        for mid in snapshot:
            m = re.fullmatch(r"model_(\d+)", mid)
            if m:
                cls._model_counter = max(cls._model_counter, int(m.group(1)))

        print(f"init: {len(snapshot)} model(s) available on the orchestrator")
        for mid, entry in snapshot.items():
            print(f"  {mid}")
            print(f"    image_path:  {entry.get('image_path') or '?'}")
            print(f"    vllm_config: {entry.get('vllm_config', {})}")

        if session is not None:
            # Replay first (with ``_session_path`` still ``None`` so
            # the per-register ``_save_session`` calls are no-ops),
            # then publish the path and write a single consolidated
            # snapshot.  This keeps the on-disk file untouched if any
            # replayed register raises mid-way.
            path = os.path.abspath(session)
            loaded: dict[str, dict | str] = {}
            if os.path.exists(path):
                with open(path) as f:
                    loaded = json.load(f)
            for job_id, spec in loaded.items():
                cls.register(job_id, spec)
            cls._session_path = path
            cls._save_session()
            print(f"init: session bound to {path} "
                  f"({len(cls._session_specs)} job(s) replayed)")

    # ------------------------------------------------------------------
    # session persistence (job directory mirrored to a JSON file)
    # ------------------------------------------------------------------

    @classmethod
    def _save_session(cls) -> None:
        if cls._session_path is None:
            return
        path = cls._session_path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cls._session_specs, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # job directory
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, job_id: str, vllm_config: dict | str) -> None:
        """Register *job_id* and bind it to a model on the orchestrator.

        Two forms for the second argument:

        * ``dict`` -- a vllm config.  If the orchestrator's registry
          already contains a model with an equal ``vllm_config``,
          *job_id* is bound to that model and **no new server
          registration is issued** (this is how two jobs end up sharing
          a single backing model).  Otherwise a fresh model id
          ``f"model_{counter}"`` is assigned and ``POST /register`` is
          sent to the server.
        * ``str`` -- an explicit orchestrator-side ``model_id``.  The
          server's ``/state`` is consulted; if no model with that id is
          registered, ``ValueError`` is raised.  No new registration is
          ever issued in this form -- it is a pure direct association.

        Idempotent on *job_id*: if *job_id* is already registered with
        the **same** *vllm_config* (compared against the cached
        ``_session_specs`` entry, by value), the call is a silent
        no-op -- no ``/state`` round-trip, no rewrite of the session
        file.  This makes it safe to re-run a registration block after
        an :meth:`init` session replay.  Re-registering a *job_id*
        with a **different** spec still raises ``ValueError`` so
        accidental config drift is not silently swallowed; explicitly
        :meth:`remove` first if you want to rebind.

        Raises ``ValueError`` if *job_id* is already registered locally
        with a different spec, or if the explicit ``model_id`` form is
        used and the model is not present in the orchestrator's
        registry.
        """
        if job_id in cls._jobs:
            existing = cls._session_specs.get(job_id)
            if existing == vllm_config:
                return
            raise ValueError(
                f"job already registered: {job_id!r} "
                f"(existing spec: {existing!r}, new spec: {vllm_config!r}); "
                f"remove() first to rebind")

        _, body = cls._get("/state")
        registered = body.get("models", {})

        if isinstance(vllm_config, str):
            model_id = vllm_config
            if model_id not in registered:
                raise ValueError(
                    f"model_id {model_id!r} is not registered on the "
                    f"orchestrator (known: {sorted(registered)})")
            cls._jobs[job_id] = model_id
            cls._session_specs[job_id] = vllm_config
            cls._save_session()
            return

        for mid, entry in registered.items():
            if entry.get("vllm_config") == vllm_config:
                cls._jobs[job_id] = mid
                cls._session_specs[job_id] = vllm_config
                cls._save_session()
                return

        cls._model_counter += 1
        model_id = f"model_{cls._model_counter}"
        cls._post("/register",
                  {"model_id": model_id, "vllm_config": vllm_config})
        cls._jobs[job_id] = model_id
        cls._session_specs[job_id] = vllm_config
        cls._save_session()

    @classmethod
    def jobs(cls) -> None:
        """Print one row per registered job: model, state, gpu, flags.

        Same JOBS table that :meth:`status` shows, but without the
        REQUESTS section or pause-records footer.  Performs one
        ``GET /state`` to look up each backing model's live state.
        Jobs whose backing model is no longer present on the server
        render with ``state=?`` -- typically a sign the model was
        removed out from under the client.
        """
        _, body = cls._get("/state")
        print(f"jobs: {len(cls._jobs)} on {cls._base or '?'}")
        cls._render_jobs(body)

    @classmethod
    def model_of(cls, job_id: str) -> None:
        """Print the orchestrator model_id bound to *job_id*."""
        print(f"{job_id}  ->  {cls._jobs[job_id]}")

    # ------------------------------------------------------------------
    # API surface (job_id-keyed; mirrors Orchestrator's single-model methods)
    #
    # Convention: per-job methods take ``job_id`` as their first
    # positional arg, and passing ``None`` (or omitting it) fans out
    # over every currently registered job.  :meth:`move` is the
    # exception -- ``target`` is also required, so the fan-out form
    # lives on a separate :meth:`move_all` to keep the positional
    # signature ``move(job_id, target)`` ergonomic.
    # ------------------------------------------------------------------

    DEFAULT_PROMPT = "Hello World!"
    DEFAULT_MAX_TOKENS = 1000

    @classmethod
    def move(cls, job_id: str, target: str,
             target_gpu: int | None = None) -> None:
        cls._post("/move", {
            "model_id": cls._jobs[job_id],
            "target": target,
            "target_gpu": target_gpu,
        })

    @classmethod
    def move_all(cls, target: str, target_gpu: int | None = None) -> None:
        """Client-side fan-out of :meth:`move` over every registered job.

        Two jobs sharing a backing model get two ``/move`` calls (the
        second is a cheap no-op on the orchestrator side).
        """
        for job_id in list(cls._jobs):
            cls.move(job_id, target, target_gpu=target_gpu)

    @classmethod
    def generate(cls, job_id: str | int | None = None,
                 prompts: list[str] | str = _UNSET,
                 sampling_params: dict | int | None = _UNSET,
                 ) -> None:
        """Submit a non-blocking generate against *job_id*.

        Accepts the same shorthand as :meth:`Orchestrator.generate`:
        ``prompts`` may be a single string, ``sampling_params`` may be an
        int (interpreted as ``{"max_tokens": N, "ignore_eos": True}``).
        The server-side ``req_id`` is not returned; track requests via
        ``status()`` or the dashboard if needed.

        Call shapes:

        * ``generate()`` -- all jobs, ``"Hello World!"``,
          ``max_tokens=1000``.
        * ``generate("job 1")`` -- single job, defaults.
        * ``generate("job 1", "foo", 50)`` -- single job, explicit
          prompts and sampling params.
        * ``generate(N)`` -- int shorthand for "fan out over all
          registered jobs with ``max_tokens=N``" (equivalent to
          ``generate(None, sampling_params=N)``).  Cannot be combined
          with explicit *prompts* / *sampling_params* args.
        * ``generate(None, "foo", 50)`` -- explicit fan-out with
          overrides.
        """
        if isinstance(job_id, int):
            if prompts is not _UNSET or sampling_params is not _UNSET:
                raise TypeError(
                    "generate(int) shorthand cannot be combined with "
                    "additional args; use generate(job_id, prompts, "
                    "sampling_params) instead")
            sampling_params = job_id
            job_id = None
        if prompts is _UNSET:
            prompts = cls.DEFAULT_PROMPT
        if sampling_params is _UNSET:
            sampling_params = cls.DEFAULT_MAX_TOKENS

        if job_id is None:
            for jid in list(cls._jobs):
                cls.generate(jid, prompts, sampling_params)
            return
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, int):
            sampling_params = {"max_tokens": sampling_params,
                               "ignore_eos": True}
        cls._post("/generate", {
            "model_id": cls._jobs[job_id],
            "prompts": prompts,
            "sampling_params": sampling_params,
        })

    @classmethod
    def wait(cls, job_id: str | None = None) -> None:
        """Block until the server's pending futures for *job_id* complete.

        With ``job_id=None`` (or omitted), acts as a client-side
        barrier across every registered job: issues ``/wait`` per job
        in registration order, so total wall time is the sum of
        per-job waits (sequential on the wire).
        """
        if job_id is None:
            for jid in list(cls._jobs):
                cls.wait(jid)
            return
        cls._post("/wait", {"model_id": cls._jobs[job_id]})

    @classmethod
    def remove(cls, job_id: str | None = None) -> None:
        """Unbind *job_id* from its backing model.

        Pure local operation -- the orchestrator-side model is *never*
        touched.  If you need to delete a model from the orchestrator,
        do it explicitly against ``model_id`` (the orchestrator's
        ``/remove`` endpoint).  Raises ``KeyError`` if *job_id* is not
        registered.

        With ``job_id=None`` (or omitted), drops every binding.  If a
        session file is bound (see the ``session`` arg of :meth:`init`),
        the file is updated atomically to match (a single rewrite for
        the all-form; one rewrite per job otherwise).  Re-call
        :meth:`init` without ``session`` if you want to detach from the
        file without rewriting it.
        """
        if job_id is None:
            cls._jobs.clear()
            cls._session_specs.clear()
            cls._paused_gpu.clear()
            cls._save_session()
            return
        cls._jobs.pop(job_id)
        cls._session_specs.pop(job_id, None)
        cls._paused_gpu.pop(job_id, None)
        cls._save_session()

    @classmethod
    def pause(cls, job_id: str | int | None = None) -> None:
        """Pause the model bound to *job_id* (running -> up slotless).

        No-op unless the bound model is currently in ``running`` state.
        Releases the slot, snapshots in-flight requests, and flips the
        published state to ``up`` (slotless) with a paused flag.
        Pending request futures resolve only after :meth:`resume`.

        Call shapes:

        * ``pause("job 1")`` -- single job.
        * ``pause()`` -- fans out over every registered job.
        * ``pause(N)`` -- ``int`` shorthand: pauses every registered
          job whose backing model is currently resident on GPU *N*
          (looked up via ``GET /state``).  Each (job_id, *N*) pair
          is recorded in client-side state so a later ``resume(N)``
          can undo exactly that set without another server
          round-trip.  Two jobs sharing one backing model on GPU
          *N* both get a ``/pause`` call (the second is a
          server-side no-op).
        """
        if isinstance(job_id, int):
            gpu = job_id
            _, body = cls._get("/state")
            on_gpu = {mid for mid, entry in body.get("models", {}).items()
                      if isinstance(entry, dict) and entry.get("gpu") == gpu}
            for jid, mid in list(cls._jobs.items()):
                if mid in on_gpu:
                    cls._paused_gpu[jid] = gpu
                    cls.pause(jid)
            return
        if job_id is None:
            for jid in list(cls._jobs):
                cls.pause(jid)
            return
        cls._post("/pause", {"model_id": cls._jobs[job_id]})

    @classmethod
    def resume(cls, job_id: str | int | None = None) -> None:
        """Resume the model bound to *job_id* (up -> running).

        No-op unless the bound model is in ``up`` state and paused.
        Re-acquires a slot via the same Tier A/B/C path as
        :meth:`generate`, re-prefills the saved sub-requests, and
        flips the published state to ``running``.

        Call shapes:

        * ``resume("job 1")`` -- single job.
        * ``resume()`` -- fans out over every registered job.
        * ``resume(N)`` -- ``int`` shorthand: resume every job that a
          previous ``pause(N)`` recorded against GPU *N*.  Pure
          client-side bookkeeping -- no extra ``GET /state`` is
          issued.  Each resumed job's record is cleared on the way
          out, so back-to-back ``resume(N)`` calls are a no-op.
          Symmetric to :meth:`pause` (``int``).
        """
        if isinstance(job_id, int):
            gpu = job_id
            targets = [jid for jid, g in list(cls._paused_gpu.items())
                       if g == gpu]
            for jid in targets:
                cls._paused_gpu.pop(jid, None)
                cls.resume(jid)
            return
        if job_id is None:
            for jid in list(cls._jobs):
                cls.resume(jid)
            return
        cls._post("/resume", {"model_id": cls._jobs[job_id]})

    @classmethod
    def paused(cls) -> None:
        """Print every paused job/model and the GPU each is paused on.

        Performs one ``GET /state`` and shows one row per job whose
        backing model has ``paused=True`` server-side.  Rows are
        sorted by how long each has been paused, **longest-paused
        first** -- the things most likely to need attention sit at
        the top.  Columns:

        * **job_id** -- the registered job id (``-`` for paused
          models with no local binding -- a "ghost" row).
        * **model** -- the bound ``model_id``.
        * **gpu** -- the GPU the paused model still sits on (pause
          deallocates the slot but preserves ``entry["gpu"]``).
        * **paused_for** -- seconds since the model was originally
          paused, derived from ``elapsed_s - paused_since_rel_s``.
          Backed by the orchestrator's dedicated ``paused_since``
          field (set when ``entry["paused"] = True``, cleared on
          resume), so the value is immune to ``move()`` walks while
          paused -- it tracks the original pause moment, not the
          current state.  ``?`` when the server hasn't published
          ``paused_since`` (e.g. talking to an older orchestrator).
        * **pause_record** -- ``GPU N (pause(N))`` if the
          ``(job_id, N)`` pair is in ``_paused_gpu`` (i.e. the pause
          came from the int-shorthand and a future ``resume(N)``
          will undo it); ``-`` otherwise (single-job ``pause(jid)``
          or no-arg ``pause()``).

        A trailing **STALE PAUSE RECORDS** section lists any
        ``_paused_gpu`` entries whose job is no longer paused
        server-side -- typically the result of an explicit
        ``resume(jid)`` between a ``pause(N)`` and the matching
        ``resume(N)``.  Section is omitted when the records and the
        live state agree.
        """
        _, body = cls._get("/state")
        models = body.get("models", {})
        paused_models = {mid: entry for mid, entry in models.items()
                         if isinstance(entry, dict) and entry.get("paused")}
        elapsed = body.get("elapsed_s")

        # Build (sort_key, columns...) tuples so we can sort the
        # combined bound + ghost row set by paused-since timestamp.
        # Older paused_since means longer paused, so ascending sort
        # puts the longest-paused rows first.  Backed by the
        # orchestrator's per-model ``paused_since`` field, which
        # survives ``move()`` walks while the model stays paused
        # (``state_since_rel_s`` would reset on each step).  Missing
        # timestamps sort to the end via +inf.
        sortable: list[tuple[float, tuple[str, ...]]] = []
        bound_paused_mids: set[str] = set()
        for jid, mid in cls._jobs.items():
            if mid not in paused_models:
                continue
            entry = paused_models[mid]
            ts = entry.get("paused_since_rel_s")
            sort_key = ts if isinstance(ts, (int, float)) else float("inf")
            for_s = (elapsed - ts
                     if elapsed is not None and isinstance(ts, (int, float))
                     else None)
            for_s_str = cls._fmt_s(for_s) if for_s is not None else "?"
            gpu = entry.get("gpu")
            gpu_s = "-" if gpu is None else str(gpu)
            rec = cls._paused_gpu.get(jid)
            rec_s = "-" if rec is None else f"GPU {rec} (pause({rec}))"
            sortable.append((sort_key, (jid, mid, gpu_s, for_s_str, rec_s)))
            bound_paused_mids.add(mid)
        for mid, entry in paused_models.items():
            if mid in bound_paused_mids:
                continue
            ts = entry.get("paused_since_rel_s")
            sort_key = ts if isinstance(ts, (int, float)) else float("inf")
            for_s = (elapsed - ts
                     if elapsed is not None and isinstance(ts, (int, float))
                     else None)
            for_s_str = cls._fmt_s(for_s) if for_s is not None else "?"
            gpu = entry.get("gpu")
            gpu_s = "-" if gpu is None else str(gpu)
            sortable.append((sort_key, ("-", mid, gpu_s, for_s_str, "-")))

        sortable.sort(key=lambda r: r[0])
        rows = [r[1] for r in sortable]

        n_jobs_paused = sum(1 for mid in cls._jobs.values()
                            if mid in paused_models)
        print(f"paused: {n_jobs_paused} of {len(cls._jobs)} job(s) paused, "
              f"{len(paused_models)} model(s) paused on {cls._base or '?'}")
        if not rows:
            print("\nPAUSED  (none)")
        else:
            cls._print_table(
                title="PAUSED",
                headers=("job_id", "model", "gpu",
                         "paused_for", "pause_record"),
                rows=rows,
            )

        stale = [(jid, gpu) for jid, gpu in cls._paused_gpu.items()
                 if cls._jobs.get(jid) not in paused_models]
        if stale:
            print("\nSTALE PAUSE RECORDS  "
                  "(in _paused_gpu but not paused server-side)")
            for jid, gpu in stale:
                print(f"  {jid}  -> recorded GPU {gpu}")

    @classmethod
    def requests(cls) -> None:
        """Print one row per in-flight or completed request on the server.

        Same REQUESTS table that :meth:`status` shows, but without
        the JOBS section or pause-records footer.  Performs one
        ``GET /state``.  Each row's ``jobs`` column lists every
        registered job_id bound to the request's backing model
        (multiple when jobs share a model; ``-`` when no local job
        is bound to it).
        """
        _, body = cls._get("/state")
        n = len(body.get("requests", []))
        print(f"requests: {n} on {cls._base or '?'}")
        cls._render_requests(body)

    @classmethod
    def status(cls) -> None:
        """Pretty-print a job-centric view of the orchestrator state.

        Renders three sections from a single ``GET /state`` (so the
        view is internally consistent):

        * **JOBS** -- the table from :meth:`jobs`.
        * **REQUESTS** -- the table from :meth:`requests`.
        * **PAUSE RECORDS** -- contents of ``_paused_gpu``: the
          (job_id, gpu) pairs recorded by the int-shorthand of
          :meth:`pause`, that the int-shorthand of :meth:`resume`
          will consume.  Section is omitted when empty.

        Use :meth:`jobs` / :meth:`requests` for the focused views,
        or :meth:`status_raw` for the verbatim JSON snapshot.
        """
        _, body = cls._get("/state")
        models = body.get("models", {})
        requests = body.get("requests", [])
        elapsed = body.get("elapsed_s")
        print(f"status: {len(cls._jobs)} job(s), {len(models)} model(s), "
              f"{len(requests)} request(s) on {cls._base or '?'}"
              + (f"   (server uptime {elapsed}s)" if elapsed else ""))
        cls._render_jobs(body)
        cls._render_requests(body)
        if cls._paused_gpu:
            print("\nPAUSE RECORDS  (set by pause(N), consumed by resume(N))")
            for jid, gpu in cls._paused_gpu.items():
                print(f"  {jid}  -> GPU {gpu}")

    @classmethod
    def _render_jobs(cls, body: dict) -> None:
        if not cls._jobs:
            print("\nJOBS  (none)")
            return
        models = body.get("models", {})
        draining = set(body.get("draining", []))
        rows = []
        for jid, mid in cls._jobs.items():
            entry = models.get(mid, {})
            state = entry.get("state", "?")
            gpu = entry.get("gpu")
            gpu_s = "-" if gpu is None else str(gpu)
            flags = []
            if entry.get("paused"):
                flags.append("paused")
            if gpu in draining:
                flags.append("draining")
            rows.append((jid, mid, state, gpu_s, ",".join(flags)))
        cls._print_table(
            title="JOBS",
            headers=("job_id", "model", "state", "gpu", "flags"),
            rows=rows,
        )

    @classmethod
    def _render_requests(cls, body: dict) -> None:
        requests = body.get("requests", [])
        if not requests:
            print("\nREQUESTS  (none)")
            return
        mid_to_jobs: dict[str, list[str]] = {}
        for jid, mid in cls._jobs.items():
            mid_to_jobs.setdefault(mid, []).append(jid)

        # Group requests under their bound jobs by sorting on the
        # registration index of the lowest-ordered job each request's
        # backing model is bound to.  Stable sort preserves the
        # server's submission order within each group.  Requests whose
        # backing model has no local job binding (orphans) sink to the
        # end so the live, job-bound rows stay at the top.
        job_order = {jid: i for i, jid in enumerate(cls._jobs)}
        sentinel = len(job_order)

        def _sort_key(r: dict) -> tuple[int]:
            jids = mid_to_jobs.get(r.get("model_id", "?"), [])
            if not jids:
                return (sentinel,)
            return (min(job_order[j] for j in jids),)

        requests = sorted(requests, key=_sort_key)

        req_rows = []
        for r in requests:
            mid = r.get("model_id", "?")
            jobs_s = ",".join(mid_to_jobs.get(mid, [])) or "-"
            pt = r.get("prompt_tokens")
            ct = r.get("completion_tokens")
            tokens_s = (f"{ct or 0}/{pt}" if pt is not None
                        else (f"{ct}" if ct is not None else "-"))
            req_rows.append((
                str(r.get("req_id", "?")),
                mid,
                jobs_s,
                str(r.get("state", "?")),
                cls._fmt_s(r.get("wait_s")),
                cls._fmt_s(r.get("gen_s")),
                cls._fmt_s(r.get("paused_s")) if r.get("paused_s") else "-",
                tokens_s,
            ))
        cls._print_table(
            title="REQUESTS",
            headers=("req_id", "model", "jobs", "state",
                     "wait", "gen", "paused", "tokens"),
            rows=req_rows,
        )

    @classmethod
    def status_raw(cls) -> None:
        """Pretty-print the full state snapshot (``GET /state``) as JSON."""
        _, body = cls._get("/state")
        print(json.dumps(body, indent=2, default=str))

    @staticmethod
    def _fmt_s(v: float | None) -> str:
        if v is None:
            return "-"
        return f"{v:.2f}s"

    @staticmethod
    def _print_table(title: str, headers: tuple[str, ...],
                     rows: list[tuple[str, ...]]) -> None:
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        sep = "  "
        head = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
        rule = sep.join("-" * widths[i] for i in range(len(headers)))
        print(f"\n{title}")
        print(f"  {head}")
        print(f"  {rule}")
        for row in rows:
            line = sep.join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            print(f"  {line}")

    # ------------------------------------------------------------------
    # GPU pool management
    # ------------------------------------------------------------------

    @classmethod
    def add(cls, gpu: int) -> None:
        """Add *gpu* to the orchestrator's pool.

        Bookkeeping-only on the server side; new placements may begin
        landing on *gpu* immediately after this returns.  The server
        warns (does not raise) if *gpu* is not visible to its NVML.
        """
        cls._post("/add", {"gpu": gpu})

    @classmethod
    def sub(cls, gpu: int) -> None:
        """Drain *gpu* from the orchestrator's pool (non-blocking).

        Refuses to drain the last non-draining GPU client-side, since
        residents would have nowhere to migrate and the drain would
        hang forever.  Use :meth:`wait_gpu` to await completion.
        """
        _, body = cls._get("/state")
        gpu_ids = set(body.get("gpu_ids", []))
        draining = set(body.get("draining", []))
        remaining = gpu_ids - draining - {gpu}
        if not remaining:
            raise ValueError(
                f"refusing to drain GPU {gpu}: it is the last "
                f"non-draining GPU (gpu_ids={sorted(gpu_ids)}, "
                f"draining={sorted(draining)})")
        cls._post("/sub", {"gpu": gpu})

    @classmethod
    def wait_gpu(cls, gpu: int) -> None:
        """Block until the pending :meth:`sub` for *gpu* completes."""
        cls._post("/wait_gpu", {"gpu": gpu})

    # ------------------------------------------------------------------
    # transport helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_connected(cls) -> None:
        if not cls._base:
            raise RuntimeError(
                "OrchestratorClient.connect(...) must be called first")

    @classmethod
    def _post(cls, path: str, payload: dict | None = None) -> dict:
        cls._ensure_connected()
        data = json.dumps(payload or {}).encode()
        req = urllib.request.Request(
            cls._base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cls._timeout_s) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            body = json.loads(e.read() or b"{}")
            msg = body.get("error", str(e))
            raise RuntimeError(f"orch_server {path} -> HTTP {e.code}: {msg}")

    @classmethod
    def _get(cls, path: str) -> tuple[int, dict]:
        cls._ensure_connected()
        try:
            with urllib.request.urlopen(
                    cls._base + path, timeout=cls._timeout_s) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
