"""Embedded HTTP server that exposes orchestrator state as JSON (GET /state)."""
from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

_t0: float = 0.0


def init_t0() -> None:
    """Record the orchestrator start time for relative request timestamps."""
    global _t0
    _t0 = time.perf_counter()


def _snapshot_slots() -> tuple[dict[int, list[dict]], int]:
    """Return ``({gpu_id: [leaf, ...]}, waiters)`` for the buddy allocator.

    Each leaf is ``{"level": L, "index": I, "alloc": bool, "mid": str|None}``
    covering ``1 / 2**(L-1)`` of GPU ``gpu_id``; ``mid`` is the model id of
    the slot's holder for ALLOC leaves and absent for FREE leaves.  Leaves
    are emitted in left-to-right order (DFS by ``index``) so a renderer
    can map them onto a bar without further sorting.
    """
    from orchestrator import Orchestrator
    from slots import Slots

    with Slots._cv:
        if not Slots._inited:
            return {}, 0
        pool_leaves: dict[int, set[tuple[int, int]]] = {}
        for (g, L), pool in Slots._pools.items():
            for s in pool:
                pool_leaves.setdefault(g, set()).add((L, s.index))
        live_leaves: dict[int, set[tuple[int, int]]] = {}
        for s in Slots._live:
            live_leaves.setdefault(s.gpu_id, set()).add((s.level, s.index))
        gpus = sorted(set(Slots._last_used.keys()))
        waiters = len(Slots._waiters)

    # Reverse-map (gpu, level, index) -> mid by scanning the registry once.
    # Slotless residents (slot is None) don't appear here; they're handled
    # by the dashboard as ghosts via ``state == "sleep" && slot_level is None``.
    slot_to_mid: dict[tuple[int, int, int], str] = {}
    for mid, entry in Orchestrator._registry.items():
        s = entry.get("slot")
        if s is not None:
            slot_to_mid[(s.gpu_id, s.level, s.index)] = mid

    out: dict[int, list[dict]] = {}
    for g in gpus:
        free = pool_leaves.get(g, set())
        live = live_leaves.get(g, set())
        leaves: list[dict] = []

        def walk(level: int, index: int) -> None:
            key = (level, index)
            if key in free:
                leaves.append({"level": level, "index": index,
                               "alloc": False})
                return
            if key in live:
                leaves.append({
                    "level": level, "index": index, "alloc": True,
                    "mid": slot_to_mid.get((g, level, index)),
                })
                return
            walk(level + 1, 2 * index)
            walk(level + 1, 2 * index + 1)

        walk(1, 0)
        out[g] = leaves
    return out, waiters


def snapshot_state() -> dict:
    """Return a JSON-serialisable snapshot of the orchestrator state."""
    from orchestrator import Orchestrator

    now = time.perf_counter()
    t0_valid = _t0 and _t0 > 0.0
    elapsed_s = round(now - _t0, 3) if t0_valid else None
    models = {}
    for mid, entry in list(Orchestrator._registry.items()):
        inst = entry.get("instance")
        pid = None
        if inst is not None and getattr(inst, "pid", None):
            pid = inst.pid
        raw_since = entry.get("state_since", 0)
        state_since_rel = (
            round(raw_since - _t0, 3)
            if t0_valid and raw_since
            else None
        )
        # ``paused_since`` is set by ``_pause_sync`` and cleared by
        # ``_resume_sync`` (both branches), so it is a strict per-model
        # "time of the original pause" timestamp -- unaffected by
        # subsequent ``move()`` walks while the model stays paused
        # (which do bump ``state_since`` per step).  Exposed alongside
        # ``state_since_rel_s`` so consumers can pick the semantic
        # they want.
        raw_paused_since = entry.get("paused_since")
        paused_since_rel = (
            round(raw_paused_since - _t0, 3)
            if t0_valid and raw_paused_since
            else None
        )
        slot = entry.get("slot")
        models[mid] = {
            "state": entry.get("state"),
            "gpu": entry.get("gpu"),
            "pid": pid,
            "pinned_cpu_bytes": entry.get("pinned_cpu_bytes", 0),
            "total_gpu_bytes": entry.get("total_gpu_bytes", 0),
            "vllm_config": entry.get("vllm_config", {}),
            "image_path": entry.get("image_dir"),
            "state_since": entry.get("state_since", 0),
            "state_since_rel_s": state_since_rel,
            "level": entry.get("level"),
            "slot_level": slot.level if slot is not None else None,
            "slot_index": slot.index if slot is not None else None,
            "paused": bool(entry.get("paused", False)),
            "paused_since_rel_s": paused_since_rel,
        }
    busy = {
        e.get("gpu") for e in Orchestrator._registry.values()
        if e.get("gpu") is not None
        and e.get("state") in ("init", "sleep", "up", "running")
    }
    free_gpus = sorted(g for g in Orchestrator._gpu_ids if g not in busy)
    from slots import Slots as _Slots
    with _Slots._cv:
        draining = sorted(_Slots._draining)

    requests = []
    with Orchestrator._request_lock:
        # Index earlier-submitted records per-model for piggy-back
        # detection.  Two flavours collapse into the same
        # ``piggyback`` field (the ``req_id`` of the *lead* request --
        # the earliest in-flight submit that R hitched onto):
        #
        #   * climb-piggyback: an earlier submit on the same model
        #     hadn't reached t_gen_start when R submitted, meaning R
        #     caught a ride on that request's up-cycle rather than
        #     driving its own climb.  Persists for R's lifetime
        #     because the condition (``L.t_gen_start > R.t_submit``)
        #     is immutable once L starts.
        #
        #   * pause-piggyback: R was submitted while its model was
        #     paused (orchestrator's one-shot ``paused_at_submit``
        #     marker) and an earlier-submit L on the same model was
        #     still alive at R's submit time (``L.t_done is None`` or
        #     ``L.t_done > R.t_submit``).  R cannot make progress on
        #     its own until the next ``resume`` re-prefills L's saved
        #     sub-requests and unfreezes the engine; that same
        #     ``resume`` is what drives R, so the bracket collapses
        #     to ``wait #L+1`` instead of replaying R's (mostly empty)
        #     state timeline.  Both inputs are immutable, so this
        #     classification persists past resume and even past L's
        #     completion -- matching how climb-piggyback persists past
        #     the lead's ``t_gen_start``.
        #
        # ``piggyback`` is ``None`` when neither flavour applies.
        per_model_subs: dict[
            str, list[tuple[int, float, float | None, float | None]]
        ] = {}
        for other in Orchestrator._request_log:
            per_model_subs.setdefault(other["model_id"], []).append(
                (other["req_id"], other["t_submit"],
                 other["t_gen_start"], other.get("t_done")))

        for rec in Orchestrator._request_log:
            state = rec["state"]
            t_submit = rec["t_submit"]
            t_gen_start = rec["t_gen_start"]
            t_done = rec["t_done"]
            rec_paused_at_submit = bool(rec.get("paused_at_submit"))

            piggyback: int | None = None
            earliest_sub = float("inf")
            for other_id, other_sub, other_gen, other_done in (
                    per_model_subs.get(rec["model_id"], [])):
                if other_sub >= t_submit:
                    continue
                is_climb_lead = (other_gen is None or other_gen > t_submit)
                # L was alive when R was submitted iff L hadn't yet
                # finished by then.  ``other_done is None`` covers
                # "still in flight"; the strict bound covers leads
                # that completed after R's pause-submit but whose
                # follower-relationship we want to keep displaying.
                other_alive_at_submit = (
                    other_done is None or other_done > t_submit)
                is_pause_lead = (rec_paused_at_submit
                                 and other_alive_at_submit)
                if is_climb_lead or is_pause_lead:
                    if other_sub < earliest_sub:
                        earliest_sub = other_sub
                        piggyback = other_id

            # Total time this request has spent suspended because its
            # model was paused.  ``paused_s`` is the closed-out
            # accumulator (updated on resume); ``t_pause_started`` is
            # set while we're currently inside a pause window and
            # contributes the live ``now - t_pause_started`` slice.
            paused_s_total = float(rec.get("paused_s") or 0.0)
            t_pause_started = rec.get("t_pause_started")
            if t_pause_started is not None:
                paused_s_total += max(0.0, now - t_pause_started)

            if state == "waiting":
                wait_s = now - t_submit
                gen_s = None
            elif state == "generating":
                if t_gen_start is not None:
                    wait_s = t_gen_start - t_submit
                    gen_s = max(0.0, now - t_gen_start - paused_s_total)
                else:
                    wait_s = now - t_submit
                    gen_s = None
            else:
                wait_s = (t_gen_start - t_submit) if t_gen_start else None
                gen_s = (max(0.0, t_done - t_gen_start - paused_s_total)
                         if (t_done and t_gen_start) else None)

            if _t0 and _t0 > 0.0:
                submit_rel_s = round(t_submit - _t0, 3)
                gen_start_rel_s = (
                    round(t_gen_start - _t0, 3)
                    if t_gen_start is not None
                    else None
                )
                done_rel_s = (
                    round(t_done - _t0, 3) if t_done is not None else None
                )
            else:
                submit_rel_s = None
                gen_start_rel_s = None
                done_rel_s = None

            if t0_valid:
                state_log = [
                    [round(t - _t0, 3), s]
                    for (t, s) in rec.get("state_log", [])
                ]
            else:
                state_log = []

            requests.append({
                "req_id": rec["req_id"],
                "model_id": rec["model_id"],
                "state": state,
                "start_state": rec.get("start_state"),
                "wait_s": round(wait_s, 2) if wait_s is not None else None,
                "gen_s": round(gen_s, 2) if gen_s is not None else None,
                # Total paused time so far on this request (closed-out
                # ``paused_s`` plus the live ``now - t_pause_started``
                # window when currently paused).  Lets the dashboard
                # surface a ``paused N.Ns`` counter alongside ``gen``.
                "paused_s": round(paused_s_total, 2),
                "prompt_tokens": rec.get("prompt_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
                "ttft_s": rec.get("ttft_s"),
                "tpot_ms": rec.get("tpot_ms"),
                "gpu_wait_s": round(rec["gpu_wait_s"], 2) if rec.get("gpu_wait_s") is not None else None,
                "migrate_s": round(rec["migrate_s"], 2) if rec.get("migrate_s") is not None else None,
                "up_s": round(rec["up_s"], 2) if rec.get("up_s") is not None else None,
                "submit_rel_s": submit_rel_s,
                "gen_start_rel_s": gen_start_rel_s,
                "done_rel_s": done_rel_s,
                "state_log": state_log,
                "piggyback": piggyback,
            })

    slots, waiters = _snapshot_slots()
    return {
        "gpu_ids": list(Orchestrator._gpu_ids),
        "gpu_pool": free_gpus,
        "draining": draining,
        "image_cache": Orchestrator._image_cache,
        "models": models,
        "requests": requests,
        "elapsed_s": elapsed_s,
        "slots": slots,
        "slot_waiters": waiters,
    }


class StateHandler(BaseHTTPRequestHandler):
    """Serves orchestrator state and control over HTTP.

    GET  /state              -- JSON snapshot of registry, slots, requests
    POST /register           -- Orchestrator.register(model_id, vllm_config)
    POST /move               -- Orchestrator.move(model_id, target, target_gpu)
    POST /generate           -- Orchestrator.submit_generate(...); returns req_id
    POST /wait               -- Orchestrator.wait(model_id) (blocks)
    POST /remove             -- Orchestrator.remove(model_id)
    POST /pause              -- Orchestrator.pause(model_id)  (running -> up)
    POST /resume             -- Orchestrator.resume(model_id) (up -> running)
    POST /add                -- Orchestrator.add(gpu)
    POST /sub                -- Orchestrator.sub(gpu) (non-blocking)
    POST /wait_gpu           -- Orchestrator.wait_gpu(gpu) (blocks)

    Multi-model fan-out (``move_all``, ``generate_all``, ``wait_all``,
    ``remove_all``) is not exposed here on purpose -- clients implement
    it as a loop over ``GET /state`` -> per-model calls.  See
    :class:`client.OrchestratorClient` for the canonical implementation.
    """

    def do_GET(self):
        if self.path != "/state":
            self.send_error(404)
            return
        self._json(200, snapshot_state())

    def do_POST(self):
        from orchestrator import Orchestrator

        body = self._read_json()
        try:
            if self.path == "/register":
                Orchestrator.register(body["model_id"], body["vllm_config"])
                self._json(200, {"ok": True})
            elif self.path == "/move":
                Orchestrator.move(
                    body["model_id"], body["target"],
                    target_gpu=body.get("target_gpu"))
                self._json(200, {"ok": True})
            elif self.path == "/generate":
                req_id, _fut = Orchestrator.submit_generate(
                    body["model_id"], body["prompts"],
                    body.get("sampling_params"))
                self._json(200, {"req_id": req_id})
            elif self.path == "/wait":
                Orchestrator.wait(body["model_id"])
                self._json(200, {"ok": True})
            elif self.path == "/remove":
                Orchestrator.remove(body["model_id"])
                self._json(200, {"ok": True})
            elif self.path == "/pause":
                Orchestrator.pause(body["model_id"])
                self._json(200, {"ok": True})
            elif self.path == "/resume":
                Orchestrator.resume(body["model_id"])
                self._json(200, {"ok": True})
            elif self.path == "/add":
                Orchestrator.add(body["gpu"])
                self._json(200, {"ok": True})
            elif self.path == "/sub":
                Orchestrator.sub(body["gpu"])
                self._json(200, {"ok": True})
            elif self.path == "/wait_gpu":
                Orchestrator.wait_gpu(body["gpu"])
                self._json(200, {"ok": True})
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before we could send the response
            # (common when a long-running call like /wait outlasts the
            # client process).  No socket to report on, just drop.
            pass
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n))

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_state_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), StateHandler)
    server.socket.set_inheritable(False)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
