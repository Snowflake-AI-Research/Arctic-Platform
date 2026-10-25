"""Pure-logic tests for the semi-persistence onboarding suite (no network).

These import the real package, so they live apart from the hermetic
``semi_persistence/tests``, whose module stubs would shadow it. From the
repository root::

    python -m pytest arctic_platform/inference/semi_persistence/onboarding/tests
"""

import json

import pytest

from arctic_platform.inference.semi_persistence.onboarding import jobs
from arctic_platform.inference.semi_persistence.onboarding import perf
from arctic_platform.inference.semi_persistence.onboarding import report
from arctic_platform.inference.semi_persistence.onboarding import startup
from arctic_platform.inference.semi_persistence.onboarding.common import log_text
from arctic_platform.inference.semi_persistence.onboarding.common import parse_kv_args


def _spec(tp=2, n_gpus=8, **extra_vc):
    return jobs.ModelSpec(
        name="m",
        sub_job={
            "job_type": "sampling",
            "model_name": "org/model",
            "inference_config": {
                "max_seq_len": 128,
                "n_gpus": n_gpus,
                "vllm_config": {"tensor_parallel_size": tp, **extra_vc},
            },
        },
    )


def test_variant_bodies_differ_only_in_what_each_variant_needs():
    spec = _spec(tp=2, extra_env={"A": "1"})
    bodies = jobs.build_variant_bodies(spec, "TAG", list(jobs.VARIANTS), copies=2, miss_image_cache="/scratch/run1")
    ic = {v: b["sub_job_configs"][0]["inference_config"] for v, b in bodies.items()}
    assert all(b["debug"] == {"job": {"image_tag": "TAG"}} for b in bodies.values())
    assert {v: c["semi_p"] for v, c in ic.items()} == {"on_hit": True, "off_default": False,
                                                        "on_miss": True, "off_matched": False}
    assert {v: c["n_gpus"] for v, c in ic.items()} == {"on_hit": 16, "off_default": 16, "on_miss": 8, "off_matched": 16}
    plain = {"tensor_parallel_size": 2, "extra_env": {"A": "1"}}
    assert ic["on_hit"]["vllm_config"] == plain and ic["off_default"]["vllm_config"] == plain
    assert ic["on_miss"]["vllm_config"]["extra_env"] == {"A": "1", "SEMIP_IMAGE_SOURCE": "",
                                                         "SEMIP_DUMP_PUBLISH": "1",
                                                         "SEMIP_IMAGE_CACHE": "/scratch/run1"}
    matched = ic["off_matched"]["vllm_config"]
    assert matched["extra_env"] == {"A": "1", **jobs.MATCHED_TP_ENV, **jobs.MATCH_FLAG_ENV}
    assert matched["extra_engine_kwargs"]["compilation_config"]["pass_config"] == {"fuse_allreduce_rms": False}
    assert "SEMIP_IMAGE_CACHE" not in jobs.build_variant_bodies(spec, None, ["on_miss"])["on_miss"][
        "sub_job_configs"][0]["inference_config"]["vllm_config"]["extra_env"]


def test_plan_variants_skips_what_cannot_differ_or_be_matched():
    assert jobs.plan_variants(_spec(tp=2)) == (list(jobs.VARIANTS), {})
    variants, skipped = jobs.plan_variants(_spec(tp=1, n_gpus=1))
    assert variants == ["on_hit", "off_default", "on_miss"] and "off_matched" in skipped
    variants, skipped = jobs.plan_variants(_spec(tp=16, n_gpus=16))
    assert "off_matched" not in variants and "pod-spanning" in skipped["off_matched"]
    assert jobs.plan_variants(_spec(tp=2), ["on_hit", "off_default"])[0] == ["on_hit", "off_default"]
    with pytest.raises(ValueError):
        jobs.plan_variants(_spec(tp=2), ["on_miss"])


def test_replicas_and_model_file_validation(tmp_path):
    assert _spec(tp=2, n_gpus=8).replicas == 4
    assert _spec(tp=16, n_gpus=16).replicas == 1
    path = tmp_path / "x.json"
    sub = _spec().sub_job
    sub["inference_config"]["semi_p"] = True
    path.write_text(json.dumps({"sub_job": sub}))
    with pytest.raises(ValueError, match="semi_p"):
        jobs.load_model(str(path))


def _shipped_model_paths():
    return [p for p in sorted(jobs.MODELS_DIR.glob("*.json")) if p != jobs.BACKENDS_FILE]


def test_shipped_model_files_load():
    for path in _shipped_model_paths():
        spec = jobs.load_model(str(path))
        assert spec.name == path.stem
        assert spec.n_gpus % spec.tensor_parallel_size == 0


def test_every_shipped_model_has_a_backend():
    for path in _shipped_model_paths():
        assert jobs.resolve_image_tag(jobs.load_model(str(path))).startswith(("dev_", "release_"))


def test_resolve_image_tag(tmp_path):
    spec = jobs.load_model("qwen36-35b-a3b-tp1")
    backends = tmp_path / "backends.json"
    backends.write_text('{"qwen36-35b-a3b-tp1": {"image_tag": "dev_A"}}')
    assert jobs.resolve_image_tag(spec, backends_file=backends) == "dev_A"
    assert jobs.resolve_image_tag(spec, "dev_B", backends_file=backends) == "dev_B"
    backends.write_text("{}")
    with pytest.raises(ValueError, match="no image_tag"):
        jobs.resolve_image_tag(spec, backends_file=backends)


RESTORE_LOG = [
    "00:00:01.000 [r0 engine] INFO x.py:1 pid=1 semi_p: resolved model_dir=/cache/k (config=k)",
    "00:00:03.500 [r0 engine] INFO x.py:1 pid=1 semi_p: materialized /cache/k from /published/k",
    "00:00:03.600 [r0 engine] INFO x.py:1 pid=1 semi_p: reading weights in place from /published/w",
    "00:00:03.700 [r0 engine] INFO x.py:1 pid=1 semi_p: restoring image model_dir=/cache/k",
    "00:00:07.000 [r0 worker] INFO w.py:1 pid=2 <<< criu_restore OK (3.300s)",
    "00:00:09.000 [r0 child] INFO c.py:1 pid=3 <<< load_weights OK (2.000s)",
    "00:00:21.000 [r0 engine] INFO x.py:1 pid=1 semi_p: restore complete model_dir=/cache/k gpus=[0]",
    "00:01:00.000 [r0 child] INFO c.py:1 pid=3 <<< generate req_id=a OK (0.1s)  finish=['stop']",
]


def test_parse_semi_p_restore():
    r = startup.parse_semi_p(RESTORE_LOG)
    assert r["markers_visible"] and r["restored"] and not r["cache_miss"]
    rep = r["replicas"]["r0"]
    assert rep["materialize_s"] == 2.5 and rep["key_to_ready_s"] == 20.0 and rep["restore_s"] == 17.3
    assert r["slowest_replica_steps"] == {"criu_restore": 3.3, "load_weights": 2.0}
    assert startup.generate_counts({"zone-device-manager-abcde.log": RESTORE_LOG}) == {"abcde r0": 1}


def test_parse_semi_p_counts_the_attach_in_key_to_ready():
    log = RESTORE_LOG[:-1] + [
        "00:00:21.500 [r0 child] INFO c.py:1 pid=3 <<< serve_core OK (0.400s)",
        "00:00:24.000 [r0 engine] INFO x.py:1 pid=1 semi_p: front end attached in 3.00s "
        "(serve_core 0.45s, AsyncLLM 2.50s)",
        RESTORE_LOG[-1],
    ]
    r = startup.parse_semi_p(log)
    rep = r["replicas"]["r0"]
    assert rep["attach_s"] == 3.0 and rep["restore_s"] == 17.3 and rep["key_to_ready_s"] == 23.0
    assert r["slowest_replica_steps"] == {"criu_restore": 3.3, "load_weights": 2.0, "serve_core": 0.4}
    assert "| r0 | published image | 2.5 | 17.3 | 3.0 | 23.0 |" in "\n".join(report.replica_table(r))


def test_restore_report_prefers_pod_logs_and_labels_each_pod():
    agent = ["00:00:05.000 [r0 worker] INFO w.py:1 pid=9 <<< criu_restore OK (4.000s)",
             "00:00:20.000 [r0 worker] INFO w.py:1 pid=9 <<< cuda_restore OK (15.000s)",
             "00:09:00.000 [r0 worker] INFO w.py:1 pid=9 <<< teardown OK (1.000s)"]
    pods = {"zone-device-manager-aaaaa.log": RESTORE_LOG, "zone-device-manager-bbbbb.log": agent}
    r = startup.restore_report({"sub-job.log": ["unrelated"]}, pods)
    assert r["steps_source"] == "pod logs" and r["restored"]
    assert r["replicas"]["aaaaa r0"]["steps"] == {"criu_restore": 3.3, "load_weights": 2.0}
    assert r["replicas"]["bbbbb r0"] == {"partition_agent": True, "failed_steps": [],
                                         "steps": {"criu_restore": 4.0, "cuda_restore": 15.0}}
    table = "\n".join(report.replica_table(r))
    assert "| bbbbb r0 | node-partition agent |" in table and "cuda_restore" in table
    assert startup.restore_report({"sub-job.log": RESTORE_LOG}, {})["steps_source"].startswith("sub-job log")


def test_parse_semi_p_sub_job_log_format_two_pods():
    def line(ip, pid, ts, msg):
        return f"2026-01-01T00:00:00Z (InferenceWorker pid={pid}, ip={ip}) 2026-01-01 {ts} INFO m.py:1 semi_p: {msg}"

    lines = []
    for ip, pid in (("10.0.0.1", 11), ("10.0.0.2", 22)):
        lines += [line(ip, pid, "00:00:01,000", "resolved model_dir=/cache/k/replica0 (config=k)"),
                  line(ip, pid, "00:00:03,000", "materialized /cache/k from /published/k"),
                  line(ip, pid, "00:00:03,100", "reading weights in place from /published/w"),
                  line(ip, pid, "00:00:03,200", "restoring image model_dir=/cache/k/replica0"),
                  line(ip, pid, "00:00:31,000", "restore complete model_dir=/cache/k/replica0 gpus=[0, 1]")]
    r = startup.parse_semi_p(lines)
    assert sorted(r["replicas"]) == ["10.0.0.1 replica0", "10.0.0.2 replica0"]
    assert r["restored"] and r["key_to_ready_s"] == 30.0
    assert r["replicas"]["10.0.0.2 replica0"]["materialize_s"] == 2.0


def test_parse_semi_p_cache_miss_and_invisible_markers():
    lines = ["00:00:01.000 [r1 engine] semi_p: resolved model_dir=/cache/k",
             "00:00:01.100 [r1 engine] semi_p: cache miss -- cold-starting",
             "00:05:00.000 [r1 engine] semi_p: dump complete model_dir=/cache/k in 298.9s"]
    r = startup.parse_semi_p(lines)
    assert r["cache_miss"] and not r["restored"]
    assert r["replicas"]["r1"]["cold_start_and_dump_s"] == 298.9
    assert not startup.parse_semi_p(["plain vLLM output"])["markers_visible"]


def test_parse_vllm_takes_slowest_replica():
    lines = ["[r0] INFO [default_loader.py:484] Loading weights took 9.81 seconds",
             "[r1] INFO [default_loader.py:484] Loading weights took 12.50 seconds",
             "INFO [core.py:373] init engine (profile, create kv cache, warmup model) took 137.22 s (compilation: 15 s)",
             "INFO [model_runner.py:1087] Graph capturing finished in 14 secs, took 0.88 GiB"]
    assert startup.parse_vllm(lines) == {"load_weights_s": 12.5, "engine_init_s": 137.22, "graph_capture_s": 14.0}


def test_perf_grid_helpers():
    assert perf.concurrency_levels(1) == [1, 8, 32]
    assert perf.concurrency_levels(4) == [1, 4, 16]
    assert perf.concurrency_levels(64) == [1, 64, 128]
    assert perf.fit_shapes([[16, 16], [64, 96]], 128) == [(16, 16)]
    with pytest.raises(ValueError):
        perf.fit_shapes([[100, 100]], 128)


def _result(**kw):
    base = {"model": {"name": "m"}, "startup": {"semi_p": {"markers_visible": True, "restored": True,
                                                            "cache_miss": False}}}
    base.update(kw)
    return base


def test_verdict():
    ok = {"name": "c", "passed": True, "detail": "", "required": True}
    info = {"name": "i", "passed": None, "detail": "", "required": False}
    assert report.verdict(_result(correctness=[ok, info])) == ("PASS", [])
    bad = {**ok, "passed": False, "detail": "differs"}
    assert report.verdict(_result(correctness=[bad]))[0] == "FAIL"
    perf_fail = {"checks": [{"name": "TPOT ratio", "value": 1.5, "limit": "<= 1.1", "passed": False}]}
    assert report.verdict(_result(perf=perf_fail))[0] == "FAIL"
    hidden = _result(startup={"semi_p": {"markers_visible": False, "restored": False, "cache_miss": False}})
    assert report.verdict(hidden)[0] == "WARN"
    miss = _result(startup={"semi_p": {"markers_visible": True, "restored": False, "cache_miss": True}})
    assert report.verdict(miss)[0] == "FAIL"


def test_book_keeps_latest_run_per_config(tmp_path):
    from arctic_platform.inference.semi_persistence.onboarding import book

    def run(started, verdict_ok, image="IMG"):
        return {"model": {"name": "m-tp1", "model_name": "org/m", "n_gpus": 1, "tensor_parallel_size": 1,
                          "replicas": 1, "copies": 1},
                "image_tag": image, "started_at": started, "on_job_id": "SECRET-ON", "off_job_id": "SECRET-OFF",
                "correctness": [{"name": "greedy", "passed": verdict_ok, "detail": "", "required": True}],
                "startup": {"semi_p": {"markers_visible": True, "restored": True, "cache_miss": False,
                                       "key_to_ready_s": 20.0, "replicas": {}},
                            "vllm_off": {"model_loading_s": 15.0, "engine_init_s": 145.0}}}

    (tmp_path / "a.json").write_text(json.dumps(run("2026-01-01T00:00:00", False)))
    (tmp_path / "b.json").write_text(json.dumps(run("2026-01-02T00:00:00", True)))
    (tmp_path / "c.json").write_text(json.dumps(run("2026-01-01T00:00:00", True, image="OTHER")))
    runs = book.load_runs(tmp_path)
    assert [(r["image_tag"], r["started_at"][:10]) for r in runs] == [("IMG", "2026-01-02"), ("OTHER", "2026-01-01")]
    text = book.build(tmp_path)
    assert "SECRET" not in text
    for section in ("## Overview", "## Start-up", "## Restore primitives", "## Latency", "## Throughput"):
        assert section in text
    assert "| 160.0 | 8.0x |" in text
    assert book.plain_cold_start({"engine_init_s": 100.0}) == (100.0, False)


class _FakeClient:
    """Answers deterministically per (prompt, sampling params); jobs in ``drift``
    answer greedy prompts differently from the rest."""

    def __init__(self, drift=()):
        self.drift, self.pending, self.n = set(drift), {}, 0

    def generate(self, job_id, prompts, sampling_params, routing_key=None):
        self.n += 1
        rid = f"r{self.n}"
        out = []
        for p in prompts:
            n = int(sampling_params.get("max_tokens", 4)) if sampling_params.get("ignore_eos") else 3
            ids = [abs(hash((str(p), json.dumps(sampling_params, sort_keys=True), i))) % 1000 + 1 for i in range(n)]
            if job_id in self.drift and sampling_params.get("temperature") == 0.0:
                ids[-1] += 1
            r = {"token_ids": ids, "text": " ".join(map(str, ids)), "finish_reason": "stop",
                 "prompt_len": 10, "generation_len": n}
            if sampling_params.get("guided_json"):
                r["text"] = json.dumps({"capital": str(ids)})
            if sampling_params.get("logprobs") == 1:
                r["logprobs"] = [-0.5] * n
            elif sampling_params.get("logprobs"):
                r["logprobs"] = [{"a": -0.1, "b": -2.0}] * n
            if sampling_params.get("prompt_logprobs"):
                r["prompt_logprobs"] = [None] + [{"x": -1.0}] * 9
            out.append(r)
        self.pending[rid] = out
        return rid

    def poll_request(self, job_id, request_id):
        return {"results": self.pending.pop(request_id)}


def test_correctness_gates_matching_jobs_and_reports_stock_plain():
    from arctic_platform.inference.semi_persistence.onboarding import correctness

    jobs_ = {"on_hit": "H", "off_default": "D", "on_miss": "M", "off_matched": "X"}
    replicas = {"on_hit": 2, "off_default": 2, "on_miss": 1, "off_matched": 1}
    checks = correctness.run(_FakeClient(drift={"D"}), jobs_, "on_hit", {"on_miss", "off_matched"}, replicas, 128)
    gated = [c for c in checks if c["required"]]
    assert gated and all(c["passed"] for c in gated)
    greedy = checks[0]
    assert greedy["variants"]["off_default"]["passed"] is False and not greedy["variants"]["off_default"]["gated"]
    assert "position 2" in greedy["variants"]["off_default"]["detail"]

    checks = correctness.run(_FakeClient(drift={"M"}), jobs_, "on_hit", {"on_miss", "off_matched"}, replicas, 128)
    assert checks[0]["passed"] is False and "on_miss" in checks[0]["detail"]
    table = "\n".join(report.correctness_table(checks))
    assert "semi_p, forced miss vs reference" in table and "**differs**" in table


def test_cleanup_finds_jobs_never_cancelled(tmp_path):
    from arctic_platform.inference.semi_persistence.onboarding import cleanup

    ledger = tmp_path / "jobs.jsonl"
    ledger.write_text("\n".join(json.dumps(e) for e in [
        {"job_id": "a", "model": "m", "variant": "on_hit"},
        {"job_id": "b", "model": "m", "variant": "off_default"},
        {"job_id": "a", "cancelled_at": "t"},
        {"job_id": "c", "model": "n", "variant": "on_miss"}]) + "\n")
    assert cleanup.open_jobs(ledger) == {"b": "m/off_default", "c": "n/on_miss"}


def test_footprint_counts_whole_nodes(monkeypatch, tmp_path):
    from arctic_platform.inference.semi_persistence.onboarding import onboard

    monkeypatch.setattr(onboard, "make_client", lambda *a, **k: type("C", (), {"poll_backoff_multiplier": 2.0})())
    runner = onboard.Runner({"config": "x", "out_dir": str(tmp_path)})
    assert runner.footprint(_spec(tp=2, n_gpus=8)) == (56, 7)  # 16 + 16 + 8 + 16 GPUs, 2 + 2 + 1 + 2 nodes
    assert runner.footprint(_spec(tp=1, n_gpus=1)) == (3, 3)   # three 1-GPU jobs, one node each
    assert runner.footprint(_spec(tp=16, n_gpus=16)) == (48, 6)  # off_matched skipped: 16 x 3


def test_perf_breakdown_splits_settings_from_semi_p():
    def c(conc, srv, tok):
        return {"concurrency": conc, "prompt": 16, "output": 16, "server_p50": srv, "tok_s": tok}

    rows = [{"on": c(1, 0.12, 100), "off": c(1, 0.10, 100), "matched": c(1, 0.11, 100)},
            {"on": c(8, 0.24, 720), "off": c(8, 0.20, 800), "matched": c(8, 0.22, 760)}]
    lat = {s: {"ttft_short_s": t, "ttft_long_s": t, "tpot_s": t / 10} for s, t in
           (("on", 0.060), ("off", 0.050), ("matched", 0.055))}
    settings, own = perf._compare(rows, lat, "matched", "off"), perf._compare(rows, lat, "on", "matched")
    assert settings["worst_throughput_ratio"] == pytest.approx(0.20 / 0.22)
    assert own["worst_throughput_ratio"] == pytest.approx(0.22 / 0.24)
    assert settings["ttft_gap_ms"] == pytest.approx(5.0) and own["tpot_ratio"] == pytest.approx(0.06 / 0.055)
    table = "\n".join(report.breakdown_table({"settings": settings, "semi_p": own}))
    assert "semi_p's collective settings" in table and "semi_p itself" in table


def test_perf_score_gates_engine_throughput_and_reports_client():
    def c(conc, out, srv, tok):
        return {"concurrency": conc, "prompt": 16, "output": out, "server_p50": srv, "tok_s": tok,
                "client_p50": srv + 0.25, "errors": 0, "zero_id_responses": 0}

    # Server latency within 5% everywhere; one client cell 0.79, as the API path produces.
    rows = [{"on": c(1, 96, 0.40, 112), "off": c(1, 96, 0.38, 111)},
            {"on": c(4, 96, 0.42, 364), "off": c(4, 96, 0.40, 417)},
            {"on": c(16, 16, 0.136, 360), "off": c(16, 16, 0.130, 457)}]
    lat = {s: {"ttft_short_s": 0.05, "ttft_long_s": 0.05, "tpot_s": 0.0035, "errors": 0} for s in ("on", "off")}
    scored = perf.score(rows, lat, 4, {})
    checks = {x["name"]: x for x in scored["checks"]}
    engine = checks["median engine throughput ratio, concurrency > 1"]
    client = checks["worst client throughput ratio, concurrency > 1"]
    assert engine["passed"] and engine["value"] == pytest.approx((0.40 / 0.42 + 0.130 / 0.136) / 2)
    assert client["gated"] is False and not client["passed"] and client["value"] == pytest.approx(360 / 457)
    assert scored["scaling"]["on"] == pytest.approx(0.40 / 0.42)
    assert report.verdict({"perf": scored})[0] == "PASS"
    assert report._perf_status(client) == "info (below)"


def test_perf_score_passes_a_gap_plain_matched_shows_too():
    def c(srv, tok):
        return {"concurrency": 2, "prompt": 16, "output": 96, "server_p50": srv, "tok_s": tok,
                "client_p50": srv + 0.2, "errors": 0, "zero_id_responses": 0}

    rows = [{"on": c(1.10, 170), "off": c(1.00, 190), "matched": c(1.10, 170)}]
    lat = {s: {"ttft_short_s": 0.05, "ttft_long_s": 0.05, "tpot_s": t, "errors": 0}
           for s, t in (("on", 0.0110), ("off", 0.0098), ("matched", 0.0112))}
    tpot = {x["name"]: x for x in perf.score(rows, lat, 1, {})["checks"]}["TPOT ratio"]
    assert tpot["passed"] and "passes against plain matched" in tpot["limit"]
    del rows[0]["matched"], lat["matched"]
    assert not {x["name"]: x for x in perf.score(rows, lat, 1, {})["checks"]}["TPOT ratio"]["passed"]


def test_guided_json_that_does_not_parse_passes_only_when_plain_matched_agrees():
    from arctic_platform.inference.semi_persistence.onboarding import correctness

    def check(**variants):
        return {"passed": True, "detail": "",
                "variants": {v: {"passed": ok, "detail": "", "gated": True} for v, ok in variants.items()}}

    agrees, differs, absent = check(off_matched=True), check(off_matched=False), check(on_miss=True)
    for c in (agrees, differs, absent):
        correctness.apply_guided_parse_rule(c, parses=False)
    assert agrees["passed"] and not differs["passed"] and not absent["passed"]
    assert "no off_matched job" in absent["detail"]
    parsed = check(on_miss=True)
    correctness.apply_guided_parse_rule(parsed, parses=True)
    assert parsed["passed"]


def test_publish_body_asks_a_miss_to_dump_and_publish():
    from arctic_platform.inference.semi_persistence.onboarding import publish

    spec = _spec(tp=2, extra_env={"A": "1"})
    body = publish.publish_body(spec, "dev_X")
    ic = body["sub_job_configs"][0]["inference_config"]
    assert body["debug"] == {"job": {"image_tag": "dev_X"}}
    assert ic["semi_p"] is True
    assert ic["vllm_config"]["extra_env"] == {"A": "1", "SEMIP_DUMP_PUBLISH": "1"}
    assert "semi_p" not in spec.sub_job["inference_config"]
    assert spec.sub_job["inference_config"]["vllm_config"]["extra_env"] == {"A": "1"}


_KEY = "/data-fast/image-cache_neutrino/aaaaaaaaaaaa_bbbbbbbbbbbb"


def _engine_log(*messages):
    return [f"12:00:{i:02d} [r0 engine] semi_p: {m}" for i, m in enumerate(messages)]


def test_publish_outcome_waits_while_the_engine_dumps_and_publishes():
    from arctic_platform.inference.semi_persistence.onboarding import publish

    lines = _engine_log(f"resolved model_dir={_KEY} (config=...)",
                        "cache miss -- cold-starting to dump",
                        f"dump complete model_dir={_KEY} in 412.0s",
                        f"restore complete model_dir={_KEY} gpus=[0]",
                        f"publishing {_KEY} to s3://bucket (log /tmp/x.log): python3 ...")
    assert publish.publish_outcome(lines) is None


def test_publish_outcome_reads_the_published_skeleton():
    from arctic_platform.inference.semi_persistence.onboarding import publish

    lines = _engine_log(f"resolved model_dir={_KEY} (config=...)", "cache miss -- cold-starting",
                        f"published {_KEY} in 655s:")
    lines.append("  skeleton s3://b/image-cache/Qwen3.8-27B/skeleton/"
                 "aaaaaaaaaaaa_bbbbbbbbbbbb_cccccccccccc/ expect it on every node")
    out = publish.publish_outcome(lines)
    assert out["status"] == "published" and out["key_dir"] == _KEY and out["publish_s"] == 655
    assert out["slug"] == "Qwen3.8-27B" and out["weight_hash"] == "cccccccccccc"


def test_publish_outcome_reports_a_publish_the_engine_declined_or_failed():
    from arctic_platform.inference.semi_persistence.onboarding import publish

    declined = publish.publish_outcome(_engine_log(
        f"dumped {_KEY} but not publishing it: the key is already published as /m/s"))
    failed = publish.publish_outcome(_engine_log(f"publishing {_KEY} failed (rc=1) after 30s; ..."))
    assert declined["status"].startswith("FAILED: semi_p: dumped")
    assert failed["status"].startswith("FAILED: semi_p: publishing")


def test_publish_outcome_needs_nothing_for_a_restore_from_the_mirror():
    from arctic_platform.inference.semi_persistence.onboarding import publish

    lines = _engine_log(f"resolved model_dir={_KEY} (config=...)",
                        f"materialized {_KEY} from /mnt/x: 3 file(s)",
                        f"restore complete model_dir={_KEY} gpus=[0]")
    assert publish.publish_outcome(lines)["status"].startswith("already published")


def test_cli_helpers():
    assert parse_kv_args(["a=1", "b=x=y"]) == {"a": "1", "b": "x=y"}
    with pytest.raises(SystemExit):
        parse_kv_args(["nokey"])
    assert log_text({"_raw": "r"}) == "r"
    assert log_text({"msg": "m", "message": "n"}) == "m"
    assert log_text("plain") == "plain"
