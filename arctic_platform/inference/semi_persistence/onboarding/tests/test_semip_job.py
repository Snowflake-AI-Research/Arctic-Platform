# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the semip-publish driver (no network: GitHub and the child commands are faked).

    python -m pytest arctic_platform/inference/semi_persistence/onboarding/tests
"""

import io
import json
import urllib.error

import pytest

from arctic_platform.inference.semi_persistence.onboarding import onboard
from arctic_platform.inference.semi_persistence.onboarding import publish
from arctic_platform.inference.semi_persistence.onboarding import semip_job

SHA = "c55112cb595918a6e927f67510dc0cb7f83a96fb"
OWNER = "sfc-gh-mhidayetoglu"
BASE = f"https://github.com/{OWNER}/cortex-training/blob"
PERMALINK = f"{BASE}/{SHA}/model-catalog/semip_models/jobs.jsonl"
BRANCH_LINK = f"{BASE}/mert/semip-jobs-sample/model-catalog/semip_models/jobs.jsonl"
TAG = "release_20261119_163502_6ed8c6d2361-h"


def _line(model, n_gpus=8, tp=8):
    """A semi_p job body as a user submits it; its label is ``<model>-32k-full``."""
    sub = {"job_type": "sampling", "model_name": f"org/{model}", "inference_config": {
        "max_seq_len": 32768, "n_gpus": n_gpus, "vllm_config": {"tensor_parallel_size": tp}, "semi_p": True}}
    return json.dumps({"hardware": "H200", "sub_job_configs": [sub]})


BIG, NODE, ONE = "big-32k-full", "node-32k-full", "one-32k-full"
CONFIGS = "\n".join([_line("big", 16, 16), _line("node"), "", _line("one", 1, 1)]) + "\n"


class _GitHub:
    """Answers the commits API for ``refs`` and serves ``CONFIGS`` at ``SHA``; records every GET."""

    def __init__(self, refs=None, text=CONFIGS):
        self.refs = {"mert/semip-jobs-sample": SHA, **(refs or {})}
        self.text, self.calls = text, []

    def __call__(self, url, accept=None):
        self.calls.append((url, accept))
        api = f"https://api.github.com/repos/{OWNER}/cortex-training/commits/"
        if url.startswith(api):
            ref = semip_job.urllib.parse.unquote(url[len(api):])
            if ref not in self.refs:
                raise ValueError(f"GET {url}: HTTP 422 Unprocessable Entity")
            return self.refs[ref].encode()
        if url == f"https://raw.githubusercontent.com/{OWNER}/cortex-training/{SHA}/model-catalog/semip_models/jobs.jsonl":
            return self.text.encode()
        raise ValueError(f"GET {url}: HTTP 404 Not Found")


@pytest.fixture
def connection(tmp_path):
    path = tmp_path / "qa6.config"
    path.write_text(json.dumps({"host": "acct.snowflakecomputing.com", "pat": "SECRET-PAT"}))
    return str(path)


def _args(tmp_path, connection, **extra):
    args = {"cluster": "qa6", "image_tag": TAG, "configs": PERMALINK, "max_gpus": "16",
            "connection": connection, "out_dir": str(tmp_path / "out")}
    return [f"{k}={v}" for k, v in {**args, **extra}.items()]


def _no_submission(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("submitted in a dry run")

    monkeypatch.setattr(publish, "Publisher", refuse)
    monkeypatch.setattr(onboard, "Runner", refuse)
    monkeypatch.setattr(semip_job.subprocess, "Popen", refuse)
    return refuse


@pytest.mark.parametrize("url,ref,path", [
    (PERMALINK, SHA, "model-catalog/semip_models/jobs.jsonl"),
    (BRANCH_LINK, "mert/semip-jobs-sample", "model-catalog/semip_models/jobs.jsonl"),
    (f"{BASE}/main/model-catalog/semip_models/h200/all.jsonl", "main", "model-catalog/semip_models/h200/all.jsonl"),
    (f" {PERMALINK}#L3 ", SHA, "model-catalog/semip_models/jobs.jsonl"),
])
def test_configs_links_split_at_the_configs_directory(url, ref, path):
    configs = semip_job.parse_configs_url(url)
    assert (configs.owner, configs.repo, configs.ref, configs.path) == (OWNER, f"{OWNER}/cortex-training", ref, path)


@pytest.mark.parametrize("url,error", [
    (PERMALINK.replace("https://github.com", "https://www.github.com"), "https://github.com/ link"),
    (PERMALINK.replace("https://", "http://"), "https://github.com/ link"),
    (f"https://raw.githubusercontent.com/{OWNER}/cortex-training/{SHA}/model-catalog/semip_models/jobs.jsonl",
     "https://github.com/ link"),
    (PERMALINK.replace("/cortex-training/", "/cortex-training-fork/"), "cortex-training repository"),
    (PERMALINK.replace("/blob/", "/tree/"), "blob link"),
    (f"https://github.com/{OWNER}/cortex-training", "blob link"),
    (PERMALINK.replace("semip_models", "models"), "under model-catalog/semip_models/"),
    (f"{BASE}/{SHA}/model-catalog/semip_models/", "under model-catalog/semip_models/"),
    (PERMALINK.replace(".jsonl", ".json"), "a .jsonl file"),
    (f"{BASE}/{SHA}/model-catalog/semip_models/../models/x.jsonl", "a .jsonl file"),
    (PERMALINK + "?plain=1", "without a query"),
    ("", "https://github.com/ link"),
])
def test_bad_configs_links_are_refused(url, error):
    with pytest.raises(SystemExit, match=error):
        semip_job.parse_configs_url(url)


def test_a_commit_is_used_as_is_and_a_branch_is_resolved_through_the_api():
    github = _GitHub()
    assert semip_job.resolve_ref(semip_job.parse_configs_url(PERMALINK), github) == SHA and github.calls == []
    assert semip_job.resolve_ref(semip_job.parse_configs_url(BRANCH_LINK), github) == SHA
    assert github.calls == [(f"https://api.github.com/repos/{OWNER}/cortex-training/commits/mert%2Fsemip-jobs-sample",
                             "application/vnd.github.sha")]
    with pytest.raises(ValueError, match="HTTP 422"):
        semip_job.resolve_ref(semip_job.parse_configs_url(BRANCH_LINK.replace("mert/semip-jobs-sample", "gone")),
                              github)
    with pytest.raises(ValueError, match="expected a commit SHA"):
        semip_job.resolve_ref(semip_job.parse_configs_url(BRANCH_LINK), _GitHub(refs={"mert/semip-jobs-sample": "<html>"}))


def test_http_get_sends_github_token_and_keeps_it_out_of_errors(monkeypatch):
    seen = []

    def urlopen(request, timeout):
        seen.append(dict(request.header_items()))
        if "missing" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return io.BytesIO(b"ok")

    monkeypatch.setattr(semip_job.urllib.request, "urlopen", urlopen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert semip_job.http_get("https://api.github.com/x", "application/vnd.github.sha") == b"ok"
    assert "Authorization" not in seen[-1] and seen[-1]["Accept"] == "application/vnd.github.sha"
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SECRET")
    semip_job.http_get("https://api.github.com/x")
    assert seen[-1]["Authorization"] == "Bearer ghp_SECRET"
    with pytest.raises(ValueError) as err:
        semip_job.http_get("https://raw.githubusercontent.com/missing")
    assert "HTTP 404" in str(err.value) and "SECRET" not in str(err.value)


@pytest.mark.parametrize("value,tests,options", [
    ("", [], None),
    (None, [], None),
    ("functional", ["functional"], {"variants": "on_hit", "checks": "none"}),
    ("correctness", ["functional", "correctness"], {"variants": "on_hit,off_default", "checks": "correctness"}),
    ("performance", ["functional", "performance"], {"variants": "on_hit,off_default", "checks": "perf"}),
    ("performance, correctness", ["functional", "correctness", "performance"],
     {"variants": "on_hit,off_default", "checks": "correctness,perf"}),
    ("functional,correctness,performance", ["functional", "correctness", "performance"],
     {"variants": "on_hit,off_default", "checks": "correctness,perf"}),
])
def test_tests_imply_functional_and_map_to_onboard_variants_and_checks(value, tests, options):
    assert semip_job.parse_tests(value) == tests
    assert semip_job.onboard_options(tests) == options


def test_unknown_tests_are_refused():
    with pytest.raises(SystemExit, match="perf"):
        semip_job.parse_tests("functional,perf")


@pytest.mark.parametrize("tag", [TAG, "release_20260101_000000_0123456789a-a", "release_20260101_000000_0123456789a-b"])
def test_image_tags(tag):
    assert semip_job.parse_image_tag(tag) == tag


@pytest.mark.parametrize("tag", ["", None, "dev_abc", TAG[:-2], TAG + "x", TAG.replace("-h", "-c"),
                                 TAG.replace("6ed8c6d2361", "6ed8c6d236"), TAG.replace("6ed8c6d2361", "6ED8C6D2361"),
                                 " " + TAG])
def test_bad_image_tags_are_refused(tag):
    with pytest.raises(SystemExit, match="image_tag="):
        semip_job.parse_image_tag(tag)


def test_max_gpus_is_a_positive_integer():
    assert semip_job.parse_max_gpus("16") == 16 and semip_job.parse_max_gpus("1") == 1
    for bad in ("", None, "0", "-8", "8.0", "016", "x", "1e3"):
        with pytest.raises(SystemExit, match="max_gpus="):
            semip_job.parse_max_gpus(bad)


@pytest.mark.parametrize("cluster,context,bucket", [("qa6", "sfc-or-dev-meta-k8s-2", "neutrino-sfc-or-dev-meta-k8s-2"),
                                                    ("prod", "sfc-or-prod-meta-k8s-2", "neutrino-sfc-or-prod-meta-k8s-2")])
def test_the_cluster_sets_the_context_bucket_and_connection(monkeypatch, tmp_path, cluster, context, bucket):
    monkeypatch.delenv("SEMIP_CONNECTION", raising=False)
    monkeypatch.setenv(f"SEMIP_CONNECTION_{cluster.upper()}", str(tmp_path / f"{cluster}.json"))
    monkeypatch.setenv(f"SEMIP_CONNECTION_{'PROD' if cluster == 'qa6' else 'QA6'}", "/wrong")
    args = {"cluster": cluster, "image_tag": TAG, "configs": PERMALINK, "max_gpus": "8", "out_dir": str(tmp_path)}
    job = semip_job.SemipJob(args)
    assert (job.kube_context, job.model_bucket, job.connection) == (context, bucket, str(tmp_path / f"{cluster}.json"))
    assert semip_job.SemipJob({**args, "kube_context": "other"}).kube_context == "other"
    assert semip_job.SemipJob({**args, "connection": str(tmp_path / "c.json")}).connection == str(tmp_path / "c.json")
    monkeypatch.delenv(f"SEMIP_CONNECTION_{cluster.upper()}")
    monkeypatch.setenv("SEMIP_CONNECTION", str(tmp_path / "any.json"))
    assert semip_job.SemipJob(args).connection == str(tmp_path / "any.json")
    with pytest.raises(SystemExit, match="cluster="):
        semip_job.SemipJob({**args, "cluster": "qa7"})


def test_a_dry_run_records_the_configs_and_the_plan_and_submits_nothing(monkeypatch, tmp_path, connection, capsys):
    _no_submission(monkeypatch)
    github = _GitHub()
    code = semip_job.main(_args(tmp_path, connection, configs=BRANCH_LINK, tests="correctness", dry_run="true"),
                          get=github, run_child=_no_submission(monkeypatch))
    assert code == 0
    out = tmp_path / "out"
    assert (out / "configs.jsonl").read_text() == CONFIGS
    record = json.loads((out / "configs.json").read_text())
    assert {k: record[k] for k in ("repo", "ref", "sha", "path", "jobs")} == {
        "repo": f"{OWNER}/cortex-training", "ref": "mert/semip-jobs-sample", "sha": SHA,
        "path": "model-catalog/semip_models/jobs.jsonl", "jobs": 3}
    plan = json.loads((out / "plan.json").read_text())
    assert (plan["cluster"], plan["kube_context"], plan["model_bucket"]) == (
        "qa6", "sfc-or-dev-meta-k8s-2", "neutrino-sfc-or-dev-meta-k8s-2")
    assert plan["tests"] == ["functional", "correctness"] and plan["dry_run"] is True
    assert [(j["line"], j["name"], j["model_name"], j["max_seq_len"], j["n_gpus"], j["tensor_parallel_size"],
             j["test_gpus"], j["onboard"]) for j in plan["jobs"]] == [
        (1, BIG, "org/big", 32768, 16, 16, 32, "needs 32 GPUs on 4 nodes; max_gpus=16"),
        (2, NODE, "org/node", 32768, 8, 8, 32, "needs 32 GPUs on 4 nodes; max_gpus=16"),
        (4, ONE, "org/one", 32768, 1, 1, 2, "yes")]
    pub, onb = plan["commands"]["publish"], plan["commands"]["onboard"]
    for want in (f"config={connection}", f"image_tag={TAG}", f"models={out / 'configs.jsonl'}", "max_gpus=16",
                 "reserve_gpus=0", "model_bucket=neutrino-sfc-or-dev-meta-k8s-2", f"out_dir={out / 'publish'}"):
        assert want in pub.split()
    for want in (f"debug_image_tag={TAG}", f"models={out / 'onboard.jsonl'}", "variants=on_hit,off_default",
                 "checks=correctness", "require_restore=true", "max_gpus=16", "reserve_gpus=0",
                 "kube_context=sfc-or-dev-meta-k8s-2", f"publish_dir={out / 'publish'}", f"out_dir={out / 'onboard'}"):
        assert want in onb.split()
    assert (out / "onboard.jsonl").read_text() == CONFIGS.splitlines()[3] + "\n"
    assert any("2 of 3 config(s) need more than max_gpus=16" in w for w in plan["warnings"])
    text = (out / "plan.txt").read_text()
    assert f"@{SHA} model-catalog/semip_models/jobs.jsonl (requested ref mert/semip-jobs-sample)" in text
    assert f"| 4 | {ONE} | org/one | 32768 | 1 | 1 | 2 | yes |" in text
    assert f"publish: {pub}" in text and f"onboard: {onb}" in text
    assert "SECRET-PAT" not in text + json.dumps(plan) + capsys.readouterr().err


def test_a_publish_only_dry_run_plans_no_onboard(monkeypatch, tmp_path, connection):
    run_child = _no_submission(monkeypatch)
    assert semip_job.main(_args(tmp_path, connection, dry_run="true"), get=_GitHub(), run_child=run_child) == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["onboard"] is None and "onboard" not in plan["commands"] and plan["warnings"] == []
    assert all("test_gpus" not in j for j in plan["jobs"])


def test_a_dry_run_without_a_connection_warns(monkeypatch, tmp_path):
    for env in ("SEMIP_CONNECTION", "SEMIP_CONNECTION_QA6"):
        monkeypatch.delenv(env, raising=False)
    run_child = _no_submission(monkeypatch)
    args = [a for a in _args(tmp_path, "", dry_run="true") if not a.startswith("connection=")]
    assert semip_job.main(args, get=_GitHub(), run_child=run_child) == 0
    plan = json.loads((tmp_path / "out" / "plan.json").read_text())
    assert plan["warnings"] == ["no connection file: pass connection= or set SEMIP_CONNECTION"]
    with pytest.raises(SystemExit, match="no connection file"):
        semip_job.main([a for a in args if not a.startswith("dry_run=")], get=_GitHub(), run_child=run_child)


def test_a_budget_below_a_configs_n_gpus_is_refused_before_anything_runs(monkeypatch, tmp_path, connection):
    run_child = _no_submission(monkeypatch)
    with pytest.raises(SystemExit, match=rf"max_gpus=8 is below the n_gpus of 1 config\(s\): {BIG} \(16\)"):
        semip_job.main(_args(tmp_path, connection, max_gpus="8", dry_run="true"), get=_GitHub(), run_child=run_child)
    assert not (tmp_path / "out" / "plan.json").exists()


def test_bad_configs_files_fail_the_dry_run_with_their_line(monkeypatch, tmp_path, connection):
    run_child = _no_submission(monkeypatch)
    for text, error in ((CONFIGS + _line("one", 1, 1) + "\n", "configs.jsonl:5: the same job as line 4"),
                        (CONFIGS + _line("one", 1, 1).replace("true", "false") + "\n",
                         "configs.jsonl:5: inference_config.semi_p must be true"),
                        ("not json\n", "configs.jsonl:1: not JSON")):
        with pytest.raises(SystemExit, match=error):
            semip_job.main(_args(tmp_path, connection, dry_run="true"), get=_GitHub(text=text), run_child=run_child)
    with pytest.raises(SystemExit, match="HTTP 404"):
        semip_job.main(_args(tmp_path, connection, dry_run="true",
                             configs=PERMALINK.replace("jobs.jsonl", "other.jsonl")), get=_GitHub(), run_child=run_child)


def test_fit_onboard_mirrors_onboards_gpu_and_node_budget(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text("\n".join([_line("big", 16, 16), _line("node"), _line("one", 1, 1), _line("two", 1, 1)]) + "\n")
    big, node, one, two = semip_job.load_models(str(path))
    fits, too_big = semip_job.fit_onboard([big, node, one], ["on_hit"], 16)
    assert [s.name for s in fits] == [BIG, NODE, ONE] and too_big == {}
    fits, too_big = semip_job.fit_onboard([big, node, one], ["on_hit", "off_default"], 16)
    assert [s.name for s in fits] == [ONE] and set(too_big) == {BIG, NODE}
    # Two 1-GPU jobs take two nodes; onboard's default node budget is max_gpus // 8 + one per config.
    assert semip_job.fit_onboard([one], ["on_hit", "off_default"], 4) == (
        [], {ONE: "needs 2 GPUs on 2 nodes; max_gpus=4"})
    assert semip_job.fit_onboard([one, two], ["on_hit", "off_default"], 4) == ([one, two], {})


def _report(name, restored=True, cache_miss=False, correctness=(), perf=(), error=None, not_measured=None):
    r = {"model": {"name": name}, "started_at": "2099-01-01T00:00:00",
         "startup": {"semi_p": {"markers_visible": True, "restored": restored, "cache_miss": cache_miss},
                     "on_timeline": {"submit_to_running_s": 30.0}},
         "correctness": [{"name": n, "required": True, "passed": p} for n, p in correctness],
         "perf": {"checks": [{"name": n, "passed": p} for n, p in perf]}}
    if error:
        r["error"] = error
    if not_measured:
        r["not_measured"] = not_measured
    return r


def test_a_run_publishes_then_tests_what_published_and_fits(monkeypatch, tmp_path, connection):
    calls = []

    def run_child(argv):
        calls.append(argv)
        module = argv[2].rsplit(".", 1)[1]
        opts = dict(a.split("=", 1) for a in argv[3:])
        if module == "publish":
            summary = tmp_path / "out" / "publish" / TAG / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({BIG: {"status": "published"}, NODE: {"status": "FAILED: boom"},
                                           ONE: {"status": "already published (restored from it); nothing to publish"}}))
            return 1
        out = tmp_path / "out" / "onboard"
        out.mkdir()
        assert [s.name for s in semip_job.load_models(opts["models"])] == [ONE] and opts["checks"] == "correctness"
        (out / f"{ONE}-20990101T000000.json").write_text(json.dumps(
            _report(ONE, correctness=[("greedy", True), ("stop strings", False)])))
        (out / "stale.json").write_text(json.dumps({**_report(BIG), "started_at": "2000-01-01T00:00:00"}))
        return 1

    code = semip_job.main(_args(tmp_path, connection, tests="correctness"), get=_GitHub(), run_child=run_child)
    assert code == 1 and [c[2].rsplit(".", 1)[1] for c in calls] == ["publish", "onboard"]
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    rows = {r["job"]: (r["publish"], r["tests"]) for r in result["jobs"]}
    assert rows == {
        BIG: ("published", {"functional": "not run: needs 32 GPUs on 4 nodes; max_gpus=16",
                            "correctness": "not run: needs 32 GPUs on 4 nodes; max_gpus=16"}),
        NODE: ("FAILED: boom", {"functional": "not run: not published", "correctness": "not run: not published"}),
        ONE: ("already published (restored from it); nothing to publish",
              {"functional": "PASS", "correctness": "FAIL: stop strings"})}
    assert result["failed"] == [BIG, NODE, ONE]
    md = (tmp_path / "out" / "result.md").read_text()
    assert md.startswith("| job | publish | functional | correctness |") and f"| {ONE} | already published" in md


def test_a_publish_only_run_passes_when_every_config_published(monkeypatch, tmp_path, connection):
    def run_child(argv):
        assert argv[2].endswith(".publish")
        summary = tmp_path / "out" / "publish" / TAG / "summary.json"
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({n: {"status": "published"} for n in (BIG, NODE, ONE)}))
        return 0

    assert semip_job.main(_args(tmp_path, connection), get=_GitHub(), run_child=run_child) == 0
    assert json.loads((tmp_path / "out" / "result.json").read_text())["failed"] == []


def test_a_missing_publish_summary_fails_every_config(monkeypatch, tmp_path, connection):
    assert semip_job.main(_args(tmp_path, connection), get=_GitHub(), run_child=lambda argv: 2) == 1
    rows = json.loads((tmp_path / "out" / "result.json").read_text())["jobs"]
    assert {r["publish"] for r in rows} == {"missing (publish exited 2)"}


@pytest.mark.parametrize("result,tests,want", [
    (_report("m"), ["functional"], {"functional": "PASS"}),
    (_report("m", restored=False, cache_miss=True, error="RuntimeError: semi_p job cold-started"), ["functional"],
     {"functional": "FAIL: on_hit cold-started (cache miss)"}),
    (_report("m", restored=False), ["functional"], {"functional": "FAIL: not every replica restored from the published image"}),
    ({**_report("m", restored=False), "startup": {"semi_p": {"markers_visible": False},
                                                  "on_timeline": {"submit_to_running_s": 1.0}}},
     ["functional"], {"functional": "WARN: restore markers not in the logs; restore not confirmed"}),
    ({**_report("m"), "startup": {}, "error": "RuntimeError: on_hit job j ended failed: no capacity"}, ["functional"],
     {"functional": "FAIL: RuntimeError: on_hit job j ended failed: no capacity"}),
    (_report("m", correctness=[("greedy", True)], perf=[("ttft", True)]), ["functional", "correctness", "performance"],
     {"functional": "PASS", "correctness": "PASS", "performance": "PASS"}),
    (_report("m", correctness=[("greedy", True)], perf=[("ttft", False), ("tpot", False)]),
     ["functional", "correctness", "performance"],
     {"functional": "PASS", "correctness": "PASS", "performance": "FAIL: ttft, tpot"}),
    (_report("m", correctness=[("greedy", True)], error="x",
             not_measured={"off_default": "perf and correctness: RuntimeError: ended failed"}),
     ["functional", "correctness"],
     {"functional": "PASS", "correctness": "FAIL: not measured (off_default: perf and correctness: RuntimeError: ended failed)"}),
    ({**_report("m"), "perf": {}, "perf_note": "not measured: no plain job"}, ["functional", "performance"],
     {"functional": "PASS", "performance": "FAIL: not measured (not measured: no plain job)"}),
    (None, ["functional", "correctness"], {"functional": "FAIL: no onboard report (onboard exited 1)",
                                           "correctness": "FAIL: no onboard report (onboard exited 1)"}),
])
def test_verdicts_per_test_from_an_onboard_report(result, tests, want):
    assert semip_job.verdicts(result, tests, 1) == want


def test_a_signal_is_passed_on_to_the_child_and_ends_the_driver(monkeypatch):
    sent = []

    class Proc:
        def __init__(self, argv, cwd, env):
            assert env["PYTHONPATH"].split(semip_job.os.pathsep)[0] == str(semip_job.REPO_ROOT)
            self.handler = None

        def send_signal(self, signum):
            sent.append(signum)

        def wait(self):
            semip_job.signal.getsignal(semip_job.signal.SIGTERM)(semip_job.signal.SIGTERM, None)
            return -15

    monkeypatch.setattr(semip_job.subprocess, "Popen", Proc)
    before = semip_job.signal.getsignal(semip_job.signal.SIGTERM)
    with pytest.raises(SystemExit) as exit_:
        semip_job.child_process(["python", "-m", "x"])
    assert exit_.value.code == 128 + semip_job.signal.SIGTERM and sent == [semip_job.signal.SIGTERM]
    assert semip_job.signal.getsignal(semip_job.signal.SIGTERM) == before
