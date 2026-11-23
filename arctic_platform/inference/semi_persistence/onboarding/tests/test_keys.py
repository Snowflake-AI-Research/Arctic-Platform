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

"""Tests for the offline skeleton keys and the node-label lookup.

``fixtures/published_keys.jsonl`` holds published jobs: the job body, the
``vllm_config`` the engine logged for it and the skeleton publish wrote. The
engine-side steps are held to their source here, read with ``ast`` because the
engine modules import vLLM, Ray and NVML.
"""

import ast
import hashlib
import json
import logging
import re
from pathlib import Path

import pytest

from arctic_platform.inference.semi_persistence.onboarding import keys
from arctic_platform.inference.semi_persistence.onboarding import semip_job

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[2] / "server"
PUBLISHED = [json.loads(line) for line in (HERE / "fixtures" / "published_keys.jsonl").read_text().splitlines()]


def _sub(case):
    return case["job"]["sub_job_configs"][0]


def _source(path: Path, *names: str) -> dict:
    """The top-level functions and assignments ``names`` of ``path``, executed on their own."""
    tree = ast.parse(path.read_text())
    wanted = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
              or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    assert {getattr(n, "name", None) or n.targets[0].id for n in wanted} == set(names)
    namespace = {"json": json, "hashlib": hashlib, "re": re, "logger": logging.getLogger("test"), "Any": object}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("case", PUBLISHED, ids=[c["label"] for c in PUBLISHED])
def test_cfg12_of_published_jobs(case):
    sub, cfg12 = _sub(case), case["published"].split("_")[0]
    k = keys.keying(sub, case["job"]["hardware"])
    assert k.reason is None and k.slug == case["vllm_config"]["model"].rsplit("/", 1)[1]
    if keys.reasoner_of(sub)[0]:
        assert (k.cfg12, k.reasoning_parser) == (cfg12, case["reasoning_parser"]) and not k.candidates
    else:
        assert k.cfg12 is None and set(k.candidates) == set(keys.CHAT_REASONERS)
        assert k.candidates[case["reasoning_parser"]] == cfg12
        assert len(set(k.candidates.values())) == len(keys.CHAT_REASONERS)
    n_gpus, nnodes = keys.topology(sub)
    kwargs = keys.engine_kwargs(keys.dss_kwargs(sub), multi_node=nnodes > 1)
    assert keys.vllm_config(keys.restore_kwargs(kwargs, case["reasoning_parser"])) == case["vllm_config"]


def test_the_two_pod_job_hashes_nnodes_and_the_mp_backend():
    case = next(c for c in PUBLISHED if c["job"]["sub_job_configs"][0]["inference_config"]["n_gpus"] == 16)
    assert keys.topology(_sub(case)) == (16, 2)
    assert case["vllm_config"]["distributed_executor_backend"] == "mp" and "nnodes" not in case["vllm_config"]
    binding = keys.WHOLE_NODE_DEVICES["H200"]
    assert keys.config_hash({**case["vllm_config"], "nnodes": 2}, binding) == case["published"][:12]
    assert keys.config_hash(case["vllm_config"], binding) != case["published"][:12]


def test_hash_binding_and_projection_match_the_engine_source():
    engine = _source(SERVER / "semip_engine.py", "_HASH_LEN", "_config_hash", "_GPUS_PER_NODE",
                     "_TP1_REPLICA_LAYOUT", "_tp1_replica_layout", "_NON_VLLM_ENGINE_KEYS",
                     "_vllm_config_from_engine_kwargs", "_DEVICE_BOUND_MIN_TP", "_DEVICE_GLOBS")
    engine["_strip_kwargs_vllm_rejects"] = lambda c: {k: v for k, v in c.items() if k not in keys.VLLM_REJECTED}
    assert (engine["_HASH_LEN"], engine["_DEVICE_BOUND_MIN_TP"]) == (keys.HASH_LEN, keys.DEVICE_BOUND_MIN_TP)
    assert set(engine["_NON_VLLM_ENGINE_KEYS"]) == set(keys.NON_VLLM_ENGINE_KEYS)
    assert engine["_DEVICE_GLOBS"] == ("/dev/nvidia[0-9]*", "/dev/infiniband/uverbs[0-9]*")
    for replicas, n_gpus in ((8, 8), (2, 2), (4, 4)):
        assert keys.device_binding({"tensor_parallel_size": 1}, n_gpus, "H200") == engine["_tp1_replica_layout"](replicas)
    for case in PUBLISHED:
        config = case["vllm_config"]
        for binding in (None, ["replica-layout"], keys.WHOLE_NODE_DEVICES["H200"][::-1]):
            assert keys.config_hash(config, binding) == engine["_config_hash"](config, binding)
        sub = _sub(case)
        kwargs = keys.restore_kwargs(keys.engine_kwargs(keys.dss_kwargs(sub), keys.topology(sub)[1] > 1),
                                     case["reasoning_parser"])
        nested = {**kwargs, "fp32_lm_head": False, "ray_num_gpus": 0.0,
                  "extra_engine_kwargs": {"router_replay_max_cache_bytes": 1}}
        assert keys.vllm_config(nested) == engine["_vllm_config_from_engine_kwargs"](nested) == config


def _model_config_fields():
    tree = ast.parse((SERVER / "config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelConfig")
    dumped, excluded = {}, []
    for node in cls.body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)) or node.target.id == "model":
            continue
        value = node.value
        if isinstance(value, ast.Call) and getattr(value.func, "id", None) == "Field":
            kw = {k.arg: k.value for k in value.keywords}
            if "exclude" in kw and ast.literal_eval(kw["exclude"]):
                excluded.append(node.target.id)
                continue
            value = kw["default"]
        dumped[node.target.id] = ast.literal_eval(value)
    return dumped, excluded


def test_model_config_defaults_match_config_py():
    dumped, excluded = _model_config_fields()
    assert dumped == {name: default for name, (_, default) in keys.MODEL_CONFIG_FIELDS.items()}
    assert set(excluded) == set(keys.MODEL_CONFIG_EXCLUDED)


def test_chat_reasoners_are_chat_models_reasoners():
    tree = ast.parse((SERVER / "chat.py").read_text())
    named = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "ChatModel" and n.args and isinstance(n.args[0], ast.Constant)}
    assert named | {None} == set(keys.CHAT_REASONERS)


def test_worker_pops_and_defaults_are_the_workers():
    worker = (SERVER / "worker.py").read_text()
    assert f'"worker_extension_cls",\n            "{keys.WORKER_EXTENSION_CLS}"' in worker
    for key in keys.WORKER_POPPED:
        assert f'"{key}"' in worker, key
    frontend = (SERVER / "semip_frontend.py").read_text()
    assert 'kwargs.setdefault("disable_log_stats", False)' in frontend
    assert 'kwargs.setdefault("distributed_executor_backend", "mp")' in (SERVER / "replica_pool.py").read_text()


def _job(**ic):
    vc = ic.pop("vllm_config", {"tensor_parallel_size": 1})
    return {"job_type": "sampling", "model_name": "Qwen/Qwen3-0.6B",
            "inference_config": {"max_seq_len": 32768, "n_gpus": 1, "semi_p": True, "vllm_config": vc, **ic}}


@pytest.mark.parametrize("sub,hardware,why", [
    (_job(n_gpus=2, vllm_config={"tensor_parallel_size": 2}), "H200", "whichever GPUs the pod gets"),
    (_job(n_gpus=8, vllm_config={"tensor_parallel_size": 8}), "B200", "whole B200 node is not known"),
    (_job(n_gpus=32, vllm_config={"tensor_parallel_size": 16}), "H200", "not one engine over whole nodes"),
    (_job(n_gpus=3, vllm_config={"tensor_parallel_size": 2}), "H200", "not a multiple of the world size"),
    (_job(max_num_batched_tokens=None), "H200", "vLLM's default"),
    (_job(text_config_overrides={"a": 1}), "H200", "not replayed offline"),
    (_job(vllm_config={"speculative_config": {"model": "x/y"}}), "H200", "speculative_config"),
    (_job(vllm_config={"fp32_lm_head": True}), "H200", "refused by the semi_p dump"),
    (_job(vllm_config={"swap_space": 4}), "H200", "not a ModelConfig field"),
    (_job(vllm_config={"extra_env": {"ARCTIC_INFERENCE_ENABLED": "1"}}), "H200", "ARCTIC_INFERENCE_ENABLED"),
    ({**_job(), "resume_checkpoint_tag": "step10"}, "H200", "loads a checkpoint"),
])
def test_jobs_whose_cfg12_is_not_known_offline_say_why(sub, hardware, why):
    k = keys.keying(sub, hardware)
    assert k.cfg12 is None and not k.candidates and why in k.reason


def test_dss_defaults_reach_the_config():
    base = keys.vllm_config(keys.restore_kwargs(keys.engine_kwargs(keys.dss_kwargs(_job()), False), None))
    assert (base["max_model_len"], base["max_num_batched_tokens"], base["gpu_memory_utilization"]) == (32768, 8192, 0.95)
    unset = _job()
    del unset["inference_config"]["max_seq_len"]
    assert keys.dss_kwargs(unset).get("max_model_len") is None
    coerced = keys.engine_kwargs(keys.dss_kwargs(_job(vllm_config={"gpu_memory_utilization": 1,
                                                                   "max_num_seqs": "64"})), False)
    assert (coerced["gpu_memory_utilization"], coerced["max_num_seqs"]) == (1.0, 64)
    lora = {"peft_type": "Lora", "target_modules": ["q_proj"]}
    assert keys.lora_rank({**lora, "r": 33}) == 64 and keys.lora_rank(lora) == 8


# Labels read off QA6 for published skeletons, with the keys they name.
LIVE = {
    "image-cache.Qwen3-0.6B.skeleton.a65858a869ba_fc37441c0c-ec92de0":
        "image-cache/Qwen3-0.6B/skeleton/a65858a869ba_fc37441c0c27_4aa63a1e185a",
    "image-cache.DeepSeek-V4.1-Flash-1017.skeleton.ed76d4778-5cc75fd":
        "image-cache/DeepSeek-V4.1-Flash-1017/skeleton/ed76d47780e3_fc37441c0c27_4008d4360385",
    "image-cache.GLM-5.3.skeleton.44c0d7a2a3c1_fc37441c0c27-17159ff":
        "image-cache/GLM-5.3/skeleton/44c0d7a2a3c1_fc37441c0c27_c65f56541db7",
    "image-cache.Inkling-2-Small-MXFP8.skeleton.ef8257249a3c-1ae167d":
        "image-cache/Inkling-2-Small-MXFP8/skeleton/ef8257249a3c_263a697cd91b_1b4fc9b3215d",
    "image-cache.NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.skeleto-e26e8c2":
        "image-cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/skeleton/876cb6ac4a4f_fc37441c0c27_885c2ca97043",
    "image-cache.Qwen3-0.6B.weight.4aa63a1e185a": "image-cache/Qwen3-0.6B/weight/4aa63a1e185a",
    "image-cache.NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.weight-d510c85":
        "image-cache/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/weight/885c2ca97043",
}


@pytest.mark.parametrize("label,directory", LIVE.items())
def test_labels_follow_the_node_sync_rule(label, directory):
    assert keys._publisher().sanitize_label_segment(directory) == label


def _node(*segments, value="ready", **other):
    labels = {keys.LABEL_PREFIX + s: value for s in segments}
    return {"metadata": {"name": "n", "labels": {**labels, **other}}}


def _label(slug, key, kind="skeleton"):
    return keys._publisher().sanitize_label_segment(f"image-cache/{slug}/{kind}/{key}")


QWEN, DS, INK, NEMO = "Qwen3-0.6B", "DeepSeek-V4.1-Flash-1017", "Inkling-2-Small-MXFP8", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ENV, OLD = "fc37441c0c27", "263a697cd91b"


def _cluster():
    """Four cache nodes and one without model labels. Every env12 is cut from the skeleton
    labels below, except OLD's on GLM-5.3, so the others are confirmed through their digests."""
    common = [_label(QWEN, "4aa63a1e185a", "weight"), _label(DS, "4008d4360385", "weight"),
              _label(INK, "1b4fc9b3215d", "weight"),
              _label("GLM-5.3", f"44c0d7a2a3c1_{OLD}_c65f56541db7"), _label("GLM-5.3", f"44c0d7a2a3c1_{ENV}_c65f56541db7")]
    full = [_label(QWEN, f"a65858a869ba_{ENV}_4aa63a1e185a"), _label(QWEN, f"a65858a869ba_{OLD}_4aa63a1e185a"),
            _label(DS, f"ed76d47780e3_{OLD}_4008d4360385"), _label(INK, f"ef8257249a3c_{'9' * 12}_1b4fc9b3215d"),
            _label(NEMO, f"876cb6ac4a4f_{ENV}_885c2ca97043")]
    partial = [_label(DS, f"ed76d47780e3_{ENV}_4008d4360385")]
    syncing = {keys.LABEL_PREFIX + partial[0]: "syncing"}
    return [_node(*common, *full, *partial), _node(*common, *full, *partial), _node(*common, *full),
            _node(*common, *full, **syncing), {"metadata": {"labels": {"workergroup": "cpu"}}}]


def test_lookup_confirms_cut_labels_and_counts_nodes():
    cache = keys.NodeCache(_cluster())
    assert cache.nodes == 4 and cache.envs == {ENV, OLD}
    found = {f.env: (f.env_exact, f.nodes) for f in cache.lookup(QWEN, "a65858a869ba").found}
    assert found == {ENV: (True, 4), OLD: (True, 4)}
    # cfg12 cut to 9 characters and env12 not shown at all.
    assert {f.env: f.nodes for f in cache.lookup(DS, "ed76d47780e3").found} == {OLD: 4, ENV: 2}
    # Cut labels show 9 characters of DeepSeek's cfg12, so only those tell its skeletons apart.
    assert cache.lookup(DS, "ed76d4770000").found == [] and cache.lookup(QWEN, "a65858a869bb").found == []
    # An env12 no label shows in full stays unconfirmed.
    ink = cache.lookup(INK, "ef8257249a3c")
    assert [(f.env, f.env_exact, f.nodes) for f in ink.found] == [("", False, 4)]
    nemo = cache.lookup(NEMO, "876cb6ac4a4f")
    assert nemo.found == [] and nemo.unreadable == 1
    assert keys.NodeCache(_cluster(), env_hash="9" * 12).lookup(INK, "ef8257249a3c").found[0].env == "9" * 12


def test_an_uncut_label_is_read_as_is():
    cache = keys.NodeCache([_node(_label("m", f"{'a' * 12}_{ENV}_{'b' * 12}"))])
    assert len(_label("m", f"{'a' * 12}_{ENV}_{'b' * 12}")) <= 63
    assert [(f.env, f.env_exact, f.nodes) for f in cache.lookup("m", "a" * 12).found] == [(ENV, True, 1)]


JOBS = "\n".join(json.dumps({"hardware": "H200", "sub_job_configs": [c["job"]["sub_job_configs"][0]]})
                 for c in PUBLISHED if c["label"] in ("qwen3-0.6b-32k-full", "deepseek-v4.1-flash-1017-64k-lora")) + "\n"


SHA = "ec015a7f4b954c09ba21f30c5533f3d8c43ac11f"
CONFIGS = f"https://github.com/o/cortex-training/blob/{SHA}/model-catalog/semip_models/jobs.jsonl"


def _dry_run(tmp_path, monkeypatch, nodes, **extra):
    def get(url, accept=None):
        assert url == f"https://raw.githubusercontent.com/o/cortex-training/{SHA}/model-catalog/semip_models/jobs.jsonl"
        return JOBS.encode()

    def refuse(*_a, **_k):
        raise AssertionError("submitted in a dry run")

    monkeypatch.setattr(semip_job.subprocess, "Popen", refuse)
    (tmp_path / "c.json").write_text(json.dumps({"host": "h", "pat": "p"}))
    args = {"cluster": "qa6", "image_tag": "release_20261119_163502_6ed8c6d2361-h", "configs": CONFIGS,
            "max_gpus": "16", "connection": str(tmp_path / "c.json"), "out_dir": str(tmp_path / "out"),
            "dry_run": "true", **extra}
    assert semip_job.main([f"{k}={v}" for k, v in args.items()], get=get, run_child=refuse, nodes=nodes) == 0
    return json.loads((tmp_path / "out" / "plan.json").read_text()), (tmp_path / "out" / "plan.txt").read_text()


def test_a_dry_run_reports_existing_skeletons_for_this_image(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _cluster(), env_hash=ENV)
    qwen, ds = plan["jobs"]
    assert (qwen["cfg12"], qwen["reasoning_parser"], qwen["reasoning_parser_from"]) == ("a65858a869ba", "qwen3",
                                                                                         "node labels")
    assert (qwen["skeleton"], ds["skeleton"]) == ("restore", "partial")
    assert ("| 1 | qwen3-0.6b-32k-full | Qwen/Qwen3-0.6B | 32768 | 1 | 1 | a65858a869ba | "
            "fc37441c0c: 4/4 nodes (exists); exists for another image: 263a697cd9 4/4 |") in text
    assert "| ed76d47780e3 | fc37441c0c: 2/4 nodes (exists); exists for another image: 263a697cd9 4/4 |" in text
    assert "skeletons: 1 will restore (all nodes), 1 exist but not on all nodes, 0 will dump\n" in text
    assert "node cache  4 model-cache node(s) on sfc-or-dev-meta-k8s-2, env_hash fc37441c0c27" in text
    assert plan["node_cache"]["summary"] == {"restore": 1, "partial": 1, "dump": 0, "unknown": 0, "other_image": 0}


def test_a_dry_run_without_env_hash_leaves_the_env_unconfirmed(tmp_path, monkeypatch):
    nodes = [_node(_label(QWEN, "4aa63a1e185a", "weight"), _label(QWEN, f"a65858a869ba_{OLD}_4aa63a1e185a"),
                   _label("GLM-5.3", f"44c0d7a2a3c1_{OLD}_c65f56541db7"))]
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes)
    assert "| a65858a869ba | 263a697cd9: 1/1 nodes (env unconfirmed) |" in text
    assert "| ed76d47780e3 | none |" in text and plan["jobs"][1]["skeleton"] == "dump"
    assert text.count("skeletons: 1 will restore (all nodes), 0 exist but not on all nodes, 1 will dump "
                      "(env unconfirmed: pass env_hash= to match this image)") == 1
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes, env_hash=ENV)
    assert plan["jobs"][0]["skeleton"] == "dump" and "0 will restore" in text
    assert "| a65858a869ba | exists for another image: 263a697cd9 1/1 |" in text
    assert "2 will dump (1 of them exist for another image)" in text


def test_a_failing_kubectl_skips_the_lookup_with_a_note(tmp_path, monkeypatch):
    def fail(argv, **kwargs):
        assert argv[:4] == ["kubectl", "--context=sfc-or-dev-meta-k8s-2", "--request-timeout=30s", "get"]
        return type("P", (), {"returncode": 1, "stdout": "", "stderr": "error: context not found\n"})()

    monkeypatch.setattr(semip_job.shutil, "which", lambda name: "/bin/kubectl")
    with pytest.raises(semip_job.NodesUnavailable, match="get nodes failed: error: context not found"):
        semip_job.cluster_nodes("sfc-or-dev-meta-k8s-2", fail)
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: semip_job.cluster_nodes(ctx, fail))
    assert plan["node_cache"]["checked"] is False and "skeletons: not checked (kubectl --context=" in text
    assert plan["jobs"][1]["cfg12"] == "ed76d47780e3" and plan["jobs"][0]["cfg12"] is None


def test_env_hash_must_be_twelve_hex():
    assert semip_job.parse_env_hash(ENV) == ENV and semip_job.parse_env_hash("") is None
    for bad in ("fc37441c0c2", "FC37441C0C27", "fc37441c0c27x"):
        with pytest.raises(SystemExit, match="env_hash="):
            semip_job.parse_env_hash(bad)
