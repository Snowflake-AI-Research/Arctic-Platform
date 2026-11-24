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


TAG = "release_20261119_163502_6ed8c6d2361-h"
OLD_TAG = "release_20261115_163548_99d52631769-h"
# The digests QA6's nodes list for TAG and OLD_TAG; with driver 580.159.03 they give ENV and OLD.
DIGEST = "a544c618585e5f020aef245387f5f6ec4206d696a4c6e3ca8787fee8d9aa3358"
OLD_DIGEST = "437137899d34d1b7603dae5017be5fd422ac658d77db8475b0e18b40e926566d"
QA6_REPO = "937559181915.dkr.ecr.us-west-2.amazonaws.com/neutrino/dss-backend"


def _pulled(node, *images, repo=QA6_REPO):
    """``node`` listing each ``(tag, digest)`` image the way kubelet reports it in status.images."""
    listed = [{"names": [f"{repo}@sha256:{digest}", f"{repo}:{tag}"], "sizeBytes": 9252435633}
              for tag, digest in images]
    return {**node, "status": {"images": [*(node.get("status") or {}).get("images", []), *listed]}}


def _label(slug, key, kind="skeleton"):
    return keys._publisher().sanitize_label_segment(f"image-cache/{slug}/{kind}/{key}")


QWEN, DS, INK, NEMO = "Qwen3-0.6B", "DeepSeek-V4.1-Flash-1017", "Inkling-2-Small-MXFP8", "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ENV, OLD = "fc37441c0c27", "263a697cd91b"


def _cluster(images=((TAG, DIGEST), (OLD_TAG, OLD_DIGEST))):
    """Four cache nodes and one without model labels; the first two list ``images``. Every
    env12 is cut from the skeleton labels below, except on GLM-5.3, so the others are
    confirmed through their weight hashes."""
    nodes = _labelled_cluster()
    return [_pulled(n, *images) if i < 2 else n for i, n in enumerate(nodes)]


def _labelled_cluster():
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
    assert keys.NodeCache(_cluster(), envs=["9" * 12]).lookup(INK, "ef8257249a3c").found[0].env == "9" * 12
    assert cache.shows_env(ENV, 6) == "whole" and cache.shows_env("9" * 12, 6) is None


def test_env_hash_matches_the_engine_source_and_the_published_keys():
    engine = _source(SERVER / "semip_engine.py", "_HASH_LEN", "_NVIDIA_VERSION_PROC", "_NVRM_VERSION_RE",
                     "_driver_version", "_env_hash")
    engine["_pod_image_ref"] = lambda: f"{QA6_REPO}@sha256:{DIGEST}"
    engine["_read_text"] = lambda path: ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.159.03  "
                                         "Release Build  (dvs-builder@U22-I3-AF03-09-1)  Thu Oct 16 2025")
    assert engine["_env_hash"]() == (ENV, f"{QA6_REPO}@sha256:{DIGEST}", "580.159.03")
    assert engine["_NVRM_VERSION_RE"].pattern.endswith(f"({keys.DRIVER_VERSION.pattern})")
    assert keys.env_hash(DIGEST, "580.159.03") == ENV and keys.env_hash(OLD_DIGEST, "580.159.03") == OLD
    assert semip_job.DEFAULT_DRIVER_VERSION == "580.159.03"


def test_image_digests_come_from_the_tag_under_any_registry():
    prod = "686874466970.dkr.ecr.us-west-2.amazonaws.com/neutrino/dss-backend"
    nodes = [_pulled(_node(), (TAG, DIGEST), (OLD_TAG, OLD_DIGEST)), _pulled(_node(), (TAG, DIGEST), repo=prod),
             _pulled(_node(), (TAG, "b" * 64)), _pulled(_node(), (TAG + "x", "c" * 64)), _node(),
             {"metadata": {}, "status": {"images": [{"names": [f"{QA6_REPO}:{TAG}"]}]}}]
    assert keys.image_digests(nodes, TAG) == {DIGEST: 2, "b" * 64: 1}
    assert keys.image_digests(nodes, OLD_TAG) == {OLD_DIGEST: 1}
    assert keys.image_digests(nodes, "release_20260101_000000_0123456789a-h") == {}


@pytest.mark.parametrize("labels,want", [
    ({"nvidia.com/cuda.driver-version.full": "580.159.03"}, ("580.159.03", "nvidia.com/cuda.driver-version.full")),
    ({"nvidia.com/cuda.driver-version.major": "580", "nvidia.com/cuda.driver-version.minor": "159",
      "nvidia.com/cuda.driver-version.revision": "03"}, ("580.159.03", "nvidia.com/cuda.driver-version.*")),
    ({"nvidia.com/cuda.driver.major": "570", "nvidia.com/cuda.driver.minor": "86", "nvidia.com/cuda.driver.rev": ""},
     ("570.86", "nvidia.com/cuda.driver.*")),
    ({"nvidia.com/cuda.driver-version.full": "garbage"}, None),
    ({"nvidia.com/gpu.product": "NVIDIA-H200"}, None),
])
def test_a_node_driver_comes_from_gpu_feature_discovery_labels(labels, want):
    assert keys.node_driver(labels) == want


def test_node_drivers_counts_model_cache_nodes_only():
    gfd = {"nvidia.com/cuda.driver-version.full": "580.159.03"}
    nodes = [_node("m", **gfd), _node("m", **gfd), _node("m", **{"nvidia.com/cuda.driver-version.full": "570.1.2"}),
             {"metadata": {"labels": {**gfd, "workergroup": "cpu"}}}, {"metadata": {"labels": {"x": "1"}}}]
    assert keys.node_drivers(nodes) == ({"580.159.03": 2, "570.1.2": 1}, "nvidia.com/cuda.driver-version.full")
    assert keys.node_drivers(_cluster()) == ({}, None)


def test_an_uncut_label_is_read_as_is():
    cache = keys.NodeCache([_node(_label("m", f"{'a' * 12}_{ENV}_{'b' * 12}"))])
    assert len(_label("m", f"{'a' * 12}_{ENV}_{'b' * 12}")) <= 63
    assert [(f.env, f.env_exact, f.nodes) for f in cache.lookup("m", "a" * 12).found] == [(ENV, True, 1)]


JOBS = "\n".join(json.dumps({"hardware": "H200", "sub_job_configs": [c["job"]["sub_job_configs"][0]]})
                 for c in PUBLISHED if c["label"] in ("qwen3-0.6b-32k-full", "deepseek-v4.1-flash-1017-64k-lora")) + "\n"


SHA = "ec015a7f4b954c09ba21f30c5533f3d8c43ac11f"
OFFICIAL = f"https://github.com/snowflakedb/cortex-training/blob/{SHA}/model-catalog/semip_models/jobs.jsonl"
FORK = OFFICIAL.replace("/snowflakedb/", "/o/")
COMPARE = f"https://api.github.com/repos/snowflakedb/cortex-training/compare/{SHA}...main"


def _dry_run(tmp_path, monkeypatch, nodes, configs=OFFICIAL, **extra):
    """A dry run of ``JOBS``; the official catalog's ``SHA`` is on main."""
    owner = configs.split("/")[3]

    def get(url, accept=None):
        if url == COMPARE:
            return json.dumps({"status": "ahead"}).encode()
        assert url == f"https://raw.githubusercontent.com/{owner}/cortex-training/{SHA}/model-catalog/semip_models/jobs.jsonl"
        return JOBS.encode()

    def refuse(*_a, **_k):
        raise AssertionError("submitted in a dry run")

    monkeypatch.setattr(semip_job.subprocess, "Popen", refuse)
    (tmp_path / "c.json").write_text(json.dumps({"host": "h", "pat": "p"}))
    args = {"cluster": "qa6", "image_tag": "release_20261119_163502_6ed8c6d2361-h", "configs": configs,
            "max_gpus": "16", "connection": str(tmp_path / "c.json"), "out_dir": str(tmp_path / "out"),
            "dry_run": "true", **extra}
    assert semip_job.main([f"{k}={v}" for k, v in args.items()], get=get, run_child=refuse, nodes=nodes) == 0
    return json.loads((tmp_path / "out" / "plan.json").read_text()), (tmp_path / "out" / "plan.txt").read_text()


ENV_LINE = (f"env         {ENV} (digest a544c618585e from node images, driver 580.159.03 from "
            "DEFAULT_DRIVER_VERSION (assumed); a skeleton label carries it)\n")


def test_a_dry_run_reports_existing_skeletons_for_this_image(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _cluster())
    assert ENV_LINE in text and "node cache  4 model-cache node(s) on sfc-or-dev-meta-k8s-2\n" in text
    env = plan["env"]
    assert (env["env12"], env["digests"], env["drivers"], env["driver_source"], env["driver_assumed"]) == (
        [ENV], {DIGEST: 2}, {"580.159.03": 0}, "DEFAULT_DRIVER_VERSION", True)
    assert env["candidates"] == {ENV: {"digest": DIGEST, "driver": "580.159.03", "confirmed": "whole"}}
    assert plan["warnings"] == ["GPU driver assumed to be 580.159.03 (DEFAULT_DRIVER_VERSION): no read-only cluster "
                                f"source names it; a skeleton label on the nodes carries the env12 it gives ({ENV})"]
    assert "WARNING: GPU driver assumed to be 580.159.03" in text
    qwen, ds = plan["jobs"]
    assert (qwen["cfg12"], qwen["reasoning_parser"], qwen["reasoning_parser_from"]) == ("a65858a869ba", "qwen3",
                                                                                         "node labels")
    assert (qwen["skeleton"], ds["skeleton"]) == ("restore", "partial")
    assert ("| 1 | qwen3-0.6b-32k-full | Qwen/Qwen3-0.6B | 32768 | 1 | 1 | a65858a869ba | "
            "fc37441c0c: 4/4 nodes (exists); exists for another image: 263a697cd9 4/4 |") in text
    assert "| ed76d47780e3 | fc37441c0c: 2/4 nodes (exists); exists for another image: 263a697cd9 4/4 |" in text
    assert "skeletons: 1 will restore (all nodes), 1 exist but not on all nodes, 0 will dump\n" in text
    assert plan["node_cache"]["summary"] == {"restore": 1, "partial": 1, "dump": 0, "unknown": 0, "other_image": 0}
    assert "visibility  official catalog (snowflakedb/cortex-training main; ec015a7f4b95 is on main)\n" in text
    assert plan["visibility"] == {"official": True, "show_all": False,
                                  "reason": "official catalog (snowflakedb/cortex-training main; ec015a7f4b95 is on main)"}


def test_a_skeleton_of_another_images_env_is_not_this_images(tmp_path, monkeypatch):
    nodes = [_pulled(_node(_label(QWEN, "4aa63a1e185a", "weight"), _label(QWEN, f"a65858a869ba_{OLD}_4aa63a1e185a"),
                           _label("GLM-5.3", f"44c0d7a2a3c1_{OLD}_c65f56541db7")), (TAG, DIGEST))]
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes)
    assert plan["jobs"][0]["skeleton"] == "dump" and "0 will restore" in text
    assert "| a65858a869ba | exists for another image: 263a697cd9 1/1 |" in text
    assert "| ed76d47780e3 | none |" in text and plan["jobs"][1]["skeleton"] == "dump"
    assert "2 will dump (1 of them exist for another image)\n" in text
    assert f"{ENV} (digest a544c618585e from node images, driver 580.159.03 from DEFAULT_DRIVER_VERSION (assumed); " \
           "no skeleton label shows it)" in text
    assert plan["warnings"] == ["GPU driver assumed to be 580.159.03 (DEFAULT_DRIVER_VERSION): no read-only cluster "
                                f"source names it; no skeleton label confirms env12 {ENV}, so if the nodes run "
                                "another driver this image's skeletons are reported missing"]


NOT_PULLED_SUMMARY = "(image not pulled on any node yet, so no skeleton of it is on the cluster)"


def test_an_image_no_node_pulled_will_dump_every_job(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _cluster(images=((OLD_TAG, OLD_DIGEST),)))
    assert (f"env         none: {TAG} is not pulled on any node yet, so no skeleton of it can be on the cluster\n"
            in text)
    assert plan["env"]["pulled"] is False and plan["env"]["env12"] == [] and plan["warnings"] == []
    assert ("| a65858a869ba | none (will dump; image not pulled on any node yet); exists for another image: "
            "263a697cd9 4/4, fc37441c0c 4/4 |") in text
    assert [j["skeleton"] for j in plan["jobs"]] == ["dump", "dump"]
    assert ("skeletons: 0 will restore (all nodes), 0 exist but not on all nodes, 2 will dump (2 of them exist for "
            f"another image) {NOT_PULLED_SUMMARY}") in text


def test_a_restricted_list_of_an_image_no_node_pulled_will_dump_every_job(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _cluster(images=()), configs=FORK)
    assert text.count("| none (will dump; image not pulled on any node yet) |") == 2
    assert ("skeletons: 0 will restore (all nodes), 0 exist but not on all nodes, 2 will dump "
            f"{NOT_PULLED_SUMMARY}") in text
    assert all(j["skeleton"] == "dump" and j["existing"] == [] for j in plan["jobs"])
    for other in (ENV, OLD, ENV[:10], OLD[:10], "another image"):
        assert other not in json.dumps(plan) and other not in text, other


def test_a_tag_with_two_digests_gives_two_candidates(tmp_path, monkeypatch):
    nodes = _labelled_cluster()
    nodes[0], nodes[1] = _pulled(nodes[0], (TAG, DIGEST)), _pulled(nodes[1], (TAG, OLD_DIGEST))
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes, configs=FORK)
    assert plan["env"]["ambiguous"] is True and set(plan["env"]["env12"]) == {ENV, OLD}
    assert plan["env"]["digests"] == {OLD_DIGEST: 1, DIGEST: 1}
    assert "env         ambiguous, any of 2: " in text and "digest 437137899d34 from node images" in text
    assert any(w.startswith(f"{TAG} is listed with 2 digests across the nodes (tags can move): 437137899d34 on 1 "
                            "node(s), a544c618585e on 1 node(s)") for w in plan["warnings"])
    assert "| a65858a869ba | exists: 4/4 nodes |" in text and "| ed76d47780e3 | exists: 4/4 nodes |" in text


GFD = "nvidia.com/cuda.driver-version.full"


@pytest.mark.parametrize("driver,want", [("580.159.03", "exists: 4/4 nodes"), ("570.1.2", "none (will dump)")])
def test_a_driver_from_node_labels_is_used_instead_of_the_default(tmp_path, monkeypatch, driver, want):
    nodes = _cluster()
    for n in nodes[:4]:
        n["metadata"]["labels"][GFD] = driver
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes, configs=FORK)
    env = keys.env_hash(DIGEST, driver)
    assert (plan["env"]["drivers"], plan["env"]["driver_source"], plan["env"]["driver_assumed"]) == (
        {driver: 4}, f"node label {GFD}", False)
    assert f"env         {env} (digest a544c618585e from node images, driver {driver} from node label {GFD}; " in text
    assert plan["warnings"] == [] and "assumed" not in text
    assert f"| a65858a869ba | {want} |" in text if driver == "580.159.03" else f"| ? | {want} |" in text


def test_a_failing_kubectl_skips_the_lookup_with_a_note(tmp_path, monkeypatch):
    def fail(argv, **kwargs):
        assert argv[:4] == ["kubectl", "--context=sfc-or-dev-meta-k8s-2", "--request-timeout=30s", "get"]
        return type("P", (), {"returncode": 1, "stdout": "", "stderr": "error: context not found\n"})()

    monkeypatch.setattr(semip_job.shutil, "which", lambda name: "/bin/kubectl")
    with pytest.raises(semip_job.NodesUnavailable, match="get nodes failed: error: context not found"):
        semip_job.cluster_nodes("sfc-or-dev-meta-k8s-2", fail)
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: semip_job.cluster_nodes(ctx, fail))
    assert plan["node_cache"]["checked"] is False and "skeletons: not checked (kubectl --context=" in text
    assert "env         not derived (kubectl --context=sfc-or-dev-meta-k8s-2 get nodes failed: " in text
    assert plan["env"]["derived"] is False and plan["warnings"] == []
    assert plan["jobs"][1]["cfg12"] == "ed76d47780e3" and plan["jobs"][0]["cfg12"] is None


def _table(text):
    return [line for line in text.splitlines() if line.startswith("| ")]


def test_a_restricted_dry_run_reports_only_this_images_own_skeletons(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _cluster(), configs=FORK)
    reason = f"restricted: o/cortex-training@{SHA} is not the official catalog on main"
    assert f"visibility  {reason}\n" in text and ENV_LINE in text
    assert plan["visibility"] == {"official": False, "reason": reason, "show_all": False}
    qwen, ds = plan["jobs"]
    assert (qwen["cfg12"], qwen["reasoning_parser"], qwen["reasoning_parser_from"]) == ("a65858a869ba", "qwen3",
                                                                                         "node labels")
    assert "| a65858a869ba | exists: 4/4 nodes |" in text and "| ed76d47780e3 | exists: 2/4 nodes |" in text
    assert (qwen["skeleton"], ds["skeleton"]) == ("restore", "partial")
    assert ds["existing"] == [{"env": ENV, "nodes": 2, "of": 4, "match": "exists"}]
    assert "skeletons: 1 will restore (all nodes), 1 exist but not on all nodes, 0 will dump\n" in text
    dumped = json.dumps(plan)
    assert "env_candidates" not in plan["node_cache"] and plan["node_cache"]["restricted"] is True
    # The backend inventory lists GLM-5.3's skeleton of this env and counts Inkling's cut label, nothing else of them.
    assert [m["model"] for m in plan["backend_inventory"]["models"]] == [DS, "GLM-5.3", QWEN]
    assert plan["backend_inventory"]["unattributed"] == {"labels": 2, "models": [INK, NEMO], "kind_unknown": 0}
    for other in (OLD, OLD[:10], "another image", "9" * 12):
        assert other not in dumped and other not in text, other


def test_a_restricted_dry_run_says_none_for_another_env_and_never_names_the_others(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _cluster(images=((TAG, "0" * 64),)), configs=FORK)
    assert plan["env"]["env12"] == [keys.env_hash("0" * 64, "580.159.03")]
    assert "| ? | none (will dump) |" in text and "| ed76d47780e3 | none (will dump) |" in text
    assert [j["skeleton"] for j in plan["jobs"]] == ["dump", "dump"] and plan["jobs"][0]["existing"] == []
    assert "cfg12 ?: the job names no reasoning_parser and no candidate's skeleton was found for this image" in text
    assert "skeletons: 0 will restore (all nodes), 0 exist but not on all nodes, 2 will dump\n" in text
    for other in (ENV, ENV[:10], OLD, OLD[:10], "another image"):
        assert other not in json.dumps(plan) and other not in text, other


def test_a_restricted_dry_run_without_kubectl_says_why_it_is_not_checked(tmp_path, monkeypatch):
    def nodes(context):
        raise semip_job.NodesUnavailable("kubectl is not on PATH")

    plan, text = _dry_run(tmp_path, monkeypatch, nodes, configs=FORK)
    assert "| a65858a869ba |" not in text and text.count("| not checked: kubectl is not on PATH |") == 2
    assert "skeletons: not checked (kubectl is not on PATH)" in text
    assert "node cache  not checked: kubectl is not on PATH" in text
    assert "env         not derived (kubectl is not on PATH)" in text
    assert plan["node_cache"]["checked"] is False and plan["node_cache"]["summary"] is None
    assert all(j["existing"] == [] and j["skeleton"] == "not checked" for j in plan["jobs"])
    assert "cfg12 ?: the job names no reasoning_parser and the node labels were not checked" in text


def test_a_restricted_dry_run_cannot_confirm_a_label_cut_too_short(tmp_path, monkeypatch):
    nodes = [_pulled(_node(_label(DS, f"ed76d47780e3_{OLD}_4008d4360385")), (TAG, DIGEST))]
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes, configs=FORK)
    assert plan["jobs"][1]["skeleton"] == "unknown" and OLD[:6] not in json.dumps(plan)
    assert f"| ed76d47780e3 | ? (labels of {DS} are cut too short to confirm this image's skeleton) |" in text


def test_a_restricted_job_with_no_label_of_its_cfg12_will_dump(tmp_path, monkeypatch):
    nodes = [_pulled(_node(_label(QWEN, "4aa63a1e185a", "weight"), _label(QWEN, f"a65858a869ba_{ENV}_4aa63a1e185a")),
                     (TAG, DIGEST))]
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes, configs=FORK)
    # DeepSeek's labels would show no env12 and no weight hash, but it has no label at all.
    assert "| a65858a869ba | exists: 1/1 nodes |" in text and "| ed76d47780e3 | none (will dump) |" in text
    assert plan["env"]["candidates"][ENV]["confirmed"] == "start" and "; a cut skeleton label shows its start)" in text
    assert plan["warnings"][0].endswith("a cut skeleton label on the nodes shows the start of the env12 it gives "
                                        f"({ENV})")


def test_show_all_restores_the_full_output_for_a_restricted_list(tmp_path, monkeypatch, caplog):
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
    official, official_text = _dry_run(tmp_path / "a", monkeypatch, lambda ctx: _cluster())
    with caplog.at_level(logging.WARNING, logger=semip_job.__name__):
        plan, text = _dry_run(tmp_path / "b", monkeypatch, lambda ctx: _cluster(), configs=FORK, show_all="true")
    assert _table(text) == _table(official_text) and "exists for another image: 263a697cd9 4/4" in text
    assert plan["node_cache"]["env_candidates"] == official["node_cache"]["env_candidates"] == sorted([ENV, OLD])
    assert plan["visibility"]["official"] is False and plan["visibility"]["show_all"] is True
    assert "is not the official catalog on main; show_all=true (admin override)\n" in text
    assert any("show_all=true (admin override)" in r.getMessage() for r in caplog.records)


LIGHT = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
OTHER_TAG, OTHER_DIGEST = "dev_20261115_234921_a2c56b581ba-h", "e" * 64
OTHER = keys.env_hash(OTHER_DIGEST, "580.159.03")
QWEN_OLD_ONLY, GLM_OLD_ONLY, M_OLD_ONLY = "0123456789ab", "b74ddf49aa1e", "c" * 12


def _two_env_cluster(images=((TAG, DIGEST), (OLD_TAG, OLD_DIGEST), (OTHER_TAG, OTHER_DIGEST))):
    """Four cache nodes with skeletons of ENV and OLD, their labels cut every way, and one node
    without model labels; the first two list ``images``. Every label's shape:

    Qwen3-0.6B  cut inside env12 (10 shown); 4b84547e3782 on 2 nodes; one cfg12 only under OLD
    DeepSeek    cut inside cfg12 (9 shown), completed by trying every value; ENV on 3 nodes
    GLM-5.3     cut after a whole env12, wt12 from its weight label
    m           not cut
    mini        cut inside wt12 (3 shown), whose weight label is not on the nodes
    Inkling     cut after cfg12: one of OTHER (an image the nodes list), one of an unknown env12
    Nemotron    cut inside ``skeleton``, its weight label cut too: unattributable
    Lightning   cut before saying skeleton or weight: unattributable
    """
    weights = [_label(QWEN, "4aa63a1e185a", "weight"), _label(DS, "4008d4360385", "weight"),
               _label("GLM-5.3", "c65f56541db7", "weight"), _label("m", "b" * 12, "weight"),
               _label(INK, "1b4fc9b3215d", "weight"), _label(NEMO, "885c2ca97043", "weight"), f"nvidia.{LIGHT}"]
    every = [_label(QWEN, f"a65858a869ba_{ENV}_4aa63a1e185a"), _label(QWEN, f"a65858a869ba_{OLD}_4aa63a1e185a"),
             _label(QWEN, f"{QWEN_OLD_ONLY}_{OLD}_4aa63a1e185a"), _label(DS, f"ed76d47780e3_{OLD}_4008d4360385"),
             _label("GLM-5.3", f"44c0d7a2a3c1_{ENV}_c65f56541db7"),
             _label("GLM-5.3", f"{GLM_OLD_ONLY}_{OLD}_c65f56541db7"),
             _label("m", f"{'a' * 12}_{ENV}_{'b' * 12}"), _label("m", f"{M_OLD_ONLY}_{OLD}_{'b' * 12}"),
             _label("mini", f"{'d' * 12}_{ENV}_{'e' * 12}"),
             _label(INK, f"ef8257249a3c_{OTHER}_1b4fc9b3215d"), _label(INK, f"155abe39cfc9_{'9' * 12}_1b4fc9b3215d"),
             _label(NEMO, f"876cb6ac4a4f_{ENV}_885c2ca97043"), _label(LIGHT, f"{'f' * 12}_{ENV}_{'1' * 12}")]
    two = [_label(QWEN, f"4b84547e3782_{ENV}_4aa63a1e185a")]
    three = [_label(DS, f"ed76d47780e3_{ENV}_4008d4360385")]
    nodes = [_node(*weights, *every, *two, *three), _node(*weights, *every, *two, *three),
             _node(*weights, *every, *three), _node(*weights, *every), {"metadata": {"labels": {"workergroup": "cpu"}}}]
    return [_pulled(n, *images) if i < 2 else n for i, n in enumerate(nodes)]


def test_the_fixture_labels_are_cut_where_the_docstring_says():
    shown = {slug: _label(slug, key).rsplit("-", 1)[0][len(f"image-cache.{slug}."):] for slug, key in (
        (QWEN, f"a65858a869ba_{ENV}_4aa63a1e185a"), (DS, f"ed76d47780e3_{ENV}_4008d4360385"),
        ("GLM-5.3", f"44c0d7a2a3c1_{ENV}_c65f56541db7"), ("mini", f"{'d' * 12}_{ENV}_{'e' * 12}"),
        (INK, f"155abe39cfc9_{'9' * 12}_1b4fc9b3215d"), (NEMO, f"876cb6ac4a4f_{ENV}_885c2ca97043"))}
    assert shown == {QWEN: f"skeleton.a65858a869ba_{ENV[:10]}", DS: "skeleton.ed76d4778",
                     "GLM-5.3": f"skeleton.44c0d7a2a3c1_{ENV}", "mini": f"skeleton.{'d' * 12}_{ENV}_eee",
                     INK: "skeleton.155abe39cfc9", NEMO: "skeleto"}
    assert _label("m", f"{'a' * 12}_{ENV}_{'b' * 12}") == f"image-cache.m.skeleton.{'a' * 12}_{ENV}_{'b' * 12}"
    assert _label(LIGHT, "x").rsplit("-", 1)[0] == f"image-cache.{LIGHT}"
    assert keys.repository_digests(_two_env_cluster(), TAG) == {DIGEST, OLD_DIGEST, OTHER_DIGEST}
    assert keys.repository_digests(_two_env_cluster(), "release_20260101_000000_0123456789a-h") == set()


def test_inventory_reads_every_models_skeletons_of_one_env():
    cache = keys.NodeCache(_two_env_cluster(), [ENV])
    inv = cache.inventory({ENV}, others={OTHER})
    assert [(s.slug, s.cfg12, s.env12, s.wt12, s.wt_exact, s.nodes) for s in inv.skeletons] == [
        (DS, "ed76d47780e3", ENV, "4008d4360385", True, 3),
        ("GLM-5.3", "44c0d7a2a3c1", ENV, "c65f56541db7", True, 4),
        ("m", "a" * 12, ENV, "b" * 12, True, 4),
        ("mini", "d" * 12, ENV, "eee", False, 4),
        (QWEN, "4b84547e3782", ENV, "4aa63a1e185a", True, 2),
        (QWEN, "a65858a869ba", ENV, "4aa63a1e185a", True, 4)]
    assert inv.weights == {(DS, "4008d4360385"): 4, ("GLM-5.3", "c65f56541db7"): 4, ("m", "b" * 12): 4,
                           ("mini", "eee"): None, (QWEN, "4aa63a1e185a"): 4}
    assert inv.unattributed == {INK: 1, NEMO: 1, LIGHT: 1} and inv.either == 1
    # Without the other image's env12 its Inkling label cannot be told either.
    assert cache.inventory({ENV}).unattributed == {INK: 2, NEMO: 1, LIGHT: 1}
    old = cache.inventory({OLD}, others={OTHER})
    assert {(s.slug, s.cfg12) for s in old.skeletons} == {
        (QWEN, "a65858a869ba"), (QWEN, QWEN_OLD_ONLY), (DS, "ed76d47780e3"), ("GLM-5.3", GLM_OLD_ONLY), ("m", M_OLD_ONLY)}
    assert old.unattributed == inv.unattributed


INVENTORY_ROWS = [
    f"backend inventory (env {ENV}, image {TAG})",
    "| model | kind | cfg12 | wt12 | nodes | this list |",
    "|---|---|---|---|---|---|",
    f"| {DS} | skeleton | ed76d47780e3 | 4008d4360385 | 3/4 | in this list |",
    f"| {DS} | weights | - | 4008d4360385 | 4/4 | |",
    "| GLM-5.3 | skeleton | 44c0d7a2a3c1 | c65f56541db7 | 4/4 | |",
    "| GLM-5.3 | weights | - | c65f56541db7 | 4/4 | |",
    f"| m | skeleton | {'a' * 12} | {'b' * 12} | 4/4 | |",
    f"| m | weights | - | {'b' * 12} | 4/4 | |",
    f"| mini | skeleton | {'d' * 12} | eee.. | 4/4 | |",
    "| mini | weights | - | eee.. | ? | |",
    f"| {QWEN} | skeleton | 4b84547e3782 | 4aa63a1e185a | 2/4 | |",
    f"| {QWEN} | skeleton | a65858a869ba | 4aa63a1e185a | 4/4 | in this list |",
    f"| {QWEN} | weights | - | 4aa63a1e185a | 4/4 | |",
    "backend inventory: 6 skeletons across 5 models (4 on all nodes), 5 weight sets",
    f"3 skeleton label(s) of models {INK}, {NEMO}, {LIGHT} could not be attributed to an image (labels cut; 1 of "
    "them are cut before saying skeleton or weight, so they may be weight labels)",
]


def _inventory_text(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("backend inventory"))
    return lines[start:lines.index("", start)]


def test_a_restricted_dry_run_lists_this_envs_skeletons_of_every_model_and_nothing_else(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _two_env_cluster(), configs=FORK)
    assert plan["visibility"]["official"] is False
    assert _inventory_text(text) == INVENTORY_ROWS
    assert text.index("skeletons: ") < text.index("backend inventory (") < text.index("publish: ")
    inv = plan["backend_inventory"]
    assert {k: inv[k] for k in ("checked", "pulled", "image_tag", "env12", "nodes")} == {
        "checked": True, "pulled": True, "image_tag": TAG, "env12": [ENV], "nodes": 4}
    assert inv["totals"] == {"skeletons": 6, "models": 5, "on_all_nodes": 4, "weights": 5}
    assert inv["unattributed"] == {"labels": 3, "models": [INK, NEMO, LIGHT], "kind_unknown": 1}
    assert [m["model"] for m in inv["models"]] == [DS, "GLM-5.3", "m", "mini", QWEN]
    qwen = inv["models"][-1]
    assert qwen["skeletons"][1] == {"cfg12": "a65858a869ba", "env12": ENV, "wt12": "4aa63a1e185a", "wt12_exact": True,
                                    "nodes": 4, "of": 4, "in_list": True, "jobs": ["qwen3-0.6b-32k-full"]}
    assert qwen["weights"] == [{"wt12": "4aa63a1e185a", "wt12_exact": True, "nodes": 4, "of": 4}]
    assert inv["models"][3]["weights"] == [{"wt12": "eee", "wt12_exact": False, "nodes": None, "of": 4}]
    # The per-job lookup stays restricted to the jobs' own keys.
    assert "| a65858a869ba | exists: 4/4 nodes |" in text and "| ed76d47780e3 | exists: 3/4 nodes |" in text
    dumped = json.dumps(plan)
    for other in (OLD, OLD[:10], OTHER, OTHER[:10], "9" * 12, QWEN_OLD_ONLY, GLM_OLD_ONLY, M_OLD_ONLY,
                  "155abe39cfc9", "ef8257249a3c", "876cb6ac4a4f", "image-cache.", "another image"):
        assert other not in dumped and other not in text, other


def test_the_inventory_is_the_same_for_official_and_show_all_lists(tmp_path, monkeypatch):
    for run in ("fork", "official", "show_all"):
        (tmp_path / run).mkdir()
    fork, fork_text = _dry_run(tmp_path / "fork", monkeypatch, lambda ctx: _two_env_cluster(), configs=FORK)
    official, official_text = _dry_run(tmp_path / "official", monkeypatch, lambda ctx: _two_env_cluster())
    admin, admin_text = _dry_run(tmp_path / "show_all", monkeypatch, lambda ctx: _two_env_cluster(), configs=FORK,
                                 show_all="true")
    assert official["visibility"]["official"] is True and admin["visibility"]["show_all"] is True
    assert fork["backend_inventory"] == official["backend_inventory"] == admin["backend_inventory"]
    assert _inventory_text(fork_text) == _inventory_text(official_text) == _inventory_text(admin_text) == INVENTORY_ROWS
    # The official per-job lookup still names the other image.
    assert "exists for another image: 263a697cd9 4/4" in official_text


def test_the_inventory_says_none_when_the_image_is_not_pulled(tmp_path, monkeypatch):
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: _two_env_cluster(images=((OLD_TAG, OLD_DIGEST),)),
                          configs=FORK)
    assert plan["backend_inventory"] == {"checked": True, "pulled": False,
                                         "note": "none (image not pulled on any node yet)"}
    assert "\nbackend inventory: none (image not pulled on any node yet)\n" in text
    assert "backend inventory (" not in text
    for other in (OLD, OLD[:10], ENV, ENV[:10], "Inkling", "GLM"):
        assert other not in json.dumps(plan) and other not in text, other


def test_the_inventory_is_not_checked_without_kubectl(tmp_path, monkeypatch):
    def nodes(context):
        raise semip_job.NodesUnavailable("kubectl is not on PATH")

    plan, text = _dry_run(tmp_path, monkeypatch, nodes, configs=FORK)
    assert plan["backend_inventory"] == {"checked": False, "note": "not checked: kubectl is not on PATH"}
    assert "\nbackend inventory: not checked: kubectl is not on PATH\n" in text


def test_the_inventory_of_an_ambiguous_tag_holds_each_candidates_skeletons(tmp_path, monkeypatch):
    nodes = _two_env_cluster(images=())
    nodes[0], nodes[1] = _pulled(nodes[0], (TAG, DIGEST)), _pulled(nodes[1], (TAG, OLD_DIGEST))
    plan, text = _dry_run(tmp_path, monkeypatch, lambda ctx: nodes, configs=FORK)
    inv = plan["backend_inventory"]
    assert inv["env12"] == sorted([ENV, OLD]) and inv["totals"]["skeletons"] == 11
    assert f"backend inventory (env {' or '.join(sorted([ENV, OLD]))}, image {TAG})" in text
    assert "| model | kind | cfg12 | env12 | wt12 | nodes | this list |" in text
    assert f"| GLM-5.3 | skeleton | {GLM_OLD_ONLY} | {OLD} | c65f56541db7 | 4/4 | |" in text


def test_label_shows_the_cfg12_and_env12_characters_left_by_the_cut():
    assert keys.label_shows("m") == (12, 12)
    assert keys.label_shows(QWEN) == (12, 10) and keys.label_shows(DS) == (9, 0) and keys.label_shows(NEMO) == (0, 0)
    for label, directory in LIVE.items():
        _, slug, kind, key = directory.split("/")
        if kind == "skeleton":
            cfg_chars, env_chars = keys.label_shows(slug)
            shown = label.rsplit("-", 1)[0][len(keys._segment_prefix(f"image-cache/{slug}/skeleton/")):]
            assert shown == key[:cfg_chars] + (f"_{key[13:13 + env_chars]}" if env_chars else ""), label

