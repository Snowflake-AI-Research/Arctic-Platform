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


"Inference tensor schemas and bounded checkpoint export."

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlsplit

import requests

from arctic_platform.client.requests import export_checkpoint_request

from .storage import parse_hostfile
from .storage import require_data_fast_path

FAMILIES = {
    "qwen3_5": ("Qwen3_5ForConditionalGeneration", "qwen3_5_text", "Qwen3_5ForCausalLM"),
    "qwen3_5_moe": ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
}
DIMENSION_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
)
QWEN4_INTEGER_BUFFERS = frozenset({"layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"})


def validate_initial_training_model_dir(value: str | Path) -> Path:
    """Validate local storage, not model identity or readiness on other allocation nodes."""
    if not isinstance(value, (str, Path)):
        raise ValueError("inference initial model directory must be an absolute path")
    path = Path(value)
    canonical = require_data_fast_path(path, label="inference initial model directory")
    if not path.is_absolute() or path != canonical or path.is_symlink() or not path.is_dir():
        raise ValueError("inference initial model directory must be a canonical regular FAST directory")
    return canonical


def schema_sha(schema):
    raw = json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def schema_bytes(schema):
    if not isinstance(schema, dict) or not schema:
        raise ValueError("nonempty tensor header schema required")
    total = 0
    for name, row in schema.items():
        if not isinstance(name, str) or not name or not isinstance(row, list) or len(row) != 2:
            raise ValueError("invalid tensor header schema")
        dtype, shape = row
        if (
            dtype not in ("BF16", "F32", "I64")
            or (dtype == "I64" and name.rsplit(".", 1)[-1] not in QWEN4_INTEGER_BUFFERS)
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim <= 0 for dim in shape)
        ):
            raise ValueError("invalid tensor dtype or shape")
        total += math.prod(shape) * {"BF16": 2, "F32": 4, "I64": 8}[dtype]
    return total


def validate_qwen4_full_schema(source_schema, export_schema):
    """Preserve original Qwen4 integer buffers while exporting floating weights as BF16."""
    schema_bytes(source_schema)
    schema_bytes(export_schema)
    projected = {}
    for name, (dtype, shape) in source_schema.items():
        integer_buffer = name.rsplit(".", 1)[-1] in QWEN4_INTEGER_BUFFERS
        if integer_buffer and dtype != "I64":
            raise ValueError("original Qwen4 integer buffer must retain I64 dtype: " + name)
        if dtype == "I64" and not integer_buffer:
            raise ValueError("unexpected original Qwen4 integer tensor: " + name)
        projected[name] = ["I64" if integer_buffer else "BF16", shape]
    if export_schema != projected:
        raise ValueError("exact original Qwen4 names, shapes and integer buffer dtypes required")
    return schema_bytes(export_schema)


def _json_identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_qwen4_full_config(source_config, export_config):
    """Preserve the original composite configuration, including nested text and vision settings."""
    if (
        not isinstance(source_config, dict)
        or not isinstance(export_config, dict)
        or source_config.get("model_type") != "qwen4_exp"
        or source_config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]
    ):
        raise ValueError("original Qwen4 composite configuration required")
    _json_identity(source_config)
    _json_identity(export_config)

    def compare(original, exported, prefix):
        if not isinstance(exported, dict) or exported.get("auto_map"):
            raise ValueError("local Qwen4 configuration without remote code required: " + prefix)
        extra = set(exported) - set(original) - {"transformers_version", "use_cache"}
        if extra:
            raise ValueError("unexpected Qwen4 configuration fields: " + prefix + ", ".join(sorted(extra)))
        if "use_cache" in exported and type(exported["use_cache"]) is not bool:
            raise ValueError("explicit boolean Qwen4 cache configuration required: " + prefix + "use_cache")
        for field, value in original.items():
            name = prefix + field
            if field == "transformers_version":
                continue
            if field == "use_cache":
                if type(value) is not bool or type(exported.get(field)) is not bool:
                    raise ValueError("explicit boolean Qwen4 cache configuration required: " + name)
                continue
            if field not in exported:
                raise ValueError("original Qwen4 configuration field missing: " + name)
            if isinstance(value, dict):
                compare(value, exported[field], name + ".")
            elif _json_identity(value) != _json_identity(exported[field]):
                raise ValueError("original Qwen4 configuration differs: " + name)

    compare(source_config, export_config, "")


def validate_text_projection(source_config, export_config, source_schema, export_schema):
    """Return None for ordinary exports; validate named text projections against admitted originals."""
    if not isinstance(source_config, dict) or not isinstance(export_config, dict):
        raise ValueError("raw configuration objects required")
    if source_config.get("model_type") == export_config.get("model_type"):
        return None
    family = FAMILIES.get(source_config.get("model_type"))
    if family is None:
        if isinstance(source_config.get("text_config"), dict):
            raise ValueError("unknown composite-to-text projection")
        return None
    source_architecture, text_type, export_architecture = family
    text = source_config.get("text_config")
    if (
        source_config.get("architectures") != [source_architecture]
        or not isinstance(text, dict)
        or text.get("model_type") != text_type
        or export_config.get("model_type") != text_type
        or export_config.get("architectures") != [export_architecture]
        or "text_config" in export_config
        or "vision_config" in export_config
        or any(config.get("auto_map") for config in (source_config, text, export_config))
    ):
        raise ValueError("named original composite-to-text configuration required")
    _json_identity(source_config)
    _json_identity(export_config)
    for field in DIMENSION_FIELDS:
        if (
            type(text.get(field)) is not int
            or text[field] <= 0
            or type(export_config.get(field)) is not int
            or export_config[field] != text[field]
        ):
            raise ValueError("full original text dimension differs: " + field)
    if (
        type(text.get("tie_word_embeddings")) is not bool
        or export_config.get("tie_word_embeddings") is not text["tie_word_embeddings"]
        or text.get("dtype") != "bfloat16"
        or export_config.get("dtype") != "bfloat16"
    ):
        raise ValueError("original text embedding and BF16 configuration required")
    # These are the observed serializer/cache differences, not model-field exemptions.
    allowed = {"architectures", "transformers_version", "use_cache"}
    if set(export_config) - set(text) - allowed:
        raise ValueError("unexpected text export configuration fields")
    for field, value in text.items():
        if field not in allowed and (
            field not in export_config or _json_identity(export_config[field]) != _json_identity(value)
        ):
            raise ValueError("original text configuration differs: " + field)
    if type(text.get("use_cache")) is not bool or type(export_config.get("use_cache")) is not bool:
        raise ValueError("explicit boolean cache configuration required")
    source_bytes = schema_bytes(source_schema)
    text_names = {
        name for name in source_schema if name.startswith("model.language_model.") or name == "lm_head.weight"
    }
    vision_names = {name for name in source_schema if name.startswith("model.visual.")}
    mtp_names = {name for name in source_schema if name.startswith("mtp.")}
    if not text_names or set(source_schema) - text_names - vision_names - mtp_names:
        raise ValueError("source tensors outside original text, named vision and MTP namespaces")
    layers = {
        int(match[1]) for name in text_names if (match := re.match(r"model\.language_model\.layers\.(\d+)\.", name))
    }
    if layers != set(range(text["num_hidden_layers"])):
        raise ValueError("all original text layers required")
    for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        if source_schema.get(name, [None, None])[1] != [text["vocab_size"], text["hidden_size"]]:
            raise ValueError("full original vocabulary tensor shape required")
    if not isinstance(export_schema, dict) or set(export_schema) != text_names:
        raise ValueError("exact original text inventory required; missing or extra export tensor")
    projected = {name: ["BF16", source_schema[name][1]] for name in text_names}
    if export_schema != projected:
        raise ValueError("exact original text shapes and BF16 projection required")
    export_bytes = schema_bytes(export_schema)
    return {
        "contract": source_config["model_type"] + "_composite_to_full_original_text",
        "schema_contract_valid": True,
        "source_schema_sha256": schema_sha(source_schema),
        "export_schema_sha256": schema_sha(export_schema),
        "source_tensor_count": len(source_schema),
        "text_tensor_count": len(text_names),
        "excluded_vision_tensor_count": len(vision_names),
        "excluded_mtp_tensor_count": len(mtp_names),
        "source_payload_bytes": source_bytes,
        "export_payload_bytes": export_bytes,
        "text_layer_count": text["num_hidden_layers"],
        "tensor_payloads_read": False,
        "runtime_admitted": False,
        "numerical_pass": None,
    }


def _fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate export metadata key: " + key)
        result[key] = value
    return result


def read_export_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular HF metadata required")
    before = _fingerprint(path)
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2 or _fingerprint(path) != before:
        raise ValueError("oversized or concurrently changed HF metadata")
    result = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("HF metadata object required")
    return result


def read_inventory(root, index):
    """Validate index/header coverage without materializing tensor values."""
    from safetensors import safe_open

    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("canonical regular HF directory required")
    if not isinstance(index, dict):
        raise ValueError("HF index object required")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("nonempty HF tensor index required")
    for tensor, shard in mapping.items():
        if (
            not isinstance(tensor, str)
            or not tensor
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
        ):
            raise ValueError("invalid indexed tensor or shard path")
    shards = set(mapping.values())
    if {path.name for path in root.glob("*.safetensors")} != shards:
        raise ValueError("weights inventory differs from its shard index")
    schema = {}
    for name in sorted(shards):
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("regular contained HF weight shard required")
        before = _fingerprint(path)
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
            if (
                len(keys) != len(set(keys))
                or set(keys) != {tensor for tensor, shard in mapping.items() if shard == name}
                or schema.keys() & set(keys)
            ):
                raise ValueError("HF tensor headers disagree with the index")
            for tensor in keys:
                view = stream.get_slice(tensor)
                schema[tensor] = [view.get_dtype(), view.get_shape()]
        if _fingerprint(path) != before:
            raise ValueError("HF tensor header changed during admission")
    total = schema_bytes(schema)
    metadata = index.get("metadata")
    expected = metadata.get("total_size") if isinstance(metadata, dict) else None
    # An original admitted index may encode an integral byte count as a JSON float.
    if type(expected) not in (int, float) or expected != total:
        raise ValueError("HF header byte inventory disagrees with the index")
    return schema


_LOCAL_RECEIPT = "inference_locality_receipt.json"
_LOCAL_METADATA_LIMIT = 16 * 1024**2
_LOCAL_HOSTFILE = Path("/job/hostfile")
_LOCAL_HOSTFILE_TARGET = Path("/data-fast/hostfile")


def _local_remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("inference export locality deadline exceeded")
    return remaining


def _regular_sha(path, deadline):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("regular inference export file required: " + str(path))
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        if (os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("inference export file identity changed")
        while chunk := stream.read(1024**2):
            _local_remaining(deadline)
            digest.update(chunk)
    _local_remaining(deadline)
    if _fingerprint(path) != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns):
        raise ValueError("inference export file changed while hashing")
    return digest.hexdigest()


def _local_manifest(root, deadline, fingerprints=None):
    """Hash every saved byte; inspect tensor headers without materializing tensors."""
    root = Path(root)
    if root != require_data_fast_path(root) or root.is_symlink() or not root.is_dir():
        raise ValueError("canonical FAST inference export required")
    entries = []
    file_fingerprints = {}
    root_before = _fingerprint(root)
    for path in sorted(root.iterdir()):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", path.name) or path.name == _LOCAL_RECEIPT:
            raise ValueError("unexpected inference export filename")
        before = _fingerprint(path)
        digest = _regular_sha(path, deadline)
        if _fingerprint(path) != before:
            raise ValueError("saved inference export changed")
        entries.append({"path": path.name, "size_bytes": before[2], "sha256": digest})
        file_fingerprints[path.name] = list(before)
    names = {entry["path"] for entry in entries}
    if not {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"} <= names:
        raise ValueError("complete saved weights, configuration and tokenizer required")
    schema = read_inventory(root, read_export_json(root / "model.safetensors.index.json"))
    config = read_export_json(root / "config.json")
    if config.get("auto_map"):
        raise ValueError("local inference export must not require remote code")
    _local_remaining(deadline)
    if _fingerprint(root) != root_before or file_fingerprints != {
        row["path"]: list(_fingerprint(root / row["path"])) for row in entries
    }:
        raise ValueError("inference export directory changed")
    if fingerprints is not None:
        fingerprints.update(file_fingerprints)
    return {
        "schema": "same-allocation-inference-export-v1",
        "files": entries,
        "config_sha256": schema_sha(config),
        "tensor_schema_sha256": schema_sha(schema),
        "tensor_payload_bytes": schema_bytes(schema),
        "tokenizer_files": {row["path"]: row["sha256"] for row in entries if "token" in row["path"]},
    }


def _local_hostfile_identity():
    """Permit only the allocation's prescribed link, not arbitrary file aliases."""
    hostfile = _LOCAL_HOSTFILE
    original = hostfile.lstat()
    link = None
    target = hostfile
    if stat.S_ISLNK(original.st_mode):
        link = os.readlink(hostfile)
        target = _LOCAL_HOSTFILE_TARGET
        if link != str(target) or hostfile.resolve(strict=True) != target:
            raise ValueError("only the prescribed allocation hostfile symlink is supported")
    elif not stat.S_ISREG(original.st_mode):
        raise ValueError("regular allocation hostfile or prescribed symlink required")
    resolved = target.lstat()
    if not stat.S_ISREG(resolved.st_mode):
        raise ValueError("regular allocation hostfile target required")

    def identity(value):
        return [value.st_mode, value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    return {
        "path": str(hostfile),
        "link": link,
        "original": identity(original),
        "target_path": str(target),
        "target": identity(resolved),
    }


def _validate_local_allocation(admission, deadline, hostfile_identity=None):
    """Bind a supplied admission to the live head; never discover or combine jobs."""
    if not isinstance(admission, dict) or not re.fullmatch(r"[A-Za-z0-9_-]+", admission.get("allocation_id", "")):
        raise ValueError("explicit current allocation admission required")
    before = _local_hostfile_identity()
    if hostfile_identity and hostfile_identity != before:
        raise ValueError("allocation hostfile link or target identity changed")
    hostfile = Path(before["target_path"])
    content = admission.get("hostfile_content")
    if not isinstance(content, str) or _regular_sha(hostfile, deadline) != admission.get("hostfile_sha256"):
        raise ValueError("current allocation hostfile hash differs from admission")
    if hostfile.read_text() != content:
        raise ValueError("current allocation hostfile content differs from admission")
    parsed = parse_hostfile(hostfile)
    nodes = admission.get("nodes")
    if not isinstance(nodes, list) or not nodes or len(nodes) != len(parsed):
        raise ValueError("one admitted identity per allocation node required")
    if [(row.get("host"), row.get("slots")) for row in nodes] != [(row.host, row.slots) for row in parsed]:
        raise ValueError("ordered allocation hosts and slots differ")
    if len({row["host"] for row in nodes}) != len(nodes) or len({row.get("hostname") for row in nodes}) != len(nodes):
        raise ValueError("unique allocation node identities required")
    for node in nodes:
        if (
            not re.fullmatch(r"[A-Za-z0-9_.:-]+", node["host"])
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", node.get("hostname", ""))
            or not re.fullmatch(r"[0-9a-f-]{36}", node.get("boot_id", ""))
            or type(node["slots"]) is not int
            or node["slots"] <= 0
            or not isinstance(node.get("python"), str)
            or not Path(node["python"]).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", node.get("python_sha256", ""))
        ):
            raise ValueError("actual node hostname, boot, slots and executable pins required")
    _validate_local_node(nodes[0], deadline)
    if _local_hostfile_identity() != before:
        raise ValueError("allocation hostfile link or target changed during validation")
    if hostfile_identity is not None:
        hostfile_identity.update(before)
    return nodes


def validate_local_inference_allocation(admission):
    """Require live same-job identity before a local inference training run."""
    return _validate_local_allocation(admission, time.monotonic() + 30)


def _validate_local_node(node, deadline):
    if (
        socket.gethostname() != node["hostname"]
        or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != node["boot_id"]
        or Path(sys.executable).resolve() != Path(node["python"]).resolve()
        or _regular_sha(Path(sys.executable).resolve(), deadline) != node["python_sha256"]
    ):
        raise ValueError("inference export node or executable identity differs from admission")


def _stop_local_children(children):
    # Signal only sessions created by this export operation, including a stalled writer.
    def live_groups():
        pending = []
        for child in children:
            child.poll()
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                continue
            pending.append(child.pid)
        return pending

    pending = live_groups()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in pending:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 3
        while pending := live_groups():
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        if not pending:
            return
    raise RuntimeError("owned inference export process groups did not empty: " + str(pending))


def _local_worker_argv(python, role, request):
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from . import storage

    storage_sha = hashlib.sha256(Path(storage.__file__).read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parents[3]
    # SSH does not inherit PYTHONPATH. Bind the released source before importing the receiver.
    code = (
        "import hashlib,pathlib,sys;"
        f"root=pathlib.Path({str(source_root)!r});sys.path.insert(0,str(root));"
        "assert hashlib.sha256((root/'arctic_platform/correctness/harness/inference_export.py').read_bytes())"
        f".hexdigest()=={source_sha!r};"
        "assert hashlib.sha256((root/'arctic_platform/correctness/harness/storage.py').read_bytes())"
        f".hexdigest()=={storage_sha!r};"
        "from arctic_platform.correctness.harness import inference_export as m;"
        f"m._local_worker({role!r},{request!r})"
    )
    return [python, "-B", "-c", code]


def _bounded_local_json(argv, deadline):
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        output, error = child.communicate(timeout=_local_remaining(deadline))
        _local_remaining(deadline)
        if child.returncode:
            raise RuntimeError("inference export worker failed: " + error.decode(errors="replace")[-8000:])
        if len(output) > _LOCAL_METADATA_LIMIT:
            raise ValueError("inference export worker receipt exceeds metadata bound")
        return json.loads(output, object_pairs_hook=_unique_object)
    finally:
        _stop_local_children([child])


def _stream_local_export(root, manifest, deadline):
    stream = sys.stdout.buffer
    stream.write(json.dumps(manifest, sort_keys=True, allow_nan=False).encode() + b"\n")
    for row in manifest["files"]:
        path = Path(root) / row["path"]
        before = _fingerprint(path)
        digest = hashlib.sha256()
        count = 0
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
            while chunk := source.read(1024**2):
                _local_remaining(deadline)
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != row["size_bytes"] or digest.hexdigest() != row["sha256"] or _fingerprint(path) != before:
            raise ValueError("saved inference export changed during transfer")
    stream.flush()
    _local_remaining(deadline)


def _receive_local_export(request, deadline):
    _validate_local_node(request["node"], deadline)
    root = Path(request["root"])
    if root != _destination_parent(root):
        raise ValueError("canonical FAST export destination required")
    manifest = json.loads(sys.stdin.buffer.readline(_LOCAL_METADATA_LIMIT + 1), object_pairs_hook=_unique_object)
    if schema_sha(manifest) != request["manifest_sha256"]:
        raise ValueError("inference export manifest identity differs")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("complete export file manifest required")
    names = set()
    for row in files:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", row["path"])
            or row["path"] in {".", "..", _LOCAL_RECEIPT}
            or row["path"] in names
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))
        ):
            raise ValueError("invalid export file manifest")
        names.add(row["path"])
    # Existing directories, including failed partial transfers, are never overwritten.
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    for row in files:
        digest = hashlib.sha256()
        remaining = row["size_bytes"]
        with _open_exclusive(root, row["path"]) as output:
            while remaining:
                _local_remaining(deadline)
                chunk = sys.stdin.buffer.read(min(1024**2, remaining))
                if not chunk:
                    raise ValueError("truncated saved inference export")
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("saved inference export bytes differ")
    if sys.stdin.buffer.read(1):
        raise ValueError("extra bytes after saved inference export")
    # Rehash persisted files and re-read config and tensor headers, not merely the stream.
    if _local_manifest(root, deadline) != manifest:
        raise ValueError("persisted inference export identity differs")
    _validate_local_node(request["node"], deadline)
    receipt = {
        "semantic_exit": 0,
        "allocation_id": request["allocation_id"],
        "node": request["node"],
        "root": str(root),
        "manifest_sha256": request["manifest_sha256"],
        "files_verified": len(files),
    }
    _local_remaining(deadline)
    with _open_exclusive(root, _LOCAL_RECEIPT) as output:
        output.write(json.dumps(receipt, sort_keys=True, allow_nan=False).encode() + b"\n")
        output.flush()
        os.fsync(output.fileno())
    _local_remaining(deadline)
    return receipt


def _local_worker(role, request):
    budget = request["timeout_s"]
    if "deadline_epoch" in request:
        budget = min(budget, request["deadline_epoch"] - time.time())
    deadline = time.monotonic() + budget
    _local_remaining(deadline)
    if role == "manifest":
        fingerprints = {}
        manifest = _local_manifest(request["root"], deadline, fingerprints)
        result = {"manifest": manifest, "head_files": fingerprints}
    elif role == "send":
        _stream_local_export(request["root"], request["manifest"], deadline)
        return
    elif role == "receive":
        result = _receive_local_export(request, deadline)
    elif role == "supervise":
        result = _bounded_local_json(_local_worker_argv(sys.executable, "receive", request), deadline)
    else:
        raise ValueError("unknown inference export worker operation")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def _local_ssh_argv(admission, node, request, deadline):
    transport = admission.get("ssh")
    if (
        not isinstance(transport, dict)
        or type(transport.get("port")) is not int
        or not 1 <= transport["port"] <= 65535
        or not isinstance(transport.get("known_hosts"), str)
        or not Path(transport["known_hosts"]).is_absolute()
        or _regular_sha(transport["known_hosts"], deadline) != transport.get("known_hosts_sha256")
    ):
        raise ValueError("explicit admitted strict SSH transport required")
    command = _local_worker_argv(node["python"], "supervise", request)
    return [
        "ssh",
        "-T",
        "-p",
        str(transport["port"]),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=" + transport["known_hosts"],
        "-o",
        "ConnectTimeout=" + str(max(1, min(30, int(_local_remaining(deadline))))),
        node["host"],
        shlex.join(command),
    ]


def distribute_local_inference_export(root, admission, *, timeout_s=1800.0):
    """Publish exact step-save bytes on every admitted same-job sampling node."""
    if type(timeout_s) not in (float, int) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("finite positive inference export deadline required")
    deadline = time.monotonic() + timeout_s
    hostfile_identity = {}
    nodes = _validate_local_allocation(admission, deadline, hostfile_identity)
    saved = _bounded_local_json(
        _local_worker_argv(sys.executable, "manifest", {"root": str(root), "timeout_s": _local_remaining(deadline)}),
        deadline,
    )
    manifest = saved["manifest"]
    manifest_sha = schema_sha(manifest)
    receipts = []
    for node in nodes[1:]:
        request = {
            "root": str(root),
            "node": node,
            "allocation_id": admission["allocation_id"],
            "manifest_sha256": manifest_sha,
            "timeout_s": _local_remaining(deadline),
            "deadline_epoch": time.time() + _local_remaining(deadline),
        }
        argv = _local_ssh_argv(admission, node, request, deadline)
        children = []
        with tempfile.TemporaryFile() as sender_errors, tempfile.TemporaryFile() as receiver_errors:
            try:
                sender = subprocess.Popen(
                    _local_worker_argv(
                        sys.executable,
                        "send",
                        {"root": str(root), "manifest": manifest, "timeout_s": _local_remaining(deadline)},
                    ),
                    stdout=subprocess.PIPE,
                    stderr=sender_errors,
                    start_new_session=True,
                )
                children.append(sender)
                receiver = subprocess.Popen(
                    argv,
                    stdin=sender.stdout,
                    stdout=subprocess.PIPE,
                    stderr=receiver_errors,
                    start_new_session=True,
                )
                children.append(receiver)
                sender.stdout.close()
                output, _ = receiver.communicate(timeout=_local_remaining(deadline))
                sender.wait(timeout=_local_remaining(deadline))
                _local_remaining(deadline)
                if sender.returncode or receiver.returncode:
                    details = []
                    for error in (sender_errors, receiver_errors):
                        error.seek(max(0, error.tell() - 8000))
                        details.append(error.read().decode(errors="replace"))
                    raise RuntimeError("saved export transport failed: " + "\n".join(details))
                if len(output) > _LOCAL_METADATA_LIMIT:
                    raise ValueError("inference export locality receipt exceeds bound")
                receipt = json.loads(output, object_pairs_hook=_unique_object)
                if receipt != {
                    "semantic_exit": 0,
                    "allocation_id": admission["allocation_id"],
                    "node": node,
                    "root": str(root),
                    "manifest_sha256": manifest_sha,
                    "files_verified": len(manifest["files"]),
                }:
                    raise ValueError("sampling node did not publish the exact inference export")
                receipts.append(receipt)
            finally:
                _stop_local_children(children)
    _validate_local_allocation(admission, deadline, hostfile_identity)
    if saved["head_files"] != {row["path"]: list(_fingerprint(Path(root) / row["path"])) for row in manifest["files"]}:
        raise ValueError("trained export changed before locality completion")
    return {
        "root": str(root),
        "admission_sha256": schema_sha(admission),
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "remote_receipts": receipts,
        "head_files": saved["head_files"],
        "hostfile_identity": hostfile_identity,
    }


def require_local_inference_export_locality(ctx, root):
    """A sampling request must consume the completed save's same-allocation proof."""
    admission = getattr(ctx, "inference_export_allocation", None)
    proof = getattr(ctx, "_inference_export_locality", None)
    if (
        admission is None
        or not isinstance(proof, dict)
        or proof.get("root") != str(root)
        or proof.get("admission_sha256") != schema_sha(admission)
        or proof.get("manifest_sha256") != schema_sha(proof.get("manifest"))
        or not isinstance(proof.get("hostfile_identity"), dict)
        or not proof["hostfile_identity"]
    ):
        raise ValueError("completed same-allocation trained export locality receipt required before sampling")
    nodes = _validate_local_allocation(admission, time.monotonic() + 30, proof["hostfile_identity"])
    expected = [
        {
            "semantic_exit": 0,
            "allocation_id": admission["allocation_id"],
            "node": node,
            "root": str(root),
            "manifest_sha256": proof["manifest_sha256"],
            "files_verified": len(proof["manifest"]["files"]),
        }
        for node in nodes[1:]
    ]
    if proof.get("remote_receipts") != expected:
        raise ValueError("complete sampling allocation export receipts required")
    if proof.get("head_files") != {
        row["path"]: list(_fingerprint(Path(root) / row["path"])) for row in proof["manifest"]["files"]
    }:
        raise ValueError("trained export changed after locality publication")
    if {path.name for path in Path(root).iterdir()} != {row["path"] for row in proof["manifest"]["files"]}:
        raise ValueError("trained export inventory changed after locality publication")


_CHUNK_BYTES = 1024 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_RECEIPT_NAME = "export_receipt.json"
_OPTIMIZER_NAME = re.compile(r"(bf16_)?zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_00_optim_states\.pt")


class CheckpointExportError(RuntimeError):
    """An export is unavailable, unsafe, incomplete, or outside its download budget."""


def _positive_int(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CheckpointExportError(f"{label} must be a positive integer")
    return value


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise CheckpointExportError(f"invalid {label}")
    return value


def _size(value) -> int:
    # SnowAPI may encode uint64 sizes as decimal strings.
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    return _positive_int(value, "declared file size")


def _https_url(value) -> str:
    if not isinstance(value, str) or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise CheckpointExportError("export URL must be an HTTPS URL")
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme == "https"
            and parts.hostname
            and parts.username is None
            and parts.password is None
            and not parts.fragment
        )
        parts.port
    except ValueError:
        valid = False
    if not valid:
        raise CheckpointExportError("export URL must be an HTTPS URL")
    return value


def _relative_filename(value) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise CheckpointExportError("unsafe export filename")
    return value


def _manifest(reply, checkpoint_id: str, dp_size: int, max_total_bytes: int) -> tuple[list[dict], str]:
    if not isinstance(reply, dict) or reply.get("checkpoint_id") != checkpoint_id:
        raise CheckpointExportError("export checkpoint identity mismatch")
    files = reply.get("files")
    if not isinstance(files, list) or len(files) != dp_size + 2:
        raise CheckpointExportError("export must contain the expected optimizer shards, model and metadata")
    entries = []
    names = set()
    ranks = set()
    tags = set()
    prefixes = set()
    model_count = 0
    metadata_count = 0
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise CheckpointExportError("invalid export file record")
        name = _relative_filename(item.get("filename"))
        if name in names:
            raise CheckpointExportError("duplicate export filename")
        names.add(name)
        if item.get("type", "file") != "file" or item.get("is_symlink", False) or "link_target" in item:
            raise CheckpointExportError("export symlinks are not supported")
        if name == "metadata.json":
            metadata_count += 1
        else:
            parts = name.split("/")
            if len(parts) != 2 or not re.fullmatch(r"global_step[0-9]+", parts[0]):
                raise CheckpointExportError("unsupported dense checkpoint layout")
            tags.add(parts[0])
            if parts[1] == "mp_rank_00_model_states.pt":
                model_count += 1
            else:
                match = _OPTIMIZER_NAME.fullmatch(parts[1])
                if match is None:
                    raise CheckpointExportError("unsupported dense optimizer shard filename")
                rank = int(match[2])
                if rank in ranks or rank >= dp_size:
                    raise CheckpointExportError("duplicate or unexpected optimizer rank")
                ranks.add(rank)
                prefixes.add(match[1] or "")
        size = _size(item.get("size_bytes"))
        total += size
        if total > max_total_bytes:
            raise CheckpointExportError("export exceeds the total download byte budget")
        # Only the documented public field is accepted; never guess URL aliases.
        entries.append({"path": name, "size_bytes": size, "url": _https_url(item.get("url"))})
    if ranks != set(range(dp_size)) or len(tags) != 1 or len(prefixes) != 1 or model_count != 1 or metadata_count != 1:
        raise CheckpointExportError("incomplete or inconsistent dense checkpoint layout")
    return sorted(entries, key=lambda entry: entry["path"]), tags.pop()


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CheckpointExportError("checkpoint download deadline exceeded")
    return remaining


def _download_response(session: requests.Session, url: str, deadline: float):
    # Follow presigned cross-host redirects without the Cortex PAT or netrc auth,
    # and reject HTTPS downgrades before sending a redirected request.
    for hop in range(6):
        remaining = _remaining(deadline)
        response = session.get(
            url,
            stream=True,
            allow_redirects=False,
            timeout=(min(30.0, remaining), min(60.0, remaining)),
            headers={"Accept-Encoding": "identity"},
        )
        if response.status_code not in _REDIRECT_STATUSES:
            return response
        try:
            if hop == 5:
                raise CheckpointExportError("checkpoint download redirect limit exceeded")
            location = response.headers.get("Location")
            if not isinstance(location, str) or not location:
                raise CheckpointExportError("checkpoint download redirect has no location")
            url = _https_url(urljoin(url, location))
        finally:
            response.close()
    raise CheckpointExportError("checkpoint download redirect limit exceeded")


def _open_exclusive(root: Path, name: str):
    # Anchor directory traversal to no-follow directory descriptors, not paths
    # that might have been replaced with symlinks between mkdir and open.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = name.split("/")
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        file_fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode=0o600, dir_fd=fd)
        return os.fdopen(file_fd, "wb")
    finally:
        os.close(fd)


def _download_file(session: requests.Session, root: Path, entry: dict, deadline: float) -> dict:
    response = _download_response(session, entry["url"], deadline)
    try:
        if response.status_code != 200:
            raise CheckpointExportError(f"checkpoint download HTTP status {int(response.status_code)}")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise CheckpointExportError("checkpoint download must be uncompressed")
        expected = entry["size_bytes"]
        length = response.headers.get("Content-Length")
        if length is not None and _size(length) != expected:
            raise CheckpointExportError("checkpoint Content-Length differs from the declared size")
        raw = getattr(response, "raw", None)
        if response.headers.get("Transfer-Encoding") or getattr(raw, "chunked", False):
            raise CheckpointExportError("checkpoint download chunked framing cannot enforce the deadline")
        read1 = getattr(raw, "read1", None)
        sock = getattr(getattr(raw, "connection", None), "sock", None)
        if sock is None:
            # Connection: close transfers ownership from the connection to the
            # http.client response's buffered SocketIO while the body stays open.
            buffer = getattr(getattr(raw, "_fp", None), "fp", None)
            sock = getattr(getattr(buffer, "raw", None), "_sock", None)
        settimeout = getattr(sock, "settimeout", None)
        if not callable(read1) or not callable(settimeout):
            raise CheckpointExportError("checkpoint download requires bounded raw reads")
        digest = hashlib.sha256()
        count = 0
        with _open_exclusive(root, entry["path"]) as output:
            # read1 returns available bytes after at most one socket read, unlike
            # iter_content which can wait for a full chunk despite a slow trickle.
            while count < expected or length is None:
                settimeout(min(60.0, _remaining(deadline)))
                limit = min(_CHUNK_BYTES, expected - count + 1)
                chunk = read1(limit, decode_content=False)
                _remaining(deadline)
                if not isinstance(chunk, bytes) or len(chunk) > limit:
                    raise CheckpointExportError("checkpoint download requires bounded raw bytes")
                if not chunk:
                    break
                count += len(chunk)
                if count > expected:
                    raise CheckpointExportError("checkpoint download exceeds the declared size")
                output.write(chunk)
                digest.update(chunk)
            if count != expected:
                raise CheckpointExportError("checkpoint download is truncated")
            _remaining(deadline)
            output.flush()
            os.fsync(output.fileno())
        return {"path": entry["path"], "size_bytes": count, "sha256": digest.hexdigest()}
    finally:
        response.close()


def _destination_parent(output_dir) -> Path:
    parent = Path(output_dir).expanduser().absolute()
    if any(path.is_symlink() for path in (parent, *parent.parents)):
        raise CheckpointExportError("checkpoint destination must not traverse a symlink")
    try:
        parent = require_data_fast_path(parent, label="checkpoint export destination")
    except ValueError as exc:
        raise CheckpointExportError(str(exc)) from None
    if parent.exists() and not parent.is_dir():
        raise CheckpointExportError("checkpoint destination must be a directory")
    return parent


def export_checkpoint(
    client,
    job_id: str,
    checkpoint_id: str,
    output_dir: Path,
    *,
    expected_dp_world_size: int,
    max_total_bytes: int = 128 << 30,
    timeout_s: float = 1800.0,
) -> Path:
    """Return a fresh complete snapshot root; the caller owns successful cleanup.

    Only dense single-model-rank layouts are supported. The receipt binds bytes,
    not tensor semantics: the optimizer decoder must validate stage and identity.
    """
    job_id = _identifier(job_id, "job ID")
    checkpoint_id = _identifier(checkpoint_id, "checkpoint ID")
    dp_size = _positive_int(expected_dp_world_size, "expected DP world size")
    _positive_int(max_total_bytes, "maximum total bytes")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise CheckpointExportError("download timeout must be finite and positive")
    transport = client.transport
    if transport.job_id != job_id:
        raise CheckpointExportError("export job does not match the client parent job")
    root = None
    failure = None
    complete = False
    try:
        parent = _destination_parent(output_dir)
        # Preserve the original public client's framing, auth and retry policy.
        reply = transport.call(export_checkpoint_request(job_id, checkpoint_id))
        entries, tag = _manifest(reply, checkpoint_id, dp_size, max_total_bytes)
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir() or any(path.is_symlink() for path in (parent, *parent.parents)):
            raise CheckpointExportError("checkpoint destination must be a directory without symlinks")
        root = Path(tempfile.mkdtemp(prefix="checkpoint-export-", dir=parent))
        deadline = time.monotonic() + timeout_s
        with requests.Session() as session:
            session.trust_env = False
            session.auth = None
            files = [_download_file(session, root, entry, deadline) for entry in entries]
        receipt = {
            "schema_version": 1,
            "complete": True,
            "source_job_id": job_id,
            "checkpoint_id": checkpoint_id,
            "dp_world_size": dp_size,
            "tag": tag,
            "total_bytes": sum(entry["size_bytes"] for entry in files),
            "files": files,
        }
        with _open_exclusive(root, _RECEIPT_NAME) as output:
            output.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii"))
            output.flush()
            os.fsync(output.fileno())
        complete = True
        return root
    except CheckpointExportError as exc:
        failure = str(exc)
    except Exception:
        # HTTP/IO exception text can include signed URLs, bodies or credentials.
        failure = "checkpoint export or download failed"
    finally:
        if not complete and root is not None:
            try:
                shutil.rmtree(root)
            except OSError:
                raise CheckpointExportError("checkpoint export failed and snapshot cleanup failed") from None
    raise CheckpointExportError(failure) from None
