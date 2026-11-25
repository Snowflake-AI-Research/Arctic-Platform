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

"""Real Gloo/DCP cold-cache loads; the CLI also runs on one allocated multi-node job."""

import argparse
import json
import multiprocessing
import os
import signal
import time
from datetime import timedelta
from pathlib import Path

import pytest


def _write_fixture(root):
    import torch
    from safetensors.torch import save_file

    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from tests.model.test_glm5_next import _tiny_config

    source = root / "source"
    source.mkdir(parents=True)
    config = _tiny_config()
    config.text_config.mlp_layer_types = ["dense", "sparse", "sparse", "sparse"]
    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(config)
    native = {
        name: (torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape) + index) / 100
        for index, (name, value) in enumerate(model.state_dict().items())
    }
    save_file(native, root / "expected.safetensors")
    config.save_pretrained(source)
    hf = model.convert_to_hf({name: value.clone() for name, value in native.items()})
    names = tuple(hf)
    weight_map = {}
    for number in range(2):
        filename = f"model-{number}.safetensors"
        selected = names[number::2]
        save_file({name: hf[name].contiguous() for name in selected}, source / filename)
        weight_map.update({name: filename for name in selected})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _load(root, wave, *, kill_writer=False):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig

    from arctic_platform.model.implementations.glm52 import model_builder
    from arctic_platform.model.implementations.glm52.config import ModelConfig
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL
    from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready
    from arctic_platform.model.implementations.moe.conversion_cache import get_checkpoint_converter
    from arctic_platform.model.implementations.moe.conversion_cache import resolve_conversion_cache_path
    from arctic_platform.testing_utils import torch_assert_equal

    rank = int(os.environ["RANK"])
    source = root / "source"
    config = ModelConfig(name=str(source), fsdp_cpu_offload=True, weight_conversion_cache_dir=str(root / "cache"))
    with torch.device("meta"):
        model = Glm5NextForConditionalGenerationPrimeRL(Glm5NextConfig.from_pretrained(source))
    converter = get_checkpoint_converter(model, "prime")

    def convert(source_path, destination):
        # Instrument only the model callback; storage, conversion, collectives and DCP remain real.
        (root / f"conversion-{wave}-rank{rank}").write_text(str(destination))
        converter(source_path, destination)
        if kill_writer:
            assert conversion_cache_ready(destination)
            os.kill(os.getpid(), signal.SIGKILL)

    model.stream_convert_to_prime = convert
    # Exercise real streamed-cache publication and recovery, not direct HF loading.
    model_builder.supports_direct_hf_loading = lambda model: False
    model_builder.load_dcp_from_hf(model, config, None)
    expected = load_file(root / "expected.safetensors")
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch_assert_equal(actual[name], value, msg=name)
    destination = resolve_conversion_cache_path(config, source, "prime")
    assert conversion_cache_ready(destination)
    (root / f"loaded-{wave}-rank{rank}.json").write_text(
        json.dumps({"cache": str(destination), "tensor_count": len(expected)})
    )
    dist.barrier()


def _worker(rank, rendezvous, root, wave, kill_writer):
    import torch
    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE="2",
        DSS_WEIGHT_CONVERSION_CACHE_SCOPE="shared",
        DSS_WEIGHT_CONVERSION_TIMEOUT="5",
    )
    os.environ.pop("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", None)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _load(Path(root), wave, kill_writer=kill_writer)
    except TimeoutError as error:
        if not kill_writer or rank == 0:
            raise
        (Path(root) / "killed-writer-reader.json").write_text(json.dumps({"error": str(error)}))
    finally:
        dist.destroy_process_group()


def _run_pair(root, wave, *, kill_writer=False):
    context = multiprocessing.get_context("spawn")
    rendezvous = f"file://{root / ('rendezvous-' + wave)}"
    processes = [
        context.Process(target=_worker, args=(rank, rendezvous, str(root), wave, kill_writer)) for rank in range(2)
    ]
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        deadline = started + 90
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes), "GLM cache load did not exit within 90 seconds"
        assert [process.exitcode for process in processes] == ([-signal.SIGKILL, 0] if kill_writer else [0, 0])
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)


@pytest.mark.parametrize("kill_writer", [False, True])
def test_glm_cold_shared_cache_real_dcp_and_killed_writer_retry(tmp_path, kill_writer):
    from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready
    from arctic_platform.model.implementations.moe.conversion_cache import hashed_conversion_cache_path

    _write_fixture(tmp_path)
    cache = hashed_conversion_cache_path(tmp_path / "cache", tmp_path / "source", "prime")
    assert not conversion_cache_ready(cache)
    if kill_writer:
        _run_pair(tmp_path, "killed", kill_writer=True)
        assert not conversion_cache_ready(cache)
        assert list(cache.parent.glob("prime.tmp-rank0-pid*"))
        error = json.loads((tmp_path / "killed-writer-reader.json").read_text())["error"]
        assert "did not become ready" in error
    _run_pair(tmp_path, "retry" if kill_writer else "cold")
    assert conversion_cache_ready(cache)
    assert not list(cache.parent.glob("prime.tmp-rank*-pid*"))
    wave = "retry" if kill_writer else "cold"
    assert [path.name for path in tmp_path.glob(f"conversion-{wave}-rank*")] == [f"conversion-{wave}-rank0"]
    reports = [json.loads((tmp_path / f"loaded-{wave}-rank{rank}.json").read_text()) for rank in range(2)]
    assert reports[0] == reports[1]
    assert reports[0]["cache"] == str(cache)
    _run_pair(tmp_path, "warm")
    assert not list(tmp_path.glob("conversion-warm-rank*"))


@pytest.mark.parametrize(
    "index",
    [
        [],
        None,
        {"weight_map": {"weight": []}},
        {"weight_map": {"weight": {}}},
        {"weight_map": {"weight": None}},
        {"weight_map": {"weight": "../weights.safetensors"}},
        {"weight_map": {"weight": "absolute"}},
    ],
)
def test_malformed_conversion_cache_is_not_ready(tmp_path, index):
    from arctic_platform.model.implementations.moe.conversion_cache import conversion_cache_ready

    destination = tmp_path / "prime"
    destination.mkdir()
    (tmp_path / "weights.safetensors").touch()
    if index == {"weight_map": {"weight": "absolute"}}:
        index = {"weight_map": {"weight": str(tmp_path / "weights.safetensors")}}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index))
    assert conversion_cache_ready(destination) is False


def _main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--wave", choices=["cold", "killed", "retry", "warm"], required=True)
    args = parser.parse_args()
    os.environ["DSS_WEIGHT_CONVERSION_CACHE_SCOPE"] = "shared"
    os.environ.pop("DSS_WEIGHT_CONVERSION_ATTEMPT_ID", None)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        if args.wave in {"cold", "killed"}:
            if dist.get_rank() == 0:
                _write_fixture(args.shared_root)
            dist.barrier()
        _load(args.shared_root, args.wave, kill_writer=args.wave == "killed")
        if dist.get_rank() == 0:
            converters = [path.name for path in args.shared_root.glob(f"conversion-{args.wave}-rank*")]
            assert converters == ([] if args.wave == "warm" else [f"conversion-{args.wave}-rank0"])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _main()
