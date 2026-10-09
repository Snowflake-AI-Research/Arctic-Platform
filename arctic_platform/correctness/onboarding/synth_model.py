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

"""Build and materialize the reduced model both engines load.

Every dimension of the source checkpoint is preserved and only the layer count is reduced, which keeps the
architecture-specific kernels in the Arctic Platform path under test. A generic small transformer would exercise none
of them.

The kept layers carry the checkpoint's own weights. A freshly initialized model on uniform random tokens
sits at maximum entropy, where each parameter's gradient is a small residual of many opposing per-token
contributions and bf16 rounding is largest relative to the result; comparing two engines there measures the
fixture. On a 2,048-token case the random-init model put Arctic Platform 4.98e-04 above the reference with 312 of 374
tensors on the same side, and the checkpoint's weights bring that to -2.01e-04 with a 161/374 split.
"""

from __future__ import annotations

import glob
import shutil
from pathlib import Path
from typing import List
from typing import Tuple

from ..harness.seeds import SEED
from ..harness.spec import ModelSpec
from ..harness.spec import hash_directory

# A single H200 must hold the reference's weights, its fp32 gradient accumulator, the live gradients of
# one backward pass, one microbatch of activations, and the chunked cross-entropy workspace. The margin
# below that keeps the largest case from sitting at the edge of the card.
SINGLE_GPU_BUDGET_GIB = 100.0

# bfloat16 weights (2) + the fp32 gradient accumulator (4) + the bfloat16 gradients a backward pass leaves
# on the parameters before they are drained into that accumulator (2). All three are resident at once, so
# sizing on weights and accumulator alone understates the requirement by a quarter and selects a layer
# count that does not fit.
BYTES_PER_PARAM = 8


def detect_block_period(layer_types: List[str]) -> int:
    """Smallest period for which the layer list is its first ``p`` entries repeated.

    Derived from the list rather than read from an interval field, because not every architecture has such
    a field and the ones that do are not always consistent with the list.
    """
    n = len(layer_types)
    for p in range(1, n + 1):
        if n % p == 0 and all(layer_types[i] == layer_types[i % p] for i in range(n)):
            return p
    return n


def reduce_config(source: str, num_layers: int, vision_depth: int = 2):
    """Rewrite the model config to ``num_layers``, truncating every entry correlated with the layer count.

    Correlated entries are found by shape rather than by name: any list whose length equals the original
    layer count is truncated alongside it. That catches per-layer type lists without naming them, which is
    what lets this work on an architecture the harness has not seen.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(source, trust_remote_code=True)
    text = getattr(cfg, "text_config", cfg)
    original = int(text.num_hidden_layers)

    period = detect_block_period(list(getattr(text, "layer_types", []))) if hasattr(text, "layer_types") else 1
    if period > 1 and num_layers % period != 0:
        raise ValueError(
            f"num_layers={num_layers} must be a multiple of the block period {period}; a partial block "
            "would drop a layer type from the comparison entirely"
        )

    text.num_hidden_layers = num_layers
    for attr in dir(text):
        if attr.startswith("_"):
            continue
        value = getattr(text, attr, None)
        if isinstance(value, list) and len(value) == original:
            setattr(text, attr, list(value[:num_layers]))

    if hasattr(cfg, "vision_config") and vision_depth is not None:
        cfg.vision_config.depth = vision_depth
    return cfg, original, period


def count_parameters(cfg) -> int:
    """Exact parameter count with no memory cost and no weight loading, for any architecture."""
    import torch
    import transformers

    cls = getattr(transformers, cfg.architectures[0])
    with torch.device("meta"):
        model = cls(cfg)
    return sum(p.numel() for p in model.parameters())


def size_for_single_gpu(
    source: str, budget_gib: float = SINGLE_GPU_BUDGET_GIB, bytes_per_param: int = BYTES_PER_PARAM
) -> Tuple[int, int]:
    """Largest whole-block layer count whose parameter-resident memory fits the budget.

    Activations and the cross-entropy workspace are not counted here; they are bounded separately by the
    reference's own microbatch token budget, and the margin in ``budget_gib`` covers them.
    """
    from transformers import AutoConfig

    probe = AutoConfig.from_pretrained(source, trust_remote_code=True)
    text = getattr(probe, "text_config", probe)
    full = int(text.num_hidden_layers)
    period = detect_block_period(list(getattr(text, "layer_types", []))) or 1

    for n in range(full - (full % period), 0, -period):
        cfg, _, _ = reduce_config(source, n)
        params = count_parameters(cfg)
        if params * bytes_per_param / 1024**3 <= budget_gib:
            return n, params
    raise RuntimeError(f"no layer count fits {budget_gib} GiB; the embedding pair alone may exceed it")


# Tokenizer, merge table, vocabulary, chat template and image/video processor files: everything a serving engine
# needs beside the weights, and nothing a reduced ``save_pretrained`` writes for itself. vLLM builds a multimodal
# checkpoint's input processor at engine start and refuses to initialize without ``preprocessor_config.json``.
SIDECAR_PATTERNS = ("*token*", "*.txt", "*processor*", "chat_template*")


def sidecar_files(source: str) -> List[Path]:
    """The source checkpoint's tokenizer sidecars, as paths on this filesystem.

    ``source`` is a Hugging Face hub id as often as it is a directory, and a glob over a hub id matches
    nothing. Resolving the snapshot first makes both forms local. The failure this prevents is silent in
    two stages: a reduced model written without tokenizer sidecars still loads, and
    ``AutoTokenizer.from_pretrained`` on it returns a single-entry vocabulary that encodes every string to
    an empty id list rather than raising. The weights-only checkpoint a serving engine is pointed at then
    inherits the gap, because that save copies its sidecars from the job's model directory.
    """
    root = source
    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download

        root = snapshot_download(source, allow_patterns=list(SIDECAR_PATTERNS))
    found: List[Path] = []
    for pattern in SIDECAR_PATTERNS:
        found.extend(Path(path) for path in glob.glob(f"{root}/{pattern}"))
    return sorted(set(found))


def copy_sidecars(source: str, dest_path: Path) -> List[Path]:
    """Place the source's tokenizer sidecars beside a reduced model's weights.

    Run whether or not the weights were just written, so a cache produced before this copy existed gains
    them on the next call rather than staying unusable. Copying is idempotent: a hub snapshot's files are
    read-only, so an earlier copy is removed rather than opened for writing, and the replacement is created
    with the ambient mode instead of inheriting the snapshot's.
    """
    copied: List[Path] = []
    for sidecar in sidecar_files(source):
        target = dest_path / sidecar.name
        target.unlink(missing_ok=True)
        shutil.copyfile(sidecar, target)
        copied.append(target)
    if not any(path.name == "tokenizer_config.json" for path in copied):
        raise RuntimeError(
            f"no tokenizer sidecars were found for {source!r}; a reduced model without them loads but "
            "tokenizes every string to nothing, and so does the weights-only checkpoint saved from it"
        )
    return copied


def materialize_pretrained(
    source: str, dest: str, num_layers: int, vision_depth: int = 2, force: bool = False
) -> ModelSpec:
    """Write the reduced config filled with the source checkpoint's own weights for the layers it keeps.

    The layers above ``num_layers`` are unexpected keys and are dropped. The result is not a trained model
    -- a prefix of a stack feeding the final norm and the lm_head predicts badly -- but every tensor holds
    a value the checkpoint was trained to, so the gradients are not the near-cancelling residual that a
    fresh initializer on random tokens produces.

    The written tree carries the source's tokenizer sidecars, so it can be tokenized against and a
    weights-only checkpoint saved from a job that loaded it is self-contained for a serving engine.
    """
    import torch
    import transformers

    dest_path = Path(dest)
    cfg, _, _ = reduce_config(source, num_layers, vision_depth)

    if force or not (dest_path / "config.json").exists():
        dest_path.mkdir(parents=True, exist_ok=True)
        cls = getattr(transformers, cfg.architectures[0])
        model = cls.from_pretrained(source, config=cfg, dtype=torch.bfloat16, trust_remote_code=True)
        model.save_pretrained(dest_path, safe_serialization=True, max_shard_size="5GB")
        del model
    copy_sidecars(source, dest_path)

    params = count_parameters(cfg)
    return ModelSpec(
        source_checkpoint=source,
        num_hidden_layers=num_layers,
        vision_depth=vision_depth,
        seed=0,
        param_count=params,
        cache_path=str(dest_path),
        content_hash=hash_directory(dest_path),
        sized_against_gib=SINGLE_GPU_BUDGET_GIB,
    )


def materialize(
    source: str, dest: str, num_layers: int, vision_depth: int = 2, seed: int = SEED, force: bool = False
) -> ModelSpec:
    """Generate the model once and write it to ``dest``; both engines then load that directory by path.

    The weights are regenerable from the seed, so ``dest`` is a cache rather than an input. A caller that
    finds a matching content hash reuses it; anything else rebuilds.
    """
    import torch
    import transformers

    dest_path = Path(dest)
    cfg, original, period = reduce_config(source, num_layers, vision_depth)

    if force or not (dest_path / "config.json").exists():
        dest_path.mkdir(parents=True, exist_ok=True)
        cls = getattr(transformers, cfg.architectures[0])
        torch.manual_seed(seed)
        model = cls(cfg).to(torch.bfloat16)
        model.save_pretrained(dest_path, safe_serialization=True, max_shard_size="5GB")
        for extra in glob.glob(f"{source}/*token*") + glob.glob(f"{source}/*.txt"):
            shutil.copy2(extra, dest_path)
        del model

    params = count_parameters(cfg)
    return ModelSpec(
        source_checkpoint=source,
        num_hidden_layers=num_layers,
        vision_depth=vision_depth,
        seed=seed,
        param_count=params,
        cache_path=str(dest_path),
        content_hash=hash_directory(dest_path),
        sized_against_gib=SINGLE_GPU_BUDGET_GIB,
    )
