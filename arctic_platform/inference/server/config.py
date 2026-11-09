from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from arctic_platform.inference.envs import arctic_inference_effective_enabled

_QUANT_SUFFIXES = re.compile(r"-(FP8|NVFP4|FP4)$", re.IGNORECASE)


def normalize_model_name(name: str) -> str:
    basename = name.rstrip("/").split("/")[-1]
    return _QUANT_SUFFIXES.sub("", basename)


class GPUType(str, Enum):
    H200 = "H200"
    B200 = "B200"


class ModelConfig(BaseModel):
    """Configuration for vLLM engine initialization.

    Field names match AsyncEngineArgs where possible so to_engine_kwargs()
    can pass them through directly.
    """

    model: str = Field(description="Model name, HuggingFace ID, or local path")
    tensor_parallel_size: int = 1
    # Pipeline-parallel stages. world_size = tensor_parallel_size *
    # pipeline_parallel_size ranks; the pool sizes its GPU reservation / PG and
    # its multi-node check off that product, not tp alone. Name matches
    # AsyncEngineArgs so to_engine_kwargs() passes it straight through.
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str | None = None
    # The server/DSS path does not currently support Ulysses SP. Keep the field
    # explicit so Pydantic cannot silently ignore an unsupported request.
    ulysses_sequence_parallel_size: int = Field(default=1, exclude=True)
    max_model_len: int = 32768
    max_num_seqs: int = 1024
    gpu_memory_utilization: float = 0.95
    quantization: str | None = None
    enable_chunked_prefill: bool = True
    kv_cache_dtype: str | None = None
    enable_expert_parallel: bool = False
    trust_remote_code: bool = True
    fp32_lm_head: bool = False
    gdn_prefill_backend: str | None = None
    structured_outputs_config: dict[str, Any] | None = None
    speculative_config: dict[str, Any] | None = None

    # Optional vLLM reasoning parser name (e.g. "qwen3", "deepseek_r1"). This
    # is a serving-layer concept, not an AsyncEngineArgs field, so the worker
    # pops it before building the engine.
    reasoning_parser: str | None = None
    # Chat-mode overrides of chat.CHAT_MODELS, which picks parsers by
    # architecture. tool_call_parser is vllm serve's --tool-call-parser.
    # chat_reasoning_parser applies to chat streams only, so /generate keeps
    # reasoning_parser; an engine that already has a reasoner must name the
    # same one. Either field enables chat on an architecture not listed there.
    tool_call_parser: str | None = None
    chat_reasoning_parser: str | None = None
    # When true, include parsed `reasoning` / `content` fields in generate
    # responses. The parser may still be used internally when this is false.
    return_reasoning_content: bool = False

    # Forest Cascade Attention: groups concurrent requests by shared KV-cache
    # prefix and runs one prefix FA call + per-request suffix FA + log-sum-exp
    # merge, eliminating redundant KV reads when many requests share long
    # prompt prefixes (e.g. multi-turn rollouts pinned to the same replica
    # via routing_key).  Default-on with `'{}'` (all backend defaults); FCA
    # transparently falls back to standard FlashAttention when batch
    # conditions don't apply (see attention/README.md).  Set to ``None`` to
    # disable.  Only passed to vLLM when ARCTIC_INFERENCE_ENABLED=1 in the
    # process or ``extra_env`` (otherwise omitted so vanilla AsyncEngineArgs
    # does not error).
    forest_cascade_attn_configs: str | None = "{}"

    yarn_factors: list[float] = Field(default_factory=list, exclude=True)
    text_config_overrides: dict[str, Any] = Field(default_factory=dict, exclude=True)

    # Semi-persistence: restore a pre-warmed CRIU image via
    # semi_persistence.Instance instead of a cold vLLM load. Popped by
    # InferenceWorker.initialize before AsyncEngineArgs is built.
    # The image directory is not a job field: it is derived from a hash of this
    # config and of the container image, under a cache root semip_engine
    # defaults (SEMIP_IMAGE_CACHE in ``extra_env`` still overrides it). That is
    # what lets a dump and a later restore agree on a path, and it is why a job
    # that names its own directory would be naming a cache key it cannot
    # compute.
    semi_p: bool = False

