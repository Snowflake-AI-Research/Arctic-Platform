"""Observe real GDN metadata builds in a worker with its own process-global vLLM patches."""

import logging
import os
import subprocess
import sys
from types import SimpleNamespace

SETTINGS = {
    "counts-only": {},
    "glm-kda": dict(kda_prefill_backend="flashkda"),
    "qwen-gdn": dict(
        gdn_prefill_backend="flashinfer", gdn_decode_kernel="triton", enable_packed_recurrent_decode=False
    ),
    "kimi-k3": dict(kda_prefill_backend="flashinfer"),
}


def run_worker():
    import torch
    from vllm.config import SpeculativeConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
    from vllm.utils import torch_utils
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends import utils as attn_utils
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    from arctic_platform.inference.vllm.required_patches import apply_required_vllm_patches

    # CUDA wheels can request pinned staging buffers even when building metadata on CPU.
    torch_utils.is_pin_memory_available = lambda: False
    torch_utils.PIN_MEMORY = attn_utils.PIN_MEMORY = False
    parent = logging.getLogger("vllm")
    parent.handlers = [logging.StreamHandler(sys.stdout)]
    parent.propagate = False
    parent.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.WARNING)

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={name: SimpleNamespace(**fields) for name, fields in SETTINGS.items()},
            cudagraph_mode=CUDAGraphMode.NONE, max_cudagraph_capture_size=None,
        ),
        speculative_config=SpeculativeConfig(method="ngram", num_speculative_tokens=1),
        use_v2_model_runner=False, model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        additional_config={}, parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(mamba_cache_mode="none", use_kda_recoverssm=False),
    )
    spec = MambaSpec(block_size=16, shapes=((2, 2),), dtypes=(torch.float32,), num_speculative_blocks=1)
    locs = torch.tensor([0, 2, 3], dtype=torch.int32)
    lengths = torch.tensor([4, 5], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=locs, query_start_loc_cpu=locs, seq_lens=lengths, seq_lens_cpu_upper_bound=lengths,
        num_reqs=2, num_actual_tokens=3, max_query_len=2, max_seq_len=5, is_prefilling=torch.tensor([False, False]),
        block_table_tensor=torch.tensor([[3, 4], [7, 8]]), slot_mapping=torch.arange(3),
    )
    # The variable is set before the first install, as in a worker; a second install must not wrap again.
    for _ in range(2):
        apply_required_vllm_patches()
        for name in SETTINGS:
            cls = KimiK3KDAMetadataBuilder if name == "kimi-k3" else GDNAttentionMetadataBuilder
            builder = cls(spec, [name], config, torch.device("cpu"))
            metadata = builder.build(0, common, torch.tensor([1, 1]), torch.tensor([1, -1]))
            assert metadata.non_spec_state_indices_tensor.tolist() == [7]


def worker_lines(enabled):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), ARCTIC_INFERENCE_LOG_DISPATCH=enabled)
    output = subprocess.check_output([sys.executable, __file__], env=env, text=True)
    return [line for line in output.splitlines() if line.startswith("linear-attention dispatch")]


def test_each_gdn_family_build_logs_its_counts_and_layer_settings():
    lines = []
    for name, settings in SETTINGS.items():
        counts = ("prefills=0 prefill_tokens=0 decodes=1 decode_tokens=1" if name == "kimi-k3"
                  else "prefills=1 prefill_tokens=1 decodes=0 decode_tokens=0")
        fields = "".join(f" {key}={value}" for key, value in settings.items())
        lines.append(f"linear-attention dispatch group={name} layer=SimpleNamespace{fields} {counts} "
                     "spec_decodes=1 spec_decode_tokens=2")
    assert worker_lines("1") == lines * 2


def test_builds_log_nothing_when_off():
    assert worker_lines("0") == []


if __name__ == "__main__":
    run_worker()
