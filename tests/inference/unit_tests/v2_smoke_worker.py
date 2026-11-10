"""Subprocess worker for offline v2 runner integration tests."""

import json
import sys
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.plugins import load_general_plugins


class SmokeWorkerExtension:
    def configure_smoke(self, all_rows, forest):
        if all_rows:
            state = self.model_runner.model_state
            prepare = state.prepare_attn

            def prepare_all(input_batch, *args, **kwargs):
                metadata = prepare(input_batch, *args, **kwargs)
                for meta in metadata.values():
                    meta.swiftkv_logits_indices = state._token_indices[:input_batch.num_tokens_after_padding]
                return metadata

            state.prepare_attn = prepare_all
        if forest:
            from arctic_platform.inference.vllm.attention.flash_attn_forest_cascade import FlashAttentionImpl
            forward = FlashAttentionImpl.forward

            def observed_forward(impl, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
                if getattr(attn_metadata, "use_forest_cascade", False):
                    layer._forest_steps = getattr(layer, "_forest_steps", 0) + 1
                return forward(impl, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)

            FlashAttentionImpl.forward = observed_forward

    def inspect_smoke(self):
        runner = self.model_runner
        draft = runner.get_draft_model()
        if draft is not None:
            expected = {name: param.detach().clone() for name, param in draft.named_parameters()
                if not name.startswith("qhead.")}
            weights = []
            for name, tensor in expected.items():
                # Checkpoint vocabulary tensors exclude the padded embedding rows.
                module = draft.get_submodule(name.rsplit(".", 1)[0])
                vocab_size = getattr(module, "org_vocab_size", None)
                if vocab_size is not None:
                    valid_rows = module.shard_indices.org_vocab_end_index - module.shard_indices.org_vocab_start_index
                    checkpoint = module.TP_GROUP.all_gather(tensor, dim=0)[:vocab_size]
                    tensor[valid_rows:] = 0
                    weights.append((name, checkpoint))
                    continue
                if (getattr(draft, "method", None) == "sum_lstm"
                        and draft.tie_lstm_embs and name.startswith("projs.")):
                    for gate, chunk in zip(("forget", "input", "output", "cell"), tensor.chunk(4)):
                        weights.append((name.replace("projs", f"{gate}_proj"), chunk))
                    continue
                weights.append((name, tensor[:vocab_size] if vocab_size is not None else tensor))
            draft.load_weights(weights)
            assert all(torch.equal(param, expected[name]) for name, param in draft.named_parameters()
                if name in expected)
        return {
            "draft_parameters": 0 if draft is None else sum(p.numel() for p in draft.parameters()),
            "proposal_graphs": len(getattr(draft, "cuda_graphs", {})),
            "forest_steps": sum(getattr(layer, "_forest_steps", 0)
                for layer in runner.compilation_config.static_forward_context.values()),
        }


def main():
    model_dir, output_path, options_json = sys.argv[1:]
    options = json.loads(options_json)
    torch.set_float32_matmul_precision("highest")
    load_general_plugins()
    kwargs = options.pop("engine", {})
    kwargs.setdefault("async_scheduling", False)
    kwargs.setdefault("block_size", 16)
    graph_mode = options.pop("graphs", False)
    if graph_mode:
        # Isolate graph replay from compiler fusion changes in numerical checks.
        kwargs.setdefault("compilation_config", {"mode": 0, "cudagraph_mode": "FULL"})
    neural = options.pop("neural", False)
    lstm = options.pop("lstm", False)
    swiftkv = options.pop("swiftkv", False)
    if neural:
        from arctic_platform.inference.vllm.spec_dec.config import MLPSpeculatorConfig
        draft_dir = Path(output_path).with_suffix(".draft")
        draft_options = dict(input_hidden_dim=256, proj_dim=128, tie_weights=True,
            tie_lstm_embs=True, method="sum_lstm") if lstm else {}
        MLPSpeculatorConfig(
            vocab_size=130, emb_dim=128 if lstm else 256, inner_dim=128, n_predict=3,
            architectures=["ArcticLSTMSpeculatorPreTrainedModel" if lstm else "ArcticMLPSpeculatorPreTrainedModel"],
            base_model_archs=["LlamaForCausalLM"], max_position_embeddings=256,
            **draft_options,
        ).save_pretrained(draft_dir)
        kwargs["speculative_config"] = {
            "method": "arctic", "model": str(draft_dir), "num_speculative_tokens": 3,
            "enable_suffix_decoding": options.pop("hybrid", False),
            "draft_load_config": {"load_format": "dummy"},
        }
    if swiftkv:
        model_dir = str(Path(model_dir).with_name("swiftkv"))
    all_swiftkv_rows = options.pop("all_swiftkv_rows", False)
    forest = options.pop("forest", False)
    if neural or forest or all_swiftkv_rows:
        kwargs["worker_extension_cls"] = "tests.inference.unit_tests.v2_smoke_worker.SmokeWorkerExtension"
    if forest:
        kwargs["forest_cascade_attn_configs"] = json.dumps({
            "min_batch_size": 4, "min_group_size": 2,
            "min_additional_prefix_blocks": 1,
        })
    llm = LLM(
        model=model_dir, skip_tokenizer_init=True, dtype=options.pop("dtype", "bfloat16"), seed=17,
        max_model_len=128, max_num_seqs=4, max_num_batched_tokens=64,
        gpu_memory_utilization=0.03, enforce_eager=not graph_mode,
        enable_sleep_mode=True, **kwargs,
    )
    if forest or all_swiftkv_rows:
        llm.collective_rpc("configure_smoke", args=(all_swiftkv_rows, forest))
    repeats = options.pop("prefix_repeats", 8 if options.pop("long_prefix", forest) else 1)
    prefix = [1, 2, 3, 4] * repeats
    prompts = [
        {"prompt_token_ids": prefix + [7, 8, 9]},
        {"prompt_token_ids": prefix + [7, 8, 10]},
        {"prompt_token_ids": [1, 5, 6, 7] * repeats + [11, 12]},
        {"prompt_token_ids": [1, 5, 6, 7] * repeats + [11, 13]},
    ]
    max_tokens = options.pop("max_tokens", 12)
    def generate(prompt_logprobs):
        params = SamplingParams(
            temperature=0, max_tokens=max_tokens, logprobs=3,
            prompt_logprobs=3 if prompt_logprobs else None, ignore_eos=True,
        )
        return [{
            "tokens": list(r.outputs[0].token_ids),
            "logprobs": [d[t].logprob for t, d in zip(r.outputs[0].token_ids, r.outputs[0].logprobs)],
            "prompt_logprobs": [None if d is None else d[t].logprob
                for t, d in zip(r.prompt_token_ids, r.prompt_logprobs or [])],
        } for r in llm.generate(prompts, params, use_tqdm=False)]

    def compare(reference, actual, check_logprobs=True):
        from arctic_platform.testing_utils import torch_assert_close
        for expected, output in zip(reference, actual):
            assert expected["tokens"] == output["tokens"]
            if not check_logprobs:
                continue
            for key in ("logprobs", "prompt_logprobs"):
                torch_assert_close(
                    torch.tensor([v for v in expected[key] if v is not None]),
                    torch.tensor([v for v in output[key] if v is not None]), rtol=0,
                    atol=1e-5 if llm.llm_engine.vllm_config.model_config.dtype == torch.float32 else 1e-3,
                )

    result = {"with_prompt_logprobs": generate(True), "cached_prefix": generate(False)}
    # Sleep clears prefix caching, so repeat-token checks are independent of
    # kernel rounding from a different prefill schedule. Mode parity below
    # compares both token ids and logprobs before any sleep.
    for level in options.pop("sleep_levels", [1]):
        llm.sleep(level=level)
        llm.wake_up()
        compare(result["with_prompt_logprobs"], generate(True), check_logprobs=False)
        compare(result["cached_prefix"], generate(False), check_logprobs=False)
    if neural or forest:
        stats = llm.collective_rpc("inspect_smoke")
        assert not neural or all(s["draft_parameters"] > 0 for s in stats)
        assert not (neural and graph_mode) or all(s["proposal_graphs"] > 0 for s in stats)
        assert not forest or all(s["forest_steps"] > 0 for s in stats)
        compare(result["with_prompt_logprobs"], generate(True), check_logprobs=False)
    Path(output_path).write_text(json.dumps(result))
    llm.llm_engine.engine_core.shutdown()
    print("offline v2 generation and sleep/wake passed")


if __name__ == "__main__":
    main()
