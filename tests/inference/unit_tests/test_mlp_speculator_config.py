from arctic_platform.inference.vllm.spec_dec.config import MLPSpeculatorConfig


def test_speculator_checkpoint_config_round_trip(tmp_path):
    config = MLPSpeculatorConfig(
        emb_dim=128, inner_dim=256, n_predict=2,
        top_k_tokens_per_head=[4, 3], base_model_arch="Qwen2ForCausalLM",
    )
    config.save_pretrained(tmp_path)
    loaded = MLPSpeculatorConfig.from_pretrained(tmp_path)

    assert loaded.hidden_size == loaded.emb_dim == 128
    assert loaded.num_attention_heads == 1
    assert loaded.num_lookahead_tokens == loaded.n_predict == 2
    assert loaded.top_k_tokens_per_head == [4, 3]
    assert loaded.base_model_arch == "Qwen2ForCausalLM"
