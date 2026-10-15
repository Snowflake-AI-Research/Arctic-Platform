from __future__ import annotations

from arctic_platform.inference.vllm.gemma3_vision_mapper import (
    install_gemma3_flat_vision_mapper,
)


class _Mapper:
    def __init__(self, vision_prefix: str = "vision_tower."):
        self.orig_to_new_regex = {}
        self.orig_to_new_prefix = {
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": vision_prefix,
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }


def _map_name(mapper: _Mapper, key: str) -> str:
    for pattern, new_key in mapper.orig_to_new_regex.items():
        if pattern.search(key):
            key = pattern.sub(new_key, key)
    for prefix, new_key in mapper.orig_to_new_prefix.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_key, 1)
    return key


def test_flattened_transformers5_vision_names_gain_vision_model():
    mapper = _Mapper()
    assert install_gemma3_flat_vision_mapper(mapper)
    mapped = _map_name(mapper, "model.vision_tower.embeddings.patch_embedding.weight")
    assert mapped == "vision_tower.vision_model.embeddings.patch_embedding.weight"
    encoder = _map_name(
        mapper,
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
    )
    assert encoder == "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"


def test_pre_transformers5_vision_names_are_not_doubled():
    mapper = _Mapper()
    install_gemma3_flat_vision_mapper(mapper)
    mapped = _map_name(
        mapper,
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    )
    assert mapped == "vision_tower.vision_model.embeddings.patch_embedding.weight"


def test_language_model_prefix_is_unchanged():
    mapper = _Mapper()
    install_gemma3_flat_vision_mapper(mapper)
    mapped = _map_name(mapper, "model.language_model.layers.0.mlp.down_proj.weight")
    assert mapped == "language_model.model.layers.0.mlp.down_proj.weight"


def test_install_is_idempotent_and_skips_a_rewritten_prefix():
    mapper = _Mapper()
    assert install_gemma3_flat_vision_mapper(mapper)
    assert not install_gemma3_flat_vision_mapper(mapper)
    assert len(mapper.orig_to_new_regex) == 1

    rewritten = _Mapper(vision_prefix="vision_tower.vision_model.")
    assert not install_gemma3_flat_vision_mapper(rewritten)
    assert rewritten.orig_to_new_regex == {}
