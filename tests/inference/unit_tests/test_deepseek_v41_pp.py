"""DeepSeek-V4.1's decoder SWA bounded replay on the stages of a PP2 (20/20) split.

Runs vLLM's own ``DeepseekV4Model._decoder_replay_supported`` on a stand-in for
the model's attributes it reads.
"""

from types import SimpleNamespace

import pytest

from arctic_platform.inference.vllm import deepseek_v41_pp

NUM_LAYERS = 40
CUT = 20  # max(kv_source_layer_ids) of DeepSeek-V4.1-Flash


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning_once(self, message, *args):
        self.warnings.append(message % args)


def _vllm_decoder_replay_supported():
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

    fn = DeepseekV4Model._decoder_replay_supported
    return getattr(fn, "_orig", fn)


def _model_class(logger=None):
    class Model:
        _decoder_replay_supported = _vllm_decoder_replay_supported()

    if logger is not None:
        deepseek_v41_pp.patch_decoder_replay_supported(Model, logger)
    return Model


def _stage(model_cls, start_layer, end_layer):
    stage = model_cls()
    stage.start_layer = start_layer
    stage.end_layer = end_layer
    stage.use_sequence_parallel = False
    stage.config = SimpleNamespace(
        num_hidden_layers=NUM_LAYERS, sliding_window=128, engram_layer_ids=[1, 14]
    )
    return stage


VLLM_CONFIG = SimpleNamespace(
    parallel_config=SimpleNamespace(use_ubatching=False), speculative_config=None
)


def test_vllm_allows_replay_on_a_stage_starting_at_the_cut():
    stage = _stage(_model_class(), CUT, NUM_LAYERS)
    assert stage._decoder_replay_supported(VLLM_CONFIG, CUT)


def test_replay_is_off_on_a_stage_starting_at_the_cut():
    logger = _Logger()
    stage = _stage(_model_class(logger), CUT, NUM_LAYERS)
    assert not stage._decoder_replay_supported(VLLM_CONFIG, CUT)
    assert logger.warnings and "starts at the last KV source layer 20" in logger.warnings[0]


@pytest.mark.parametrize(
    ("start_layer", "end_layer"),
    [
        (0, NUM_LAYERS),  # PP1
        (0, CUT),  # stage 1 of 20/20: does not hold the layers after the cut
        (14, NUM_LAYERS),  # a stage that runs layers before the cut
    ],
)
def test_other_stages_get_vllms_answer(start_layer, end_layer):
    expected = _stage(_model_class(), start_layer, end_layer)._decoder_replay_supported(
        VLLM_CONFIG, CUT
    )
    logger = _Logger()
    stage = _stage(_model_class(logger), start_layer, end_layer)
    assert stage._decoder_replay_supported(VLLM_CONFIG, CUT) == expected
    assert not any("starts at the last KV source layer" in w for w in logger.warnings)


def test_patch_applies_once():
    logger = _Logger()
    model_cls = _model_class(logger)
    patched = model_cls._decoder_replay_supported
    deepseek_v41_pp.patch_decoder_replay_supported(model_cls, logger)
    assert model_cls._decoder_replay_supported is patched


def test_the_deepseek_v41_entry_point_installs_the_fix():
    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4Model

    import arctic_platform.inference.vllm.deepseek_v41_lora_model  # noqa: F401

    assert getattr(DeepseekV4Model._decoder_replay_supported, "_arctic_deepseek_v41_pp", False)
