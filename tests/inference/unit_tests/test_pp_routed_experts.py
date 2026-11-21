from types import SimpleNamespace

import torch
from vllm.sequence import IntermediateTensors

from arctic_platform.inference.vllm import pp_routed_experts as pp
from arctic_platform.testing_utils import torch_assert_equal

NUM_LAYERS = 6
TOPK = 2
MAX_TOKENS = 8


def _capturer(output_dtype=torch.uint8):
    return SimpleNamespace(
        device_buffer=torch.zeros((MAX_TOKENS, NUM_LAYERS, TOPK), dtype=torch.int32),
        output_dtype=output_dtype,
    )


def _routes(num_tokens, high=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, high, (num_tokens, NUM_LAYERS, TOPK), dtype=torch.int32, generator=generator
    )


def _runner_class(layers_by_stage, routes):
    """A stand-in for Model Runner V2: its forward captures only its stage's layers."""

    class Runner:
        def __init__(self, stage, num_stages=len(layers_by_stage)):
            self.layers = layers_by_stage[stage]
            self.use_pp = num_stages > 1
            self.is_last_pp_rank = stage == num_stages - 1
            self.capturer = _capturer()
            self.aux_output_connector = SimpleNamespace(_capturer=self.capturer)
            self.execute_model_state = None
            self.seen_keys = None

        def execute_model(
            self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kwargs
        ):
            if intermediate_tensors is not None:
                self.seen_keys = set(intermediate_tensors.tensors)
            n = scheduler_output.num_tokens
            for layer in self.layers:
                self.capturer.device_buffer[:n, layer] = routes[:n, layer]
            self.execute_model_state = SimpleNamespace(
                input_batch=SimpleNamespace(num_tokens_after_padding=n)
            )
            if self.is_last_pp_rank:
                return None
            return IntermediateTensors({"hidden_states": torch.ones(n, 4)})

    pp.patch_execute_model(Runner)
    return Runner


def _run_pipeline(runners, num_tokens, dummy_run=False):
    tensors = None
    step = SimpleNamespace(num_tokens=num_tokens)
    for runner in runners:
        if tensors is not None:
            # What the next stage receives: every tensor, freshly allocated.
            tensors = IntermediateTensors({k: v.clone() for k, v in tensors.items()})
        tensors = runner.execute_model(step, tensors, dummy_run=dummy_run)
    return tensors


def test_last_stage_holds_every_layer_like_pp1():
    # Uneven stages, and a layer without MoE (5) that no stage captures.
    layers_by_stage = [range(0, 1), range(1, 3), range(3, 5)]
    routes = _routes(MAX_TOKENS, seed=1)
    num_tokens = 5
    one_stage = _runner_class([range(0, 6)], routes)(0)
    one_stage.layers = [layer for stage in layers_by_stage for layer in stage]
    _run_pipeline([one_stage], num_tokens)

    cls = _runner_class(layers_by_stage, routes)
    stages = [cls(i) for i in range(len(layers_by_stage))]
    assert _run_pipeline(stages, num_tokens) is None

    last = stages[-1].capturer.device_buffer
    torch_assert_equal(last[:num_tokens], one_stage.capturer.device_buffer[:num_tokens])
    # The relay key never reaches the original runner.
    assert all(pp.ROUTES_KEY not in stage.seen_keys for stage in stages[1:])
