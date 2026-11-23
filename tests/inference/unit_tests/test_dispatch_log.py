"""Linear-attention metadata builds log the step's batch class and kernel path when switched on."""

import logging
from types import SimpleNamespace

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

from arctic_platform.inference.vllm import dispatch_log

_LAYER = "model.layers.0.linear_attn"


def _builder(layer):
    context = SimpleNamespace(static_forward_context={_LAYER: layer})
    return SimpleNamespace(
        layer_names=[_LAYER], gdn_prefill_backend="triton", vllm_config=SimpleNamespace(compilation_config=context)
    )


def _metadata(prefills, decodes, spec_decodes=0):
    return SimpleNamespace(
        num_prefills=prefills,
        num_prefill_tokens=7 * prefills,
        num_decodes=decodes,
        num_decode_tokens=decodes,
        num_spec_decodes=spec_decodes,
        num_spec_decode_tokens=2 * spec_decodes,
    )


def test_decodes_beside_a_prefill_are_named_on_the_chunked_kernel():
    line = dispatch_log.describe(_builder(SimpleNamespace(kda_prefill_backend="flashkda")), _metadata(1, 3))
    assert "prefills=1 prefill_tokens=7 decodes=3 decode_tokens=3" in line
    assert line.endswith("non_spec_path=chunked:flashkda")


def test_a_decode_only_step_is_named_recurrent_and_a_gdn_layer_uses_the_builders_backend():
    assert dispatch_log.describe(_builder(SimpleNamespace()), _metadata(0, 2, 1)).endswith(
        "spec_decodes=1 spec_decode_tokens=2 non_spec_path=recurrent"
    )
    assert dispatch_log.describe(_builder(SimpleNamespace()), _metadata(2, 0)).endswith("non_spec_path=chunked:triton")


def test_every_build_logs_one_line_when_switched_on(monkeypatch, caplog):
    monkeypatch.setattr(GDNAttentionMetadataBuilder, "build", lambda self, *args, **kwargs: _metadata(0, 4))
    monkeypatch.setattr(dispatch_log, "_APPLIED", False)
    monkeypatch.setenv("ARCTIC_INFERENCE_LOG_DISPATCH", "1")
    dispatch_log.ensure_dispatch_log()
    with caplog.at_level(logging.INFO, logger=dispatch_log.__name__):
        GDNAttentionMetadataBuilder.build(_builder(SimpleNamespace()), 0, None)
        GDNAttentionMetadataBuilder.build(_builder(SimpleNamespace()), 0, None)
    assert [record.getMessage().split()[-1] for record in caplog.records] == ["non_spec_path=recurrent"] * 2


def test_builds_are_untouched_when_off(monkeypatch):
    original = lambda self, *args, **kwargs: _metadata(0, 4)  # noqa: E731
    monkeypatch.setattr(GDNAttentionMetadataBuilder, "build", original)
    monkeypatch.setattr(dispatch_log, "_APPLIED", False)
    monkeypatch.delenv("ARCTIC_INFERENCE_LOG_DISPATCH", raising=False)
    dispatch_log.ensure_dispatch_log()
    assert GDNAttentionMetadataBuilder.build is original
