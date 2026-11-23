"""Linear-attention metadata builds log the step's batch counts when switched on."""

import logging
from types import SimpleNamespace

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

from arctic_platform.inference.vllm import dispatch_log

_LAYER = "model.layers.0.linear_attn"


def _builder(layer):
    context = SimpleNamespace(static_forward_context={_LAYER: layer})
    return SimpleNamespace(layer_names=[_LAYER], vllm_config=SimpleNamespace(compilation_config=context))


def _metadata(prefills, decodes, spec_decodes=0):
    return SimpleNamespace(
        num_prefills=prefills,
        num_prefill_tokens=7 * prefills,
        num_decodes=decodes,
        num_decode_tokens=decodes,
        num_spec_decodes=spec_decodes,
        num_spec_decode_tokens=2 * spec_decodes,
    )


def test_a_kda_layer_names_its_prefill_backend_and_the_batch_counts():
    line = dispatch_log.describe(_builder(SimpleNamespace(kda_prefill_backend="flashkda")), _metadata(1, 3, 2))
    assert line == (
        f"linear-attention dispatch group={_LAYER} layer=SimpleNamespace kda_prefill_backend=flashkda "
        "prefills=1 prefill_tokens=7 decodes=3 decode_tokens=3 spec_decodes=2 spec_decode_tokens=4"
    )


def test_a_layer_without_a_recorded_backend_logs_the_counts_only():
    line = dispatch_log.describe(_builder(SimpleNamespace()), _metadata(0, 2))
    assert "kda_prefill_backend" not in line and line.endswith("decodes=2 decode_tokens=2 spec_decodes=0 spec_decode_tokens=0")


def test_every_build_logs_one_line_visible_at_vllms_info_level(monkeypatch, caplog):
    result = _metadata(0, 4)
    monkeypatch.setattr(GDNAttentionMetadataBuilder, "build", lambda self, *args, **kwargs: result)
    monkeypatch.setattr(dispatch_log, "_APPLIED", False)
    monkeypatch.setenv("ARCTIC_INFERENCE_LOG_DISPATCH", "1")
    dispatch_log.ensure_dispatch_log()
    dispatch_log.ensure_dispatch_log()
    vllm_logger = logging.getLogger("vllm")
    monkeypatch.setattr(vllm_logger, "handlers", [caplog.handler])
    monkeypatch.setattr(vllm_logger, "propagate", False)
    with caplog.at_level(logging.WARNING), caplog.at_level(logging.INFO, logger="vllm"):
        for _ in range(2):
            assert GDNAttentionMetadataBuilder.build(_builder(SimpleNamespace()), 0, None) is result
    assert [record.getMessage().split()[-3] for record in caplog.records] == ["decode_tokens=4"] * 2


def test_builds_are_untouched_when_off(monkeypatch):
    original = lambda self, *args, **kwargs: _metadata(0, 4)  # noqa: E731
    monkeypatch.setattr(GDNAttentionMetadataBuilder, "build", original)
    monkeypatch.setattr(dispatch_log, "_APPLIED", False)
    monkeypatch.delenv("ARCTIC_INFERENCE_LOG_DISPATCH", raising=False)
    dispatch_log.ensure_dispatch_log()
    assert GDNAttentionMetadataBuilder.build is original
