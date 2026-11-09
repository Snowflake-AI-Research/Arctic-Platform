from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch


def _install_fix(monkeypatch, *, stop_accepted):
    import arctic_platform.inference.vllm.xgrammar_stop_mask as stop_mask
    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    base_calls = []

    def base_fill_bitmask(_self, bitmask, idx):
        base_calls.append(idx)
        bitmask[idx, 0] = 1 << 5

    monkeypatch.setattr(XgrammarGrammar, "fill_bitmask", base_fill_bitmask)
    monkeypatch.setattr(
        stop_mask,
        "require_supported_vllm_version",
        lambda _feature: "0.31.0",
    )
    stop_mask.ensure_xgrammar_stop_mask_fix()

    accept_calls = []
    rollback_calls = []

    def accept_token(token_id):
        accept_calls.append(token_id)
        return stop_accepted

    matcher = SimpleNamespace(
        stop_token_ids=[5],
        accept_token=accept_token,
        rollback=lambda count: rollback_calls.append(count),
        accept_calls=accept_calls,
        rollback_calls=rollback_calls,
    )
    return stop_mask, XgrammarGrammar, matcher, base_calls


def test_xgrammar_stop_mask_clears_rejected_stop_token(monkeypatch):
    _, grammar_cls, matcher, _ = _install_fix(
        monkeypatch,
        stop_accepted=False,
    )
    bitmask = torch.zeros((1, 1), dtype=torch.int32)

    grammar_cls.fill_bitmask(SimpleNamespace(matcher=matcher), bitmask, 0)

    assert int(bitmask[0, 0]) & (1 << 5) == 0
    assert matcher.accept_calls == [5]
    assert matcher.rollback_calls == []


def test_xgrammar_stop_mask_preserves_accepted_stop_token(monkeypatch):
    _, grammar_cls, matcher, _ = _install_fix(
        monkeypatch,
        stop_accepted=True,
    )
    bitmask = torch.zeros((1, 1), dtype=torch.int32)

    grammar_cls.fill_bitmask(SimpleNamespace(matcher=matcher), bitmask, 0)

    assert int(bitmask[0, 0]) & (1 << 5)
    assert matcher.accept_calls == [5]
    assert matcher.rollback_calls == [1]


def test_xgrammar_stop_mask_install_is_idempotent(monkeypatch):
    stop_mask, grammar_cls, matcher, base_calls = _install_fix(
        monkeypatch,
        stop_accepted=False,
    )
    first_wrapper = grammar_cls.fill_bitmask

    stop_mask.ensure_xgrammar_stop_mask_fix()
    bitmask = torch.zeros((1, 1), dtype=torch.int32)
    grammar_cls.fill_bitmask(SimpleNamespace(matcher=matcher), bitmask, 0)

    assert grammar_cls.fill_bitmask is first_wrapper
    assert base_calls == [0]


def test_xgrammar_stop_mask_checks_vllm_version(monkeypatch):
    import arctic_platform.inference.vllm.xgrammar_stop_mask as stop_mask
    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    original = XgrammarGrammar.fill_bitmask
    monkeypatch.setattr(
        stop_mask,
        "require_supported_vllm_version",
        lambda _feature: (_ for _ in ()).throw(RuntimeError("unsupported")),
    )

    with pytest.raises(RuntimeError, match="unsupported"):
        stop_mask.ensure_xgrammar_stop_mask_fix()

    assert XgrammarGrammar.fill_bitmask is original


def test_xgrammar_stop_mask_matches_pinned_accept_rollback_contract():
    import xgrammar as xgr

    assert version("xgrammar") == "0.2.7"
    vocab = [
        b'{"',
        b"value",
        b'":',
        b'"',
        b"hello",
        b'"}',
        b"x",
        b"<eos>",
        b"<im_end>",
    ]
    tokenizer_info = xgr.TokenizerInfo(
        vocab,
        xgr.VocabType.RAW,
        stop_token_ids=[7, 8],
    )
    matcher = xgr.GrammarMatcher(
        xgr.GrammarCompiler(tokenizer_info).compile_regex(
            r'\{"value":"[A-Za-z0-9 <>|_]+"\}'
        ),
        override_stop_tokens=[7, 8],
    )

    for token_id in [0, 1, 2, 3]:
        assert matcher.accept_token(token_id)
    assert not matcher.accept_token(7)
    assert matcher.accept_token(4)
    assert matcher.accept_token(5)
    assert matcher.accept_token(7)
    assert matcher.is_terminated()

    matcher.rollback(1)

    assert not matcher.is_terminated()
    assert matcher.accept_token(7)
