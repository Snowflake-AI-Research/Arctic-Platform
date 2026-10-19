import json

import pytest
import xgrammar as xgr

from arctic_platform.inference.vllm import xgrammar_request_errors

VOCAB = [chr(i) for i in range(32, 127)] + ["<eos>"]
OVERFLOW = (
    "Internal check failed: (additional_size <= kMaxRepresentableSize - data_.size()) is false: Compact2DArray data "
    "size would exceed the int32_t limit, likely due to an unbounded grammar pattern causing state explosion"
)


def _grammar(monkeypatch, *, overflow_after=None, overflow_in_fill=False):
    from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

    monkeypatch.setattr(XgrammarGrammar, "accept_tokens", XgrammarGrammar.accept_tokens)
    monkeypatch.setattr(XgrammarGrammar, "fill_bitmask", XgrammarGrammar.fill_bitmask)
    info = xgr.TokenizerInfo(VOCAB, stop_token_ids=[len(VOCAB) - 1])
    schema = {"type": "object", "properties": {"a": {"type": "string", "minLength": 200}}, "required": ["a"]}
    ctx = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
    matcher = xgr.GrammarMatcher(ctx)
    if overflow_after is not None or overflow_in_fill:
        accepted = []

        class Overflowing:  # the real matcher until its history is "full", then the RuntimeError xgrammar 0.2.7 raises
            def __getattr__(self, name):
                return getattr(matcher, name)

            def accept_token(self, token_id):
                if len(accepted) == overflow_after:
                    raise RuntimeError(OVERFLOW)
                accepted.append(token_id)
                return matcher.accept_token(token_id)

            def fill_next_token_bitmask(self, bitmask, idx):
                if overflow_in_fill:
                    raise RuntimeError(OVERFLOW)
                return matcher.fill_next_token_bitmask(bitmask, idx)

        matcher = Overflowing()
    return XgrammarGrammar, XgrammarGrammar(matcher=matcher, vocab_size=len(VOCAB), ctx=ctx)


def _ids(text):
    return [VOCAB.index(ch) for ch in text]


def test_overflow_finishes_only_that_request(monkeypatch):
    grammar_cls, grammar = _grammar(monkeypatch, overflow_after=8)
    with pytest.raises(RuntimeError, match="Compact2DArray"):
        grammar.accept_tokens("req-0", _ids('{"a":"xxxxx'))

    _, grammar = _grammar(monkeypatch, overflow_after=8)
    xgrammar_request_errors.ensure_xgrammar_request_error_isolation()
    xgrammar_request_errors.ensure_xgrammar_request_error_isolation()
    assert grammar_cls.accept_tokens._arctic_request_error_isolation
    assert grammar.accept_tokens("req-0", _ids('{"a":"xxxxx')) is False


def test_overflow_while_filling_the_mask_fails_the_request_at_its_next_accept(monkeypatch):
    _, grammar = _grammar(monkeypatch, overflow_in_fill=True)
    bitmask = xgr.allocate_token_bitmask(2, len(VOCAB))
    with pytest.raises(RuntimeError, match="Compact2DArray"):
        grammar.fill_bitmask(bitmask, 1)

    xgrammar_request_errors.ensure_xgrammar_request_error_isolation()
    bitmask.zero_()
    grammar.fill_bitmask(bitmask, 1)
    assert bool((bitmask[1] == -1).all()) and bool((bitmask[0] == 0).all())
    assert grammar.accept_tokens("req-2", _ids("{")) is False


def test_isolation_leaves_accepting_and_rejecting_unchanged(monkeypatch):
    _, grammar = _grammar(monkeypatch)
    xgrammar_request_errors.ensure_xgrammar_request_error_isolation()
    assert grammar.accept_tokens("req-1", _ids('{"a":"' + "x" * 199)) is True
    assert grammar.accept_tokens("req-1", _ids('"')) is False  # minLength 200: the grammar's own rejection
    assert grammar.accept_tokens("req-1", _ids('x"}')) is True
