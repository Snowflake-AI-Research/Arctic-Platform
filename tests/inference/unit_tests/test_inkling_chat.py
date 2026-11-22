import pytest

pytest.importorskip("vllm.parser.inkling")

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.inkling import INKLING_SPECIAL_TOKENS

from arctic_platform.inference.server.chat import inkling_chat_parser, without_tool_parser


class Tokenizer:
    all_special_tokens = list(INKLING_SPECIAL_TOKENS)
    all_special_ids = list(range(len(INKLING_SPECIAL_TOKENS)))

    def get_vocab(self):
        return {
            **dict(zip(self.all_special_tokens, self.all_special_ids)),
            "thought": 100,
            "answer": 101,
        }

    def decode(self, token_ids):
        vocab = {value: key for key, value in self.get_vocab().items()}
        return "".join(vocab[token_id] for token_id in token_ids)


@pytest.mark.parametrize("thinking", [False, True])
def test_inkling_content_block_closers_are_not_visible_after_reasoning(thinking):
    tokenizer = Tokenizer()
    parser = without_tool_parser(inkling_chat_parser())(tokenizer)
    request = ChatCompletionRequest(model="m", messages=[])
    pieces = (
        ["<|message_model|>", "<|content_thinking|>", "thought", "<|end_message|>"]
        if thinking else []
    ) + [
        "<|message_model|>", "<|content_text|>", "answer",
        "<|end_message|>", "<|content_model_end_sampling|>",
    ]
    deltas = []
    token_ids = [tokenizer.get_vocab()[piece] for piece in pieces]
    for index, (piece, token_id) in enumerate(zip(pieces, token_ids)):
        delta = parser.parse_delta(
            piece,
            [token_id],
            request,
            prompt_token_ids=[tokenizer.get_vocab()["<|message_model|>"]],
            finished=index == len(pieces) - 1,
        )
        if delta is not None:
            deltas.append(delta)

    assert "".join(delta.content or "" for delta in deltas) == "answer"
    assert "".join(delta.reasoning or "" for delta in deltas) == ("thought" if thinking else "")
    assert parser.count_reasoning_tokens(token_ids) == int(thinking)
    assert not any(delta.tool_calls for delta in deltas)
