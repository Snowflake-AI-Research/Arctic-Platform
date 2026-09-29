# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

from arctic_platform.integrations.verl.cortex_generate import prompt_text_from_ids
from arctic_platform.integrations.verl.cortex_generate import strip_pad_ids


class _Tok:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return ",".join(str(i) for i in ids)


def test_strip_pad_ids_both_sides():
    assert strip_pad_ids([0, 0, 10, 11, 12, 0], 0) == [10, 11, 12]
    assert strip_pad_ids([10, 11], 0) == [10, 11]
    assert strip_pad_ids([0, 0, 0], 0) == []
    assert strip_pad_ids([10, 11], None) == [10, 11]


def test_prompt_text_drops_pads_and_keeps_specials():
    text = prompt_text_from_ids(_Tok(), [0, 151644, 10, 151645, 0])
    assert text == "151644,10,151645"
    assert "0" not in text.split(",")
