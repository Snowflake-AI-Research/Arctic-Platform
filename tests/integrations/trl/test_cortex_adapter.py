# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the Cortex adapter must translate, and what it must refuse.

A zone rejects a mistranslated envelope, so translation bugs are loud. The two
that would *not* be loud are a mis-framed ``labels`` tensor and a ``loss_mask``
one position too wide: both produce a running job that optimises the wrong
thing. Those get the most attention here.
"""

import pytest
import torch

cortex = pytest.importorskip("arctic_platform.integrations.trl.cortex")

CortexTRLAdapter = cortex.CortexTRLAdapter
next_token_labels = cortex.next_token_labels


class _RecordingClient:
    """Captures what the adapter sends, and answers plausibly."""

    def __init__(self, *, logprobs=None, entropy=None, omit_logprobs=False):
        self.fwd_no_grad_payload = None
        self.fwd_bwd_payload = None
        self._logprobs = logprobs
        self._entropy = entropy
        self._omit_logprobs = omit_logprobs
        self.generate_calls = 0

    def fwd_no_grad(self, payload):
        self.fwd_no_grad_payload = payload
        if self._omit_logprobs:
            return {"job_id": "j"}
        rows = payload["kwargs"]["input_ids"]
        lp = self._logprobs if self._logprobs is not None else torch.zeros_like(rows, dtype=torch.float32)
        out = {"logprobs": lp}
        if self._entropy is not None:
            out["entropy"] = self._entropy
        return out

    def fwd_bwd(self, payload):
        self.fwd_bwd_payload = payload
        return {"avg_loss": 0.5}

    def generate(self, *a, **k):
        self.generate_calls += 1
        return "generated"


def _payload(lens=(4, 3), width=5, *, processing=None, extra=None):
    """An on-prem-dialect payload with right-padded rows of the given lengths."""
    b = len(lens)
    input_ids = torch.zeros(b, width, dtype=torch.long)
    attention_mask = torch.zeros(b, width, dtype=torch.long)
    for row, n in enumerate(lens):
        input_ids[row, :n] = torch.arange(1, n + 1) + 100 * row
        attention_mask[row, :n] = 1
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    if extra:
        batch.update(extra)
    return {
        "batch": batch,
        "meta": {"temperature": 1.0},
        "processing": (
            processing if processing is not None else {"post": ["apply_temperature", "compute_entropy_and_logprobs"]}
        ),
    }


class TestLabelFraming:
    """The quiet failure mode: labels off by one supervise the wrong tokens."""

    def test_labels_are_the_next_token(self):
        p = _payload(lens=(4,), width=5)
        labels = next_token_labels(p["batch"])
        ids = p["batch"]["input_ids"]
        # Positions 0..2 predict tokens 1..3.
        assert torch.equal(labels[0, :3], ids[0, 1:4])

    def test_last_real_token_is_not_a_target(self):
        # Row of length 4 in a width-5 tensor: index 3 is the last real token and
        # has no successor, so it must not be supervised.
        labels = next_token_labels(_payload(lens=(4,), width=5)["batch"])
        assert labels[0, 3] == -100

    def test_padding_is_not_a_target(self):
        labels = next_token_labels(_payload(lens=(4,), width=6)["batch"])
        assert torch.all(labels[0, 4:] == -100)

    def test_a_full_width_row_still_drops_its_final_position(self):
        # No padding to hide behind: the last column must still be excluded.
        labels = next_token_labels(_payload(lens=(5,), width=5)["batch"])
        assert labels[0, -1] == -100
        assert torch.all(labels[0, :-1] != -100)

    def test_supervised_count_is_one_less_than_the_row_length(self):
        labels = next_token_labels(_payload(lens=(4, 3), width=5)["batch"])
        assert int((labels[0] != -100).sum()) == 3
        assert int((labels[1] != -100).sum()) == 2

    def test_caller_supplied_labels_are_not_overwritten(self):
        mine = torch.full((1, 5), 7, dtype=torch.long)
        client = _RecordingClient()
        CortexTRLAdapter(client).fwd_no_grad(_payload(lens=(4,), width=5, extra={"labels": mine}))
        assert torch.equal(client.fwd_no_grad_payload["kwargs"]["labels"], mine)


class TestEnvelopeTranslation:
    def test_batch_is_rehomed_under_kwargs(self):
        client = _RecordingClient()
        CortexTRLAdapter(client).fwd_no_grad(_payload())
        sent = client.fwd_no_grad_payload
        assert set(sent) == {"kwargs", "processing"}
        assert "batch" not in sent and "meta" not in sent

    def test_on_prem_post_processors_are_replaced(self):
        client = _RecordingClient()
        CortexTRLAdapter(client).fwd_no_grad(_payload())
        assert client.fwd_no_grad_payload["processing"]["post"] == ["compute_logprobs"]

    def test_loss_fn_and_config_survive(self):
        client = _RecordingClient()
        CortexTRLAdapter(client).fwd_bwd(
            _payload(processing={"post": [], "loss_fn": "grpo", "config": {"dp_size": 1}})
        )
        processing = client.fwd_bwd_payload["processing"]
        assert processing["loss_fn"] == "grpo"
        assert processing["config"] == {"dp_size": 1}

    def test_absent_config_is_not_invented(self):
        client = _RecordingClient()
        CortexTRLAdapter(client).fwd_bwd(_payload(processing={"post": [], "loss_fn": "grpo"}))
        assert "config" not in client.fwd_bwd_payload["processing"]

    def test_an_unrecognised_post_processor_is_refused(self):
        # Dropping it would leave a run that looks healthy and optimises
        # something other than what was asked for.
        client = _RecordingClient()
        with pytest.raises(ValueError, match="cannot translate post-processors"):
            CortexTRLAdapter(client).fwd_no_grad(_payload(processing={"post": ["something_new"]}))
        assert client.fwd_no_grad_payload is None

    def test_unrelated_methods_pass_through(self):
        client = _RecordingClient()
        assert CortexTRLAdapter(client).generate() == "generated"
        assert client.generate_calls == 1


class TestLossMaskTightening:
    """The other quiet failure: a mask one position wider than the labels."""

    def test_loss_mask_is_narrowed_to_the_labels(self):
        client = _RecordingClient()
        lens, width = (4, 3), 5
        attention = _payload(lens=lens, width=width)["batch"]["attention_mask"]
        CortexTRLAdapter(client).fwd_bwd(
            _payload(lens=lens, width=width, extra={"loss_mask": attention.to(torch.float32)})
        )
        sent = client.fwd_bwd_payload["kwargs"]
        assert torch.equal(sent["loss_mask"] != 0, sent["labels"] != -100)
        # Strictly tighter than what the caller passed: one position per row.
        assert int(attention.sum()) - int(sent["loss_mask"].sum()) == len(lens)

    def test_loss_mask_dtype_is_preserved(self):
        client = _RecordingClient()
        attention = _payload()["batch"]["attention_mask"]
        CortexTRLAdapter(client).fwd_bwd(_payload(extra={"loss_mask": attention.to(torch.float64)}))
        assert client.fwd_bwd_payload["kwargs"]["loss_mask"].dtype is torch.float64

    def test_no_loss_mask_is_not_fabricated(self):
        client = _RecordingClient()
        CortexTRLAdapter(client).fwd_bwd(_payload())
        assert "loss_mask" not in client.fwd_bwd_payload["kwargs"]


class TestTheLimitsAreEnforcedNotHidden:
    def test_a_temperature_other_than_one_is_refused_up_front(self):
        # Fail at construction, not on the first request: the zone has no
        # apply_temperature, so any other value is silently mis-scaled logits.
        with pytest.raises(ValueError, match="apply_temperature"):
            CortexTRLAdapter(_RecordingClient(), temperature=0.7)

    def test_temperature_one_is_accepted(self):
        assert CortexTRLAdapter(_RecordingClient(), temperature=1.0).temperature == 1.0

    def test_entropy_is_zeros_when_the_zone_returns_none(self):
        client = _RecordingClient()
        out = CortexTRLAdapter(client).fwd_no_grad(_payload())
        entropy = out["batch"]["entropy"]
        assert entropy.shape == out["batch"]["logprobs"].shape
        assert torch.all(entropy == 0)

    def test_a_zone_supplied_entropy_is_passed_through(self):
        # So this stops being a fabricated zero the moment a zone computes it.
        real = torch.full((2, 5), 0.25)
        out = CortexTRLAdapter(_RecordingClient(entropy=real)).fwd_no_grad(_payload())
        assert torch.equal(out["batch"]["entropy"], real)

    def test_a_forward_without_logprobs_is_an_error(self):
        with pytest.raises(RuntimeError, match="no logprobs"):
            CortexTRLAdapter(_RecordingClient(omit_logprobs=True)).fwd_no_grad(_payload())
