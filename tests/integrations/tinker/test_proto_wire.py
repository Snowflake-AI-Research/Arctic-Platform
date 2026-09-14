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
"""Protobuf codec round-trips, with the installed SDK as the oracle.

``test_wire_schema.py`` checks the router's models against hard-coded copies of
the SDK's types. That is what let the wire move to protobuf while every schema
test still passed, so these tests deliberately do the opposite: they call the
SDK's own serializer to build requests and its own deserializer to read
responses. A drift that matters now fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

tinker = pytest.importorskip("tinker", reason="the proto wire schema ships with the tinker SDK")

from tinker.proto import request_conv  # noqa: E402
from tinker.proto import response_conv  # noqa: E402

from arctic_platform.integrations.tinker.proto_wire import decode_forward_backward_request  # noqa: E402
from arctic_platform.integrations.tinker.proto_wire import encode_forward_backward_output  # noqa: E402
from arctic_platform.integrations.tinker.proto_wire import encode_sample_response  # noqa: E402
from arctic_platform.integrations.tinker.proto_wire import wants_proto  # noqa: E402


def _sdk_request(loss_fn="importance_sampling", forward_only=False):
    """A ForwardBackwardRequest serialized exactly as the SDK would send it."""
    data = [
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints([11, 12, 13, 14]),
            loss_fn_inputs={
                "target_tokens": [12, 13, 14, 15],
                "weights": [0.0, 1.0, 1.0, 1.0],
                "advantages": [0.0, 0.25, 0.5, 0.75],
                "logprobs": [0.0, -0.5, -1.5, -2.5],
            },
        ),
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints([21, 22]),
            loss_fn_inputs={
                "target_tokens": [22, 23],
                "weights": [1.0, 1.0],
                "advantages": [-1.0, 2.0],
                "logprobs": [-0.1, -0.2],
            },
        ),
    ]
    request = tinker.types.ForwardBackwardRequest(
        model_id="main",
        seq_id=1,
        forward_backward_input=tinker.types.ForwardBackwardInput(
            data=data, loss_fn=loss_fn, loss_fn_config={"eps_clip": 0.2}
        ),
    )
    msg = request_conv.forward_backward_request_to_proto(request)
    msg.forward_only = forward_only
    return msg.SerializeToString()


class TestRequestDecode:
    def test_tokens_and_tensors_survive_the_sdks_encoder(self):
        decoded, forward_only = decode_forward_backward_request(_sdk_request())
        assert forward_only is False
        assert decoded.model_id == "main"
        assert decoded.seq_id == 1
        fbi = decoded.forward_backward_input
        assert fbi.loss_fn == "importance_sampling"
        assert fbi.loss_fn_config == {"eps_clip": pytest.approx(0.2)}

        assert [c.tokens for c in fbi.data[0].model_input.chunks] == [[11, 12, 13, 14]]
        assert [c.tokens for c in fbi.data[1].model_input.chunks] == [[21, 22]]
        assert fbi.data[0].loss_fn_inputs["target_tokens"].data == [12, 13, 14, 15]
        assert fbi.data[0].loss_fn_inputs["advantages"].data == pytest.approx([0.0, 0.25, 0.5, 0.75])
        assert fbi.data[1].loss_fn_inputs["advantages"].data == pytest.approx([-1.0, 2.0])

    def test_dtypes_are_not_collapsed(self):
        """`target_tokens` is int64 and `weights` float32; mixing them up would
        reinterpret the bytes rather than fail."""
        decoded, _ = decode_forward_backward_request(_sdk_request())
        inputs = decoded.forward_backward_input.data[0].loss_fn_inputs
        assert inputs["target_tokens"].dtype == "int64"
        assert inputs["weights"].dtype == "float32"

    def test_forward_only_is_carried(self):
        """`forward` rides this endpoint behind the flag, so losing it would
        silently run a backward pass and corrupt the gradient."""
        _, forward_only = decode_forward_backward_request(_sdk_request(forward_only=True))
        assert forward_only is True


class TestForwardBackwardOutputEncode:
    def _payload(self, per_datum_lengths=(4, 2)):
        return {
            "loss_fn_output_type": "ArrayRecord",
            "metrics": {"loss:mean": 0.5, "grad_norm:mean": 1.25},
            "loss_fn_outputs": [
                {
                    "logprobs": {
                        "dtype": "float32",
                        "data": [round(-0.1 * (i + 1), 4) for i in range(n)],
                        "shape": [n],
                    }
                }
                for n in per_datum_lengths
            ],
        }

    def test_sdk_reads_back_what_we_wrote(self):
        payload = self._payload()
        output = response_conv.deserialize_forward_backward_output(
            encode_forward_backward_output(payload)
        )
        assert output.metrics == {"loss:mean": pytest.approx(0.5), "grad_norm:mean": pytest.approx(1.25)}
        assert len(output.loss_fn_outputs) == 2
        for expected, actual in zip(payload["loss_fn_outputs"], output.loss_fn_outputs):
            np.testing.assert_allclose(
                np.asarray(actual["logprobs"].data, dtype=np.float32),
                np.asarray(expected["logprobs"]["data"], dtype=np.float32),
            )

    def test_ragged_datums_keep_their_own_lengths(self):
        """Per-datum offsets are byte offsets; getting that wrong would hand the
        second datum a slice of the first."""
        output = response_conv.deserialize_forward_backward_output(
            encode_forward_backward_output(self._payload((5, 1, 3)))
        )
        assert [len(d["logprobs"].data) for d in output.loss_fn_outputs] == [5, 1, 3]

    def test_metrics_only_response_is_valid(self):
        """`optim_step`-style replies carry no per-token output at all."""
        output = response_conv.deserialize_forward_backward_output(
            encode_forward_backward_output({"metrics": {"loss:mean": 2.0}, "loss_fn_outputs": []})
        )
        assert output.loss_fn_outputs == []
        assert output.metrics == {"loss:mean": pytest.approx(2.0)}

    def test_fields_missing_from_some_datums_are_dropped_not_misaligned(self):
        """A field absent from one datum cannot be packed: its offsets would not
        cover the batch, shifting every later datum's slice."""
        payload = {
            "metrics": {},
            "loss_fn_outputs": [
                {"logprobs": {"dtype": "float32", "data": [-1.0, -2.0], "shape": [2]},
                 "extra": {"dtype": "float32", "data": [1.0, 2.0], "shape": [2]}},
                {"logprobs": {"dtype": "float32", "data": [-3.0], "shape": [1]}},
            ],
        }
        output = response_conv.deserialize_forward_backward_output(
            encode_forward_backward_output(payload)
        )
        assert [sorted(d) for d in output.loss_fn_outputs] == [["logprobs"], ["logprobs"]]
        np.testing.assert_allclose(np.asarray(output.loss_fn_outputs[1]["logprobs"].data), [-3.0])


class TestSampleResponseEncode:
    def test_sdk_reads_back_tokens_logprobs_and_stop_reason(self):
        payload = {
            "sequences": [
                {"tokens": [5, 6, 7], "logprobs": [-0.1, -0.2, -0.3], "stop_reason": "stop"},
                {"tokens": [8, 9], "logprobs": [-1.0, -2.0], "stop_reason": "length"},
            ]
        }
        response = response_conv.deserialize_sample_response(encode_sample_response(payload))
        assert [s.stop_reason for s in response.sequences] == ["stop", "length"]
        assert response.sequences[0].tokens_np.tolist() == [5, 6, 7]
        assert response.sequences[1].tokens_np.tolist() == [8, 9]
        np.testing.assert_allclose(response.sequences[0].logprobs_np, [-0.1, -0.2, -0.3], rtol=1e-6)

    def test_missing_logprobs_stay_absent(self):
        payload = {"sequences": [{"tokens": [1, 2], "logprobs": None, "stop_reason": "stop"}]}
        response = response_conv.deserialize_sample_response(encode_sample_response(payload))
        assert response.sequences[0].logprobs_np is None

    def test_unknown_stop_reason_is_refused(self):
        """The SDK raises on an unmapped enum, so guessing here would turn a
        clear server-side error into a confusing client-side one."""
        with pytest.raises(ValueError, match="unknown stop_reason"):
            encode_sample_response({"sequences": [{"tokens": [1], "stop_reason": "truncated"}]})


class TestContentNegotiation:
    @pytest.mark.parametrize(
        "accept,content_type,expected",
        [
            ("application/x-protobuf", None, True),
            (None, "application/x-protobuf", True),
            ("application/json", None, False),
            (None, None, False),
            ("application/json, application/x-protobuf", None, True),
        ],
    )
    def test_proto_is_detected_from_either_header(self, accept, content_type, expected):
        assert wants_proto(accept, content_type) is expected
