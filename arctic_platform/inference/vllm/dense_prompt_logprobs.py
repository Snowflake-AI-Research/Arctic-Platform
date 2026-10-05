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

"""Opt-in dense ``prompt_logprobs``: hand back tensors instead of a dict per position.

Teacher scoring for on-policy distillation sends ``max_tokens=1`` with
``prompt_logprobs=k`` over rows of 100-220k tokens. vLLM answers by turning the
GPU's rectangular logprob tensors into one ``{token_id: Logprob}`` dict per
prompt position -- roughly a million Python objects for one 220k row. Nothing
downstream wants those objects: ArcticInference reads ``.logprob`` and ``.rank``
back out immediately, and the DSS zone head then has to walk the whole tree
again and write it into a JSON header because it contains no tensors.

This patch lets a request opt out of that round trip. When the request carries
``extra_args["dss_prompt_logprobs_format"] == "dense"``, the processor keeps the
tensors vLLM already computed and skips pythonization entirely -- no
``.tolist()``, no ``Logprob`` objects, no dicts, and no detokenization of the
top-k ids whose decoded strings are discarded anyway.

Requests that do not carry the key are byte-for-byte unaffected: they run the
stock vLLM code path.


Why a monkeypatch and not ``ArcticPatch``
-----------------------------------------
``ArcticPatch`` subclasses are applied from ``apply_arctic_patches()``, which
only runs when ``ARCTIC_INFERENCE_ENABLED`` is set, and it raises if an
attribute is patched twice. Dense scoring has to work regardless of whether the
full Arctic stack is enabled, so this follows the convention used by the other
two unconditional patches -- ``router_replay.ensure_router_replay_vllm_patches``
and ``xgrammar_stop_mask.ensure_xgrammar_stop_mask_fix``: replace the function,
tag it, and make re-application a no-op.


.. _coupling:

vLLM coupling surface -- read this before bumping vLLM
------------------------------------------------------
Everything this module assumes about vLLM, so an upgrade is mechanical. Line
numbers are from vLLM 0.30; the names have been stable since v0.10.0.

1. ``vllm.v1.engine.logprobs.LogprobsProcessor``
   - A plain ``@dataclass`` (no ``slots``), so we can tag instances with
     :data:`_WANTS_DENSE_ATTR`. **If vLLM ever adds ``slots=True`` the tag stops
     working** and we would need a side table keyed by ``id(processor)``.
   - ``from_new_request(cls, tokenizer, request)`` is where the request -- and
     therefore ``sampling_params.extra_args`` -- is visible. The processor keeps
     no reference to the request afterwards, which is why the decision has to be
     recorded here rather than in ``_update_prompt_logprobs``.
   - ``_update_prompt_logprobs(self, prompt_logprobs_tensors)`` signature has not
     changed since ``0630d4537a`` (v0.10.0) across 20 minor releases.

2. ``vllm.v1.outputs.LogprobsTensors`` -- the payload. We read exactly two
   fields, **by name**:
   - ``logprob_token_ids``  ``[n-1, k+1]`` int, column 0 = observed token
   - ``logprobs``           ``[n-1, k+1]`` float, column 0 = observed score
   A third, ``selected_token_ranks`` ``[n-1]`` int, carries the observed token's
   vocab rank. We deliberately do not read it: see the contract note below.
   Two further fields were added later (``cu_num_generated_tokens`` in v0.11.1,
   ``cu_num_generated_tokens_tensor`` in v0.29.0) and are ``None`` on the prompt
   path. **Never unpack this NamedTuple positionally** -- that is exactly what
   breaks when vLLM appends another field.

3. The ``n-1`` row count is not a quirk, it is arithmetic. A logprob at prompt
   position ``i`` is ``P(t_i | t_0..t_{i-1})``; position 0 has no preceding
   context, so only positions ``1..n-1`` are scoreable. The producer allocates
   accordingly::

       # vllm/v1/worker/gpu_model_runner.py:5725
       logprobs_tensors = LogprobsTensors.empty_cpu(
           num_prompt_tokens - 1, num_prompt_logprobs + 1)

   and offsets the target tokens by one::

       # vllm/v1/worker/gpu_model_runner.py:5760
       # Get the "target" tokens for each index. For prompt at index i,
       # the token at prompt index i+1 is the "sampled" token we want
       # to gather the logprob for.

   **Row ``i`` of the tensors scores prompt position ``i + 1``.**

4. Chunked prefill is already reassembled for us. The model runner keeps a
   full-size CPU buffer in ``request.in_progress_prompt_logprobs_cpu``, copies
   each chunk into a slice, and only publishes it on the final chunk
   (``gpu_model_runner.py:5721-5744``). So ``_update_prompt_logprobs`` is called
   **once per request with complete tensors**. If that ever changes, the guard
   in :func:`_update_prompt_logprobs_dense` raises instead of silently keeping
   the last chunk.

5. That CPU buffer is engine-owned and reused, so :func:`densify` copies rather
   than views. See ``test_densify_does_not_alias_the_engine_tensors``.

6. ``RequestOutput.prompt_logprobs`` is assigned straight from
   ``logprobs_processor.prompt_logprobs`` (``output_processor.py:373-385``), so
   replacing that field is all it takes to reach the caller -- no second patch.
   In ``RequestOutputKind.DELTA`` the read goes through ``pop_prompt_logprobs``,
   which tests the value for truthiness; a populated dict is truthy, so both
   output kinds work.

7. ``vllm.logprobs.create_prompt_logprobs`` seeds the list with ``None`` at index
   0 so that ``prompt_logprobs[i]`` lines up with ``prompt_token_ids[i]``. The
   dense arrays keep that alignment: they are length ``n``, and index 0 carries
   the not-scored markers.


What the caller has to change
-----------------------------
Opting in is not free of client work; it is only free of *indexing* work.

Unchanged: the length is ``n``, not ``n - 1``, and index 0 is still a
placeholder. Any loop, slice, or zip against ``prompt_token_ids`` keeps the same
offsets, which is the whole reason the length was held at ``n``.

Changed, and a caller has to handle both:

- The opt-in itself -- ``sampling_params["dss_prompt_logprobs_format"] =
  "dense"``. Absent the key, the request is byte-for-byte the stock path.
- The result key -- ``prompt_logprobs_dense`` (three tensors) instead of
  ``prompt_logprobs`` (one dict per position).

Two values the dict format carried are not in the dense one:

- **The observed token's id.** The dict held ``k + 1`` entries, the observed
  token plus the top-k; :func:`densify` keeps only the top-k columns. The
  observed token's *score* is in ``logprobs[i]``; its id is
  ``prompt_token_ids[i]``, which the caller already has because it sent the
  prompt. A caller reading the id back out of the dict has to switch source.
- **``rank``.** Every dict entry carried one. Teacher scoring does not use it,
  and materialising ``selected_token_ranks`` would add a third ``[n]`` tensor to
  every response to serve no current caller. If one appears, it is one slice to
  add here and a key to add in ``worker.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from arctic_platform.inference.utils import require_supported_vllm_version

#: Sampling-param key, passed through ``SamplingParams.extra_args``. The DSS
#: worker pops it off the top-level params and moves it here before building
#: ``SamplingParams``, the same way router replay carries
#: ``dss_stop_token_sequences``.
FORMAT_KEY = "dss_prompt_logprobs_format"

#: The only accepted value for :data:`FORMAT_KEY`.
DENSE = "dense"

#: Key under which the dense arrays travel on the result dict.
RESULT_KEY = "prompt_logprobs_dense"

#: The three arrays :func:`densify` produces, and the contract clients decode.
_DENSE_FIELDS = frozenset({"logprobs", "topk_ids", "topk_logprobs"})

#: Marks a processor whose request asked for dense output. Set in the patched
#: ``from_new_request`` because that is the only place the request is in scope.
_WANTS_DENSE_ATTR = "_dss_wants_dense_prompt_logprobs"

#: Tag used to make :func:`ensure_dense_prompt_logprobs_patch` idempotent.
_PATCH_ATTR = "_dss_dense_prompt_logprobs_patch"

#: Position 0 has no preceding context, so it has no score. These markers fill
#: index 0 and mirror what the dict path expresses as ``None``.
_UNSCORED_LOGPROB = float("nan")
_UNSCORED_TOKEN_ID = -1
_UNSCORED_TOPK_LOGPROB = float("-inf")


def wants_dense(sampling_params: Any) -> bool:
    """Whether this request asked for dense prompt logprobs.

    Duck-typed on ``extra_args`` so it can be unit tested without vLLM.

    Raises:
        ValueError: the key is present with a value we cannot honour. Failing
            loudly beats silently returning the dict format to a caller that is
            going to look for tensors.
    """
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    if FORMAT_KEY not in extra_args:
        return False
    value = extra_args[FORMAT_KEY]
    if value != DENSE:
        raise ValueError(
            f"{FORMAT_KEY} must be {DENSE!r}, got {value!r}"
        )
    return True


def stage_sampling_params(sampling_params: dict) -> bool:
    """Move :data:`FORMAT_KEY` from the top level into ``extra_args``, in place.

    The DSS worker receives ``dss_*`` params at the top level of the sampling
    dict and has to strip them before ``SamplingParams(**params)``, which would
    reject the unknown key. ``extra_args`` is vLLM's sanctioned passthrough --
    *"Arbitrary additional args, that can be used by custom sampling
    implementations, plugins, etc."* -- so parking it there is what carries the
    decision all the way to :func:`wants_dense` inside the engine. Router replay
    moves ``dss_stop_token_sequences`` the same way (``worker.py:910-914``).

    Returns:
        Whether dense output was requested, for the caller's own bookkeeping.

    Raises:
        ValueError: the key is present with a value we cannot honour.
    """
    if FORMAT_KEY not in sampling_params:
        return False
    value = sampling_params.pop(FORMAT_KEY)
    if value != DENSE:
        raise ValueError(f"{FORMAT_KEY} must be {DENSE!r}, got {value!r}")
    extra_args = dict(sampling_params.pop("extra_args", None) or {})
    extra_args[FORMAT_KEY] = value
    sampling_params["extra_args"] = extra_args
    return True


def take_dense(prompt_logprobs: Any) -> dict[str, torch.Tensor] | None:
    """Return the dense arrays if the patch produced them, else ``None``.

    The stock vLLM value is a list (or ``FlatLogprobs``); only the patch puts a
    dict there. Keeping the test here rather than in the worker means the shape
    of the contract lives in one file.
    """
    if isinstance(prompt_logprobs, dict) and _DENSE_FIELDS <= prompt_logprobs.keys():
        return prompt_logprobs
    return None


def densify(
    prompt_logprobs_tensors: Any,
    num_prompt_logprobs: int,
) -> dict[str, torch.Tensor]:
    """Slice ``LogprobsTensors`` into three arrays aligned with the prompt.

    Args:
        prompt_logprobs_tensors: vLLM's ``LogprobsTensors``. Only
            ``logprob_token_ids`` and ``logprobs`` are read, **by name** -- see
            item 2 of the :ref:`coupling surface <coupling>`.
        num_prompt_logprobs: ``k``, i.e. ``SamplingParams.prompt_logprobs``.
            The tensors are ``k + 1`` wide: the observed token plus the top-k.

    Returns:
        Three tensors, each of length ``n`` (the full prompt), index 0 carrying
        the not-scored markers:

        ``logprobs``       ``[n]``     float32, the observed token's score
        ``topk_ids``       ``[n, k]``  int32, most likely first
        ``topk_logprobs``  ``[n, k]``  float32, matching scores

    Raises:
        ValueError: the tensor width does not match ``num_prompt_logprobs + 1``,
            which would mean vLLM changed the layout.
    """
    token_ids = prompt_logprobs_tensors.logprob_token_ids
    values = prompt_logprobs_tensors.logprobs

    scored, width = values.shape
    expected_width = num_prompt_logprobs + 1
    if width != expected_width:
        raise ValueError(
            f"prompt logprob tensors are {width} wide but k="
            f"{num_prompt_logprobs} implies {expected_width} "
            "(observed token + top-k); vLLM changed the layout"
        )

    # Length n: the tensors cover positions 1..n-1, index 0 is the unscored
    # first prompt token. Pre-allocating and assigning also copies us out of the
    # engine's reusable buffer (item 5).
    total = scored + 1
    observed = torch.full((total,), _UNSCORED_LOGPROB, dtype=torch.float32)
    topk_ids = torch.full(
        (total, num_prompt_logprobs), _UNSCORED_TOKEN_ID, dtype=torch.int32
    )
    topk_logprobs = torch.full(
        (total, num_prompt_logprobs), _UNSCORED_TOPK_LOGPROB, dtype=torch.float32
    )

    # Column 0 is the observed token, columns 1.. are the top-k.
    observed[1:] = values[:, 0].to(torch.float32)
    topk_ids[1:] = token_ids[:, 1:].to(torch.int32)
    topk_logprobs[1:] = values[:, 1:].to(torch.float32)

    return {
        "logprobs": observed,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_logprobs,
    }


def ensure_dense_prompt_logprobs_patch() -> None:
    """Teach ``LogprobsProcessor`` to skip pythonization when asked.

    Idempotent. Applied unconditionally from the plugin entry point, because a
    scoring request must be able to opt in whether or not the rest of the Arctic
    stack is enabled.
    """
    require_supported_vllm_version("dense prompt logprobs")

    from vllm.v1.engine.logprobs import LogprobsProcessor

    if getattr(LogprobsProcessor.from_new_request, _PATCH_ATTR, False):
        return

    original_from_new_request = LogprobsProcessor.from_new_request
    original_update = LogprobsProcessor._update_prompt_logprobs

    def from_new_request(cls, tokenizer, request):
        # The request is only in scope here; the processor does not keep it.
        # Decide now and record the answer on the instance.
        processor = original_from_new_request(tokenizer, request)
        setattr(
            processor,
            _WANTS_DENSE_ATTR,
            wants_dense(getattr(request, "sampling_params", None)),
        )
        return processor

    def _update_prompt_logprobs_dense(self, prompt_logprobs_tensors):
        if not getattr(self, _WANTS_DENSE_ATTR, False):
            original_update(self, prompt_logprobs_tensors)
            return

        # vLLM reassembles chunked prefill before publishing (item 4), so this
        # runs once with the complete tensors. Refuse rather than overwrite if
        # that assumption ever stops holding -- a silent partial result would
        # become training data.
        if isinstance(self.prompt_logprobs, dict):
            raise RuntimeError(
                "dense prompt logprobs received a second tensor chunk; vLLM no "
                "longer reassembles chunked prefill before publishing "
                "(see gpu_model_runner._get_prompt_logprobs_dict)"
            )

        assert self.num_prompt_logprobs is not None
        # Replacing the field is enough to reach RequestOutput (item 6).
        self.prompt_logprobs = densify(
            prompt_logprobs_tensors, self.num_prompt_logprobs
        )

    # Tag the plain function: attribute lookup on the bound classmethod falls
    # through to ``__func__``, so the guard at the top of this function sees it.
    setattr(from_new_request, _PATCH_ATTR, True)

    LogprobsProcessor.from_new_request = classmethod(from_new_request)
    LogprobsProcessor._update_prompt_logprobs = _update_prompt_logprobs_dense
