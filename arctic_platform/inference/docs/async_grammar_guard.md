# Asynchronous grammar guard for Model Runner V2

The required speculative-decoding grammar patch rejects every draft from the
first unchecked grammar position. It must preserve that rejection even when
the scheduler supplies placeholder draft IDs after a pause or weight update.

The guard stages its row indices in pinned CPU memory and copies them to the
input device without waiting on the CUDA stream. It uses `index_fill_` on a
clone of the input IDs. The original batch remains available to the drafter.
CPU execution uses ordinary unpinned memory.

No serving flag is needed: this is part of the existing required grammar
patch, including when Arctic inference optimizations are disabled. It does
not change sampling parameters, precision, grammar constraints, or the
checked/unchecked draft boundary.

## Validation

With the repository's inference test dependencies installed:

```sh
python -m pytest tests/inference/unit_tests/test_spec_decode_grammar.py
```

CPU tests cover noncontiguous logit positions, mixed draft widths, shortened
scheduler lists, prefill-only requests, unstructured requests, both integer
input dtypes, and preservation of the original batch. The CUDA-only test
checks repeated asynchronous copies for correct values and rejects a
`cudaStreamSynchronize` in the guard's profiler events. It skips without CUDA.

This patch addresses host-side waiting in the guard. It makes no claim about
an end-to-end throughput gain on a different vLLM release or GPU allocation.
