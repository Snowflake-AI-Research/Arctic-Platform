# Sampling Configuration

Sampling sub-jobs currently configure model name, precision, sequence length,
GPU count, vLLM settings, and optional PEFT settings.

Generation requests can specify temperature, top-p, maximum tokens, stop
strings, and stop token IDs. See the [CLI reference](../cli.md#run-a-generate-smoke-test)
and [REST API reference](../rest-api.md#66-generate---post-job_idgenerate).

A shared typed sampling configuration and documented defaults remain planned.
