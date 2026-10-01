## Rollout Replay Patch (vLLM 0.18.0)

This patch extends `SamplingParams` to specify the `max_tokens` of each
child sequence when `n > 1`. With it you can replay a recorded rollout
trace where each of the `n` generations is forced to emit a specific,
precomputed number of tokens.

> Originally written for vLLM 0.14.1 (see git history). The patches in this folder are now rebased to vLLM 0.18.0.

### Apply

```bash
source patch_sampling.sh
```

The script is idempotent (re-running it on an already-patched install
is a no-op) and refuses to apply if the install is not pristine
v0.18-compatible.

### Use the field directly

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
]

sampling_params = [
    SamplingParams(
        n=2,
        temperature=0.8, top_p=1.0,
        max_tokens_n=[25, 50],   # length must == n
        ignore_eos=True,
    ),
    SamplingParams(
        n=3,
        temperature=0.8, top_p=1.0,
        max_tokens_n=[5, 10, 15],
        ignore_eos=True,
    ),
    SamplingParams(
        n=1,
        temperature=0.8, top_p=1.0,
        max_tokens=100,
        ignore_eos=True,
        # max_tokens_n=[100],  # ignored when n == 1
    ),
]

outputs = llm.generate(prompts, sampling_params=sampling_params)
```

Resulting input/output token counts:

```
prompt 0 seq 0: input 5 output 25
prompt 0 seq 1: input 5 output 50
prompt 1 seq 0: input 7 output 5
prompt 1 seq 1: input 7 output 10
prompt 1 seq 2: input 7 output 15
prompt 2 seq 0: input 5 output 100
```

