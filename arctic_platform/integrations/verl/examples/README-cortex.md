# verl × Arctic-Platform × Cortex-training

Companion to [`run_gsm8k_grpo_arl.sh`](run_gsm8k_grpo_arl.sh): same verl
`RemoteBackend` adapter, same `arctic_platform.integrations.verl` plugin,
same GRPO recipe on Qwen3-0.6B / GSM8K. `ARCTIC_BACKEND=cortex` routes every
GPU op (fwd/bwd/step, generate, weight-sync) through
[`arctic_platform/rl/_cortex_dispatch.py`](../../../rl/_cortex_dispatch.py)
into Cortex-training instead of a local Ray server. Driver is CPU-only.
Nothing in `arctic_platform/integrations/verl/` had to change.

## Prereqs

```bash
pip install "arctic_platform[verl]"

# verl fork carrying the RemoteBackend abstraction
# (upstream verl-project/verl#6422 not yet in a stable release).
pip install -e 'git+https://github.com/Snowflake-AI-Research/verl.git@arctic_rl_share_v0.7.1#egg=verl' --no-deps

# verl.third_party.vllm gates on vllm's version at import time even on the
# CPU-only driver.
pip install "vllm==0.18.0"

# vllm 0.18.0 can pull a mismatched torchvision; uninstall or reinstall
# against your torch channel.
pip uninstall -y torchvision
```

Two CPU-driver-only stubs are needed:

- `arctic_inference.server.replica_pool` — imported at module load by
  `arctic_platform.rl.ray_server` but never called on Cortex.
- `flash_attn.bert_padding` — verl's driver-side padding helpers import
  four functions unconditionally; real attention kernels still run
  inside the Cortex training sub-job.

```bash
python - <<'PY'
import arctic_inference, pathlib
d = pathlib.Path(arctic_inference.__file__).parent / "server"
d.mkdir(exist_ok=True)
(d / "__init__.py").touch(exist_ok=True)
(d / "replica_pool.py").write_text(
    'class ReplicaPool:\n'
    '    def __init__(self, *a, **k):\n'
    '        raise RuntimeError("stub; local Ray server disabled on Cortex client.")\n'
)
PY

python - <<'PY'
import site, pathlib, textwrap
d = pathlib.Path(site.getsitepackages()[0]) / "flash_attn"
d.mkdir(exist_ok=True)
(d / "__init__.py").write_text('__version__ = "0.0.0-cpu-stub"\n')
(d / "bert_padding.py").write_text(textwrap.dedent("""
    import torch
    import torch.nn.functional as F

    def index_first_axis(tensor, indices):
        return tensor.index_select(0, indices)

    def unpad_input(hidden_states, attention_mask):
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int64)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_s = int(seqlens.max().item()) if seqlens.numel() else 0
        cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int64), (1, 0))
        b, t = hidden_states.shape[:2]
        flat = hidden_states.reshape(b * t, *hidden_states.shape[2:]) if hidden_states.dim() >= 3 else hidden_states.reshape(b * t)
        return index_first_axis(flat, indices), indices, cu, max_s

    def pad_input(hidden_unpad, indices, batch, seqlen):
        if hidden_unpad.dim() >= 2:
            out = hidden_unpad.new_zeros((batch * seqlen, *hidden_unpad.shape[1:]))
            out.index_copy_(0, indices, hidden_unpad)
            return out.reshape(batch, seqlen, *hidden_unpad.shape[1:])
        out = hidden_unpad.new_zeros((batch * seqlen,))
        out.index_copy_(0, indices, hidden_unpad)
        return out.reshape(batch, seqlen)

    def rearrange(tensor, pattern, **axes):
        from einops import rearrange as _r
        return _r(tensor, pattern, **axes)
""").lstrip())
PY
```

## Cortex credentials

```bash
export CORTEX_PAT=<snowflake programmatic access token>
export ARCTIC_CORTEX_HOST=<account>.<region>.snowflakecomputing.com
# Defaults (override if your endpoint differs):
export ARCTIC_CORTEX_DATABASE=NEUTRINO_DB
export ARCTIC_CORTEX_SCHEMA=PUBLIC
export ARCTIC_CORTEX_ENDPOINT=cortex-training
```

## Data

Uses the same verl-schema parquets as the sibling arl example — download via
[`recipes/rl/verl/simple/download_data.py`](../../../../recipes/rl/verl/simple/download_data.py):

```bash
python recipes/rl/verl/simple/download_data.py --output_dir ~/data/gsm8k
```

## Launch

```bash
NGPU_TRAIN=4 NGPU_SAMPLE=4 \
    bash arctic_platform/integrations/verl/examples/run_gsm8k_grpo_cortex.sh
```

Overrides mirror the sibling arl launcher — `MODEL`, `LR`, `TRAIN_BSZ`,
`MINI_BSZ`, `ROLLOUT_N`, `TOTAL_STEPS`, `TOTAL_EPOCHS`, `GPU_MEM_UTIL`.

## End-to-end status

40-step GSM8K GRPO on Qwen3-0.6B, 4 training + 4 sampling GPUs:

| metric | value |
|---|---|
| Steps | 40 / 40 |
| Wall time | 15m31s (~22.6 s/step) |
| Per-step (median) | `gen ≈ 19s`, `update_actor ≈ 15.5s`, `update_weights ≈ 0.95s` |
| `critic/rewards/mean` | 0.163 → 0.350 (peak 0.45 @ step 37) |
| `actor/loss` | bounded [−0.006, 0.070], no drift |
| `actor/grad_norm` | 1.3–5.0, no NaN |
| `actor/entropy` | 1.5–1.7, no collapse |
| `actor/approx_kl` / `clip_ratio` | 0 / 0 (single-epoch on-policy) |

40 × `train_batch_size=16` = 640 examples is a smoke run; bump
`TOTAL_TRAINING_STEPS` for a real convergence comparison against on-prem.

## Plugin-only changes

Two behaviors live in `arctic_platform` (verl core untouched):

**Rollout coalescing** (`ArcticLLMEngine._batched_generate`). verl's
`AgentLoopManager` fires N concurrent per-prompt Ray calls into one actor.
On-prem those are ms-scale; on Cortex each is a 5–15 s HTTPS RTT and the
sync client blocks the event loop, so 80 prompts serialize to ~15 min.
A `~50 ms` latch collects arrivals with matching sampling params, then
flushes them in one `_client.generate(prompts=[...])` call and fans results
back in prompt-major order. Set `ARCTIC_ROLLOUT_BATCH_WINDOW_MS=0` to
disable.

**Response normalization** (`_normalize_fwd_bwd_response` in the shim).
Cortex returns `fwd_bwd` / `step` scalars flat (`avg_loss`, `grad_norm`,
`last_lr`, …); verl's adapter reads `response["metrics"]["loss"]`. The
shim lifts known scalars into `metrics` and aliases `avg_loss → loss`.
Idempotent when already nested.
