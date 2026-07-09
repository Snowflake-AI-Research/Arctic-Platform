# verl integration

Arctic RL ⇄ [verl](https://github.com/volcengine/verl) adapter, packaged
as an opt-in subpackage so verl can pick it up via its
`VERL_USE_EXTERNAL_MODULES` plugin hook. This lets us own the Arctic
runtime and its verl adapter in one repo, and to ship the plugin
without patching verl's source tree.

## Install

```bash
pip install "arctic_platform[verl]"
# verl itself is user-supplied; pin the version in your launcher.
pip install "verl==<the version you tested against>"
```

## Use

Two things need to be wired at launch: the plugin hook and Hydra's
config search path.

```bash
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register

# Point Hydra at the plugin's config dir so `remote_backend=arctic`
# resolves to `remote_backend/arctic.yaml` under this package.
CFG_DIR=$(python -c "import arctic_platform.integrations.verl as p, os; \
                     print(os.path.join(os.path.dirname(p.__file__), 'config'))")

python -m verl.trainer.main_ppo \
    hydra.searchpath="[file://${CFG_DIR}]" \
    trainer.remote_backend=arctic \
    remote_backend=arctic \
    ...
```

The `VERL_USE_EXTERNAL_MODULES` hook triggers our
[`register.py`](./register.py) on verl bootstrap, which lazily binds
three loaders against the name `"arctic"`:

- `RemoteBackendRegistry("arctic")`
  &rarr; [`ArcticRLClientWrapper`](./adapter.py) (`RemoteBackend` impl).
- `RemoteBackendRegistry` actor-rollout worker slot
  &rarr; [`ArcticRLActorRolloutRefWorker`](./worker.py) (the CPU-only
  forwarder verl instantiates for the `ActorRollout(Ref)` role).
- `RolloutReplicaRegistry("arctic")`
  &rarr; [`ArcticReplica`](./rollout.py) (rollout replica hosting our own
  vLLM engine, shared with the training backend).

All three are wired up as loader callables, not eager imports &mdash; a
plain `import arctic_platform.integrations.verl.register` stays cheap
and does not pull in DeepSpeed, vLLM, or the Arctic RL runtime unless
verl actually resolves the backend at `fit()` time.

## Example launchers

Reference scripts (matching Golden Runs 1 & 2 in
[Arctic-Platform#35](https://github.com/Snowflake-AI-Research/Arctic-Platform/issues/35)):

- [`examples/run_gsm8k_grpo_arl.sh`](./examples/run_gsm8k_grpo_arl.sh)
  &mdash; 0.6B GSM8K, single GPU, GRPO.
- [`examples/run_bird_grpo_arl.sh`](./examples/run_bird_grpo_arl.sh)
  &mdash; 0.6B BIRD, single GPU, GRPO with ZoRRo + CUDA-IPC weight sync.

## Files

| File | Role |
|---|---|
| `register.py` | Loaded by `VERL_USE_EXTERNAL_MODULES`; registers "arctic" on both verl registries via lazy loaders. |
| `adapter.py` | `ArcticRLClientWrapper` &mdash; implements verl's `RemoteBackend` ABC on top of `arctic_platform.rl`. |
| `rollout.py` | `ArcticReplica` / `ArcticLLMServer` &mdash; hosts Arctic's own vLLM engine as a verl `RolloutReplica`. |
| `worker.py` | `ArcticRLActorRolloutRefWorker` &mdash; per-backend forwarder worker with verl dispatch decorators. |
| `grpo_loss.py` | Server-side verl-shaped GRPO loss (registered as `"verl_grpo"` on `arctic_platform.rl.processors.LOSS_FNS`). |
| `config/remote_backend/arctic.yaml` | Per-backend Hydra config block loaded into `config.remote_backend` when `remote_backend=arctic`. |

## Backward compatibility

The GRPO loss previously lived at
`arctic_platform/rl/processors/verl_grpo.py`. That module remains as a
one-line re-export shim so any existing import path still resolves.

## Companion verl-core change

The paired verl-core change lives at
[verl-project/verl#6422](https://github.com/verl-project/verl/pull/6422)
&mdash; it lands the generic `RemoteBackend` ABC + registry + V1-hook
trainer with **zero Arctic-specific code**, and enables the plugin-hook
direction that makes this integration possible.

## Smoke tests

Both runs use the shipped recipe launchers **unchanged** &mdash;
plugin is loaded solely via
`VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register`
+ `hydra.searchpath=[file://.../integrations/verl/config]` &mdash; no
verl-core patches. Weight sync runs over CUDA-IPC and the receiver logs
`[weight-sync names validated] context=cuda_ipc sender=310 expected=310`
on every step.

### Golden Run 1 &mdash; GSM8K (Qwen3-1.7B, single H200, 4 steps)

Recipe: [`recipes/rl/verl/simple/run_qwen3_1.7b_gsm8k_grpo_arl.sh`](../../../recipes/rl/verl/simple/run_qwen3_1.7b_gsm8k_grpo_arl.sh)

| step | MFU (actor) | throughput (tok/s) | step time (s) |
|---:|---:|---:|---:|
| 1 | 0.261 | 5164 | 48.4 |
| 2 | 0.256 | 5860 | 42.4 |
| 3 | 0.260 | 6283 | 39.0 |
| 4 | 0.264 | 6478 | 38.4 |

Validation at step 4 (`val-core/openai/gsm8k/acc/mean@1`): `0.00152`
(Qwen3-1.7B baseline; expected to be near-zero at 4 steps, we're just
confirming the eval path lights up).

### Golden Run 2 &mdash; BIRD text-to-SQL (Qwen3-0.6B, single H200, 20 steps)

Recipe: [`recipes/rl/verl/txt2sql/run_qwen3_32b_bird_grpo_arl_zorro_yes.sh`](../../../recipes/rl/verl/txt2sql/run_qwen3_32b_bird_grpo_arl_zorro_yes.sh)
(shipped 32B/32-GPU launcher, invoked with CLI overrides for Qwen3-0.6B / 1 GPU
so the trajectory is directly comparable to the pre-plugin arctic-verl reference log
`bird_grpo_Qwen3-0.6B_ngpu1_gbs8_mbs16_rolln16_arl_zorro_yes.log`).

Per-step training reward (`critic/rewards/mean`, format bonus is 0.1):

| steps | mean reward | resp\_len | MFU | step (s) |
|---:|---:|---:|---:|---:|
| 1&ndash;5   | 0.335 | 1116 | 0.487 | 53.5 |
| 6&ndash;10  | 0.322 | 1073 | 0.491 | 46.3 |
| 11&ndash;15 | 0.309 | 1061 | 0.484 | 44.9 |
| 16&ndash;20 | 0.415 |  920 | 0.507 | 44.1 |

Reward window average moves from **0.335 &rarr; 0.415** across 20 steps
while response length shrinks (1116 &rarr; 920 tokens) &mdash; the model
is learning to produce shorter, more-often-correct SQL, not just format
compliance. Validation @ step 20:

| metric | ours (plugin + verl PR-B) | pre-plugin reference |
|---|---:|---:|
| `val-core/bird/reward/mean@1`      | 0.279 | 0.294 |
| `val-aux/bird/execution_success/mean@1` | 0.531 | 0.522 |
| `val-aux/bird/format_correct/mean@1`    | 0.960 | 0.943 |

All three within noise of the reference; format compliance and executable-SQL
rate are slightly higher.
