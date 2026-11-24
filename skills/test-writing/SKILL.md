---
name: test-writing
description: >-
  Write and review tests: harness style, skips, seeds, tensor asserts, numeric
  parity (tight atol, rtol=0, loss+grad_norm), CPU/GPU test topology,
  pytest-xdist ports and gpu_serial. Use when adding or adapting tests, reviewing
  atol/rtol, or comparing a new implementation against a baseline.
---

# Test writing

Follow the existing harness so CPU, GPU, and pytest-xdist runs stay consistent.

**Paths:** this clone can live anywhere. Resolve the checkout at runtime (`git rev-parse --show-toplevel`, or walk up from this `SKILL.md` to the directory that contains `arctic_platform/` and `skills/`). Use that absolute root for `PYTHONPATH`, pytest cwd, and autorun `cd`. Do not hardcode a machine path, clone folder name, or GPU hostname. Autorun root = `$AUTORUN_DIR` or the tree that contains `enqueue.sh` (ask if missing); host from `status.<host>.txt`.

## Sources of truth

| Piece | Where |
| --- | --- |
| Shared helpers / skips / asserts | `arctic_platform/testing_utils.py` |
| Root xdist GPU partition + `MASTER_PORT` | `tests/conftest.py` |
| Shared client requests / Cortex transport tests | `tests/client/` |
| RL client/session harness + fake batches | `tests/rl/rl_harness.py` |
| `gpu_serial` lock wiring (RL) | `tests/rl/conftest.py` |
| `gpu_serial` lock wiring (SFT) | `tests/sft/conftest.py` |
| Pytest markers / timeouts | `pyproject.toml` `[tool.pytest.ini_options]` |

Prefer importing helpers from `arctic_platform.testing_utils`. Do not reinvent skips, port picking, or tensor asserts in the test file.

## Current client API

**The shared frontend does not imply identical backend payloads.** Public clients are exported from [arctic_platform.client](../../arctic_platform/client/__init__.py); shared sync/async operations live in [client/base.py](../../arctic_platform/client/base.py), request builders in [client/requests.py](../../arctic_platform/client/requests.py), and `Request` / `JobHandles` / operation registries in [client/transport.py](../../arctic_platform/client/transport.py). Extend the shared frontend tests in `tests/client/test_client_ops.py` when changing a shared operation.

- Configuration uses nested `training` and `sampling` blocks in [client/config.py](../../arctic_platform/client/config.py). `OnPremConfig(type="onprem")` launches in-process Ray; `CortexConfig(type="remote", protocol="cortex")` talks to a remote HTTP endpoint. There is no on-prem HTTP transport.
- Test the routing target, body, omitted options, and sync/async behavior separately from transport-specific serialization. Canonical `JobId` is `int | str`; preserve either type rather than coercing job handles.
- `fwd_bwd` accepts backend-specific batch contents: Cortex RPC `args` / `kwargs`, versus on-prem pretokenized `batch` / `meta` / `processing`. Assert each backend's documented request and response contract, not byte-identical bodies or metrics across backends.
- On-prem server behavior lives in [common/ray_server.py](../../arctic_platform/common/ray_server.py), with request models in [common/utils/server_models.py](../../arctic_platform/common/utils/server_models.py). Use those implementation anchors rather than removed transport modules.

## In-tree inference tests

`tests/inference` covers `arctic_platform.inference`. It is not part of the platform harness above. Default `pytest` skips this tree unless explicitly targeted; the `[sft,testing]` CI job runs the isolated CPU streaming tests separately. `[cortex]` and `[sft]` do not install the `[inference]` extra. `[rl]` selects `arctic_platform[sft,inference]`; use the vLLM pin in [pyproject.toml](../../pyproject.toml), not a copied version here.

```text
pip install -e ".[inference]"
ARCTIC_INFERENCE_PRECOMPILED_OPS=1 pip install --no-build-isolation -e ".[inference]"
pytest -sv tests/inference
```

The second install is what compiles `csrc/`. Without `ARCTIC_INFERENCE_PRECOMPILED_OPS`, hatchling does not run `setup.py`.

The wheel excludes `arctic_platform/inference/docs/`, `benchmark/`, `projects/`, and `README.md`; see the Hatch exclude list in `pyproject.toml`. `csrc/`, `setup.py`, and `semi_persistence/scripts` remain in the wheel.

Do not pull the whole inference tree into the default suite; its GPU tests import vLLM. Sampling and NCCL tests need a GPU.

## Base style

1. **Prefer `TestCasePlus`** (`unittest.TestCase` subclass) for anything that needs repo paths, temp dirs, or subprocess env. Plain pytest classes/fixtures are fine for pure client/unit tests that only need `monkeypatch` (see `tests/sft/test_sft_client_ops.py`).
2. **Skip with harness decorators**, not ad-hoc `pytest.mark.skipif`:
   - `@require_torch_gpu` — needs CUDA in the test process
   - `@require_torch`, `@require_deepspeed`, `@require_flash_attn`, …
3. **Seeds**: `set_seed(n)` (covers `random` / `numpy` / `torch` / CUDA), not bare `torch.manual_seed`.
4. **Tensor compares**: `torch_assert_close` / `torch_assert_equal` (pass `atol` / `rtol` / `msg=`). Avoid `self.assertTrue(torch.allclose(...))`. See **Numeric parity** below — do not paper over bugs with loose tolerances.
5. **Temp dirs**: `self.get_auto_remove_tmp_dir()` / `_str()` — auto-cleaned, repo-safe.
6. **Subprocess env**: `env = self.get_env()` so `PYTHONPATH` includes the package root (`arctic_platform/…` checkout). Then override keys as needed.
7. **Subprocess launch**: prefer `execute_subprocess_async(cmd, env=…, timeout=…)` over raw `subprocess.run` when you want streamed logs and a non-zero → failure.
8. **Standalone.** Class/method names, docstrings, and comments state the **contract** for a reader who was not in the chat ([../good-code/SKILL.md](../good-code/SKILL.md)). Do not bleed conversation leftovers (`#3:`, `G01`, attack ids, “as discussed”, clone nicknames). If the fact matters, write the reason in full — `save_total_limit` prunes the job dir, not its parent.

## Numeric parity (do not cheat)

When a new path claims to implement the **same math** as a baseline (e.g. `logits_optimization=compute|memory` vs `none`, tiled CE vs full logits, AP/SFT vs native Axolotl under identical DeepSpeed config):

1. **Prefer exact match.** Use `self.assertEqual` on scalars / lists, or `torch_assert_equal` / `torch_assert_close(..., rtol=0, atol=<fp noise>)`.
2. **Default: `rtol=0`.** Absolute `atol` only for documented dtype / kernel noise (fp32 ≈ `1e-5`…`1e-6`; bf16 logprobs often `atol=1e-3` with `rtol=0`). Never start from `atol=1e-2, rtol=1e-2` “just in case.”
3. **Compare more than loss.** Training-step parity should check **`loss` and `grad_norm`** (and any other metrics that must agree). A wrong backward can still print a plausible loss.
4. **Baseline first.** Run the reference path, then the new path on the same seed / batch / config; assert the new curve equals the baseline — not that each is merely finite or decreasing.
5. **Justify any looseness in a comment** next to the assert (what noise source, why that bound still catches real bugs). If you need a wide tolerance, the implementations are probably not equivalent — fix them or redefine the claim.

Anti-patterns: widening `atol`/`rtol` until green; comparing only shapes; asserting `grad_norm > 0` when the claim is “same as baseline.”

## CPU tests vs GPU tests

- **Client / routing / wire-format tests** (CPU): stub kernels or use FakeTransport; never require CUDA. Example: `TestSftCeLogitsOptimizationRouting` in `tests/sft/test_sft_losses.py`.
- **Kernel / numeric / on-prem e2e on CUDA**: mark with `@require_torch_gpu` and run via autorun on the GPU box. Do **not** “prove” GPU CE by forcing the torch fallback on a CPU client — that is not the production path.
- On-prem CUDA e2e runs the Ray server in the client process, so the client subprocess must retain its assigned `CUDA_VISIBLE_DEVICES` slice. Do not hide its GPUs or overwrite the harness's xdist partition with a fixed device id.
- Remote Cortex e2e can run from a CPU client against a live GPU-backed endpoint. A mocked HTTP endpoint validates transport contracts, not the server's CUDA kernels.

## pytest-xdist: ports and GPUs

### Port blocks

Each xdist worker owns **8 contiguous ports**:

```text
base = get_unique_port_number()  # DEFAULT_MASTER_PORT + 8 * worker_id
# base .. base+7 belong to this worker only
```

- Root `tests/conftest.py` claims **`base`** for torch.distributed `MASTER_PORT`.
- Extra listen ports (secondary rendezvous, …) must come from **`base+1` … `base+7`**, probed with:

```python
from arctic_platform.testing_utils import get_unique_port_number, reserve_free_port

_PORT_BASE = get_unique_port_number()
extra_port = reserve_free_port(_PORT_BASE + 1, span=7)
```

**Do not** use `socket.bind(('', 0))` / `getsockname` for suite servers under xdist: OS-ephemeral ports can collide across workers and with other jobs on the host. `reserve_free_port` re-probes inside the worker’s block so a squatted port from a prior crashed session is stepped over.

RL heavyweight tests also stride Ray GCS / dashboard / weight-sync ports by `worker_id * 50` (see `tests/rl/rl_harness.py`) — follow that pattern when adding similar multi-daemon e2e.

### GPU partitioning vs serial lock

Root conftest may set `CUDA_VISIBLE_DEVICES` per worker and `ARL_GPU_PARTITIONED=1` when there are enough GPUs for every worker to get a large enough slice. Otherwise workers share GPUs.

For **heavyweight GPU bodies** that must not overlap when sharing devices:

1. Mark the test (or class) `@pytest.mark.gpu_serial`.
2. Ensure the package `conftest.py` under that tree engages the lock (`tests/rl/conftest.py` or `tests/sft/conftest.py`). The mark alone does nothing outside those directories.
3. The autouse fixture calls `gpu_serial_lock()` (no-op when serial or when partitioned). Those tests also get a longer pytest-timeout (900s).

Lightweight in-process CUDA unit tests (tiny tensors, no DeepSpeed/vLLM cluster) usually need `@require_torch_gpu` only — not `gpu_serial`.

## What belongs where

| Kind of test | Typical location | Notes |
| --- | --- | --- |
| Pure helpers / shared math | `tests/common/` | No network |
| Shared sync/async client ops / Cortex transport | `tests/client/` | FakeTransport or mocked HTTP; CPU by default |
| SFT losses, batch wire, client ops | `tests/sft/` | CPU by default |
| SFT CUDA kernels / local demo | `tests/sft/` | `@require_torch_gpu`; e2e → `gpu_serial` |
| RL engine / generate / e2e | `tests/rl/` | Use `rl_harness` session helpers |
| ZoRRO / model patchers | `tests/zorro_train/` | GPU when kernels matter |
| Arctic Inference package | `tests/inference/` | Skipped by default `pytest`. Not the platform harness |

## Checklist for a new test

- [ ] Uses `TestCasePlus` or a justified plain pytest style
- [ ] Skips via `require_*` (not hand-rolled CUDA checks)
- [ ] `set_seed` + `torch_assert_*` where relevant
- [ ] Parity tests: tight tolerances (`rtol=0`); compare **loss and grad_norm** vs baseline
- [ ] Temp dirs / subprocess env from `TestCasePlus` helpers
- [ ] Any listen port from `get_unique_port_number` + `reserve_free_port`
- [ ] Heavyweight multi-GPU / server spin-up marked `gpu_serial` under `tests/rl` or `tests/sft`
- [ ] GPU visibility matches the tested transport and is documented in the module docstring when it matters
- [ ] GPU kernel tests left for the GPU box (autorun); CPU machines only run routing/wire tests
