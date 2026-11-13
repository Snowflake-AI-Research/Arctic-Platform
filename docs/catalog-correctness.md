# Catalog correctness through Cortex Training

This prototype compares a catalog training job with a separately generated Hugging Face reference on the client. It implements comparisons corresponding to the five SFT correctness checks: gradients, optimizer state, checkpoint resume, saved-checkpoint sampling, and a 100-step training and validation trajectory. It has been exercised with the Qwen3-0.6B full-training H200 recipe. The reference requires one GPU, batch size one, BF16 full-parameter training, ZeRO stage 2, and AdamW with zero weight decay. It rejects unsupported recipes rather than silently changing them.

It does not yet reproduce all of the original suite's coverage:

- Gradient and optimizer comparisons follow the recipe's accumulation precision and do not collect the original worker participation receipts. The tested Qwen recipe uses BF16 accumulation.
- Resume loads the checkpoint into a job that has already trained ten steps; it does not test initialization of an untouched job from the checkpoint.
- Training, validation and sampling score every token after the first. Answer-only masks, padding and representative SFT data are not covered by the supplied workload.
- Distributed training, LoRA, MoE and the RL checks are not supported.

Install the Cortex Training client, PyTorch, Transformers, DeepSpeed and the attention dependencies in the client environment. Put the complete Hugging Face snapshot and temporary checkpoints on local fast storage; keep the token IDs, compact reference and reports in durable storage. The reference uses its own `sdpa` attention by default; the service keeps the catalog recipe's attention configuration. The four-call workload lowers only its microbatch token limit and verifies the number of calls in the service response.

Set and export `TRAINING_RECIPE`, `SAMPLING_RECIPE`, `MODEL_SNAPSHOT`, and `RESULTS`. Prepare 101 token batches, including one held-out batch. Each batch must contain one token-ID sequence; the first four must have the same length. For a synthetic workload:

```bash
mkdir -p "$RESULTS"
python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from arctic_platform.catalog_correctness.checks import batches

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_SNAPSHOT"], local_files_only=True)
path = Path(os.environ["RESULTS"]) / "batches.json"
path.write_text(json.dumps(batches(tokenizer, rows=1, width=64, count=101, seed=1234)))
PY
```

Every token after the first is scored. This short synthetic example does not cover masked labels, padding, or the recipe's maximum sequence length.

Create the reference on a CUDA host with the snapshot on local fast storage:

```bash
python -m arctic_platform.catalog_correctness.reference \
  --training-config "$TRAINING_RECIPE" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --batches "$RESULTS/batches.json" \
  --output "$RESULTS/reference.json"
```

For gradient and optimizer gates, run 15 more fresh reference processes with the same arguments and distinct output files. Then calculate the original reference-repeatability gate (four times the largest same-parameter range, rounded up to 0.001):

```bash
python -m arctic_platform.catalog_correctness.calibrate \
  --output "$RESULTS/calibration.json" "$RESULTS/reference.json" "$RESULTS"/repeat-*.json
```

Review the recorded ranges and gates before supplying them to the suite. The calibration uses only the independent reference runs.

Run the suite from a client able to access the service. `SCRATCH` must be a new path on local fast storage; checkpoint exports and decoded optimizer tensors can occupy tens of GiB.

```bash
python -m arctic_platform.catalog_correctness \
  --training-config "$TRAINING_RECIPE" \
  --sampling-config "$SAMPLING_RECIPE" \
  --reference "$RESULTS/reference.json" \
  --calibration "$RESULTS/calibration.json" \
  --batches "$RESULTS/batches.json" \
  --scratch "$SCRATCH" \
  --report "$RESULTS/report.json" \
  --connection "$CONNECTION_PROFILE"
```

For PAT authentication, replace `--connection` with `--host`, `--database`, `--schema` and `--pat-env`; the named environment variable holds the PAT.

The suite enables the service's per-parameter gradient diagnostics, saves step-zero and step-one resumable checkpoints, and checks every initial parameter against the reference. It validates the checkpoint's DeepSpeed version and accumulation setting, optimizer groups, parameter shapes, step counters, and clipping before comparing norms. It uses a second training job for the checkpoint control and a new sampling job initialized from the saved weights-only checkpoint. It reports the service image and submits jobs with the original catalog backend. A different service image must expose these diagnostics and support the checkpoint APIs.

`--calibration` supplies the gradient and optimizer gates after checking that the reference is one of the calibrated runs. Alternatively, pass reviewed `--gradient-tolerance` and `--optimizer-tolerance` as maximum per-parameter absolute norm differences. Pass `--resume-tolerance`, `--sampler-tolerance` and `--e2e-tolerance` as absolute loss differences. The optimizer check compares Adam moments and update residuals using the same criterion as the original check. Without a tolerance the corresponding check is **measured**, not passed. A missing diagnostic, mismatched checkpoint, or missing parameter is an error. The sampler also reports per-token differences, but its acceptance criterion is mean loss. Norm agreement is not elementwise tensor agreement. Exit codes are 0 when all five checks pass, 2 when at least one is measured and none fail, and 1 for failures or errors. Review thresholds using independent repeated runs; do not widen them automatically from the candidate result.

The workload and HF snapshot hashes identify the reference; the exported step-zero checkpoint authenticates the target's actual starting tensors. Store only token IDs and measurements appropriate for the chosen storage location. Reference records do not need to be accessible from service workers.
