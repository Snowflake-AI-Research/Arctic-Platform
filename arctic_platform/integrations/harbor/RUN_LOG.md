# Harbor CLI + Arctic Cortex — end-to-end run log

Every LLM call happens inside a `harbor run` trial: Harbor's trial
runner spawns `CortexRLAgent` (BaseAgent) under `HostEnvironment`
(BaseEnvironment), scored by Harbor's stock `Verifier` execing each
task's `tests/test.sh` and reading `/logs/verifier/reward.txt`. No
custom `BaseVerifier` subclass, no `--verifier` override.

Between trials the driver reads `result.json`, hands the rollouts to
`ArcticCortexBackend.train` on Cortex QA6, and `sync_weights`
propagates the new weights so the next `harbor run` samples from an
improved model at the same sub-job endpoint.

## Configuration

- Model: `Qwen/Qwen3-0.6B`
- Task: 3-digit × 2-digit MUL (a ∈ [100, 999], b ∈ [10, 99])
- GRPO: 15 steps × 24 rollouts/step (6 prompts × 4 attempts), lr=5e-6, temp=0.8
- Held-out: 80 problems, greedy (temperature=0)
- Verifier: Harbor's stock `Verifier` running our `tests/test.sh`
  (last-integer match, comma-normalized, dense partial credit by
  relative error).
- 3 independent seeds

## Headline

```
                 seed 0            seed 1            seed 2            n=3 aggregate
pass@1           0.362 -> 0.350    0.350 -> 0.400    0.375 -> 0.425
mean held-out r  0.580 -> 0.696    0.600 -> 0.690    0.648 -> 0.741

aggregate over 3 runs (bootstrap 95% CI on the mean delta, 10k resamples):
  pass@1               delta +0.029 ± 0.029   95% CI [-0.013, +0.050]
  mean held-out reward delta +0.100 ± 0.011   95% CI [+0.090, +0.116]
```

`pass@1` CI spans zero — a 0.6B model doesn't fix arithmetic in 15 GRPO
steps. Mean held-out reward moves +10.0 pp, 95% CI [+9.0, +11.6]:
improvement in output quality that pass@1's binary threshold hides.

### Reward distribution shift (seed 0)

Distribution of the 80 held-out rewards, baseline vs. after 15 GRPO steps:

| bucket | baseline | final | Δ |
| ------ | -------- | ----- | - |
| far (0.0-0.05) | 26 | 7 | -19 |
| close_wrong (0.15-0.5) | 7 | 21 | +14 |
| verbose_correct (0.7) | 18 | 24 | +6 |
| exact (1.0) | 29 | 28 | -1 |

19 problems moved OUT of the "catastrophic" bucket (reward ≤ 0.05,
model stuck in a repetition loop or emitted no integer). Most landed
in the "close but wrong" or "verbose correct" buckets.

### Notable held-out flips on seed 0 (Δ reward ≥ 0.5)

| task | baseline (greedy) | after 15 GRPO steps (greedy) | Δ reward |
| ---- | ----------------- | ---------------------------- | -------- |
| `heldout_017_994x74` | `994 * 74 = 72, 994 * 74 = 72, 994 * 74 = 72, 994 * 74 = 72, 994 * 74 =` (r=0.05) | `994 * 74 = 72, 856.   · Final integer: **72856**.` (r=0.70) | +0.65 |
| `heldout_022_709x69` | `709 × 69 = 49, 0 709 × 69 = 49, 0 709 × 69 = 49, 0 709 × 69 = 49, 0 70` (r=0.05) | `709 * 69 = 48, 981.   · Final integer: **48981**.` (r=0.70) | +0.65 |
| `heldout_029_839x93` | `839 × 93 = 77, 839 × 93 = 77, 839 × 93 = 77, 839 × 93 = 77, 839 × 93 =` (r=0.05) | `839 * 93 = 78, 277. The final integer is **78277**.` (r=0.70) | +0.65 |
| `heldout_031_914x45` | `914 * 45 = 41, 110. So the final integer is **4110**.` (r=0.05) | `914 * 45 = 41,190.   · Final integer: **41190**.` (r=0.70) | +0.65 |
| `heldout_052_489x77` | `489 × 77 = 37,  489 × 77 = 37,  489 × 77 = 37,  489 × 77 = 37,  489 × ` (r=0.05) | `489 * 77 = 37,  933.  ·  · Final integer: 37933.` (r=0.70) | +0.65 |
| `heldout_054_152x89` | `152 * 89 = 13, 578.` (r=0.05) | `152 * 89 = 13, 608.   · Final integer: 13608` (r=0.70) | +0.65 |
| `heldout_055_892x69` | `892 * 69 = 62, 892 * 69 = 62, 892 * 69 = 62, 892 * 69 = 62, 892 * 69 =` (r=0.05) | `892 * 69 = 62, 108.   · Final integer: 62108` (r=0.70) | +0.65 |
| `heldout_071_694x45` | `694 * 45 = 31, 315.  ·  · Final integer: 315.` (r=0.05) | `694 * 45 = 31, 330.   · Final integer: **31330**.` (r=0.70) | +0.65 |
| `heldout_078_563x74` | `563 * 74 = 41, 563 * 74 = 41, 563 * 74 = 41, 563 * 74 = 41, 563 * 74 =` (r=0.05) | `563 * 74 = 41, 822.   · Final integer: 41822` (r=0.70) | +0.65 |

Example (`heldout_017_994x74`, actual answer 73556): baseline was
stuck in a repetition loop `994 * 74 = 72, 994 * 74 = 72, ...`
(reward 0.05). Final produces a coherent (but still wrong) answer
`994 * 74 = 72,856` — 0.95% off the true value, worth 0.7 reward.

## Cortex sub-job identifiers (seed 0)

- Cortex run id: `run_2bfa8e1e`
- Training sub-job id: `80abd123-5026-493c-8704-eeab95d30f34:training:0`
- Sampling sub-job id: `80abd123-5026-493c-8704-eeab95d30f34:sampling:0`

Same sampling sub-job for baseline and post-training re-eval — the
delta is entirely what `sync_weights` pushed over 15 GRPO steps.

## harbor CLI transcript (seed 0)

```
[19:44:33] harbor_runner: work_dir = /tmp/harbor_e2e_cpj6wule
[19:44:36] harbor_runner: tokenizer cache warm for Qwen/Qwen3-0.6B
[19:44:36] harbor_runner: creating Cortex job (training + sampling sub-jobs); cold-start can take a few minutes ...
[19:48:42] harbor_runner: connected: run=run_2bfa8e1e train_job=80abd123-5026-493c-8704-eeab95d30f34:training:0 sample_job=80abd123-5026-493c-8704-eeab95d30f34:sampling:0
[19:48:42] harbor_runner: reconnect config -> /tmp/harbor_e2e_cpj6wule/reconnect_config.json  (train_job_id='80abd123-5026-493c-8704-eeab95d30f34:training:0')
[19:48:42] harbor_runner: operand ranges: a in [100,999], b in [10,99], op=mul
[19:48:42] harbor_runner: BASELINE harbor run (greedy, k=1) ...
[19:48:42] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_heldout -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name baseline -n 4 -k 1 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.0 --ak max_tokens=64
  80/80 Mean: 0.580 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:01:21 0:00:00
Total runtime: 1m 21s
[19:50:08] harbor_runner: BASELINE pass@1 = 0.362  (29/80)
[19:50:08] harbor_runner: STEP 00 harbor run (k=4, temp=0.8) ...
[19:50:08] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_00 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_00 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.375 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:24 0:00:00
Total runtime: 24s
[19:50:37] harbor_runner:   rollouts=24 reward_mean=0.375 correct=1
[19:50:52] harbor_runner:   step 00: loss=0.11398084461688995 grad_norm=11.34607982635498
[19:50:52] harbor_runner: STEP 01 harbor run (k=4, temp=0.8) ...
[19:50:52] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_01 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_01 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.490 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:26 0:00:00
Total runtime: 26s
[19:51:24] harbor_runner:   rollouts=24 reward_mean=0.490 correct=2
[19:51:27] harbor_runner:   step 01: loss=0.08569996058940887 grad_norm=23.015792846679688
[19:51:27] harbor_runner: STEP 02 harbor run (k=4, temp=0.8) ...
[19:51:27] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_02 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_02 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.867 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:24 0:00:00
Total runtime: 24s
[19:51:56] harbor_runner:   rollouts=24 reward_mean=0.867 correct=19
[19:51:58] harbor_runner:   step 02: loss=-0.02910039946436882 grad_norm=13.278753280639648
[19:51:58] harbor_runner: STEP 03 harbor run (k=4, temp=0.8) ...
[19:51:58] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_03 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_03 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.731 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:22 0:00:00
Total runtime: 22s
[19:52:25] harbor_runner:   rollouts=24 reward_mean=0.731 correct=11
[19:52:27] harbor_runner:   step 03: loss=0.13956864178180695 grad_norm=20.006765365600586
[19:52:27] harbor_runner: STEP 04 harbor run (k=4, temp=0.8) ...
[19:52:27] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_04 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_04 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.688 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:24 0:00:00
Total runtime: 24s
[19:52:56] harbor_runner:   rollouts=24 reward_mean=0.688 correct=8
[19:52:58] harbor_runner:   step 04: loss=0.0731629878282547 grad_norm=11.550259590148926
[19:52:58] harbor_runner: STEP 05 harbor run (k=4, temp=0.8) ...
[19:52:58] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_05 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_05 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.727 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:25 0:00:00
Total runtime: 25s
[19:53:28] harbor_runner:   rollouts=24 reward_mean=0.727 correct=8
[19:53:30] harbor_runner:   step 05: loss=-0.01078917644917965 grad_norm=20.566085815429688
[19:53:30] harbor_runner: STEP 06 harbor run (k=4, temp=0.8) ...
[19:53:30] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_06 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_06 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.975 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:24 0:00:00
Total runtime: 24s
[19:54:00] harbor_runner:   rollouts=24 reward_mean=0.975 correct=22
[19:54:02] harbor_runner:   step 06: loss=0.028516920283436775 grad_norm=7.46567440032959
[19:54:02] harbor_runner: STEP 07 harbor run (k=4, temp=0.8) ...
[19:54:02] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_07 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_07 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.592 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:22 0:00:00
Total runtime: 22s
[19:54:29] harbor_runner:   rollouts=24 reward_mean=0.592 correct=3
[19:54:30] harbor_runner:   step 07: loss=0.21964552998542786 grad_norm=20.63243293762207
[19:54:30] harbor_runner: STEP 08 harbor run (k=4, temp=0.8) ...
[19:54:30] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_08 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_08 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.685 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:22 0:00:00
Total runtime: 22s
[19:54:58] harbor_runner:   rollouts=24 reward_mean=0.685 correct=5
[19:55:00] harbor_runner:   step 08: loss=0.030171938240528107 grad_norm=23.26975440979004
[19:55:00] harbor_runner: STEP 09 harbor run (k=4, temp=0.8) ...
[19:55:00] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_09 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_09 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.765 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:25 0:00:00
Total runtime: 25s
[19:55:30] harbor_runner:   rollouts=24 reward_mean=0.765 correct=12
[19:55:32] harbor_runner:   step 09: loss=-0.0029197190888226032 grad_norm=7.181921482086182
[19:55:32] harbor_runner: STEP 10 harbor run (k=4, temp=0.8) ...
[19:55:32] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_10 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_10 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.715 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:23 0:00:00
Total runtime: 23s
[19:56:00] harbor_runner:   rollouts=24 reward_mean=0.715 correct=8
[19:56:02] harbor_runner:   step 10: loss=0.014213311485946178 grad_norm=7.516313552856445
[19:56:02] harbor_runner: STEP 11 harbor run (k=4, temp=0.8) ...
[19:56:02] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_11 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_11 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.567 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:24 0:00:00
Total runtime: 24s
[19:56:31] harbor_runner:   rollouts=24 reward_mean=0.567 correct=1
[19:56:33] harbor_runner:   step 11: loss=0.11922883987426758 grad_norm=17.599130630493164
[19:56:33] harbor_runner: STEP 12 harbor run (k=4, temp=0.8) ...
[19:56:33] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_12 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_12 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.746 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:28 0:00:00
Total runtime: 28s
[19:57:06] harbor_runner:   rollouts=24 reward_mean=0.746 correct=9
[19:57:07] harbor_runner:   step 12: loss=0.030542073771357536 grad_norm=16.21636199951172
[19:57:07] harbor_runner: STEP 13 harbor run (k=4, temp=0.8) ...
[19:57:07] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_13 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_13 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.721 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:23 0:00:00
Total runtime: 23s
[19:57:36] harbor_runner:   rollouts=24 reward_mean=0.721 correct=9
[19:57:37] harbor_runner:   step 13: loss=0.004016334190964699 grad_norm=7.457709312438965
[19:57:37] harbor_runner: STEP 14 harbor run (k=4, temp=0.8) ...
[19:57:37] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_step_14 -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name step_14 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64
  24/24 Mean: 0.840 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:26 0:00:00
Total runtime: 26s
[19:58:08] harbor_runner:   rollouts=24 reward_mean=0.840 correct=16
[19:58:10] harbor_runner:   step 14: loss=0.006514507811516523 grad_norm=5.4572625160217285
[19:58:10] harbor_runner: FINAL harbor run (greedy, k=1) ...
[19:58:10] harbor_runner: $ harbor run -p /tmp/harbor_e2e_cpj6wule/dataset_heldout -a arctic_platform.integrations.harbor.agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.env:HostEnvironment -o /tmp/harbor_e2e_cpj6wule/harbor_jobs --job-name final -n 4 -k 1 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_cpj6wule/reconnect_config.json --ak temperature=0.0 --ak max_tokens=64
  80/80 Mean: 0.696 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:01:10 0:00:00
Total runtime: 1m 10s
[19:59:25] harbor_runner: FINAL pass@1 = 0.350  (28/80)
[19:59:25] harbor_runner: reward curve: [0.375, 0.49, 0.867, 0.731, 0.688, 0.727, 0.975, 0.592, 0.685, 0.765, 0.715, 0.567, 0.746, 0.721, 0.84]
[19:59:25] harbor_runner: RESULT pass@1  0.362 -> 0.350  (delta -0.013)
[19:59:25] harbor_runner: summary -> /tmp/harbor_e2e_cpj6wule/summary.json
[19:59:25] harbor_runner: shutting down Cortex job
```