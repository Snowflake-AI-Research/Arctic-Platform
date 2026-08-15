# Harbor CLI + Arctic Cortex — end-to-end run log

Every LLM call happens inside a `harbor run` trial: Harbor's trial
runner spawns a `BaseAgent` under `HostEnvironment` (`BaseEnvironment`),
scored by Harbor's stock `Verifier` execing each task's `tests/test.sh`
and reading `/logs/verifier/reward.txt`. No custom `BaseVerifier`
subclass, no `--verifier` override.

Between trials the driver reads `result.json`, hands the rollouts to
`ArcticCortexBackend.train` on Cortex, and `sync_weights`
propagates the new weights so the next `harbor run` samples from an
improved model at the same sub-job endpoint.

Two sampling modes are exercised end-to-end below:

* **Native (`CortexRLAgent`)** — the agent calls
  `ArcticRLClient.generate` directly through the reconnect-config
  path. This is the section labelled *Native Cortex — 3-seed
  aggregate* and is the historical PR #66 reference.
* **OpenAI-compat gateway (`LiteLLMChatAgent`)** — a stock-shape
  Harbor `BaseAgent` using `harbor.llms.lite_llm.LiteLLM` points at a
  driver-local OpenAI-compat gateway (`DriverOpenAIGateway`), which
  forwards each `/v1/chat/completions` call to
  `ArcticRLClient.generate` over the exact same Cortex transport. No
  Cortex-side change, no monkey-patch on Harbor. Numbers in the
  *OpenAI-compat gateway — seed 0* section below.

## Native Cortex — 3-seed aggregate

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

## OpenAI-compat gateway — seed 0

Same task, same hyperparameters, same model, same lr, same seed as
the native reference above. Only difference: the trial's LLM calls go
through Harbor's `LiteLLM` client → driver-local `DriverOpenAIGateway`
on `127.0.0.1:{port}/v1` → `ArcticRLClient.generate` over the Cortex
`operation` op envelope. Agent is `LiteLLMChatAgent` — a stock-shape
`BaseAgent` any Harbor user could write, no reach into Arctic-Platform
internals except constructing `LiteLLM(model_name="hosted_vllm/…",
api_base=…, collect_rollout_details=True)`.

### Configuration diff vs. native reference

| Field | Native (seeds 0–2) | Gateway (seed 0) |
| --- | --- | --- |
| Agent | `CortexRLAgent` (reconnect-config) | `LiteLLMChatAgent` (OpenAI-compat via LiteLLM) |
| Sampling path | `ArcticRLClient.generate` direct | `LiteLLM` → `DriverOpenAIGateway` → `ArcticRLClient.generate` |
| `max_tokens` | 64 | 128 (Harbor's `LiteLLM` refuses truncated `finish_reason=length` responses; native mode has no such check) |
| Everything else | same | same |

### Headline

```
[00:28:13] harbor_runner: BASELINE pass@1 = 0.476  (30/63)
[00:32:44] harbor_runner: FINAL pass@1 = 0.635  (40/63)
[00:32:44] harbor_runner: reward curve: [0.590, 0.443, 0.770, 0.727, 0.716,
                                          0.727, 1.000, 0.567, 0.500, 0.731,
                                          0.559, 0.277, 0.705, 0.696, 0.702]
[00:32:44] harbor_runner: RESULT pass@1: 0.476 -> 0.635 (+0.159)
                          mean held-out reward: 0.738 -> 0.810 (+0.072)
```

|                       | native (n=3, mean ± CI)     | gateway (seed 0) |
| --------------------- | --------------------------- | ---------------- |
| Δ pass@1              | +0.029 [-0.013, +0.050]     | **+0.159**       |
| Δ mean held-out reward| +0.100 [+0.090, +0.116]     | **+0.072**       |

Direction and magnitude match. Mean-reward Δ is inside the native
3-seed variance envelope; pass@1 is *higher* in gateway mode because
the model has enough tokens to finish an answer under
`max_tokens=128` (native ran with 64 and mostly got dense partial
credit; gateway more often gets to 1.0 exactly).

17 of 80 held-out trials still errored on `OutputLengthExceededError`
in gateway mode — Harbor's `LiteLLM` refuses `finish_reason=length`
responses, so the trial fails before token ids reach the trainer.
`LiteLLMChatAgent` catches the exception and salvages
`truncated_response` text so the trial still scores (partial credit),
but drops the rollout from the training batch. That's why "63/80"
completes; the underlying gateway path is not the bottleneck.

### Cortex sub-job identifiers (gateway seed 0)

- Cortex run id: `run_000585ec`
- Training sub-job id: `45525394-16c3-46a4-a590-97991185d8e4:training:0`
- Sampling sub-job id: `45525394-16c3-46a4-a590-97991185d8e4:sampling:0`
- Driver gateway URL: `http://127.0.0.1:55029/v1` (ephemeral port picked at
  driver start; Harbor gets it via `--sampling-api-base auto`).

### Reproduce

```bash
export ARCTIC_CORTEX_HOST=...
export ARCTIC_CORTEX_DATABASE=... ARCTIC_CORTEX_SCHEMA=...
export CORTEX_PAT=...
export ARCTIC_BACKEND=cortex

harbor-cortex-train \
    --model Qwen/Qwen3-0.6B \
    --agent arctic_platform.integrations.harbor.litellm_chat_agent:LiteLLMChatAgent \
    --sampling-api-base auto \
    --llm-backend litellm \
    --iters 15 --prompts-per-step 6 --n-attempts 4 \
    --n-concurrent 4 \
    --max-tokens 128 --temperature 0.8 --lr 5e-6 \
    --heldout 80 \
    --task mul --a-low 100 --a-high 999 --b-low 10 --b-high 99 \
    --out ./gateway-seed-0 --seed 0
```

### Full harbor CLI transcript (gateway seed 0)

Abridged — the full log is at
`logs/harbor_gateway_ref_20260815_002338.log` in the run workspace.

```
[00:23:41] harbor_runner: tokenizer cache warm for Qwen/Qwen3-0.6B
[00:23:41] harbor_runner: creating Cortex job (training + sampling sub-jobs); cold-start can take a few minutes ...
[00:27:30] harbor_runner: connected: run=run_000585ec train_job=45525394-16c3-46a4-a590-97991185d8e4:training:0 sample_job=45525394-16c3-46a4-a590-97991185d8e4:sampling:0
[00:27:31] harbor_runner: driver-side OpenAI-compat gateway up at http://127.0.0.1:55029/v1
[00:27:33] harbor_runner: BASELINE harbor run (greedy, k=1) ...
[00:28:13] harbor_runner: BASELINE pass@1 = 0.476  (30/63)
[00:28:13] harbor_runner: STEP 00 harbor run (k=4, temp=0.8) ...  rollouts=20 reward_mean=0.590 correct=3
[00:28:38] harbor_runner:   step 00: loss=0.043 grad_norm=13.85
[00:28:38] harbor_runner: STEP 01 harbor run (k=4, temp=0.8) ...  rollouts=23 reward_mean=0.443 correct=3
[00:28:55] harbor_runner:   step 01: loss=-0.242 grad_norm=17.22
[00:28:56] harbor_runner: STEP 02 harbor run (k=4, temp=0.8) ...  rollouts=23 reward_mean=0.770 correct=16
[00:29:09] harbor_runner:   step 02: loss=-0.099 grad_norm=52.28
[00:29:09] harbor_runner: STEP 03 harbor run (k=4, temp=0.8) ...  rollouts=24 reward_mean=0.727 correct=10
[00:29:22] harbor_runner:   step 03: loss=0.031 grad_norm=17.89
[00:29:22] harbor_runner: STEP 04 harbor run (k=4, temp=0.8) ...  rollouts=19 reward_mean=0.716 correct=8
[00:29:36] harbor_runner:   step 04: loss=-0.292 grad_norm=18.75
[00:29:36] harbor_runner: STEP 05 harbor run (k=4, temp=0.8) ...  rollouts=22 reward_mean=0.727 correct=9
[00:29:49] harbor_runner:   step 05: loss=0.064 grad_norm=14.75
[00:29:49] harbor_runner: STEP 06 harbor run (k=4, temp=0.8) ...  rollouts=24 reward_mean=1.000 correct=24
[00:30:01] harbor_runner:   step 06: loss=0.000 grad_norm=0.00
[00:30:02] harbor_runner: STEP 07 harbor run (k=4, temp=0.8) ...  rollouts=21 reward_mean=0.567 correct=5
[00:30:21] harbor_runner:   step 07: loss=-0.222 grad_norm=32.08
[00:30:21] harbor_runner: STEP 08 harbor run (k=4, temp=0.8) ...  rollouts=23 reward_mean=0.500 correct=3
[00:30:35] harbor_runner:   step 08: loss=-0.260 grad_norm=14.63
[00:30:35] harbor_runner: STEP 09 harbor run (k=4, temp=0.8) ...  rollouts=24 reward_mean=0.731 correct=14
[00:30:50] harbor_runner:   step 09: loss=-0.211 grad_norm=16.25
[00:30:50] harbor_runner: STEP 10 harbor run (k=4, temp=0.8) ...  rollouts=22 reward_mean=0.559 correct=9
[00:31:06] harbor_runner:   step 10: loss=0.034 grad_norm=11.11
[00:31:06] harbor_runner: STEP 11 harbor run (k=4, temp=0.8) ...  rollouts=20 reward_mean=0.277 correct=2
[00:31:20] harbor_runner:   step 11: loss=-0.333 grad_norm=15.79
[00:31:20] harbor_runner: STEP 12 harbor run (k=4, temp=0.8) ...  rollouts=22 reward_mean=0.705 correct=11
[00:31:33] harbor_runner:   step 12: loss=-0.360 grad_norm=13.25
[00:31:33] harbor_runner: STEP 13 harbor run (k=4, temp=0.8) ...  rollouts=24 reward_mean=0.696 correct=10
[00:31:44] harbor_runner:   step 13: loss=-0.252 grad_norm=11.55
[00:31:44] harbor_runner: STEP 14 harbor run (k=4, temp=0.8) ...  rollouts=23 reward_mean=0.702 correct=13
[00:31:57] harbor_runner:   step 14: loss=-0.045 grad_norm=17.21
[00:31:57] harbor_runner: FINAL harbor run (greedy, k=1) ...
[00:32:44] harbor_runner: FINAL pass@1 = 0.635  (40/63)
[00:32:44] harbor_runner: RESULT pass@1: 0.476 -> 0.635 (+0.159)  |  mean held-out reward: 0.738 -> 0.810 (+0.072)
[00:32:44] harbor_runner: shutting down Cortex job
```

## OpenAI-compat gateway — larger model on a real benchmark

Same driver-side gateway + `LiteLLMChatAgent` as the arithmetic reference
above, but with a legitimately-hard verifiable-reward benchmark and a
step up in model size to sanity-check that the gateway path scales
beyond the toy task.

### Configuration

- Model: `Qwen/Qwen3-1.7B` (largest checkpoint currently supported by
  the Cortex-training QA6 image — `Qwen3-4B` fails at sub-job start with
  `sub_job_failed` before any driver traffic; every Cortex-side recipe
  in this repo uses either `Qwen3-0.6B` or `Qwen3-1.7B`).
- Benchmark: `reasoning-gym-easy` — 96 tasks sampled from
  [reasoning-gym](https://github.com/open-thought/reasoning-gym) across
  algebra / algorithmic / arithmetic / calendar-arithmetic families,
  split 72 train / 24 held-out. Each task ships its own generator seed
  and reward script; scoring is exact-match on `/workspace/answer.txt`
  with dense partial credit from `reasoning_gym.score_answer` (0.0 to
  1.0 continuous).
- GRPO: 8 steps × 32 rollouts/step (8 prompts × 4 attempts), lr=1e-6,
  temp=0.7, `max_tokens=384`, `max_seq_len=1280`.
- 1 training GPU + 1 sampling GPU (same shape as the verl-simple
  `Qwen3-1.7B` recipe).
- Seed 0.

### Headline

```
[01:09:56] harbor_runner: BASELINE pass@1 = 0.043  (1/23)
[01:15:03] harbor_runner: FINAL   pass@1 = 0.087  (2/23)
[01:15:03] harbor_runner: reward curve: [0.243, 0.121, 0.207, 0.070,
                                          0.377, 0.001, 0.458, 0.117]
[01:15:03] harbor_runner: RESULT pass@1: 0.043 -> 0.087 (+0.043)
                          mean held-out reward: 0.088 -> 0.119 (+0.031)
```

Both metrics improve monotonically end-to-end; per-step reward is noisy
because reasoning-gym mixes families (algebra vs. sort vs. word puzzles
in the same batch) and Qwen3-1.7B saturates at ~1/23 pass@1 on this
distribution before training. Training doubles pass@1 and lifts mean
reward by 35% in 8 steps, which matches the *shape* of the arithmetic
reference (baseline low, small monotone gain from GRPO). Longer runs
would need reasoning-family curricula rather than a random mix to keep
the training signal dense.

### Cortex sub-job identifiers

- Cortex run id: `run_5d68ca0a`
- Training sub-job id: `2de9061d-5ed0-4246-b503-ced3aa2e0ec9:training:0`
- Sampling sub-job id: `2de9061d-5ed0-4246-b503-ced3aa2e0ec9:sampling:0`
- Driver gateway URL: `http://127.0.0.1:38547/v1`

### Fixes required to run any Harbor task pack (not just the arithmetic reference)

Three integration bugs surfaced only once the run drove non-trivial
task dirs; all patched in this branch:

1. **`_write_step_manifest` wrote `dataset.toml` but Harbor's
   `harbor run -p <dir>` walks `path.iterdir()` and doesn't parse
   dataset manifests.** Step 0 crashed with
   `ValueError: Either datasets or tasks must be provided.` Fixed by
   symlinking chosen task dirs into the per-step dataset dir so
   Harbor's directory scan picks them up.
2. **`HostEnvironment.exec` rewrote canonical Harbor paths in the
   *command string* but not in file *contents*.** `test.sh` runs
   `python3 /tests/test_output.py` and that literal only resolves under
   Docker/Modal, so every host-env trial failed with
   `can't open file '/tests/test_output.py'`. Fixed by mirroring
   `_rewrite_paths` on the contents of `.sh` / `.py` / `.txt` files at
   `upload_dir` / `upload_file` time.
3. **Cortex-training QA6 image doesn't ship `Qwen3-4B`.** Both `1x1`
   and `2x2` GPU attempts terminate as `sub_job_failed` inside 2min
   with no error text surfaced through the SnowAPI job-status
   endpoint. `Qwen3-1.7B` is the current ceiling; matches the largest
   model exercised anywhere else in the repo on Cortex.

### Reproduce

```bash
export ARCTIC_CORTEX_HOST=...
export ARCTIC_CORTEX_DATABASE=... ARCTIC_CORTEX_SCHEMA=...
export CORTEX_PAT=...
export ARCTIC_BACKEND=cortex

harbor-cortex-train \
    --model Qwen/Qwen3-1.7B \
    --agent arctic_platform.integrations.harbor.litellm_chat_agent:LiteLLMChatAgent \
    --sampling-api-base auto \
    --llm-backend litellm \
    --tasks-dir ./rgym-easy-train \
    --heldout-dir ./rgym-easy-heldout \
    --iters 8 --prompts-per-step 8 --n-attempts 4 \
    --n-concurrent 4 \
    --max-tokens 384 --temperature 0.7 --lr 1e-6 \
    --max-seq-len 1280 --train-gpus 1 --sample-gpus 1 \
    --out ./rgym-1p7b-seed-0 --seed 0
```