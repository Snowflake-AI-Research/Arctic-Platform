# Real Harbor CLI + Arctic Cortex — end-to-end run log

Every LLM call below happens inside a real `harbor run` trial. This is
Harbor's own trial runner driving `CortexRLAgent`, `HostEnvironment`,
and `ArithmeticVerifier` (all in this package). The middle step reads
Harbor's `result.json` files, hands them to `ArcticCortexBackend.train`
on Cortex QA6, and `sync_weights` propagates the new weights so the
next `harbor run` samples from an improved model.

## Configuration

- Model: `Qwen/Qwen3-0.6B`
- Task: 3-digit × 2-digit MUL (a ∈ [100, 999], b ∈ [10, 99])
- GRPO: 8 steps × 24 rollouts/step (6 prompts × 4 attempts), lr=5e-6, temp=0.8
- Held-out: 16 problems, greedy (temperature=0)
- Verifier: last-integer extraction, thousand-comma-normalized, dense partial credit by relative error

## Headline

```
BASELINE pass@1 = 0.250  (4/16)
FINAL    pass@1 = 0.312  (5/16)
                   delta = +0.062

training reward curve (per-step mean, partial credit):
  step 0  0.375
  step 1  0.427
  step 2  0.877
  step 3  0.823
  step 4  0.635
  step 5  0.708
  step 6  0.960
  step 7  0.615
```

- Cortex run id: `run_42879727`
- Training sub-job id: `fc5e01c2-8787-43d8-a6d9-9a8d19c4bd4f:training:0`
- Sampling sub-job id: `fc5e01c2-8787-43d8-a6d9-9a8d19c4bd4f:sampling:0`

## harbor CLI transcript

```
[09:15:46] harbor_runner: work_dir = /tmp/harbor_e2e_kw5pz07f
[09:15:46] harbor_runner: creating Cortex job (training + sampling sub-jobs); cold-start can take a few minutes ...
[09:19:45] harbor_runner: connected: run=run_42879727 train_job=fc5e01c2-8787-43d8-a6d9-9a8d19c4bd4f:training:0 sample_job=fc5e01c2-8787-43d8-a6d9-9a8d19c4bd4f:sampling:0
[09:19:45] harbor_runner: reconnect config -> /tmp/harbor_e2e_kw5pz07f/reconnect_config.json  (train_job_id='fc5e01c2-8787-43d8-a6d9-9a8d19c4bd4f:training:0')
[09:19:45] harbor_runner: operand ranges: a in [100,999], b in [10,99], op=mul
[09:19:45] harbor_runner: BASELINE harbor run (greedy, k=1) ...
[09:19:45] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_heldout -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name baseline -n 4 -k 1 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.0 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  16/16 Mean: 0.697 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:22 0:00:00
Total runtime: 22s
[09:20:12] harbor_runner: BASELINE pass@1 = 0.250  (4/16)
[09:20:12] harbor_runner: STEP 00 harbor run (k=4, temp=0.8) ...
[09:20:12] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_00 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_00 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.375 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:34 0:00:00
Total runtime: 34s
[09:20:52] harbor_runner:   rollouts=24 reward_mean=0.375 correct=1
[09:21:04] harbor_runner:   step 00: loss=0.11398084461688995 grad_norm=11.346251487731934
[09:21:04] harbor_runner: STEP 01 harbor run (k=4, temp=0.8) ...
[09:21:04] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_01 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_01 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.427 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:30 0:00:00
Total runtime: 30s
[09:21:39] harbor_runner:   rollouts=24 reward_mean=0.427 correct=1
[09:21:43] harbor_runner:   step 01: loss=0.06215127184987068 grad_norm=19.312082290649414
[09:21:43] harbor_runner: STEP 02 harbor run (k=4, temp=0.8) ...
[09:21:43] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_02 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_02 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.877 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:27 0:00:00
Total runtime: 27s
[09:22:15] harbor_runner:   rollouts=24 reward_mean=0.877 correct=19
[09:22:17] harbor_runner:   step 02: loss=0.1828266829252243 grad_norm=14.822529792785645
[09:22:17] harbor_runner: STEP 03 harbor run (k=4, temp=0.8) ...
[09:22:17] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_03 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_03 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.823 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:32 0:00:00
Total runtime: 32s
[09:22:54] harbor_runner:   rollouts=24 reward_mean=0.823 correct=12
[09:22:56] harbor_runner:   step 03: loss=0.14287319779396057 grad_norm=9.676679611206055
[09:22:56] harbor_runner: STEP 04 harbor run (k=4, temp=0.8) ...
[09:22:56] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_04 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_04 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.635 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:59 0:00:00
Total runtime: 59s
[09:24:00] harbor_runner:   rollouts=24 reward_mean=0.635 correct=9
[09:24:04] harbor_runner:   step 04: loss=0.1668817102909088 grad_norm=11.884052276611328
[09:24:04] harbor_runner: STEP 05 harbor run (k=4, temp=0.8) ...
[09:24:04] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_05 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_05 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.708 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:44 0:00:00
Total runtime: 44s
[09:24:53] harbor_runner:   rollouts=24 reward_mean=0.708 correct=9
[09:24:57] harbor_runner:   step 05: loss=0.14374659955501556 grad_norm=26.726011276245117
[09:24:57] harbor_runner: STEP 06 harbor run (k=4, temp=0.8) ...
[09:24:57] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_06 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_06 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.960 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:52 0:00:00
Total runtime: 52s
[09:25:54] harbor_runner:   rollouts=24 reward_mean=0.960 correct=23
[09:25:56] harbor_runner:   step 06: loss=-0.0384899266064167 grad_norm=4.42590856552124
[09:25:56] harbor_runner: STEP 07 harbor run (k=4, temp=0.8) ...
[09:25:56] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_step_07 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name step_07 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.615 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:33 0:00:00
Total runtime: 33s
[09:26:35] harbor_runner:   rollouts=24 reward_mean=0.615 correct=3
[09:26:36] harbor_runner:   step 07: loss=0.179788276553154 grad_norm=34.61264419555664
[09:26:36] harbor_runner: FINAL harbor run (greedy, k=1) ...
[09:26:36] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_kw5pz07f/dataset_heldout -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_kw5pz07f/harbor_jobs --job-name final -n 4 -k 1 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_kw5pz07f/reconnect_config.json --ak temperature=0.0 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  16/16 Mean: 0.744 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:19 0:00:00
Total runtime: 19s
[09:27:01] harbor_runner: FINAL pass@1 = 0.312  (5/16)
[09:27:01] harbor_runner: reward curve: [0.375, 0.427, 0.877, 0.823, 0.635, 0.708, 0.96, 0.615]
[09:27:01] harbor_runner: RESULT pass@1  0.250 -> 0.312  (delta +0.062)
[09:27:01] harbor_runner: summary -> /tmp/harbor_e2e_kw5pz07f/summary.json
[09:27:01] harbor_runner: shutting down Cortex job
```

## Held-out completions — baseline (greedy)

| Task | Reward | Model output |
| ---- | ------ | ------------ |
| `heldout_000_900x96` | 1.0 | `900 * 96 = 86400` |
| `heldout_001_181x82` | 0.7 | `181 * 82 = 14942` |
| `heldout_002_687x78` | 0.7 | `687 × 78 = 53, 894.  ·  · Final integer: 53894` |
| `heldout_003_602x71` | 1.0 | `602 * 71 = 42,742.` |
| `heldout_004_235x92` | 0.7 | `235 * 92 = 21, 780.  ·  · Final integer: 21780` |
| `heldout_005_425x92` | 0.7 | `425 * 92 = 39, 390.  ·  · Final integer: 39390` |
| `heldout_006_913x92` | 0.1 | `913 * 92 = 83, 836.  ·  · Final integer: 836.` |
| `heldout_007_965x22` | 0.5 | `965 * 22 = 21630 ·  · Final integer: 21630` |
| `heldout_008_761x34` | 0.5 | `761 * 34 = 26, 874.  ·  · Final integer: 26874` |
| `heldout_009_252x43` | 0.5 | `252 * 43 = 10, 996.  ·  · Final integer: 10996` |
| `heldout_010_684x16` | 1.0 | `684 × 16 = 10944 ·  · Final integer: 10944` |
| `heldout_011_818x42` | 0.7 | `818 * 42 = 34, 196.  ·  · Final integer: 34196` |
| `heldout_012_507x39` | 0.7 | `507 * 39 = 19833` |
| `heldout_013_953x90` | 1.0 | `953 * 90 = 85,770 ·  · Final integer: 85770` |
| `heldout_014_511x19` | 0.7 | `511 * 19 = 9689 ·  · Final integer: 9689` |
| `heldout_015_376x13` | 0.7 | `376 * 13 = 4908` |

## Held-out completions — after 8 GRPO steps (same greedy re-eval)

| Task | Reward | Model output |
| ---- | ------ | ------------ |
| `heldout_000_900x96` | 1.0 | `900 * 96 = 86400` |
| `heldout_001_181x82` | 0.7 | `181 * 82 = 14882` |
| `heldout_002_687x78` | 0.7 | `687 * 78 = 53, 894.  ·  · Final integer: 53894` |
| `heldout_003_602x71` | 1.0 | `602 * 71 = 42742` |
| `heldout_004_235x92` | 0.7 | `235 * 92 = 21740` |
| `heldout_005_425x92` | 1.0 | `425 * 92 = 39100` |
| `heldout_006_913x92` | 0.7 | `913 * 92 = 83, 836.  ·  · Final integer: 83836.` |
| `heldout_007_965x22` | 0.5 | `965 * 22 = 21630` |
| `heldout_008_761x34` | 0.5 | `761 * 34 = 26274` |
| `heldout_009_252x43` | 0.5 | `252 * 43 = 10956` |
| `heldout_010_684x16` | 1.0 | `684 * 16 = 10944` |
| `heldout_011_818x42` | 0.7 | `818 * 42 = 34116` |
| `heldout_012_507x39` | 0.7 | `507 * 39 = 19783` |
| `heldout_013_953x90` | 1.0 | `953 * 90 = 85770` |
| `heldout_014_511x19` | 0.5 | `511 * 19 = 9609` |
| `heldout_015_376x13` | 0.7 | `376 * 13 = 4908` |

Same sampling endpoint, same greedy decoding. The difference is entirely
what `sync_weights` pushed to the sampling sub-job over 8 GRPO steps.
