# Real Harbor CLI + Arctic Cortex — end-to-end run log

Every LLM call happens inside a real `harbor run` trial. This is Harbor's
own trial runner driving `CortexRLAgent`, `HostEnvironment`, and
`ArithmeticVerifier` (all in this package). The middle step reads Harbor's
`result.json` files, hands them to `ArcticCortexBackend.train` on Cortex
QA6, and `sync_weights` propagates the new weights so the next
`harbor run` samples from an improved model at the same sub-job endpoint.

## Configuration

- Model: `Qwen/Qwen3-0.6B`
- Task: 3-digit × 2-digit MUL (a ∈ [100, 999], b ∈ [10, 99])
- GRPO: 15 steps × 24 rollouts/step (6 prompts × 4 attempts), lr=5e-6, temp=0.8
- Held-out: 20 problems, greedy (temperature=0)
- Verifier: last-integer extraction, comma-normalized, dense partial credit by relative error

## Headline

```
BASELINE pass@1 = 0.250  (5/20)
FINAL    pass@1 = 0.400  (8/20)
                   delta = +0.150

training reward curve (per-step mean, partial credit):
  step  0  0.375
  step  1  0.327
  step  2  0.840
  step  3  0.744
  step  4  0.658
  step  5  0.758
  step  6  0.925
  step  7  0.606
  step  8  0.642
  step  9  0.787
  step 10  0.625
  step 11  0.610
  step 12  0.723
  step 13  0.756
  step 14  0.771
```

- Cortex run id: `run_4ca70b03`
- Training sub-job id: `6cf94f27-bbdc-4502-9376-3344361f2188:training:0`
- Sampling sub-job id: `6cf94f27-bbdc-4502-9376-3344361f2188:sampling:0`

## Held-out problems that flipped wrong → right (3)

| Task | Baseline (greedy) | After 15 GRPO steps (greedy) |
| ---- | ----------------- | ---------------------------- |
| `heldout_007_965x22` | `965 * 22 = 21630 ·  · Final integer: 21630` | `965 × 22 = 21230 ·  · Final integer: 21230` |
| `heldout_015_376x13` | `376 * 13 = 4908` | `376 × 13 = 4888 ·  · Final integer: 4888` |
| `heldout_019_991x96` | `991 * 96 = 95184 ·  · Final integer: 95184` | `991 × 96 = 95136 ·  · Final integer: 95136` |

Same sampling sub-job, same greedy decoding, same 20 held-out problems.
The difference is entirely what `sync_weights` pushed over 15 GRPO steps.

## harbor CLI transcript (excerpted)

```
[09:51:10] harbor_runner: work_dir = /tmp/harbor_e2e_g3wqgj7i
[09:51:10] harbor_runner: creating Cortex job (training + sampling sub-jobs); cold-start can take a few minutes ...
[09:55:12] harbor_runner: connected: run=run_4ca70b03 train_job=6cf94f27-bbdc-4502-9376-3344361f2188:training:0 sample_job=6cf94f27-bbdc-4502-9376-3344361f2188:sampling:0
[09:55:12] harbor_runner: reconnect config -> /tmp/harbor_e2e_g3wqgj7i/reconnect_config.json  (train_job_id='6cf94f27-bbdc-4502-9376-3344361f2188:training:0')
[09:55:12] harbor_runner: operand ranges: a in [100,999], b in [10,99], op=mul
[09:55:12] harbor_runner: BASELINE harbor run (greedy, k=1) ...
[09:55:12] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_heldout -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name baseline -n 4 -k 1 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.0 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  20/20 Mean: 0.647 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:38 0:00:00
Total runtime: 38s
[09:55:55] harbor_runner: BASELINE pass@1 = 0.250  (5/20)
[09:55:55] harbor_runner: STEP 00 harbor run (k=4, temp=0.8) ...
[09:55:55] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_00 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_00 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.375 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:39 0:00:00
Total runtime: 39s
[09:56:39] harbor_runner:   rollouts=24 reward_mean=0.375 correct=1
[09:56:53] harbor_runner:   step 00: loss=0.11398084461688995 grad_norm=11.345905303955078
[09:56:53] harbor_runner: STEP 01 harbor run (k=4, temp=0.8) ...
[09:56:53] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_01 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_01 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.327 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:29 0:00:00
Total runtime: 30s
[09:57:28] harbor_runner:   rollouts=24 reward_mean=0.327 correct=1
[09:57:31] harbor_runner:   step 01: loss=0.13074924051761627 grad_norm=14.474464416503906
[09:57:31] harbor_runner: STEP 02 harbor run (k=4, temp=0.8) ...
[09:57:31] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_02 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_02 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.840 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:29 0:00:00
Total runtime: 29s
[09:58:06] harbor_runner:   rollouts=24 reward_mean=0.840 correct=19
[09:58:08] harbor_runner:   step 02: loss=0.05097990110516548 grad_norm=11.191080093383789
[09:58:08] harbor_runner: STEP 03 harbor run (k=4, temp=0.8) ...
[09:58:08] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_03 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_03 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.744 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:27 0:00:00
Total runtime: 28s
[09:58:41] harbor_runner:   rollouts=24 reward_mean=0.744 correct=9
[09:58:42] harbor_runner:   step 03: loss=0.16561587154865265 grad_norm=21.390689849853516
[09:58:42] harbor_runner: STEP 04 harbor run (k=4, temp=0.8) ...
[09:58:42] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_04 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_04 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.658 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:28 0:00:00
Total runtime: 28s
[09:59:15] harbor_runner:   rollouts=24 reward_mean=0.658 correct=8
[09:59:17] harbor_runner:   step 04: loss=0.12927062809467316 grad_norm=10.12978458404541
[09:59:17] harbor_runner: STEP 05 harbor run (k=4, temp=0.8) ...
[09:59:17] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_05 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_05 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.758 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:35 0:00:00
Total runtime: 36s
[09:59:58] harbor_runner:   rollouts=24 reward_mean=0.758 correct=10
[10:00:01] harbor_runner:   step 05: loss=0.05037768557667732 grad_norm=10.30878734588623
[10:00:01] harbor_runner: STEP 06 harbor run (k=4, temp=0.8) ...
[10:00:01] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_06 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_06 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.925 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:37 0:00:00
Total runtime: 37s
[10:00:43] harbor_runner:   rollouts=24 reward_mean=0.925 correct=22
[10:00:45] harbor_runner:   step 06: loss=-0.046385299414396286 grad_norm=10.544990539550781
[10:00:45] harbor_runner: STEP 07 harbor run (k=4, temp=0.8) ...
[10:00:45] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_07 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_07 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.606 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:35 0:00:00
Total runtime: 35s
[10:01:25] harbor_runner:   rollouts=24 reward_mean=0.606 correct=5
[10:01:27] harbor_runner:   step 07: loss=0.22650223970413208 grad_norm=14.569156646728516
[10:01:27] harbor_runner: STEP 08 harbor run (k=4, temp=0.8) ...
[10:01:27] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_08 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_08 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.642 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:30 0:00:00
Total runtime: 30s
[10:02:02] harbor_runner:   rollouts=24 reward_mean=0.642 correct=3
[10:02:04] harbor_runner:   step 08: loss=0.07017659395933151 grad_norm=19.544095993041992
[10:02:04] harbor_runner: STEP 09 harbor run (k=4, temp=0.8) ...
[10:02:04] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_09 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_09 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.787 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:34 0:00:00
Total runtime: 34s
[10:02:43] harbor_runner:   rollouts=24 reward_mean=0.787 correct=11
[10:02:46] harbor_runner:   step 09: loss=-0.02080313302576542 grad_norm=9.930076599121094
[10:02:46] harbor_runner: STEP 10 harbor run (k=4, temp=0.8) ...
[10:02:46] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_10 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_10 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.208 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:31 0:00:00
Total runtime: 31s
[10:03:23] harbor_runner:   rollouts=8 reward_mean=0.625 correct=3
[10:03:25] harbor_runner:   step 10: loss=0.12627947330474854 grad_norm=6.149524211883545
[10:03:25] harbor_runner: STEP 11 harbor run (k=4, temp=0.8) ...
[10:03:25] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_11 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_11 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.610 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:36 0:00:00
Total runtime: 36s
[10:04:06] harbor_runner:   rollouts=24 reward_mean=0.610 correct=0
[10:04:08] harbor_runner:   step 11: loss=0.10707796365022659 grad_norm=13.94340991973877
[10:04:08] harbor_runner: STEP 12 harbor run (k=4, temp=0.8) ...
[10:04:08] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_12 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_12 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.723 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:35 0:00:00
Total runtime: 35s
[10:04:48] harbor_runner:   rollouts=24 reward_mean=0.723 correct=8
[10:04:50] harbor_runner:   step 12: loss=0.11474538594484329 grad_norm=8.565251350402832
[10:04:50] harbor_runner: STEP 13 harbor run (k=4, temp=0.8) ...
[10:04:50] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_13 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_13 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.756 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:32 0:00:00
Total runtime: 32s
[10:05:28] harbor_runner:   rollouts=24 reward_mean=0.756 correct=10
[10:05:30] harbor_runner:   step 13: loss=0.08916556090116501 grad_norm=12.936699867248535
[10:05:30] harbor_runner: STEP 14 harbor run (k=4, temp=0.8) ...
[10:05:30] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_step_14 -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name step_14 -n 4 -k 4 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.8 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  24/24 Mean: 0.771 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:36 0:00:00
Total runtime: 36s
[10:06:12] harbor_runner:   rollouts=24 reward_mean=0.771 correct=12
[10:06:13] harbor_runner:   step 14: loss=0.14711159467697144 grad_norm=10.434710502624512
[10:06:13] harbor_runner: FINAL harbor run (greedy, k=1) ...
[10:06:13] harbor_runner: $ /home/yak/miniconda3/envs/skyrl_arl/bin/harbor run -p /tmp/harbor_e2e_g3wqgj7i/dataset_heldout -a arctic_platform.integrations.harbor.cortex_agent:CortexRLAgent -m Qwen/Qwen3-0.6B -e arctic_platform.integrations.harbor.host_environment:HostEnvironment -o /tmp/harbor_e2e_g3wqgj7i/harbor_jobs --job-name final -n 4 -k 1 --yes --no-force-build --ak reconnect_config_path=/tmp/harbor_e2e_g3wqgj7i/reconnect_config.json --ak temperature=0.0 --ak max_tokens=64 --verifier arctic_platform.integrations.harbor.arithmetic_verifier:ArithmeticVerifier
  20/20 Mean: 0.758 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0:00:30 0:00:00
Total runtime: 30s
[10:06:49] harbor_runner: FINAL pass@1 = 0.400  (8/20)
[10:06:49] harbor_runner: reward curve: [0.375, 0.327, 0.84, 0.744, 0.658, 0.758, 0.925, 0.606, 0.642, 0.787, 0.625, 0.61, 0.723, 0.756, 0.771]
[10:06:49] harbor_runner: RESULT pass@1  0.250 -> 0.400  (delta +0.150)
[10:06:49] harbor_runner: summary -> /tmp/harbor_e2e_g3wqgj7i/summary.json
[10:06:49] harbor_runner: shutting down Cortex job
```

## Full baseline completions

| Task | Reward | Model output |
| ---- | ------ | ------------ |
| `heldout_000_900x96` | 1.00 | `900 * 96 = 86400` |
| `heldout_001_181x82` | 0.70 | `181 * 82 = 14942` |
| `heldout_002_687x78` | 0.70 | `687 × 78 = 53, 894.  ·  · Final integer: 53894` |
| `heldout_003_602x71` | 1.00 | `602 * 71 = 42,742.` |
| `heldout_004_235x92` | 0.70 | `235 * 92 = 21, 780.  ·  · Final integer: 21780` |
| `heldout_005_425x92` | 0.70 | `425 * 92 = 39, 390.  ·  · Final integer: 39390` |
| `heldout_006_913x92` | 0.05 | `913 * 92 = 83, 836.  ·  · Final integer: 836.` |
| `heldout_007_965x22` | 0.50 | `965 * 22 = 21630 ·  · Final integer: 21630` |
| `heldout_008_761x34` | 0.50 | `761 * 34 = 26, 874.  ·  · Final integer: 26874` |
| `heldout_009_252x43` | 0.50 | `252 * 43 = 10, 996.  ·  · Final integer: 10996` |
| `heldout_010_684x16` | 1.00 | `684 × 16 = 10944 ·  · Final integer: 10944` |
| `heldout_011_818x42` | 0.70 | `818 * 42 = 34, 196.  ·  · Final integer: 34196` |
| `heldout_012_507x39` | 0.70 | `507 * 39 = 19833` |
| `heldout_013_953x90` | 1.00 | `953 * 90 = 85,770 ·  · Final integer: 85770` |
| `heldout_014_511x19` | 0.70 | `511 * 19 = 9689 ·  · Final integer: 9689` |
| `heldout_015_376x13` | 0.70 | `376 * 13 = 4908` |
| `heldout_016_304x92` | 0.05 | `304 * 92 = 28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,` |
| `heldout_017_994x74` | 0.05 | `994 * 74 = 72, 994 * 74 = 72, 994 * 74 = 72, 994 * 74 = 72, 994 * 74 = 72,` |
| `heldout_018_313x33` | 1.00 | `313 * 33 = 10,329` |
| `heldout_019_991x96` | 0.70 | `991 * 96 = 95184 ·  · Final integer: 95184` |

## Full post-training completions

| Task | Reward | Model output |
| ---- | ------ | ------------ |
| `heldout_000_900x96` | 1.00 | `900 × 96 = 86400 ·  · Final integer: 86400` |
| `heldout_001_181x82` | 0.70 | `181 × 82 = 14882 ·  · Final integer: 14882` |
| `heldout_002_687x78` | 0.70 | `687 × 78 = 53, 826 ·  · Final integer: 53826` |
| `heldout_003_602x71` | 1.00 | `602 × 71 = 42742 ·  · Final integer: 42742` |
| `heldout_004_235x92` | 0.70 | `235 × 92 = 21,740 ·  · Final integer: 21740` |
| `heldout_005_425x92` | 0.70 | `425 × 92 = 38900 ·  · Final integer: 38900` |
| `heldout_006_913x92` | 0.70 | `913 × 92 = 83, 836 ·  · Final integer: 83836` |
| `heldout_007_965x22` | 1.00 | `965 × 22 = 21230 ·  · Final integer: 21230` |
| `heldout_008_761x34` | 0.50 | `761 × 34 = 26, 874 ·  · Final integer: 26874` |
| `heldout_009_252x43` | 0.50 | `252 × 43 = 10956 ·  · Final integer: 10956` |
| `heldout_010_684x16` | 1.00 | `684 × 16 = 10944 ·  · Final integer: 10944` |
| `heldout_011_818x42` | 0.70 | `818 × 42 = 34, 196 ·  · Final integer: 34196` |
| `heldout_012_507x39` | 0.70 | `507 × 39 = 19783 ·  · Final integer: 19783` |
| `heldout_013_953x90` | 1.00 | `953 × 90 = 85770 ·  · Final integer: 85770` |
| `heldout_014_511x19` | 0.50 | `511 × 19 = 9609 ·  · Final integer: 9609` |
| `heldout_015_376x13` | 1.00 | `376 × 13 = 4888 ·  · Final integer: 4888` |
| `heldout_016_304x92` | 0.05 | `304 × 92 = 28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,` |
| `heldout_017_994x74` | 0.70 | `994 × 74 = 72, 996 ·  · Final integer: 72996` |
| `heldout_018_313x33` | 1.00 | `313 × 33 = 10329 ·  · Final integer: 10329` |
| `heldout_019_991x96` | 1.00 | `991 × 96 = 95136 ·  · Final integer: 95136` |
