#!/usr/bin/env bash
# Tunji's gsm8k recipe config, on Cortex, at his seeds, with a sign-flip control
# arm per seed. Two jobs at a time (each takes 1 training + 1 sampling GPU) to
# stay polite on the shared QA6 account.
set -u

DEMO=/modeling-code/karthik/abstract-remote-exps/client-side-loss-demo
OUT=$DEMO/seedmatrix
mkdir -p "$OUT"

export PYTHONPATH=/code/users/karthik/ap-e2e:/code/users/karthik/thong-client:$DEMO

COMMON=(
  --cortex-config /code/users/karthik/qa6_dsa_config.json
  --client-repo /code/users/karthik/thong-client
  # The default deployed image cannot decode the /operation payload_b64 that the
  # forward path sends, so forward_only() fails to parse the frame server-side.
  # Thong's chunk-assembly work is only in this debug build.
  --debug-image-tag dev_20260828_170144_40ee0a90875
  --model Qwen/Qwen3-1.7B
  --num-prompts 512
  --num-generations 8
  --per-device-bsz 32
  --max-completion-length 256
  --max-seq-len 1024
  --max-steps 100
  --num-train-epochs 100
  --training-gpus 1
  --sampling-gpus 1
  --attn-impl flash_attention_3
  --lr 3e-6
  --client-loss-encoding grpo
)

run_one() {
  local seed=$1 arm=$2 extra=$3
  local tag="s${seed}_${arm}"
  echo "[$(date +%H:%M:%S)] start $tag"
  python3 "$DEMO/run_gsm8k_grpo_cortex.py" "${COMMON[@]}" \
    --seed "$seed" $extra \
    --metrics-out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $tag rc=$?"
}

for seed in 7 42 123; do
  echo "=== seed $seed: normal + control in parallel ==="
  run_one "$seed" normal  ""                    &
  p1=$!
  sleep 20   # stagger job creation so the two do not race on the same allocation
  run_one "$seed" control "--negate-advantages" &
  p2=$!
  wait $p1 $p2
  echo "=== seed $seed complete ==="
done

echo "ALL RUNS COMPLETE"
ls -la "$OUT"
