#!/usr/bin/env bash
# Does the gsm8k recipe run over Cortex on the DEFAULT image, with no pinned tag?
#
# The three-seed matrix in run_seed_matrix.sh had to pin
# dev_20260828_170144_40ee0a90875, because at 512 prompts the forward frame does
# not fit one /operation envelope and the then-current default image could not
# assemble a chunked payload (SafetensorError: header too large). Thong's fix
# merged Sep 1 (dss-client#83 -> 16225ef, plus the dss-platform half), and
# probe_default_image_chunked_forward.py has since shown a 33-chunk forward
# coming back bitwise identical to a single-envelope one on the default image.
#
# That probe is a single operation. This is the same recipe as the matrix, one
# seed, both arms, with the pin removed -- so it answers the question the matrix
# left open: can Tunji run this without a debug image.
#
# dss-client comes from a worktree of origin/main rather than Thong's branch, so
# a pass here is a statement about what is released, not about a local checkout.
set -u

DEMO=/modeling-code/karthik/abstract-remote-exps/client-side-loss-demo
OUT=$DEMO/defaultimage
CLIENT=/code/users/karthik/dssc-main
mkdir -p "$OUT"

export PYTHONPATH=/code/users/karthik/ap-e2e:$CLIENT:$DEMO
unset NEUTRINO_ENABLE_DEBUG_OPTIONS

COMMON=(
  --cortex-config /code/users/karthik/qa6_dsa_config.json
  --client-repo "$CLIENT"
  # No --debug-image-tag. That is the point of this run.
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
  echo "[$(date +%H:%M:%S)] start $tag (default image)"
  python3 "$DEMO/run_gsm8k_grpo_cortex.py" "${COMMON[@]}" \
    --seed "$seed" $extra \
    --metrics-out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $tag rc=$?"
}

seed=${1:-7}
echo "=== seed $seed: normal + sign-flipped control, default image ==="
run_one "$seed" normal  ""                    &
p1=$!
sleep 20   # stagger job creation so the two do not race on the same allocation
run_one "$seed" control "--negate-advantages" &
p2=$!
wait $p1 $p2
echo "RUN COMPLETE"
ls -la "$OUT"
