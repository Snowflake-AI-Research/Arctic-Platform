export PRIME_RL_ROOT=/modeling-code/boyiliu/prime-rl
export R2E_DATASET=/data/fshu/important/swe_data/r2e_family/R2E-Gym-Subset_validgold_unique_baseline/train.jsonl
set -a; . /data-fast/cortex.env; set +a
export KUBECONFIG=/data-fast/k3s/kubeconfig.yaml
K="/data-fast/k3s/bin/k3s kubectl"
$K get pods -n default --no-headers 2>/dev/null | awk '/^r2e-/{print $1}' | xargs -r $K delete pod -n default --grace-period=0 --force >/dev/null 2>&1
cd /modeling-code/karthik/abstract-remote-exps
/data-fast/ap-venv/bin/python poc/r2e_driver.py --steps 14 --group 8 --concurrency 96 --seed 42 --lr 1e-6   --mops 4 --send-logprobs --rollout-timeout 9000   --task-ids 'coveragepy@c4fc3833,pillow@bfaa0a1f,coveragepy@5c3d0946,pyramid@a43abd25,scrapy@b51b52ff,aiohttp@fa628a21,orange3@78213643,numpy@a5322429,tornado@37081d79,coveragepy@16997254,pillow@e0b95724,pyramid@24c63558,scrapy@721df895,aiohttp@274c54e4,orange3@a2c7ac74,numpy@c8a09822'   --allow-content --no-std-norm --no-length-penalty --adam-beta2 0.98   --train-gpus 2 --sample-gpus 32 --micro-batch 2 --gpu-mem-util 0.85 --max-num-seqs 56 --tensor-parallel 1   --out /modeling-code/karthik/abstract-remote-exps/runs/overfit16b-20260929-103904 2>&1
