HOSTFILE="${JOB_HOSTFILE:-/data-fast/hostfile}"
ray_path=$(which ray)

# Use all 8 local GPUs on the head. If the launching shell pins CUDA_VISIBLE_DEVICES (common on dev boxes), the
# `ray start --num-gpus=8` below fails with "Attempting to start raylet with 8 GPU, but CUDA_VISIBLE_DEVICES
# contains ['0']" and the head/GCS never comes up (workers then fail with "No node info found for head node in
# GCS"). Clear it so ray sees every local GPU; ds_ssh worker shells start fresh and are unaffected.
unset CUDA_VISIBLE_DEVICES
# PY_ENV="source ~/py_env/gsched/bin/activate"

# stop existing ray instances
stop_ray_cmd="$ray_path stop --force --grace-period 30 -v"
ds_ssh -f $HOSTFILE "eval ${stop_ray_cmd}"

head_node=$(head -n 1 $HOSTFILE | cut -d ' ' -f 1)
echo "head node is $head_node"
# launch ray on the head node
main_ray_cmd="${ray_path} start --head --port=6379 --num-gpus=8 --num-cpus=64"
eval ${main_ray_cmd}
# get the rest IPs and save it to ${HOSTFILE}-worker
tail -n +2 $HOSTFILE > ${HOSTFILE}-worker
worker_ray_cmd="$ray_path start --address ${head_node}:6379  --num-gpus 8 --num-cpus=64"
ds_ssh -f ${HOSTFILE}-worker "eval ${worker_ray_cmd}"
