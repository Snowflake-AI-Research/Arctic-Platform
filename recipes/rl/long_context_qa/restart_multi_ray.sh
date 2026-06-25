HOSTFILE="${JOB_HOSTFILE:/data-fast/hostfile}"
ray_path=$(which ray)
# PY_ENV="source ~/py_env/gsched/bin/activate"

# stop existing ray instances
stop_ray_cmd="$ray_path stop --force --grace-period 30 -v"
ds_ssh -f $HOSTFILE "eval ${stop_ray_cmd}"

echo "head node is $head_node"
head_node=$(head -n 1 $HOSTFILE | cut -d ' ' -f 1)
# launch ray on the head node
main_ray_cmd="${ray_path} start --head --port=6379 --num-gpus=8 --num-cpus=64"
eval ${main_ray_cmd}
# get the rest IPs and save it to ${HOSTFILE}-worker
tail -n +2 $HOSTFILE > ${HOSTFILE}-worker
worker_ray_cmd="$ray_path start --address ${head_node}:6379  --num-gpus 8 --num-cpus=64"
ds_ssh -f ${HOSTFILE}-worker "eval ${worker_ray_cmd}"
