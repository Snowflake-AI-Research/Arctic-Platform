# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An example POD_ENV: what belongs to one pod rather than to the suite. env.sh sources it first, so it can also set
# anything env.sh defaults (SCRATCH, CUDA_HOME, the cache paths). Copy it, edit it, and export POD_ENV=<your copy>.

# Kubernetes pods that keep their NCCL / EFA settings in the container's environment (PID 1's) do not pass them to SSH
# sessions, and a gateway started from one then runs NCCL over sockets: a multi-node file ran 2,819 tok/s that way
# against 21,567 with the settings. Export them from PID 1:
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(NCCL|FI)_[A-Z0-9_]*=' | sed -E "s/^([^=]+)=(.*)$/export \1='\2'/")"

# Node-local NVMe for the caches, the checkpoints and the baked models:
# export SCRATCH=/mnt/nvme/$USER
# A CUDA toolkit of torch's own minor version, where the image ships another one:
# export CUDA_HOME=/usr/local/cuda-13.0
