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

import builtins
import cProfile
import fcntl
import gc
import os
import pstats
import random
from collections import defaultdict
from contextlib import nullcontext
from pstats import Stats

import numpy as np
import psutil
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer
from torch.profiler import ProfilerActivity
from torch.profiler import profile

# Set to True to quickly temporarily turn off all debugging w/o needing to disable each call
#
# XXX: perhaps add API so that the operator could tweak this global from the main script and not
# mess with this module and commit wrong things by mistake
DISABLE_DEBUG = True

can_run_pynvml = True
try:
    import pynvml

    pynvml.nvmlInit()
except Exception:
    can_run_pynvml = False

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

pynvml_handle = None


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_device_id():
    """
    Derive the device id running this rank with the help of LOCAL_RANK and CUDA_VISIBLE_DEVICES env vars. The device id is
    needed for applications like pynvml.

    returns `None` if CUDA_VISIBLE_DEVICES is set to ""
    """

    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    if cuda_visible_devices == "":
        return None
    visible_device_ids = list(map(int, cuda_visible_devices.split(",")))

    if dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
    else:
        local_rank = 0

    return visible_device_ids[local_rank]


def get_nvml_mem():
    global pynvml_handle

    if not can_run_pynvml:
        return 0

    if pynvml_handle is None:
        device_id = get_device_id()
        if device_id is None:
            return 0
        pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        # pynvml.nvmlShutdown()
    try:
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml_handle)
        return memory_info.used
    except pynvml.NVMLError_NotSupported:
        # DGX Spark fails here https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#nvidia-smi-reports-memory-usage-not-supported
        return 0


def gc_empty_accelerator_cache():
    """runs gc.collect and empties cuda cache.
    this is useful when wanting to test real memory usage
    do not use in production - only during debug - as it can be very expensive
    """
    gc.collect()
    get_accelerator().empty_cache()


def see_memory_usage(message, force=False, ranks=[0]):
    """
    Arguments:
        - `message`: a pre-amble message to print before the counter dumps - useful for annotating where each measurement has been taken - e.g. "before foo" and later "after foo"
        - `force`: allows you to leave see_memory_usage in the code w/o running the code, force=True to activate
        - `ranks`: by default prints only on rank 0 but sometimes we need to debug other ranks, so pass the list like ranks=[1,3]
    """
    return
    if not force:
        return
    rank = get_rank() if dist.is_initialized() else 0
    if rank not in ranks:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # In some situations we want to flush the cache but not others, so for now let the developer
    # override this manually - by default it should not be called. when it's not enabled use the
    # MA_* numbers to get the real memory usage, rather than CA_* ones
    # torch.cuda.empty_cache()

    # collect raw memory usage outside pytorch
    nv_mem = get_nvml_mem()

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)

    accelerator_mem_str = " | ".join(
        [
            f"MA {round(get_accelerator().memory_allocated() / 2**30, 2):0.2f} GB",
            f"Max_MA {round(get_accelerator().max_memory_allocated() / 2**30, 2):0.2f} GB",
            f"CA {round(torch_memory_reserved() / 2**30, 2):0.2f} GB",
            f"Max_CA {round(torch_max_memory_reserved() / 2**30, 2):0.2f} GB",
            f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
        ]
    )
    cpu_mem_str = f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%"

    # add '[rank] mp' prefix to enable easy grep
    print(f"[{rank}] mp: {message}")
    print(f"[{rank}] mp: " + " | ".join([accelerator_mem_str, cpu_mem_str]))

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


def get_mem_metrics():

    gc.collect()
    # torch.cuda.empty_cache()

    nv_mem = get_nvml_mem()

    summary = " | ".join(
        [
            f"MA {round(get_accelerator().memory_allocated() / 2**30, 2):0.2f} GB",
            f"Max_MA {round(get_accelerator().max_memory_allocated() / 2**30, 2):0.2f} GB",
            f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
        ]
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    # this will lead to wrong peak reports if `see_mem_usage` is also used during the run,
    # as it resets the peak counter and there is only one counter
    get_accelerator().reset_peak_memory_stats()

    return summary


# fcntl.flock can be slow on shared fs, so if things are too slow especially when many ranks are
# used, you will want it off at a cost of interleaved prints from the same host. by default it'll
# be False to keep things fast, but set it to true when interleaved prints interfere with debug
#
# TODO: alternatively could try to point to some temp file on a local NVME drive - but it's hard to
# tell if say `/tmp` is on the local drive
USE_PRINTFLOCK = False
# PRINT_FLOCK_FILE = "/tmp/printflock.lock"
PRINT_FLOCK_FILE = __file__


def printflock(*args, **kwargs):
    """
    This is a wrapper around the built-in Python `print` which calls `flock` before calling
    `print` and unlocks it immediately after. This wrapper is useful for when each rank needs to
    print a message without getting it interleaved with prints from other ranks.
    The lock file is the file this wrapper is defined in.
    The output order will be random per rank.

    Example:
        >>> # assuming 4 GPUs
        >>> world_size = dist.get_world_size()
        >>> rank = dist.get_rank()
        >>> printflock(f"This is a very long message from rank {rank}/{world_size}")
       This is a very long message from rank 0/4
       This is a very long message from rank 2/4
       This is a very long message from rank 3/4
       This is a very long message from rank 1/4

    It can also be used to override normal `print`:

    from arctictraining.debug import printflock as print

    and then you don't need to change anything in your code.
    """

    #    with open(__file__, "r") as fh:
    with open(PRINT_FLOCK_FILE, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


if USE_PRINTFLOCK:
    print = printflock


def print_rank(*msg, force=True, ranks=None, **kwargs):
    """print something on all global ranks with [rank] prefix.
    if `ranks is not None` passed then only those ranks will be printed

    e.g. to print just on ranks 0 and 3:
    print_rank(*msg, ranks=[0,3]):

    """
    if DISABLE_DEBUG or not force:
        return
    global_rank = get_rank()
    if ranks is not None and global_rank not in ranks:
        return
    print(f"[{global_rank}]", *msg, **kwargs)


def print_rank0(*msg, force=True, **kwargs):
    """print something only on rank 0"""
    if DISABLE_DEBUG or not force:
        return

    global_rank = get_rank()
    if global_rank == 0:
        print(f"[{global_rank}]", *msg, **kwargs)


pr = print_rank
pr0 = print_rank0


def debug_gathered_tensor(tensor, group, name=None, dim=0):
    """gather a tensor across ranks of the given group and dump its shape and norm

    Arguments:
      - `tensor`: tensor to gather
      - `group`: process group to gather on
      - `name`: optional - the variable name for the tensor
      - `dim`: which dimension to gather on. default: 0

    """

    world_size = dist.get_world_size(group)
    prefix = f"gathered {name}" if name is not None else "gathered"

    tensor = tensor.contiguous()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)

    # concatenate on any dimension since we are just doing norm on everything
    gathered_tensor = torch.cat(tensor_list, dim=dim)
    print_rank0(f"{prefix}: shape: {gathered_tensor.shape}")
    print_rank0(f"{prefix}: norm:  {torch.norm(gathered_tensor)}")
    # print_rank0(f"{prefix}:  {gathered_tensor}")


class SynchronizedWallClockTimerSimpleDummy(object):
    """A dummy version of SynchronizedWallClockTimerSimple which can use the same API but it just won't do anything"""

    def __init__(self, *args, **kwargs):
        self.times = defaultdict(float)

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, *args, **kwargs):
        return self


class SynchronizedWallClockTimerSimple(SynchronizedWallClockTimer):
    """
    This is a simplified version of SynchronizedWallClockTimer that assumes that each timer does
    just start/stop and also takes care of not running the timers if its internal flag
    wall_clock_breakdown is False, so there is no need to litter the code with conditionals,
    leading to this:

        tname = timers.start("test2")
        import time; time.sleep(0.5)
        timers.stop(tname)
        print(timers.times[tname]) # reported time is in msecs

        # but first you need to activate the profiler
        timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
        # or if done at a later stage
        timers = SynchronizedWallClockTimerSimple()
        .... some place later ...
        timers.wall_clock_breakdown = True

        # to have even less overhead when wanting to disable timers use:
        ENABLE_TIMERS = True
        if ENABLE_TIMERS:
            from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple
            timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
        else:
            from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy
            timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


    To flip off across all files:
    $ grep --exclude-dir=.git -lIr "ENABLE_TIMERS = True" . | grep -v arctic_platform.rl.utils.debug.py | xargs -r -n1 perl -pi -e 's|ENABLE_TIMERS = True|ENABLE_TIMERS = False|g'
    To flip on:
    $ grep --exclude-dir=.git -lIr "ENABLE_TIMERS = False" . | grep -v arctic_platform.rl.utils.debug.py | xargs -r -n1 perl -pi -e 's|ENABLE_TIMERS = False|ENABLE_TIMERS = True|g'

    """

    def __init__(self, wall_clock_breakdown=False):
        self.wall_clock_breakdown = wall_clock_breakdown
        super().__init__()  # creates self.timers

        # elapsed time storage
        self.times = defaultdict(float)

    def start(self, name):
        """starts the clock if timing is enabled, returns the name of the timer to enable convenient type-once one-liners:

        name = timers.start("foo")
        timers.stop(name)

        """
        if self.wall_clock_breakdown:
            self(name).start()
        return name

    def stop(self, name, reset=True):
        """stops the clock and immediately stores the elapsed time"""
        if self.wall_clock_breakdown:
            self(name).stop()
            self.times[name] = self(name).elapsed(reset=reset)
        else:
            self.times[name] = 0

    def elapsed(self, name):
        """returns times stored by stop()"""
        return self.times[name]

    def print_elapsed(self, name, prefix="TIMER"):
        """prints elapsed timing"""
        print(f"{prefix}: {name}: {self.times[name]:.2f}msec")

    def stop_and_print_elapsed(self, name, prefix="TIMER"):
        """stop and prints elapsed timing"""
        self.stop(name)
        self.print_elapsed(name, prefix=prefix)


##### Profilers Setup Start #####


# customize the precision of cProfile to give 6 decimals
pstats.f8 = lambda x: f"{x:3.6f}"


class ProfilerContext:
    """
    A proxy Profiler context manager class that can quickly choose between cProfile, torch.profiler and no-profiler w/o changing the end user code (other than changing the profiler type flag)

    Example:

    prof_fwd = ProfilerContext(type="c", name="some context")
    with prof_fwd():
        x = 1
    prof_fwd.report()
    """

    def __init__(self, type="none", name=None):
        """

        Args:
        - type: torch: torch.profiler, c: cProfile, none: none
        - name: some context string for the reports

        For usage see example in the class docstring

        """
        self.torch = False
        self.c = False
        if type == "torch":
            self.torch = True
        elif type == "c":
            self.c = True
        elif type == "none":
            pass
        else:
            raise ValueError(f"the `type` can be one of torch|c|none but got {type}")

        self.name = name if name is not None else "unknown"

        if self.torch:
            self.ctx = profile(activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=True)
        elif self.c:
            self.ctx = cProfile.Profile()
        else:
            self.ctx = nullcontext()

    def __call__(self):
        if self.torch and torch.cuda.is_available():  # or self.c
            torch.cuda.synchronize()

        return self.ctx

    def report(self):
        if self.torch:
            # print("*** cProfile FWD ***")
            # prof_post_fwd.print_stats(sort=1)
            print(f"*** torch.profile {self.name} ***")
            print(self.ctx.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        elif self.c:
            print(f"*** cProfile {self.name} ***")
            stats = Stats(self.ctx)
            stats.sort_stats("tottime").print_stats(20)
            # cumulative is useful to understand where some of the large internal time overheads come from - because it shows you the stack of calls leading to the slow call. So `tottime` shows candidates to study and `cumulative` for finding context for those calls
            stats.sort_stats("cumulative").print_stats(50)
        else:
            pass  # report nothing


##### Profilers Setup END #####


def _is_npu_available() -> bool:
    """Best-effort check for an available Ascend NPU backend (torch_npu)."""
    return hasattr(torch, "npu") and torch.npu.is_available()


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    is_npu_available = _is_npu_available()
    if is_npu_available:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["NCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if is_npu_available:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
