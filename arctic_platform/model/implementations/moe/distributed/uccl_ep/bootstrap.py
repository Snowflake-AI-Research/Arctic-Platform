"""UCCL-EP proxy bootstrap and EventOverlap.

Adapted from uccl d56e349 ``ep/deep_ep_wrapper/deep_ep/utils.py``.
Only the pieces required to construct ``uccl.ep.Buffer`` and wait on comm events.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any
from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist
from uccl.ep import EventHandle

try:
    from uccl import ep
except ImportError:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def _gather_peer_ips(group):
    # Gather local IP strings across ranks
    world = dist.get_world_size(group)
    my_ip = ep.get_oob_ip()
    ips = [None] * world
    dist.all_gather_object(ips, my_ip, group=group)
    return ips


def detect_group_topology(
    group: dist.ProcessGroup,
) -> Tuple[int, int, int, int, int, bool]:
    """
    Infer CUDA-local rank, node-local rank, and node topology.

    Returns:
        local_rank: CUDA-visible device index for the current process.
        node_local_rank: dense rank slot within the current node, ordered by
            physical GPU BDF.
        nic_local_rank: rank of the CUDA device in UCCL's physical GPU BDF
            order.
        node_idx: compact node index within the given group.
        num_nodes: number of distinct nodes spanned by the group.
        is_intranode: whether all ranks in the group are on the same node.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = torch.cuda.current_device()

    node_token = ep.get_oob_ip() or socket.gethostname()
    nic_local_rank = ep.get_physical_gpu_rank(local_rank)

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_info = {
        "rank": rank,
        "node_token": node_token,
        "local_rank": local_rank,
        "nic_local_rank": nic_local_rank,
    }
    all_info: list[Any] = [None] * world
    dist.all_gather_object(all_info, local_info, group=group)
    same_node = [info for info in all_info if info["node_token"] == node_token]

    if nic_local_rank < 0:
        raise ValueError(
            "Could not map CUDA device to UCCL physical GPU rank: " + json.dumps(local_info, sort_keys=True)
        )

    ranks_by_physical_gpu: dict[int, list[dict[str, Any]]] = {}
    for info in same_node:
        mapped_rank = info["nic_local_rank"]
        if mapped_rank < 0:
            continue
        ranks_by_physical_gpu.setdefault(mapped_rank, []).append(info)

    duplicate_physical_ranks = {
        mapped_rank: infos for mapped_rank, infos in ranks_by_physical_gpu.items() if len(infos) > 1
    }
    if duplicate_physical_ranks:
        raise ValueError(
            "Duplicate physical GPU rank mapping on node: " + json.dumps(duplicate_physical_ranks, sort_keys=True)
        )

    node_ranks = [info["rank"] for info in sorted(same_node, key=lambda info: (info["nic_local_rank"], info["rank"]))]
    node_local_rank = node_ranks.index(rank)

    token_to_idx: dict[str, int] = {}
    for info in all_info:
        token = info["node_token"]
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    node_idx = token_to_idx[node_token]
    num_nodes = len(token_to_idx)
    is_intranode = num_nodes == 1
    return (
        local_rank,
        node_local_rank,
        nic_local_rank,
        node_idx,
        num_nodes,
        is_intranode,
    )


def get_cpu_proxies_meta(proxies, rank, scratch_ptr, scratch_bytes, num_ranks, group):
    my_ip = ep.get_oob_ip()
    meta = {
        "rank": rank,
        "ptr": int(scratch_ptr),
        "nbytes": int(scratch_bytes),
        "ip": my_ip,
        "listen_ports": [proxy.get_listen_port() for proxy in proxies],
    }
    all_meta = [None] * num_ranks
    # Use current device or fallback to LOCAL_RANK or 0
    if "LOCAL_RANK" in os.environ:
        device_index = int(os.environ["LOCAL_RANK"])
    else:
        device_index = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    dist.all_gather_object(all_meta, meta, group=group)
    rank2meta = {m["rank"]: m for m in all_meta}

    # Debug: print IP distribution
    ip_counts = {}
    for m in all_meta:
        ip = m["ip"]
        ip_counts[ip] = ip_counts.get(ip, 0) + 1
    if rank == 0:
        print(f"[DEBUG] IP distribution across {num_ranks} ranks:", flush=True)
        for ip, count in ip_counts.items():
            print(f"[DEBUG]   {ip}: {count} ranks", flush=True)

    return rank2meta


def check_nvlink_connections(group: dist.ProcessGroup):
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if "PCIE" in torch.cuda.get_device_name():
        assert group.size() <= 2, "PCIe GPUs only have pairwise NVLink connections"

        # noinspection PyUnresolvedReferences
        import pynvml

        pynvml.nvmlInit()

        # noinspection PyTypeChecker
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").strip(",").split(",")
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [
            0,
        ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink
        # Reference: https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                assert status == pynvml.NVML_P2P_STATUS_OK, (
                    f"GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via"
                    " NVLink"
                )

        # Close NVML
        pynvml.nvmlShutdown()


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(
        self,
        event: Optional[EventHandle] = None,
        extra_tensors: Optional[Tuple[torch.Tensor]] = None,
    ) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        stream_ptr = int(torch.cuda.current_stream().cuda_stream)
        self.event.current_stream_wait(stream_ptr)

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap():
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            stream_ptr = int(torch.cuda.current_stream().cuda_stream)
            self.event.current_stream_wait(stream_ptr)


def initialize_uccl(
    scratch_ptr,
    scratch_nbytes,
    rank,
    num_ranks,
    group,
    num_experts=0,
    is_intranode: Optional[bool] = None,
    use_normal_mode=False,
    rdma_buffer_is_host_allocated=False,
):
    (
        local_rank,
        node_local_rank,
        nic_local_rank,
        node_idx,
        num_nodes,
        detected_is_intranode,
    ) = detect_group_topology(group)
    if is_intranode is None:
        is_intranode = detected_is_intranode
    elif is_intranode and not detected_is_intranode:
        raise ValueError("Detected multi-node group topology, but `is_intranode=True` was requested")

    proxies = []

    for i in range(ep.get_num_proxy_threads()):
        proxy = ep.Proxy(
            thread_idx=i,
            gpu_buffer_addr=scratch_ptr,
            total_size=scratch_nbytes,
            rank=rank,
            node_idx=node_idx,
            local_rank=local_rank,
            num_experts=num_experts,
            num_ranks=num_ranks,
            num_nodes=num_nodes,
            use_normal_mode=use_normal_mode,
            is_intranode=is_intranode,
            gpu_buffer_is_host_allocated=rdma_buffer_is_host_allocated,
            barrier_local_rank=node_local_rank,
            device_index=local_rank,
            nic_local_rank=nic_local_rank,
        )
        proxies.append(proxy)

    rank2meta = get_cpu_proxies_meta(proxies, rank, scratch_ptr, scratch_nbytes, num_ranks, group)
    peers_meta_list = [rank2meta[r] for r in range(num_ranks)]

    if not is_intranode:
        for proxy in proxies:
            proxy.set_peers_meta(peers_meta_list)

    ep.register_proxies(local_rank, proxies)

    # Set atomic buffer pointer for all proxies BEFORE starting them
    # This ensures the atomic buffer info is included in connection info exchange
    # Note: Only thread 0's proxy allocates the atomic buffer in its constructor
    if not is_intranode and len(proxies) > 0:
        # Get atomic buffer pointer from thread 0 proxy (only thread 0 allocates it)
        # This must be done before start_dual() so the atomic buffer info is included
        # in the connection info exchange during init_common()
        atomic_buffer_ptr = proxies[0].get_atomic_buffer_ptr()
        if atomic_buffer_ptr:
            for proxy in proxies:
                proxy.set_atomic_buffer_ptr(atomic_buffer_ptr)

    dist.barrier(group)
    if not is_intranode:
        for proxy in proxies:
            proxy.start_dual()

    workers = None

    time.sleep(3)
    return proxies, workers


def destroy_uccl(proxies, workers):
    # Use current device or fallback to LOCAL_RANK
    if "LOCAL_RANK" in os.environ:
        device_index = int(os.environ["LOCAL_RANK"])
    else:
        device_index = torch.cuda.current_device()

    if workers is not None:
        try:
            workers.stop()
        except Exception:
            pass

    try:
        for p in proxies:
            p.stop()
    except Exception:
        pass
    try:
        ep.unregister_proxy(device_index)
    except Exception:
        pass

    # Do not sweep /dev/shm here. UCCL namespaces each local barrier by
    # node/user/mode/thread and its C++ Proxy remembers which process created
    # that exact object; Proxy::destroy unlinks only creator-owned names.
    # The names contain no job/group identifier, so Python cannot safely
    # distinguish a stale barrier from one used by a concurrent job.
