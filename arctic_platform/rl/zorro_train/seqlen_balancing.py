# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import heapq
import itertools
import logging
from typing import List, Tuple, Optional, Dict
from tensordict import TensorDict
import numpy as np

import torch
from torch import distributed as dist

from arctic_platform.rl.utils.debug import see_memory_usage, pr, pr0

logger = logging.getLogger(__name__)

ENABLE_BALANCE_STATS = False

ENABLE_TIMERS = False

if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple
    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy
    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)

is_cuda_available = torch.cuda.is_available()
def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    else:
        device = "cpu"
    return device

def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def greedy_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def log_seqlen_unbalance(seqlen_list: List[int], partitions: List[List[int]], prefix):
    """
    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
        prefix (str): A prefix to be added to each metric key in the returned dictionary.

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance.
    """
    # Get the number of partitions
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Iterate over each batch of sequence lengths
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def ceildiv(a, b):
    return -(a // -b)


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


# def rearrange_micro_batches(
#     batch,
#     max_token_len,
#     dp_group=None,
#     num_batches_divided_by=None,
#     same_micro_num_in_dp=True,
#     min_num_micro_batch=None,
#     use_prompt_deduplication=False,
#     max_group_length_threshold=None,
# ):
#     """
#     Split a batch into micro-batches by total token count, with optional DP sync and padding.

#     Args:
#         batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
#         max_token_len (int): max sum of attention_mask per micro-batch.
#         dp_group (optional): torch.distributed group for data-parallel sync.
#         num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
#         same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
#         min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
#         use_prompt_deduplication (bool): if True, use deduplication-aware batching strategy.

#     Returns:
#         List[TensorDict]: the micro-batches.
#         List[List[int]]: index lists mapping each micro-batch back to original positions.
#     """
#     # If prompt deduplication is enabled, delegate to the dedup version
#     if use_prompt_deduplication:
#         response_length = batch["responses"].size(-1)
#         #write to file the batch size and response length

#         if ENABLE_TIMERS:
#             dist.barrier() # sync ranks first before the first timer is called
#             timers.start("rearrange_micro_batches_with_dedup")

#         micro_batches, micro_bsz_idx = rearrange_micro_batches_with_dedup(
#             batch=batch,
#             response_length=response_length,
#             max_token_len=max_token_len,
#             dp_group=dp_group,
#             num_batches_divided_by=num_batches_divided_by,
#             same_micro_num_in_dp=same_micro_num_in_dp,
#             min_num_micro_batch=min_num_micro_batch,
#             max_group_length_threshold=max_group_length_threshold,
#         )
#         if ENABLE_TIMERS:
#             timers.stop("rearrange_micro_batches_with_dedup")
#             pr(f"rearrange_micro_batches_with_dedup elapsed {timers.times['rearrange_micro_batches_with_dedup']:.2f}msec")

#         return micro_batches, micro_bsz_idx



def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map

def compute_variation(l):
    """
        this computes a few useful statistics to quickly tell if the spread in the list is small.
        1. max/min ratio to see how far the outliers are from each other
        2. in particular since we want to avoid outliers on the high end this computes the distance of the max value from the median, then normalized by media - the closer to 0 the better.
    """
    import statistics
    l = sorted(l)
    min_cost, max_cost = l[0], l[-1]
    min_max_ratio = max_cost/min_cost
    divergence_cost = (max(l)-statistics.median(l))/statistics.median(l)
    return max_cost, min_max_ratio, divergence_cost

def compute_group_costs(batch, prompt_groups, prompt_len, quadratic_coeff=0):

    # I haven't found quadratic_coff>0 to be of any performance improvement at least with 16k+4k setup, but it might prove useful in longer sequence set ups
    #
    # some math to illustrate the effect of the quadratic_coeff w/ a given prompt length
    #
    # In [37]: p=16384; d=0; (p*p*d+p*(1-d))/2**10 # linear only
    # Out[37]: 16.0
    # In [36]: p=16384; d=1/p; (p*p*d+p*(1-d))/2**10 # this is just ~2*p, pointless since everything is 2x
    # Out[36]: 31.9990234375
    # In [38]: p=16384; d=1e-4; (p*p*d+p*(1-d))/2**10
    # Out[38]: 42.2128
    # In [39]: p=16384; d=1e-3; (p*p*d+p*(1-d))/2**10
    # Out[39]: 278.128

    if not 0 <= quadratic_coeff <= 1:
        raise ValueError(f"quadratic_coeff value needs to be between 0 and 1, but got {quadratic_coeff=}")

    # Step 2: Calculate deduplicated token cost for each group
    # Linear Group cost = prompt_len + (num_samples_in_group * response_len)
    # Quadratic Group cost = prompt_len^2 + (num_samples_in_group * (prompt_len+response_len)*response_len)
    # The weighted average of the above 2 is used
    group_costs = []
    total_dedup_tokens = 0
    for group in prompt_groups:
        # Use attention_mask to get actual token count
        group_prompt_tokens = batch["attention_mask"][group[0], :prompt_len].sum().item()
        dedup_cost_quadratic = group_prompt_tokens**2
        dedup_cost_linear = group_prompt_tokens

        # Response tokens for all samples in group
        group_response_tokens = 0
        for sample_idx in group:
            response_tokens = batch["attention_mask"][sample_idx, prompt_len:].sum().item()
            dedup_cost_quadratic += (group_prompt_tokens + response_tokens) * response_tokens
            dedup_cost_linear += response_tokens
            group_response_tokens += response_tokens

        dedup_cost = quadratic_coeff * dedup_cost_quadratic + (1-quadratic_coeff) * dedup_cost_linear
        total_dedup_tokens += group_prompt_tokens + group_response_tokens
        group_costs.append((dedup_cost, group))

    if ENABLE_BALANCE_STATS:
        dedup_costs = [cost[0] for cost in group_costs]
        max_cost, min_max_ratio, divergence_cost = compute_variation(dedup_costs)
        pr0(f"dedup_costs:\n", "\n".join(map(str, sorted(dedup_costs))), sep="")
        pr(f"dedup_costs max_cost {max_cost} (in {len(dedup_costs)} items)")
        pr(f"dedup_costs outlier max/min diff is of {min_max_ratio:0.2f}x (in {len(dedup_costs)} items)")
        pr(f"dedup_costs variation {divergence_cost:0.4f} (in {len(dedup_costs)} items)")

    return group_costs, total_dedup_tokens


def create_prompt_groups(input_ids, response_length, max_token_len, max_group_length_threshold):
    batch_size, seq_len = input_ids.shape
    prompt_len = seq_len - response_length

    # Step 1: Find prompt groups in the entire mini-batch
    # Extract prompts and group by equality
    prompts = input_ids[:, :prompt_len]

    prompt_groups_map = {}  # Maps prompt tuple hash -> list of sample indices
    for i in range(batch_size):
        prompt_tuple = tuple(prompts[i].tolist())
        if prompt_tuple not in prompt_groups_map:
            prompt_groups_map[prompt_tuple] = []
        prompt_groups_map[prompt_tuple].append(i)

    # Convert to list of groups
    prompt_groups = list(prompt_groups_map.values())

    # Split large groups if max_group_length_threshold is set and positive
    if prompt_groups and max_group_length_threshold is not None and max_group_length_threshold > 0:

        # first check the boundaries are configured correctly
        max_token_len_effective = prompt_len + response_length * max_group_length_threshold
        #pr0(f"sanity check: {max_token_len_effective=} {max_token_len=}")
        if max_token_len < max_token_len_effective:
            raise ValueError(
                f"""
                {max_token_len=} is smaller than {max_token_len_effective}.
                Where max_token_len_effective = {prompt_len=} + {response_length=} * {max_group_length_threshold=}.
                You can either raise actor_rollout_ref.actor.ppo_max_token_len_per_gpu to a higher value or reduce the number of rollouts (`actor_rollout_ref.rollout.n`) or to split the prompt groups by setting `arctic_rl.zorro_train.max_rollouts` to a smaller value than `actor_rollout_ref.rollout.n`, so that the math above checks out.
                Reducing prompt and/or response sizes is another way if it doesn't break the training needs.""")

        max_group_length = max(len(group) for group in prompt_groups)
        if max_group_length > max_group_length_threshold:
            # Split large groups into smaller groups of size <= max_group_length_threshold
            # This trades off some deduplication efficiency for memory/compute constraints

            split_prompt_groups = []
            for group in prompt_groups:
                if len(group) > max_group_length_threshold:
                    # Split into chunks of max_group_length_threshold
                    for i in range(0, len(group), max_group_length_threshold):
                        end_idx = min(i + max_group_length_threshold, len(group))
                        split_prompt_groups.append(group[i:end_idx])
                else:
                    split_prompt_groups.append(group)
            prompt_groups = split_prompt_groups

    return prompt_groups


def reorg_global_batch_verl(
    super_batch,
    response_length: int,
    world_size,
    max_token_len,
    max_group_length_threshold,
):
    """
    This is the verl version that works with the old super-batch DataProto data structure.

    Rearrange batches and supporting data to load balance towards zorro use, that is group by prompt groups and then re-order for the best balance.
    """


    batch = super_batch.batch

    # if there is only world size batches or less then we can't load balance anything - return immediately
    if batch.shape[0] <= world_size:
        return super_batch

    input_ids = batch["input_ids"]

    prompt_groups = create_prompt_groups(input_ids, response_length, max_token_len, max_group_length_threshold)
    batch_size, seq_len = input_ids.shape
    prompt_len = seq_len - response_length
    group_costs, total_dedup_tokens = compute_group_costs(batch, prompt_groups, prompt_len)

    # Step 3: num_micro_batches has to be of world_size because verl/ray split data world_size-way
    # it'd be very difficult to make world_size*n batches because we then won't be able to pack into bins as we won't know when to stop packing
    num_micro_batches = world_size

    # Create micro-batch bins
    micro_batch_bins = [[] for _ in range(num_micro_batches)]
    micro_batch_costs = [0 for _ in range(num_micro_batches)]

    # because of how verl splits the global batch the bins all have to end up with the same number of total batches
    batch_size_per_bin = batch.shape[0] // world_size

    def find_min_cost_index(num_micro_batches, micro_batch_bins, micro_batch_costs, group):
        # There are 2 constraints in this early version of bin packing:
        # 1. choose a bin with min cost
        # 2. while ensuring its capacity isn't going to overflow
        # if constraint 2 isn't satisfied go the next higher min cost bin
        # at the end all bins have to have the exact same number of items (batch size)
        # we can't do a more optimal version without the 2nd constraint because it may result in ranks having different batch_sizes which can't work with verl.

        # Important: there is no guarantee group-size is always the same for all items in micro_batch_costs.
        # Thus we need to make sure to put the largest costs first, in particular to deal with a situation of a duplicated prompt where 1 prompt group's size could be double of all other group sizes and then the last item won't fit into any bin as there will be 2 bins with a gap of 2 large group's size halves - so the duplicated prompt groups have to go in first. Probably could do a pre-sort by group size, but I think the 2x prompt group is already guaranting a larger micro_batch cost
        sorted_by_cost_indices = sorted(range(num_micro_batches), key=lambda i: micro_batch_costs[i])

        gs = len(group)
        for idx in sorted_by_cost_indices:
            bs = len(list(itertools.chain.from_iterable(micro_batch_bins[idx])))
            # pr(f"{idx=} {bs=} {gs=} {micro_batch_costs[idx]=}")
            if bs + gs <= batch_size_per_bin:
                return idx

        # If we hit this it means packing failed and some smaller item got where the large item could fit and thus at least one item now can't fit. I dealt with all use-cases I run into but perhaps there are some that I haven't encountered.
        raise ValueError(f"shouldn't reach here")

    # Sort groups
    # 1. by group size (descending)
    # 2. by cost (descending) (secondary)
    # for better load balancing and to ensure we can fit all groups into the exact amount of slots in the bins - because we have a constraint of needing all batches per rank to remain of the same batch_size - therefore triple and double-prompt groups have to go first to ensure everything fits - the prompt gets replicated by verl when it doesn't have enough samples.
    group_costs.sort(reverse=True, key=lambda x: (len(x[1]), x[0]))

    # Greedy bin packing: assign each group to the bin with minimum current cost
    # note that it purposefully doesn't respect max_token_len (Samyam's design) so it's possible for a bin to be longer than max_token_len
    for cost, group in group_costs:
        # Find bin with minimum cost
        min_bin_idx = find_min_cost_index(num_micro_batches, micro_batch_bins, micro_batch_costs, group)

        micro_batch_bins[min_bin_idx].append(group)
        micro_batch_costs[min_bin_idx] += cost

    # sanity check to ensure each bin is of batch_size/world_size size
    for i in range(world_size):
        assert len(list(itertools.chain.from_iterable(micro_batch_bins[i]))) == batch_size_per_bin, f"{len(list(itertools.chain.from_iterable(micro_batch_bins[i])))} != {batch_size_per_bin=}"

    if ENABLE_BALANCE_STATS:
        max_cost, min_max_ratio, divergence_cost = compute_variation(micro_batch_costs)
        pr0(f"everything micro_batch_costs unsorted:\n", "\n".join(map(str, micro_batch_costs)), sep="")
        pr(f"everything micro_batch_costs max_cost {max_cost} in {len(micro_batch_costs)} items)")
        pr(f"everything micro_batch_costs outlier max/min diff is of {min_max_ratio:0.2f}x (in {len(micro_batch_costs)} items)")
        pr(f"everything micro_batch_costs variation {divergence_cost:0.4f} (in {len(micro_batch_costs)} items)")

    # re-org batches to match the micro_batch_bin packing order, while preparing for a future /world_size split on the batch_size dimension
    from_to_indices = []
    # we already ensured micro_batch_bins is divisible by world_size
    slice_len = len(micro_batch_bins) // world_size
    # because the bins are sorted by cost, and we want each index in the micro-batch across ranks to be close in length/cost to other ranks, instead of taking a slice, we will index 1 item from each world size slice - e.g. with world_size=8:
    # rank 0: items 0, 8, 16
    # rank 1: items 1, 9, 17
    for rank in range(world_size):
        this_rank_micro_batch_bins = [micro_batch_bins[rank+world_size*i] for i in range(slice_len)]
        # remap the global batch into this rank's batch to keep only the entries it needs
        # while remapping the bin_groups to those new batch indices
        for bin_groups in this_rank_micro_batch_bins:
            for group in bin_groups:
                from_to_indices.extend(group)
    super_batch.batch = batch[from_to_indices]

    # now re-org the non-tensor part of the super-batch in the same way as the normal batch
    for key in super_batch.non_tensor_batch.keys():
        if isinstance(super_batch.non_tensor_batch[key], np.ndarray):
            super_batch.non_tensor_batch[key] = super_batch.non_tensor_batch[key][from_to_indices]

    return super_batch

def reorg_global_batch(
    batch,
    response_length: int,
    world_size,
    max_token_len,
    max_group_length_threshold,
):
    """
    This is the ARL version that works with ...

    Rearrange batches and supporting data to load balance towards zorro use, that is group by prompt groups and then re-order for the best balance.

    Args:
        batch (dict): must include "input_ids", "attention_mask", "position_ids"
        response_length (int): length of response portion in each sequence
        world_size (int): how many dp workers
        max_token_len (int): max deduplicated token count per micro-batch
        max_group_length_threshold (int): how many rollout responses per prompt group (ideally should be the same as the number of rollouts), but if the seqlen is too long half or quarter would still be OKish performance-wise. If this value is much lower than rollout.n then zorro will not only not help but may perform worse than w/o it.

    Returns the rearranged `batch` , and the indices that can be used to later revert the results back into the original order
    """

    # print("Rearranging batches")

    #max_group_length_threshold = 16

    input_ids = batch["input_ids"]
    batch_size, seq_len = input_ids.shape

    # if there is only world size batches or less then we can't load balance anything - return immediately
    if batch_size <= world_size:
        return batch

    # pr0(f"{batch_size=}")
    # pr0(f"{world_size=}")

    prompt_groups = create_prompt_groups(input_ids, response_length, max_token_len, max_group_length_threshold)
    prompt_len = seq_len - response_length
    group_costs, total_dedup_tokens = compute_group_costs(batch, prompt_groups, prompt_len)

    # Step 3: num_micro_batches has to be of world_size because verl/ray split data world_size-way
    # it'd be very difficult to make world_size*n batches because we then won't be able to pack into bins as we won't know when to stop packing
    num_micro_batches = world_size

    # Create micro-batch bins
    micro_batch_bins = [[] for _ in range(num_micro_batches)]
    micro_batch_costs = [0 for _ in range(num_micro_batches)]

    # because of how verl splits the global batch the bins all have to end up with the same number of total batches
    batch_size_per_bin = batch_size // world_size

    def find_min_cost_index(num_micro_batches, micro_batch_bins, micro_batch_costs, group):
        # There are 2 constraints in this early version of bin packing:
        # 1. choose a bin with min cost
        # 2. while ensuring its capacity isn't going to overflow
        # if constraint 2 isn't satisfied go the next higher min cost bin
        # at the end all bins have to have the exact same number of items (batch size)
        # we can't do a more optimal version without the 2nd constraint because it may result in ranks having different batch_sizes which can't work with verl.

        # Important: there is no guarantee group-size is always the same for all items in micro_batch_costs.
        # Thus we need to make sure to put the largest costs first, in particular to deal with a situation of a duplicated prompt where 1 prompt group's size could be double of all other group sizes and then the last item won't fit into any bin as there will be 2 bins with a gap of 2 large group's size halves - so the duplicated prompt groups have to go in first. Probably could do a pre-sort by group size, but I think the 2x prompt group is already guaranting a larger micro_batch cost
        sorted_by_cost_indices = sorted(range(num_micro_batches), key=lambda i: micro_batch_costs[i])

        gs = len(group)
        for idx in sorted_by_cost_indices:
            bs = len(list(itertools.chain.from_iterable(micro_batch_bins[idx])))
            # pr(f"{idx=} {bs=} {gs=} {micro_batch_costs[idx]=}")
            if bs + gs <= batch_size_per_bin:
                return idx

        # If we hit this it means packing failed and some smaller item got where the large item could fit and thus at least one item now can't fit. I dealt with all use-cases I run into but perhaps there are some that I haven't encountered.
        raise ValueError(f"shouldn't reach here")

    # Sort groups
    # 1. by group size (descending)
    # 2. by cost (descending) (secondary)
    # for better load balancing and to ensure we can fit all groups into the exact amount of slots in the bins - because we have a constraint of needing all batches per rank to remain of the same batch_size - therefore triple and double-prompt groups have to go first to ensure everything fits - the prompt gets replicated by verl when it doesn't have enough samples.
    group_costs.sort(reverse=True, key=lambda x: (len(x[1]), x[0]))

    # pr0(f"{group_costs=}")
    # pr0(f"{num_micro_batches=}")
    # pr0(f"{batch_size_per_bin=}")

    # Greedy bin packing: assign each group to the bin with minimum current cost
    # note that it purposefully doesn't respect max_token_len (Samyam's design) so it's possible for a bin to be longer than max_token_len
    for cost, group in group_costs:
        # Find bin with minimum cost
        min_bin_idx = find_min_cost_index(num_micro_batches, micro_batch_bins, micro_batch_costs, group)

        micro_batch_bins[min_bin_idx].append(group)
        micro_batch_costs[min_bin_idx] += cost

    # pr0(f"{micro_batch_costs=}")

    # sanity check to ensure each bin is of batch_size/world_size size
    for i in range(world_size):
        assert len(list(itertools.chain.from_iterable(micro_batch_bins[i]))) == batch_size_per_bin, f"{len(list(itertools.chain.from_iterable(micro_batch_bins[i])))} != {batch_size_per_bin=}"

    if ENABLE_BALANCE_STATS:
        max_cost, min_max_ratio, divergence_cost = compute_variation(micro_batch_costs)
        pr0(f"everything micro_batch_costs unsorted:\n", "\n".join(map(str, micro_batch_costs)), sep="")
        pr0(f"everything micro_batch_costs max_cost {max_cost} in {len(micro_batch_costs)} items)")
        pr0(f"everything micro_batch_costs outlier max/min diff is of {min_max_ratio:0.2f}x (in {len(micro_batch_costs)} items)")
        pr0(f"everything micro_batch_costs variation {divergence_cost:0.4f} (in {len(micro_batch_costs)} items)")

    # re-org batches to match the micro_batch_bin packing order, while preparing for a future /world_size split on the batch_size dimension
    from_to_indices = []
    # we already ensured micro_batch_bins is divisible by world_size
    slice_len = len(micro_batch_bins) // world_size
    # because the bins are sorted by cost, and we want each index in the micro-batch across ranks to be close in length/cost to other ranks, instead of taking a slice, we will index 1 item from each world size slice - e.g. with world_size=8:
    # rank 0: items 0, 8, 16
    # rank 1: items 1, 9, 17
    for rank in range(world_size):
        this_rank_micro_batch_bins = [micro_batch_bins[rank+world_size*i] for i in range(slice_len)]
        # remap the global batch into this rank's batch to keep only the entries it needs
        # while remapping the bin_groups to those new batch indices
        for bin_groups in this_rank_micro_batch_bins:
            for group in bin_groups:
                from_to_indices.extend(group)

    for key in batch.keys():
        #print(key, batch[key])
        batch[key] = batch[key][from_to_indices]
    #batch = batch[from_to_indices]

    # # now re-org the non-tensor part of the super-batch in the same way as the normal batch
    # for key in super_batch.non_tensor_batch.keys():
    #     if isinstance(super_batch.non_tensor_batch[key], np.ndarray):
    #         super_batch.non_tensor_batch[key] = super_batch.non_tensor_batch[key][from_to_indices]

    return batch, from_to_indices

def rearrange_micro_batches_with_dedup(
    batch: TensorDict,
    response_length: int,
    max_token_len: int,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
    max_group_length_threshold=None,
):
    """
    Split a batch into micro-batches considering prompt deduplication.

    Uses prompt-grouping strategy: groups samples with identical prompts together
    and packs groups into micro-batches based on deduplicated token count.

    Args:
        batch (TensorDict): must include "input_ids", "attention_mask", "position_ids"
        response_length (int): length of response portion in each sequence
        max_token_len (int): max deduplicated token count per micro-batch
        dp_group (optional): torch.distributed group for data-parallel sync
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to same count
        min_num_micro_batch (int, optional): force at least this many splits

    Returns:
        List[Dict]: micro-batches with 'reconstruction_info' and 'dedup_input_ids' added
        List[List[int]]: index lists mapping each micro-batch back to original positions
    """
    # Import here to avoid circular dependency
    from arctic_platform.rl.zorro_train import ZoRRoTrain

    if not isinstance(batch, TensorDict):
        raise ValueError(f"batch needs to be a TensorDict object, but got {type(batch)}")

    if "attention_mask" not in batch:
        raise ValueError(f"attention_mask is required to be in batch")
    if batch["attention_mask"] is None or batch["attention_mask"].ndim != 2:
        raise ValueError(f"attention_mask of shape [bs, seqlen] is required but got: {batch['attention_mask']}")

    #pr0(batch)
    #exit()

    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = get_device_name() # torch.device(f"cuda:{local_rank}")

    if torch.distributed.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    input_ids = batch["input_ids"]

    prompt_groups = create_prompt_groups(input_ids, response_length, max_token_len, max_group_length_threshold)
    batch_size, seq_len = input_ids.shape
    prompt_len = seq_len - response_length

    # Question: what if the sets are different? how is assert will help? what ensures symmetry across ranks
    # Check to see if the len of prompt_groups is same across all ranks
    if torch.distributed.is_initialized():
        local_num_groups = torch.tensor([len(prompt_groups)], dtype=torch.long, device=device)
        all_num_groups = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(all_num_groups, local_num_groups)
        all_num_groups = [t.item() for t in all_num_groups]

        assert len(set(all_num_groups)) == 1, (
            f"Number of prompt_groups must be the same across all ranks, but got: {all_num_groups}"
        )

    group_costs, total_dedup_tokens = compute_group_costs(batch, prompt_groups, prompt_len)

    # Step 3: Calculate number of micro-batches needed
    num_micro_batches = min(len(prompt_groups), ceildiv(total_dedup_tokens, max_token_len))

    if min_num_micro_batch is not None:
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)

    if dist.is_initialized() and same_micro_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches], device=device)
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()

    if num_batches_divided_by is not None:
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)

    # Sort groups by cost (descending) for better load balancing
    group_costs.sort(reverse=True, key=lambda x: x[0])

    # Create micro-batch bins
    #aname = "bin-packing"; timers.start(aname)
    micro_batch_bins = [[] for _ in range(num_micro_batches)]
    micro_batch_costs = [0 for _ in range(num_micro_batches)]

    # Greedy bin packing: assign each group to the bin with minimum current cost
    # note that it purposefully doesn't respect max_token_len (Samyam's design) so it's possible for a bin to be longer than max_token_len
    for cost, group in group_costs:
        # Find bin with minimum cost
        min_bin_idx = min(range(num_micro_batches), key=lambda i: micro_batch_costs[i])
        micro_batch_bins[min_bin_idx].append(group)
        micro_batch_costs[min_bin_idx] += cost
    #timers.stop(aname); pr(f"{aname} elapsed {timers.times[aname]:.2f}msec")

    # sort micro_batch_bins by costs to better load balance things across ranks
    micro_batch_costs, micro_batch_bins = zip(*sorted(zip(micro_batch_costs, micro_batch_bins), reverse=True))

    if ENABLE_BALANCE_STATS:
        max_cost, min_max_ratio, divergence_cost = compute_variation(micro_batch_costs)
        pr0(f"everything micro_batch_costs unsorted:\n", "\n".join(map(str, micro_batch_costs)), sep="")
        pr(f"everything micro_batch_costs max_cost {max_cost} in {len(micro_batch_costs)} items)")
        pr(f"everything micro_batch_costs outlier max/min diff is of {min_max_ratio:0.2f}x (in {len(micro_batch_costs)} items)")
        pr(f"everything micro_batch_costs variation {divergence_cost:0.4f} (in {len(micro_batch_costs)} items)")

    #aname = "create-final-mbs"; timers.start(aname)
    # Step 5: Create actual micro-batches with deduplication info
    micro_batches = []
    micro_bsz_idx = []

    for bin_groups in micro_batch_bins:
        if not bin_groups:
            # Empty bin (can happen with min_num_micro_batch)
            continue

        # Flatten groups to get all sample indices for this micro-batch
        sample_indices = []
        for group in bin_groups:
            sample_indices.extend(group)
        sample_indices.sort()  # Keep indices sorted

        # Extract samples
        curr_micro_batch_list = []
        for idx in sample_indices:
            # the weird range accessor is to keep the 0th dimension in place, only 1 item is retrieved
            # batch[idx]           : Tensor(shape=torch.Size([80]),
            # batch[idx : idx + 1] : Tensor(shape=torch.Size([1, 80])
            curr_micro_batch_list.append(batch[idx : idx + 1])
        curr_micro_batch = torch.cat(curr_micro_batch_list)

        # Convert TensorDict to regular dict to allow adding non-tensor metadata
        curr_micro_batch = {k: v for k, v in curr_micro_batch.items()}

        # Step 6: Create deduplication metadata for this micro-batch
        micro_input_ids = curr_micro_batch["input_ids"]
        micro_position_ids = curr_micro_batch["position_ids"]

        # Find prompt groups within this micro-batch
        prompt_groups_micro, unique_prompts = ZoRRoTrain.find_prompt_groups(
            micro_input_ids, response_length
        )

        # Create deduplicated batch
        dedup_input_ids, adapted_position_ids, reconstruction_info = ZoRRoTrain.create_deduplicated_batch(
            micro_input_ids,
            micro_position_ids,
            response_length,
            prompt_groups_micro,
            unique_prompts,
            attention_mask=curr_micro_batch["attention_mask"],
            use_unpad=True,
        )

        # Calculate and log token savings
        orig_tokens = micro_input_ids.numel()
        dedup_tokens = dedup_input_ids.numel()
        saved = orig_tokens - dedup_tokens
        saved_pct = 100 * saved / orig_tokens
        logger.debug(f"Dedup: {orig_tokens} -> {dedup_tokens} tokens ({saved}/{orig_tokens} = {saved_pct:.1f}% saved)")

        # Add deduplication info to micro-batch
        curr_micro_batch["dedup_input_ids"] = dedup_input_ids
        curr_micro_batch["adapted_position_ids"] = adapted_position_ids
        curr_micro_batch["reconstruction_info"] = reconstruction_info
        curr_micro_batch["prompt_groups"] = prompt_groups_micro
        # Store dedup metrics for wandb logging
        curr_micro_batch["dedup_metrics"] = {
            "orig_tokens": orig_tokens,
            "dedup_tokens": dedup_tokens,
            "tokens_saved": saved,
            "tokens_saved_pct": saved_pct,
            "num_unique_prompts": len(prompt_groups_micro),
            "batch_size": len(sample_indices),
        }

        micro_batches.append(curr_micro_batch)
        micro_bsz_idx.append(sample_indices)
    #timers.stop(aname); pr(f"{aname} elapsed {timers.times[aname]:.2f}msec")

    dedup_tokens = [mb['dedup_metrics']['dedup_tokens'] for mb in micro_batches]
    total_dedup_tokens = sum(dedup_tokens)

    if ENABLE_BALANCE_STATS:
        max_cost, min_max_ratio, divergence_cost = compute_variation(micro_batch_costs)
        pr0(f"micro_batch_costs:\n", "\n".join(map(str, sorted(micro_batch_costs))), sep="")
        pr(f"micro_batch_costs max_cost {max_cost} in {len(micro_batch_costs)} items)")
        pr(f"micro_batch_costs outlier max/min diff is of {min_max_ratio:0.2f}x (in {len(micro_batch_costs)} items)")
        pr(f"micro_batch_costs variation {divergence_cost:0.4f} (in {len(micro_batch_costs)} items)")

    # report cross-rank load balancing
    if ENABLE_BALANCE_STATS:
        micro_batch_num = len(micro_batch_costs)
        value = torch.tensor(micro_batch_costs, device=device)
        all_values = [torch.zeros_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(all_values, value)
        pr(f"{all_values=}")
        for batch_id in range(micro_batch_num):
            costs = [all_values[rank][batch_id].item() for rank in range(world_size)]
            max_cost, min_max_ratio, divergence_cost = compute_variation(costs)
            pr(f"{batch_id} cross-rank costs:\n", "\n".join(map(str, sorted(costs))), sep="")
            pr(f"{batch_id} cross-rank costs max_cost {max_cost} (in {len(costs)} items)")
            pr(f"{batch_id} cross-rank costs outlier max/min diff is of {min_max_ratio:0.2f}x (in {len(costs)} items)")
            pr(f"{batch_id} cross-rank costs variation {divergence_cost:0.4f} (in {len(costs)} items)")

    return micro_batches, micro_bsz_idx
