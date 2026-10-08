"""Correctness, memory, and latency smoke test for Arctic DeepEP 2.5."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.distributed as dist


def _routing(
    tokens: int,
    hidden: int,
    top_k: int,
    num_experts: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234 + rank)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=generator)
    # A router's top-k choices are unique per token. DeepEP rejects duplicate
    # expert IDs within one token.
    expert_ids = torch.rand(
        tokens,
        num_experts,
        device="cuda",
        generator=generator,
    ).topk(top_k, dim=-1).indices
    weights = torch.full(
        (tokens, top_k),
        1.0 / top_k,
        dtype=torch.float32,
        device="cuda",
    )
    return x, expert_ids, weights


def _time(operation, iterations: int) -> float:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    from arctic_platform.model.implementations.moe.distributed.ep_backend import (
        get_ep_comm_module,
    )

    ep_comm = get_ep_comm_module("deepep_v2")
    ep_comm.configure_num_sms(args.num_sms)
    x, expert_ids, weights = _routing(
        args.tokens,
        args.hidden,
        args.top_k,
        args.num_experts,
        rank,
    )

    def round_trip(input_x: torch.Tensor = x) -> torch.Tensor:
        pending = ep_comm.dispatch_tokens_async(
            input_x,
            expert_ids,
            weights,
            args.num_experts,
            dist.group.WORLD,
            score_before_experts=True,
        )
        routed, counts, state = ep_comm.finalize_dispatch_tokens(pending)
        combined = ep_comm.combine_tokens(routed, state)
        ep_comm.sync_combine()
        if int(counts.sum().item()) != routed.shape[0]:
            raise RuntimeError(
                f"expert counts sum to {int(counts.sum().item())}, "
                f"but dispatch returned {routed.shape[0]} rows"
            )
        return combined

    grad_input = x.detach().requires_grad_(True)
    round_trip(grad_input).float().sum().backward()
    torch.cuda.synchronize()
    grad_max_abs = float((grad_input.grad.float() - 1).abs().max().item())
    if grad_max_abs > 2e-2:
        raise RuntimeError(f"backward mismatch: max_abs={grad_max_abs}")

    torch.cuda.reset_peak_memory_stats()
    combined = round_trip()
    torch.cuda.synchronize()
    max_abs = float((combined - x).abs().max().item())
    if max_abs > 2e-2:
        raise RuntimeError(f"round-trip mismatch: max_abs={max_abs}")

    latency_ms = _time(round_trip, args.iterations)
    allocated_gib = torch.cuda.max_memory_allocated() / 2**30
    reserved_gib = torch.cuda.max_memory_reserved() / 2**30
    if rank == 0:
        print(
            f"DeepEP 2.5 Arctic round trip: tokens={args.tokens} hidden={args.hidden} "
            f"top_k={args.top_k} max_abs={max_abs:.4e} grad_max_abs={grad_max_abs:.4e} "
            f"median={latency_ms:.3f} ms "
            f"peak_allocated={allocated_gib:.3f} GiB peak_reserved={reserved_gib:.3f} GiB",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
