"""NCCLEngine — pipelined point-to-point weight transfer.

Two transfer modes:

**Bucket mode** (default):
    Weights are packed into fixed-size buckets.  A dedicated ``_nccl_stream``
    overlaps the NCCL transfer of bucket *N* with packing/processing of
    bucket *N±1* on the default stream (true double buffering).

**Direct mode** (BF16 TP=1 only):
    A manifest is sent first, then each weight tensor is transferred
    individually via ``nccl.send``/``recv`` straight into the receiver's
    pre-computed parameter views — zero intermediate buffer, zero copy.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
import json
import logging
import time

import torch

from arctic_inference.server.weight_sync.utils import stateless_init_nccl

logger = logging.getLogger(__name__)

META_BUF_SIZE = 262_144
DEFAULT_BUCKET_SIZE = 256 * 1024 * 1024
# First 8 bytes of each meta buffer are a uint64 length prefix.
_META_PAYLOAD_LIMIT = META_BUF_SIZE - 8


def _bucket_meta_nbytes(entries: list[dict], *, is_last: bool) -> int:
    """UTF-8 JSON size of a bucket metadata blob."""
    return len(
        json.dumps({"tensors": entries, "is_last": is_last}).encode("utf-8")
    )


class NCCLEngine:

    def __init__(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        device: torch.device,
        bucket_size: int = DEFAULT_BUCKET_SIZE,
        reverse: bool = False,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.bucket_size = bucket_size
        self.is_sender = rank == 0
        self._port = master_port

        torch.cuda.set_device(device)
        is_server = (not self.is_sender) if reverse else None
        self.nccl = stateless_init_nccl(
            master_addr, master_port, rank, world_size, device,
            is_server=is_server,
        )

        self._nccl_stream = torch.cuda.Stream(device=device)

        self._meta_bufs = [
            torch.zeros(META_BUF_SIZE, dtype=torch.uint8, device=device),
            torch.zeros(META_BUF_SIZE, dtype=torch.uint8, device=device),
        ]
        self._data_bufs = [
            torch.zeros(bucket_size, dtype=torch.uint8, device=device),
            torch.zeros(bucket_size, dtype=torch.uint8, device=device),
        ]
        if self._data_bufs[0].numel() != bucket_size:
            logger.warning(
                "NCCLEngine bucket buffer allocated %d bytes but requested %d",
                self._data_bufs[0].numel(), bucket_size,
            )
        else:
            logger.info(
                "NCCLEngine port=%d bucket_size=%d (%.1f MB)",
                master_port, bucket_size, bucket_size / (1024 * 1024),
            )
        self._send_events = [torch.cuda.Event() for _ in range(2)]
        self._op_count = 0

    def ensure_staging_buffers(self) -> None:
        """Reallocate 2×bucket data buffers after :meth:`release_staging_buffers`."""
        if self._data_bufs is not None:
            return
        self._data_bufs = [
            torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device),
            torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device),
        ]
        logger.info(
            "NCCLEngine port=%d re-allocated staging buffers (%.1f MB x2)",
            self._port, self.bucket_size / (1024 * 1024),
        )

    def release_staging_buffers(self) -> None:
        """Free data staging buffers; keep the NCCL communicator."""
        stream = getattr(self, "_nccl_stream", None)
        if stream is not None:
            stream.synchronize()
        bufs = self._data_bufs
        if bufs is None:
            return
        self._data_bufs = None
        del bufs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info(
            "NCCLEngine port=%d released staging buffers (kept communicator)",
            self._port,
        )

    # ------------------------------------------------------------------
    # Bucket mode — pipelined double-buffered send / receive
    # ------------------------------------------------------------------

    @torch.no_grad()
    def send_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict:
        assert self.is_sender, "Only rank 0 can send"
        self.ensure_staging_buffers()

        start = time.time()
        nccl_s = self._nccl_stream

        pack_idx = 0
        prev_idx: int | None = None
        bucket_meta: list[dict] = []
        offset = 0
        n_buckets = 0
        n_params = 0

        for name, weight in weights:
            w = weight.contiguous()
            nbytes = w.nbytes
            w_bytes = w.view(-1).view(torch.uint8)
            if w_bytes.device != self.device:
                w_bytes = w_bytes.to(self.device, non_blocking=True)

            if nbytes > self.bucket_size:
                if bucket_meta:
                    self._flush_bucket_send(
                        pack_idx, bucket_meta, is_last=False, prev_idx=prev_idx
                    )
                    prev_idx = pack_idx
                    pack_idx = 1 - pack_idx
                    bucket_meta = []
                    offset = 0
                    n_buckets += 1

                tensor_offset = 0
                while tensor_offset < nbytes:
                    chunk_nbytes = min(self.bucket_size, nbytes - tensor_offset)
                    entry = {
                        "name": name,
                        "shape": list(weight.shape),
                        "dtype": str(weight.dtype),
                        "offset": 0,
                        "nbytes": chunk_nbytes,
                        "tensor_offset": tensor_offset,
                        "tensor_nbytes": nbytes,
                    }
                    if _bucket_meta_nbytes([entry], is_last=True) > _META_PAYLOAD_LIMIT:
                        raise RuntimeError(
                            f"Single-tensor metadata for {name!r} "
                            f"({_bucket_meta_nbytes([entry], is_last=True)} B) "
                            f"exceeds META_BUF_SIZE ({META_BUF_SIZE} B)"
                        )
                    self._data_bufs[pack_idx][:chunk_nbytes].copy_(
                        w_bytes[tensor_offset:tensor_offset + chunk_nbytes],
                        non_blocking=True,
                    )
                    bucket_meta = [entry]
                    offset = chunk_nbytes
                    tensor_offset += chunk_nbytes
                    if tensor_offset < nbytes:
                        self._flush_bucket_send(
                            pack_idx,
                            bucket_meta,
                            is_last=False,
                            prev_idx=prev_idx,
                        )
                        prev_idx = pack_idx
                        pack_idx = 1 - pack_idx
                        bucket_meta = []
                        offset = 0
                        n_buckets += 1
                n_params += 1
                continue

            entry = {
                "name": name,
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "offset": offset,  # updated after a possible flush below
                "nbytes": nbytes,
            }
            # Flush on data OR meta overflow (many tiny MoE LoRA names).
            meta_overflow = bool(bucket_meta) and (
                _bucket_meta_nbytes(bucket_meta + [entry], is_last=True)
                > _META_PAYLOAD_LIMIT
            )
            data_overflow = bool(bucket_meta) and (
                offset + nbytes > self.bucket_size
            )
            if data_overflow or meta_overflow:
                self._flush_bucket_send(pack_idx, bucket_meta, is_last=False,
                                        prev_idx=prev_idx)
                prev_idx = pack_idx
                pack_idx = 1 - pack_idx
                bucket_meta = []
                offset = 0
                n_buckets += 1
                entry["offset"] = 0

            if _bucket_meta_nbytes([entry], is_last=True) > _META_PAYLOAD_LIMIT:
                raise RuntimeError(
                    f"Single-tensor metadata for {name!r} "
                    f"({_bucket_meta_nbytes([entry], is_last=True)} B) exceeds "
                    f"META_BUF_SIZE ({META_BUF_SIZE} B)"
                )

            self._data_bufs[pack_idx][offset:offset + nbytes].copy_(
                w_bytes, non_blocking=True,
            )
            entry["offset"] = offset
            bucket_meta.append(entry)
            offset += nbytes
            n_params += 1

        self._flush_bucket_send(pack_idx, bucket_meta, is_last=True,
                                prev_idx=prev_idx)
        n_buckets += 1

        nccl_s.synchronize()
        elapsed = time.time() - start
        logger.info(
            "send_weights: %d params, %d buckets, %.2fs, port=%d",
            n_params, n_buckets, elapsed, self._port,
        )
        return {
            "status": "done",
            "params_sent": n_params,
            "buckets": n_buckets,
            "elapsed": elapsed,
        }

    def _flush_bucket_send(self, idx: int, meta: list[dict], *,
                           is_last: bool, prev_idx: int | None) -> None:
        default = torch.cuda.current_stream(self.device)
        nccl_s = self._nccl_stream

        if prev_idx is not None:
            default.wait_event(self._send_events[prev_idx])

        self._write_meta(idx, meta, is_last=is_last)
        pack_done = torch.cuda.Event()
        pack_done.record(default)

        nccl_s.wait_event(pack_done)
        self._send_pair(idx, nccl_s)
        self._send_events[idx].record(nccl_s)

    @torch.no_grad()
    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Pipelined receive: prefetch next bucket while caller processes current.

        The yielded tensors are views into one of two reusable NCCL receive
        buffers. Callers normally enqueue async ``copy_`` operations from
        those views into model parameters. Before this method reuses a buffer,
        the NCCL stream must wait for the caller's current stream; otherwise a
        later receive can overwrite the staging buffer before the parameter
        copy has actually read it.
        """
        assert not self.is_sender, "Only non-zero ranks can receive"
        self.ensure_staging_buffers()

        nccl_s = self._nccl_stream
        # torch.zeros in ensure_staging_buffers runs on the default stream.
        # Wait before posting recv so an outstanding zero cannot clobber
        # bytes NCCL already wrote.
        nccl_s.wait_stream(torch.cuda.current_stream(self.device))
        consumed_events: list[torch.cuda.Event | None] = [None, None]
        partial_tensors: dict[str, tuple[torch.Tensor, int, int]] = {}

        def wait_until_consumed(idx: int) -> None:
            event = consumed_events[idx]
            if event is not None:
                nccl_s.wait_event(event)

        def mark_consumed(idx: int) -> None:
            event = consumed_events[idx]
            if event is None:
                event = torch.cuda.Event()
                consumed_events[idx] = event
            event.record(torch.cuda.current_stream(self.device))

        buf_idx = 0

        self._recv_pair(buf_idx, nccl_s)
        nccl_s.synchronize()

        while True:
            metadata = self._parse_meta(buf_idx)
            is_last = metadata["is_last"]

            if not is_last:
                next_idx = 1 - buf_idx
                wait_until_consumed(next_idx)
                self._recv_pair(next_idx, nccl_s)

            for tm in metadata["tensors"]:
                shape = torch.Size(tm["shape"])
                dtype = getattr(torch, tm["dtype"].replace("torch.", ""))
                chunk_nbytes = int(tm["nbytes"])
                chunk = self._data_bufs[buf_idx][
                    tm["offset"]:tm["offset"] + chunk_nbytes
                ]
                tensor_offset = int(tm.get("tensor_offset", 0))
                tensor_nbytes = int(tm.get("tensor_nbytes", chunk_nbytes))
                if tensor_offset == 0 and tensor_nbytes == chunk_nbytes:
                    yield tm["name"], chunk.view(dtype).reshape(shape)
                    continue

                name = tm["name"]
                state = partial_tensors.get(name)
                if state is None:
                    if tensor_offset != 0:
                        raise RuntimeError(
                            f"First chunk for {name!r} starts at byte "
                            f"{tensor_offset}, expected 0"
                        )
                    tensor = torch.empty(shape, dtype=dtype, device=self.device)
                    if tensor.nbytes != tensor_nbytes:
                        raise RuntimeError(
                            f"Chunk metadata for {name!r} declares "
                            f"{tensor_nbytes} bytes, but shape={list(shape)} "
                            f"dtype={dtype} requires {tensor.nbytes}"
                        )
                    state = (tensor, 0, tensor_nbytes)
                    partial_tensors[name] = state

                tensor, next_offset, expected_nbytes = state
                if tensor_offset != next_offset or tensor_nbytes != expected_nbytes:
                    raise RuntimeError(
                        f"Non-contiguous chunks for {name!r}: got byte "
                        f"{tensor_offset} of {tensor_nbytes}, expected byte "
                        f"{next_offset} of {expected_nbytes}"
                    )
                end = tensor_offset + chunk_nbytes
                if end > tensor_nbytes:
                    raise RuntimeError(
                        f"Chunk for {name!r} ends at byte {end}, beyond "
                        f"tensor size {tensor_nbytes}"
                    )
                tensor.view(-1).view(torch.uint8)[tensor_offset:end].copy_(
                    chunk, non_blocking=True
                )
                if end == tensor_nbytes:
                    del partial_tensors[name]
                    yield name, tensor
                else:
                    partial_tensors[name] = (tensor, end, tensor_nbytes)

            mark_consumed(buf_idx)
            if is_last:
                break
            nccl_s.synchronize()
            buf_idx = 1 - buf_idx

        if partial_tensors:
            names = ", ".join(sorted(partial_tensors))
            raise RuntimeError(f"Incomplete chunked tensors at end of transfer: {names}")

    # ------------------------------------------------------------------
    # Direct mode — per-weight send/recv into parameter views
    # ------------------------------------------------------------------

    @torch.no_grad()
    def send_weights_direct(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict:
        """Send each weight individually — fully streaming, no materialization.

        Protocol: for each weight, send a meta header (one entry,
        ``is_last=False``) then the raw bytes.  After the last weight,
        send a sentinel meta with ``is_last=True`` and no tensor entries.

        This consumes the *weights* iterable lazily, so per-param
        gather generators (e.g. FSDP ``full_tensor()`` or DeepSpeed
        ZeRO-3 ``GatheredParameters``) can yield one gathered tensor
        at a time -- the caller sends it, issues ``nccl_stream.synchronize()``
        before requesting the next item, and the generator resharding
        happens between yields.  Peak extra GPU memory is one full
        parameter.
        """
        assert self.is_sender, "Only rank 0 can send"

        start = time.time()
        nccl_s = self._nccl_stream
        n_params = 0

        for name, weight in weights:
            w = weight.contiguous()
            self._send_one_direct(name, w, nccl_s)
            n_params += 1

        self._write_meta(0, [], is_last=True)
        self._send_buf(self._meta_bufs[0], nccl_s)
        nccl_s.synchronize()

        elapsed = time.time() - start
        logger.info("send_weights_direct: %d params, %.2fs, port=%d",
                     n_params, elapsed, self._port)
        return {"status": "done", "params_sent": n_params, "elapsed": elapsed}

    @torch.no_grad()
    def receive_weights_direct(
        self,
        param_views: dict[str, torch.Tensor],
    ) -> dict:
        """Receive weights directly into pre-computed parameter views.

        Streaming protocol: the sender transmits a per-weight meta header
        followed by the raw bytes, repeated until a sentinel meta with
        ``is_last=True`` arrives.

        *param_views* maps safetensor key → contiguous parameter tensor.
        When the view's byte size matches, NCCL writes directly into the
        model parameter storage (true zero-copy).
        """
        assert not self.is_sender
        self.ensure_staging_buffers()

        start = time.time()
        nccl_s = self._nccl_stream

        loaded = 0
        orphan = 0

        while True:
            self._recv_buf(self._meta_bufs[0], nccl_s)
            nccl_s.synchronize()
            metadata = self._parse_meta(0)

            if metadata["is_last"]:
                break

            for entry in metadata["tensors"]:
                name = entry["name"]
                nbytes = entry["nbytes"]
                sender_shape = torch.Size(entry["shape"])
                sender_dtype = getattr(
                    torch, entry["dtype"].replace("torch.", "")
                )
                view = param_views.get(name)
                if view is not None and view.nbytes == nbytes:
                    recv_buf = view.view(-1).view(torch.uint8)
                    self._recv_buf(recv_buf, nccl_s)
                else:
                    # Fall back to a scratch buffer.  The permanent
                    # ``_data_bufs[0]`` is sized for bucket mode; for
                    # oversized tensors (rare: large embeddings with
                    # padded receiver views) allocate on demand instead
                    # of silently slicing short.
                    if nbytes <= self._data_bufs[0].numel():
                        scratch = self._data_bufs[0][:nbytes]
                    else:
                        logger.warning(
                            "receive_weights_direct: %s nbytes=%d exceeds "
                            "bucket_size=%d; allocating temp scratch",
                            name, nbytes, self._data_bufs[0].numel(),
                        )
                        scratch = torch.empty(
                            nbytes, dtype=torch.uint8, device=self.device,
                        )
                    self._recv_buf(scratch, nccl_s)
                    if view is not None:
                        nccl_s.synchronize()
                        sender_tensor = scratch.view(
                            sender_dtype
                        ).reshape(sender_shape)
                        if view.shape == sender_shape:
                            view.copy_(sender_tensor)
                        elif (
                            view.ndim == sender_tensor.ndim
                            and all(
                                vs >= ss for vs, ss in zip(
                                    view.shape, sender_tensor.shape
                                )
                            )
                        ):
                            # Receiver view is padded (e.g. vLLM's
                            # padded vocab).  Copy sender into the
                            # leading slice and zero any trailing pad.
                            slicer = tuple(
                                slice(0, s) for s in sender_tensor.shape
                            )
                            view[slicer].copy_(sender_tensor)
                        else:
                            raise RuntimeError(
                                f"{name}: view.shape={tuple(view.shape)} "
                                f"incompatible with sender.shape="
                                f"{tuple(sender_tensor.shape)}"
                            )
                    else:
                        orphan += 1
                loaded += 1

            nccl_s.synchronize()

        elapsed = time.time() - start
        logger.info("receive_weights_direct: %d loaded, %d orphan, %.2fs",
                     loaded, orphan, elapsed)
        return {"status": "done", "params_loaded": loaded,
                "orphan": orphan, "elapsed": elapsed}

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _send_one_direct(
        self, name: str, w: torch.Tensor, nccl_s: torch.cuda.Stream,
    ) -> None:
        """Send a single (already-full) tensor via direct-mode protocol.

        Issues the meta + data NCCL sends and synchronizes ``nccl_s``
        before returning, so callers can safely let ``w``'s storage be
        freed (e.g. ZeRO-3 reshard on ``GatheredParameters`` exit).
        """
        w_bytes = w.view(-1).view(torch.uint8)
        if w_bytes.device != self.device:
            w_bytes = w_bytes.to(self.device, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()

        # Shape-vs-nbytes sanity check: under DeepSpeed ZeRO-3 it's
        # possible (seen in practice) for ``p.data`` inside
        # ``GatheredParameters`` to be a partially-gathered / padded
        # buffer whose byte length disagrees with the logical shape.
        # Catch that early with a clear error instead of letting the
        # receiver blow up on a mysterious reshape failure.
        expected_nbytes = w.numel() * w.element_size()
        if w_bytes.numel() != expected_nbytes:
            raise RuntimeError(
                f"{name}: w.shape={tuple(w.shape)} implies "
                f"nbytes={expected_nbytes} but tensor has "
                f"{w_bytes.numel()} bytes (tensor not fully gathered?)"
            )

        entry = {
            "name": name,
            "shape": list(w.shape),
            "dtype": str(w.dtype),
            "nbytes": w_bytes.numel(),
        }
        self._write_meta(0, [entry], is_last=False)
        self._send_buf(self._meta_bufs[0], nccl_s)
        self._send_buf(w_bytes, nccl_s)
        nccl_s.synchronize()

    def _send_pair(self, idx: int, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.send(self._meta_bufs[idx], dst=1, stream=stream)
        self.nccl.send(self._data_bufs[idx], dst=1, stream=stream)

    def _recv_pair(self, idx: int, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.recv(self._meta_bufs[idx], src=0, stream=stream)
        self.nccl.recv(self._data_bufs[idx], src=0, stream=stream)

    def _send_buf(self, buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.send(buf, dst=1, stream=stream)

    def _recv_buf(self, buf: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self._op_count += 1
        self.nccl.recv(buf, src=0, stream=stream)

    def _write_meta(self, idx: int, entries: list[dict], *,
                    is_last: bool) -> None:
        meta_json = json.dumps(
            {"tensors": entries, "is_last": is_last},
        ).encode("utf-8")
        if len(meta_json) + 8 > META_BUF_SIZE:
            raise RuntimeError(
                f"Metadata ({len(meta_json)} B) exceeds "
                f"META_BUF_SIZE ({META_BUF_SIZE} B)"
            )
        buf = self._meta_bufs[idx]
        buf.zero_()
        len_t = torch.tensor([len(meta_json)], dtype=torch.int64)
        buf[:8].copy_(len_t.view(torch.uint8))
        json_t = torch.frombuffer(bytearray(meta_json), dtype=torch.uint8)
        buf[8:8 + len(meta_json)].copy_(json_t)

    def _parse_meta(self, idx: int) -> dict:
        buf = self._meta_bufs[idx]
        meta_len = buf[:8].cpu().view(torch.int64).item()
        meta_bytes = buf[8:8 + meta_len].cpu().numpy().tobytes()
        return json.loads(meta_bytes)

    def destroy(self) -> None:
        """Tear down NCCL and all GPU staging buffers."""
        if getattr(self, "nccl", None) is not None:
            del self.nccl
            self.nccl = None
        for attr in ("_data_bufs", "_meta_bufs", "_send_events"):
            bufs = getattr(self, attr, None)
            if bufs is not None:
                setattr(self, attr, None)
                del bufs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
