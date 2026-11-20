"""The real vLLM routed-experts worker connector on CPU; only the GPU capture snapshot is stubbed."""

import logging
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.distributed.aux_output_connector.connector import AuxOutputConnectorMetadata
from vllm.distributed.aux_output_connector.connector import PackedBlockHashes
from vllm.distributed.aux_output_connector.routed_experts import RoutedExpertsBuffer
from vllm.distributed.aux_output_connector.store import BackgroundBlockObjectStore
from vllm.distributed.aux_output_connector.store import BlockObjectStore
from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector

from arctic_platform.inference.vllm import routed_experts_recompute_warning

BLOCK = 4
SHAPE = (2, 2)


def warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == routed_experts_recompute_warning.__name__]


def block_hashes(*names):
    return PackedBlockHashes(b"".join(name.encode() * 8 for name in names), 8)


def step(connector, request_id, token_start, routes, hashes=None):
    """One scheduled step that captures ``routes`` from ``token_start`` and samples a token."""
    metadata = AuxOutputConnectorMetadata(0, {request_id: 0}, {request_id: hashes} if hashes else {}, ())
    connector.begin_step(metadata)
    rows = np.full((len(routes), *SHAPE), routes[:, None, None], dtype=np.uint8)
    connector._capturer = SimpleNamespace(snapshot_routing_data=lambda n: torch.from_numpy(rows[:n]))
    batch = SimpleNamespace(
        req_ids=[request_id],
        query_start_loc_np=np.array([0, len(rows)]),
        num_computed_tokens_np=np.array([token_start]),
    )
    pending = connector.prepare_output(batch)
    pending.enqueue_cpu_copy(np.array([1]), np.array([0]))
    output = pending.process_output()[request_id]
    return output.token_start, output.rows[:, 0, 0].tolist()


@pytest.fixture
def connector(caplog):
    routed_experts_recompute_warning.ensure_routed_experts_recompute_warning()
    # The attributes AuxOutputWorkerConnector.__init__ sets on the output rank, without a model to bind.
    connector = object.__new__(AuxOutputWorkerConnector)
    connector._store = BackgroundBlockObjectStore(
        BlockObjectStore(max_bytes=64 * BLOCK * 4, object_nbytes=BLOCK * 4), max_pending_batches=4
    )
    connector._buffer = RoutedExpertsBuffer(np.dtype(np.uint8), SHAPE, BLOCK, 4, 64, 1)
    connector._requests = {}
    connector._generation = 0
    connector._step_metadata = None
    connector._pending_outputs = []
    connector._lock = Lock()
    connector._max_concurrent_batches = 1
    # An earlier request stores routes 1 for blocks a, b and c. It stays live, so its keys stay
    # referenced, as they also do while vLLM keeps a finished request's KV blocks cached.
    step(connector, "earlier", 0, np.full(12, 1), block_hashes("a", "b", "c"))
    caplog.set_level(logging.WARNING, logger=routed_experts_recompute_warning.__name__)
    yield connector
    connector.close()


def test_recomputed_block_returns_stored_routes_and_warns(connector, caplog):
    # The KV hit stops one block short of the stored keys, so block c is recomputed with routes 2.
    start, rows = step(connector, "later", 8, np.full(10, 2), block_hashes("a", "b", "c", "d"))
    assert (start, rows) == (0, [1] * 12 + [2] * 6)
    [message] = warnings(caplog)
    assert "request later computed positions [8, 16) (block size 4) but 4 of them" in message


def test_capture_from_the_stored_end_is_silent(connector, caplog):
    start, rows = step(connector, "later", 12, np.full(3, 3), block_hashes("a", "b", "c"))
    assert (start, rows) == (0, [1] * 12 + [3] * 3)
    assert not warnings(caplog)


def test_routes_this_request_stored_itself_are_silent(connector, caplog):
    # Blocks e and f are new, so the store returns the routes this request computed.
    start, rows = step(connector, "other", 8, np.full(10, 4), block_hashes("a", "b", "e", "f"))
    assert (start, rows) == (0, [1] * 8 + [4] * 10)
    assert not warnings(caplog)
