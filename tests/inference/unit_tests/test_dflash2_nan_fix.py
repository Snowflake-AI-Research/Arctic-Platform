from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.model_states import mamba_hybrid

from arctic_platform.inference.vllm.dflash2_nan_fix import (
    GDNAttentionMetadataBuilderPatch,
    KVBlockZeroerPatch,
    MambaHybridModelStatePatch,
    MambaManagerPatch,
    SingleTypeKVCacheManagerPatch,
    _sanitize_stale_gdn_metadata,
)


def test_stale_gdn_rows_are_nulled_and_clamped():
    metadata = SimpleNamespace(
        num_accepted_tokens=torch.tensor([0, 2], dtype=torch.int32),
        spec_state_indices_tensor=torch.tensor(
            [[11, 12, 13], [21, 22, 23]], dtype=torch.int32),
    )

    result = _sanitize_stale_gdn_metadata(metadata)

    assert result is metadata
    torch.testing.assert_close(metadata.num_accepted_tokens,
                               torch.tensor([1, 2], dtype=torch.int32))
    torch.testing.assert_close(
        metadata.spec_state_indices_tensor,
        torch.tensor(
            [[NULL_BLOCK_ID] * 3, [21, 22, 23]], dtype=torch.int32),
    )


def test_mamba_managers_record_blocks_when_zeroing_is_enabled():
    manager = SimpleNamespace()

    def original_init(*args):
        manager._record_new_block_ids = False

    manager._orig_init = original_init
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )

    SingleTypeKVCacheManagerPatch.__init__(
        manager,
        spec,
        object(),
        False,
        0,
        4,
        needs_kv_cache_zeroing=True,
    )

    assert manager._record_new_block_ids


def test_align_mode_without_zeroing_delegates_without_recording():
    manager = SimpleNamespace(mamba_cache_mode="align")

    def original_init(*args):
        manager._record_new_block_ids = False

    manager._orig_init = original_init
    manager._orig_allocate_new_blocks = (
        lambda request_id, num_tokens, num_tokens_main_model: ["delegated"])
    spec = MambaSpec(
        block_size=4,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )

    SingleTypeKVCacheManagerPatch.__init__(
        manager,
        spec,
        object(),
        False,
        0,
        4,
        needs_kv_cache_zeroing=False,
    )
    result = MambaManagerPatch.allocate_new_blocks(manager, "request", 8, 8)

    assert not manager._record_new_block_ids
    assert result == ["delegated"]


def test_align_mode_records_only_physical_allocations():

    class BlockPool:

        def __init__(self):
            self.next_id = 7

        def get_new_blocks(self, count):
            blocks = [
                SimpleNamespace(block_id=block_id)
                for block_id in range(self.next_id, self.next_id + count)
            ]
            self.next_id += count
            return blocks

    pool = BlockPool()
    manager = SimpleNamespace(
        mamba_cache_mode="align",
        _record_new_block_ids=True,
        block_pool=pool,
        new_block_ids=[],
    )
    relocated_block = SimpleNamespace(block_id=3)
    manager._orig_allocate_new_blocks = (
        lambda request_id, num_tokens, num_tokens_main_model:
        [relocated_block, *pool.get_new_blocks(2)])

    result = MambaManagerPatch.allocate_new_blocks(manager, "request", 8, 8)

    assert [block.block_id for block in result] == [3, 7, 8]
    assert manager.new_block_ids == [7, 8]
    assert "get_new_blocks" not in vars(pool)


def test_mamba_state_tensors_are_included_in_zeroing_metadata():
    device = torch.device("cpu")
    conv_state = torch.ones((4, 3, 2), dtype=torch.float32)
    recurrent_state = torch.ones((4, 2, 2), dtype=torch.float32)
    spec = MambaSpec(
        block_size=2,
        shapes=((3, 2), (2, 2)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="align",
    )
    layer_name = "model.layers.0.linear_attn"
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=[layer_name],
    )
    zeroer = SimpleNamespace()

    KVBlockZeroerPatch.__init__(
        zeroer,
        device,
        [group],
        [2],
        {
            layer_name:
            SimpleNamespace(kv_cache=[conv_state, recurrent_state])
        },
        num_blocks=4,
    )

    assert zeroer._meta is not None
    addresses = set(zeroer._meta[0].tolist())
    assert addresses == {conv_state.data_ptr(), recurrent_state.data_ptr()}


@pytest.mark.parametrize(
    "num_scheduled_tokens, num_draft_tokens, is_prefilling, padded_reqs, "
    "expected",
    [
        # Drafted decode row, zero-draft decode row, one-token prefill tail.
        ([3, 1, 1], [2, 0, 0], [False, False, True], 3, [2, 0, -1]),
        # Nobody drafted this step; FULL-graph padding rows stay -1.
        ([1, 1], None, [False, True], 4, [0, -1, -1, -1]),
    ],
)
def test_zero_draft_decode_rows_are_spec_rows(monkeypatch,
                                              num_scheduled_tokens,
                                              num_draft_tokens, is_prefilling,
                                              padded_reqs, expected):
    num_reqs = len(num_scheduled_tokens)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(num_scheduled_tokens)
    num_tokens = int(query_start_loc[-1])
    seq_lens = torch.full((num_reqs, ), 10, dtype=torch.int32)
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=padded_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc),
        max_query_len=None,
        num_scheduled_tokens=np.array(num_scheduled_tokens, dtype=np.int32),
        seq_lens_cpu_upper_bound=seq_lens,
        seq_lens=seq_lens,
        is_prefilling_np=np.array(is_prefilling),
        prefill_runs_as_decode_np=None,
        num_draft_tokens_per_req=(None if num_draft_tokens is None else
                                  np.array(num_draft_tokens, dtype=np.int32)),
        idx_mapping=torch.arange(num_reqs),
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        prompt_lens=None,
    )
    state = object.__new__(mamba_hybrid.MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=3)
    state.num_accepted_tokens_gpu = torch.ones(num_reqs, dtype=torch.int32)
    state._align_mode = False
    state.recoverssm = None
    state._orig_prepare_attn = MethodType(
        MambaHybridModelStatePatch._orig_prepare_attn, state)
    original_metadata_cls = mamba_hybrid.MambaHybridAttnMetadata
    captured = {}
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata",
                        lambda **kwargs: captured.update(kwargs))
    mode = (CUDAGraphMode.FULL
            if padded_reqs > num_reqs else CUDAGraphMode.NONE)

    MambaHybridModelStatePatch.prepare_attn(state, input_batch, mode, (), {},
                                            [], None)

    metadata = captured["model_specific_attn_metadata"]
    assert metadata.num_decode_draft_tokens_cpu.tolist() == expected
    assert mamba_hybrid.MambaHybridAttnMetadata is original_metadata_cls


def test_all_zero_draft_batch_keeps_the_gdn_spec_path(monkeypatch):
    # No pinned host memory without a GPU driver.
    monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    builder = object.__new__(GDNAttentionMetadataBuilder)
    builder.use_spec_decode = True
    builder.num_spec = 3
    builder.use_full_cuda_graph = False
    builder.mamba_aligned_state_indices = torch.arange(
        8, dtype=torch.int32).view(2, 4)
    builder._orig_build = MethodType(
        GDNAttentionMetadataBuilderPatch._orig_build, builder)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([64, 64], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=64,
        block_table_tensor=torch.zeros((2, 4), dtype=torch.int32),
        slot_mapping=torch.zeros(2, dtype=torch.int64),
        is_prefilling=torch.zeros(2, dtype=torch.bool),
        seq_lens_cpu_upper_bound=seq_lens,
    )

    metadata = GDNAttentionMetadataBuilderPatch.build(
        builder, 0, common, torch.tensor([3, 1], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32))

    # The plain decode path reads column 0 and drops the accepted offset.
    assert metadata.num_spec_decodes == 2
    assert metadata.num_decodes == 0
    assert metadata.num_accepted_tokens.tolist() == [3, 1]
    assert metadata.spec_state_indices_tensor.tolist() == [[0, 1, 2, 3],
                                                           [4, 5, 6, 7]]
