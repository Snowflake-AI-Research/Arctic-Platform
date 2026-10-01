from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from arctic_platform.inference.vllm.dflash2_nan_fix import (
    KVBlockZeroerPatch,
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
