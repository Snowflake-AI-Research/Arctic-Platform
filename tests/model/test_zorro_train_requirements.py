"""ZoRRo checks shared by a loaded model and a config string."""

import pytest

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches
from arctic_platform.model.config import ZorroTrainPatch
from arctic_platform.model.config import require_zorro_train_parallelism
from arctic_platform.rl.zorro_train.model_types import require_supported_model_type


def test_alias_normalizes_and_unknown_type_is_rejected():
    assert require_supported_model_type("qwen3_5_text") == "qwen3_5"
    with pytest.raises(ValueError, match="llama"):
        require_supported_model_type("llama")


def test_parallelism_requires_degree_one():
    require_zorro_train_parallelism(expert_parallel=1, sequence_parallel=1)
    with pytest.raises(ValueError, match="ep_size=2"):
        require_zorro_train_parallelism(expert_parallel=2, sequence_parallel=1)
    with pytest.raises(ValueError, match="sp_size=2"):
        require_zorro_train_parallelism(expert_parallel=1, sequence_parallel=2)


def test_model_spec_rejects_zorro_above_degree_one():
    with pytest.raises(ValueError, match="sp_size=2"):
        ModelSpec(
            model_path_or_name="local",
            patches=Patches(zorro_train=ZorroTrainPatch()),
            parallelism=ParallelismConfig(sequence_parallel=2),
        )
    with pytest.raises(ValueError, match="ep_size=2"):
        ModelSpec(
            model_path_or_name="local",
            patches=Patches(zorro_train=ZorroTrainPatch()),
            parallelism=ParallelismConfig(expert_parallel=2),
        )
