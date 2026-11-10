import pytest

from arctic_platform.inference.utils import VLLM_VERSION_V2, require_supported_vllm_version


def test_require_supported_vllm_version_accepts_validated_versions():
    assert require_supported_vllm_version(version=VLLM_VERSION_V2) == VLLM_VERSION_V2


@pytest.mark.parametrize("version", ["0.18.0", "0.26.0", "0.30.0", "0.31.0", "0.32.0", "0.33.0", "0.33.1", "0.33.0rc1.dev1046+gf905e7007"])
def test_require_supported_vllm_version_rejects_unvalidated_versions(version):
    with pytest.raises(RuntimeError, match="supports vLLM .* only"):
        require_supported_vllm_version(version=version)
