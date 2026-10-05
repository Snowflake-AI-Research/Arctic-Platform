import pytest

from arctic_platform.inference.utils import require_supported_vllm_version


def test_require_supported_vllm_version_accepts_validated_versions():
    assert require_supported_vllm_version(version="0.30.0") == "0.30.0"


@pytest.mark.parametrize("version", ["0.18.0", "0.26.0", "0.29.0", "0.30.1"])
def test_require_supported_vllm_version_rejects_unvalidated_versions(version):
    with pytest.raises(RuntimeError, match="supports vLLM v0.30.0 only"):
        require_supported_vllm_version(version=version)
