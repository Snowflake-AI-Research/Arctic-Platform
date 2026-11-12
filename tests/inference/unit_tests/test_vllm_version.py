import pytest

from arctic_platform.inference.utils import require_supported_vllm_version


@pytest.mark.parametrize("version", ["0.33.0", "0.33.0+cu130", "0.33.0+build.8199363"])
def test_require_supported_vllm_version_accepts_validated_versions(version):
    assert require_supported_vllm_version(version=version) == version


@pytest.mark.parametrize(
    "version",
    ["0.18.0", "0.26.0", "0.30.0", "0.31.0", "0.32.0", "0.33.1", "0.32.0+cu130", "0.33.1+cu130"],
)
def test_require_supported_vllm_version_rejects_unvalidated_versions(version):
    with pytest.raises(RuntimeError, match="supports vLLM v0.33.0 only"):
        require_supported_vllm_version(version=version)
