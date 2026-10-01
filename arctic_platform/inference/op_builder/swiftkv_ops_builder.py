import os
from .builder import CUDAOpBuilder

class SwiftKVOpsBuilder(CUDAOpBuilder):
    def __init__(self):
        super().__init__(name="reshape_and_cache_flash_bulk")

    def absolute_name(self):
        return f'arctic_platform.inference.swiftkv_ops.{self.name}'

    def get_prefix(self):
        # borrowed from moe_op. refactor later
        ai_path = self._src_path("")
        return ai_path if os.path.isdir(os.path.join(ai_path, "csrc")) else ".."

    def sources(self):
        sources = [
            'csrc/custom_ops/torch_bindings.cpp',
            'csrc/custom_ops/reshape_and_cache_flash_bulk.cu',
        ]
        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def include_paths(self):
        sources = ['csrc/custom_ops']
        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources
