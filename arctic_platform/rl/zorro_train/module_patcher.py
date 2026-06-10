"""
Attention layer monkey-patching for prompt reconstruction.

Temporarily reconstructs full sequences during attention computation.
NO PADDING SUPPORT - assumes all sequences have the same length.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from .zorro_train import ZoRRoTrain


class ModuleReconstructionPatcher:
    """Monkey patches module forward with reconstruction logic."""

    def __init__(self, model: nn.Module, reconstruction_info: Dict, patch_with_local=False):
        self.model = model
        self.reconstruction_info = reconstruction_info
        self.original_forwards = {}
        self.patch_with_local = patch_with_local

    def __enter__(self):
        """Patch all attention layers."""
        self._patch_forward()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original attention layers."""
        self._unpatch_forward()

    def _patch_forward(self):
        """Patch all module forward methods."""
        for name, module in self.model.named_modules():
            if self._should_patch_module_forward(name, module):
                # Store original forward
                self.original_forwards[name] = module.forward

                if self.patch_with_local:
                    assert self._create_unpatched_forward_local is not None, "Subclass must implement _create_unpatched_forward_local"
                    module.forward = self._create_unpatched_forward_local(module, name)
                else:
                    # Create patched forward that optimizes QKV
                    module.forward = self._create_patched_forward(module, name)

    def _unpatch_forward(self):
        """Restore original forward methods."""
        for name, module in self.model.named_modules():
            if name in self.original_forwards:
                module.forward = self.original_forwards[name]
        self.original_forwards.clear()

    def _create_unpatched_forward_local(self, module, module_name):
        ''' implement in subclass. This should contain the original forward logic locally to allow for debugging '''
        raise NotImplementedError("Subclass should override this")

    def _create_patched_forward(self, module, module_name):
        """
    This should contain the patched forward logic that reconstructs the full sequences temporarily.
        """
        ''' Subclass should override this '''
        raise NotImplementedError("Subclass should override this")

    @staticmethod
    def _should_patch_module_forward(name, module):
        """Subclass should override this"""
        raise NotImplementedError("Subclass should override this")
