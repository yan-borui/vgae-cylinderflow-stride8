"""Architecture/weight loading used by the retained upstream VGAE."""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, arch=None, weights=None, checkpoint=None, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        if checkpoint is not None:
            if arch is not None or weights is not None:
                raise ValueError("choose architecture/weights or a checkpoint")
            saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
            arch, weights = saved["arch"], saved["weights"]
        if arch is None:
            raise ValueError("an architecture is required")
        self.load_arch(arch)
        self.to(self.device)
        if weights is not None:
            if isinstance(weights, (str, bytes)):
                weights = torch.load(
                    weights, map_location=self.device, weights_only=True
                )
            self.load_state_dict(weights)

    def load_arch(self, arch: dict) -> None:
        raise NotImplementedError
