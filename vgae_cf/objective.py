"""Original normalized UVP reconstruction and element-mean KL objective."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction and unweighted KL in normalized field units."""
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[-1] != 3:
        raise ValueError("UVP loss requires aligned normalized fields [fine nodes, 3]")
    reconstruction = F.mse_loss(prediction, target)
    kl_unweighted = -0.5 * torch.mean(1 + logvar - mean.square() - logvar.exp())
    return reconstruction + kl_weight * kl_unweighted, reconstruction, kl_unweighted
