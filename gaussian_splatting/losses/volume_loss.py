"""
VolumeLoss: Supports MSE, Dice, Tversky, KL-Divergence for volumetric supervision.
Follows PyTorch conventions and 3DGS style.
"""
import torch
import torch.nn as nn
from typing import Optional

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class VolumeLoss(nn.Module):
    def __init__(self, loss_type: str = 'mse', weight: float = 1.0, tversky_alpha: float = 0.5, tversky_beta: float = 0.5):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.mse = nn.MSELoss()
        self.dice = DiceLoss()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            return self.weight * self.mse(pred, target)
        elif self.loss_type == 'dice':
            return self.weight * self.dice(pred, target)
        elif self.loss_type == 'tversky':
            tp = (pred * target).sum()
            fp = ((1 - target) * pred).sum()
            fn = (target * (1 - pred)).sum()
            tversky = (tp + 1) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1)
            return self.weight * (1 - tversky)
        elif self.loss_type == 'kl':
            pred = torch.clamp(pred, 1e-6, 1-1e-6)
            target = torch.clamp(target, 1e-6, 1-1e-6)
            kl = (target * torch.log(target / pred)).sum() / pred.numel()
            return self.weight * kl
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
