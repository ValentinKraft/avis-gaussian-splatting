"""
Differentiable rasterization of Gaussian splats to a 3D voxel grid.
Follows PyTorch conventions and 3DGS style.
"""
import torch
from typing import Tuple

def splat_to_volume(splats: torch.Tensor, volume_shape: Tuple[int, int, int], sigma: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    """
    Accumulate splats into a 3D voxel grid using Gaussian kernel.
    Args:
        splats: (N, 3) tensor of splat positions (normalized to [0, D], [0, H], [0, W])
        volume_shape: (D, H, W)
        sigma: Gaussian stddev
        alpha: scaling factor
    Returns:
        volume: (D, H, W) tensor
    """
    device = splats.device
    volume = torch.zeros(volume_shape, dtype=torch.float32, device=device)
    D, H, W = volume_shape
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij')
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()
    for pos in splats:
        mu = pos.float()
        dist2 = ((grid - mu) ** 2).sum(dim=-1)
        gaussian = alpha * torch.exp(-dist2 / (2 * sigma ** 2))
        volume += gaussian
    return volume
