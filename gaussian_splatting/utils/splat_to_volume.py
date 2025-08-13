# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

"""
Differentiable conversion of 3D Gaussian splats to volumetric representation.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
import torch.nn.functional as F

def create_grid_points(
    volume_shape: Tuple[int, int, int],
    device: torch.device = torch.device("cuda")
) -> Tensor:
    """
    Create a grid of points for the volume.
    
    Args:
        volume_shape: (depth, height, width) of the volume
        device: Torch device to create tensors on
    
    Returns:
        Grid points tensor of shape (depth, height, width, 3)
    """
    D, H, W = volume_shape
    
    # Create normalized coordinate grid
    z = torch.linspace(0, 1, D, device=device)
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    
    # Create meshgrid
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    
    # Stack coordinates
    return torch.stack([grid_x, grid_y, grid_z], dim=-1)

def gaussian_kernel_3d(
    points: Tensor,
    mean: Tensor,
    cov: Optional[Tensor] = None,
    scale: float = 1.0
) -> Tensor:
    """
    Compute 3D Gaussian kernel values for given points.
    
    Args:
        points: Grid points (D, H, W, 3)
        mean: Gaussian center (3,)
        cov: Optional covariance matrix (3, 3)
        scale: Scale factor for isotropic Gaussian
        
    Returns:
        Kernel values at each point (D, H, W)
    """
    if cov is None:
        # Use isotropic Gaussian
        diff = points - mean.view(1, 1, 1, 3)
        return torch.exp(-0.5 * torch.sum(diff * diff, dim=-1) / (scale ** 2))
    else:
        # Use full covariance matrix
        diff = points - mean.view(1, 1, 1, 3)
        cov_inv = torch.inverse(cov)
        mahalanobis = torch.sum(diff @ cov_inv * diff, dim=-1)
        return torch.exp(-0.5 * mahalanobis)

def splat_to_volume(
    splats: Tensor,
    volume_shape: Tuple[int, int, int],
    covariances: Optional[Tensor] = None,
    scale: float = 0.1
) -> Tensor:
    """
    Convert 3D Gaussian splats to a volumetric representation.
    
    Args:
        splats: Tensor of splat centers (N, 3)
        volume_shape: Output volume shape (depth, height, width)
        covariances: Optional covariance matrices (N, 3, 3)
        scale: Scale factor for isotropic Gaussians when covariances not provided
        
    Returns:
        Volume tensor (D, H, W)
    """
    device = splats.device
    
    # Create volume grid
    points = create_grid_points(volume_shape, device)
    
    # Initialize output volume
    volume = torch.zeros(volume_shape, device=device)
    
    # Accumulate contributions from each splat
    for i, center in enumerate(splats):
        cov = covariances[i] if covariances is not None else None
        kernel = gaussian_kernel_3d(points, center, cov, scale)
        volume = volume + kernel
    
    # Normalize volume to [0, 1]
    volume = volume / (volume.max() + 1e-6)
    
    return volume

def differentiable_max_pooling(volume: Tensor, kernel_size: int = 3) -> Tensor:
    """
    Differentiable approximate maximum pooling using softmax.
    Useful for reducing noise in the volume.
    
    Args:
        volume: Input volume (D, H, W)
        kernel_size: Size of pooling kernel
        
    Returns:
        Pooled volume (D, H, W)
    """
    padding = kernel_size // 2
    
    # Add batch and channel dimensions
    x = volume.unsqueeze(0).unsqueeze(0)
    
    # Extract patches
    patches = F.unfold(
        F.pad(x, (padding, padding, padding, padding, padding, padding)),
        kernel_size=kernel_size
    )
    
    # Soft maximum using softmax
    softmax = F.softmax(patches * 10.0, dim=1)  # Scale factor for sharper maximum
    pooled = (patches * softmax).sum(dim=1)
    
    # Reshape back to volume
    return pooled.view(volume_shape)
