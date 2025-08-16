# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

"""
Initialize Gaussian points from volume data for 3D Gaussian Splatting.
"""

import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Optional
import torch.nn.functional as F
from pathlib import Path

def initialize_from_volume(
    mask_path: str,
    n_points: int = 5000,
    noise_std: float = 0.01,
    device: torch.device = torch.device('cuda')
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Initialize Gaussian points by sampling from a segmentation mask.
    The mask is expected to be pre-aligned with the volume and have the same dimensions.
    
    Args:
        mask_path: Path to segmentation mask (.nii, .npy, .mhd)
                  Values should be in [0,1], either binary or continuous
        n_points: Number of points to sample
        noise_std: Standard deviation for position noise
        device: Device to create tensors on
        
    Returns:
        Tuple of (positions, scales, opacities)
    """
    from gaussian_splatting.data.volume_loader import VolumeLoader
        
    # Load mask - keep original dimensions since it's pre-aligned
    loader = VolumeLoader(device=device)
    mask = loader.load_volume(mask_path)
    
    # Sample points based on mask values
    # Create coordinate grid
    D, H, W = mask.shape
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords = torch.stack([x, y, z], dim=-1).float()
    
    # Flatten everything
    coords_flat = coords.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    
    # Sample points based on mask values
    if mask_flat.unique().numel() == 2:  # Binary mask
        valid_idx = torch.nonzero(mask_flat > 0).squeeze(1)
        if len(valid_idx) == 0:
            raise ValueError("No valid points found in mask")
            
        # Random sampling from valid points
        if len(valid_idx) > n_points:
            selected_idx = valid_idx[torch.randperm(len(valid_idx))[:n_points]]
        else:
            selected_idx = valid_idx
            
        points = coords_flat[selected_idx]
        opacities = torch.ones(len(points), 1, device=device)
        
    else:  # Continuous mask - sample proportional to values
        # Add small epsilon to ensure some probability everywhere in mask
        probs = mask_flat + 1e-6
        probs = probs / probs.sum()  # Normalize to probability distribution
        
        # Sample point indices according to mask values
        selected_idx = torch.multinomial(probs, n_points, replacement=True)
        points = coords_flat[selected_idx]
        
        # Use mask values as initial opacities
        opacities = mask_flat[selected_idx].unsqueeze(1)
    
    # Normalize coordinates to [0, 1]
    points = points / torch.tensor([W-1, H-1, D-1], device=device)
    
    # Add noise to positions
    points = points + torch.randn_like(points) * noise_std
    points = torch.clamp(points, 0, 1)
    
    # Initialize scales (use smaller scales for denser point clouds)
    base_scale = 0.01 * (5000 / n_points) ** (1/3)  # Scale based on point density
    scales = torch.ones(len(points), 3, device=device) * base_scale
    
    return points, scales, opacities
    
    # Initialize scales and opacities
    scales = torch.ones(len(points), 3, device=device) * 0.01
    opacities = torch.ones(len(points), 1, device=device)
    
    return points, scales, opacities

def transform_points_to_world(
    points: Tensor,
    volume_transform: Optional[Tensor] = None,
    scene_bounds: Optional[Tuple[Tensor, Tensor]] = None
) -> Tensor:
    """
    Transform points from volume space to world space.
    
    Args:
        points: Points in normalized volume space [0,1]^3
        volume_transform: Optional 4x4 transform matrix
        scene_bounds: Optional (min, max) scene bounds to scale into
        
    Returns:
        Points in world space
    """
    if scene_bounds is not None:
        min_bound, max_bound = scene_bounds
        scale = max_bound - min_bound
        points = points * scale + min_bound
        
    if volume_transform is not None:
        # Add homogeneous coordinate
        points_h = torch.cat([
            points,
            torch.ones(len(points), 1, device=points.device)
        ], dim=1)
        
        # Transform
        points = (volume_transform @ points_h.T).T[:, :3]
        
    return points

def initialize_gaussians(
    model,
    volume_path: str,
    n_points: int = 5000,
    volume_transform: Optional[Tensor] = None,
    scene_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    **kwargs
):
    """
    Initialize a Gaussian model from a segmentation volume.
    
    Args:
        model: Gaussian model to initialize
        volume_path: Path to volume file
        n_points: Number of points to sample
        volume_transform: Optional 4x4 transform matrix
        scene_bounds: Optional (min, max) scene bounds
        **kwargs: Additional args for initialize_from_volume
    """
    # Get points in volume space
    points, scales, opacities = initialize_from_volume(
        volume_path, 
        n_points,
        device=model._xyz.device,
        **kwargs
    )
    
    # Transform to world space
    points = transform_points_to_world(points, volume_transform, scene_bounds)
    
    # Update model parameters
    with torch.no_grad():
        model._xyz.copy_(points)
        model._scaling.copy_(scales)
        model._opacity.copy_(opacities)
        
    return model
