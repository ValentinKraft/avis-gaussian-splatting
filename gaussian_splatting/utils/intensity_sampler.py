# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

"""
Utility functions for sampling and computing intensity values for Gaussian splats.
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

def sample_intensities_from_volume(
    points: Tensor,
    volume: Tensor,
    scale: Optional[Tensor] = None,
    radius_scale: float = 2.0
) -> Tensor:
    """
    Sample intensity values from a volume for each point position.
    Computes a mean intensity within the Gaussian's influence region.
    
    Args:
        points: Point coordinates in normalized [0,1] space, shape [N, 3]
        volume: Input volume tensor with intensity values, shape [D, H, W]
        scale: Optional scale parameters for each point, shape [N, 3] or [N]
        radius_scale: How many standard deviations to consider for intensity computation
        
    Returns:
        Intensity values for each point, shape [N, 1]
    """
    device = points.device
    D, H, W = volume.shape
    N = points.shape[0]
    
    # Convert normalized coordinates [0,1] to volume indices
    point_indices = points.clone()
    point_indices[:, 0] *= (W - 1)  # x
    point_indices[:, 1] *= (H - 1)  # y
    point_indices[:, 2] *= (D - 1)  # z
    
    # Determine sampling radius for each point
    if scale is None:
        # Default radius as fraction of volume size
        radius = torch.tensor([W, H, D], device=device).float() * 0.02
        radius = radius.unsqueeze(0).expand(N, 3)
    else:
        # Use provided scales (already in correct units)
        if scale.shape[-1] == 3:
            radius = scale * radius_scale
        else:
            radius = scale.unsqueeze(-1).expand(N, 3) * radius_scale
    
    # Create sampling grid for each point
    intensities = torch.zeros(N, 1, device=device)
    
    # Process points in batches to avoid memory issues
    batch_size = min(100, N)
    num_batches = (N + batch_size - 1) // batch_size
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, N)
        batch_points = point_indices[start_idx:end_idx]
        batch_radius = radius[start_idx:end_idx]
        
        # For each point, extract a cube around it and compute mean intensity
        batch_intensities = []
        
        for i in range(len(batch_points)):
            # Get point center and radius
            x, y, z = batch_points[i]
            rx, ry, rz = batch_radius[i]
            
            # Compute sampling bounds
            min_x = max(0, int(x - rx))
            max_x = min(W-1, int(x + rx))
            min_y = max(0, int(y - ry))
            max_y = min(H-1, int(y + ry))
            min_z = max(0, int(z - rz))
            max_z = min(D-1, int(z + rz))
            
            # Skip if out of bounds
            if min_x > max_x or min_y > max_y or min_z > max_z:
                batch_intensities.append(0.5)  # Default to mid-gray
                continue
            
            # Extract subvolume
            subvol = volume[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
            
            # Compute weighted mean based on distance
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.arange(min_z, max_z+1, device=device),
                torch.arange(min_y, max_y+1, device=device),
                torch.arange(min_x, max_x+1, device=device),
                indexing='ij'
            )
            grid_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()
            
            # Compute squared distance
            dist_sq = torch.sum((grid_pts - batch_points[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)
            
            # Create Gaussian weight
            weight = torch.exp(-0.5 * dist_sq / (batch_radius[i].mean() ** 2))
            
            # Compute weighted mean intensity
            if weight.sum() > 0:
                weighted_mean = (subvol * weight).sum() / weight.sum()
                batch_intensities.append(weighted_mean.item())
            else:
                batch_intensities.append(subvol.mean().item())
        
        # Store batch results
        intensities[start_idx:end_idx] = torch.tensor(batch_intensities, device=device).unsqueeze(-1)
    
    return intensities

def update_intensities(
    points: Tensor, 
    volume: Tensor, 
    scale: Optional[Tensor] = None
) -> Tensor:
    """
    Update intensity values for points based on their current positions.
    Should be called whenever point positions or scales change significantly.
    
    Args:
        points: Point coordinates, shape [3, N] or [N, 3]
        volume: Reference volume with intensity values
        scale: Scale parameters for points
        
    Returns:
        Updated intensity values, shape [N, 1]
    """
    # Handle different point formats
    if points.shape[0] == 3 and points.shape[0] != points.shape[1]:
        # Convert from [3, N] to [N, 3]
        points_n3 = points.permute(1, 0)
    else:
        points_n3 = points
        
    # Get volume dimensions
    D, H, W = volume.shape
    device = points.device
    
    # Normalize points to [0,1] if they're in volume index space
    if points_n3.max() > 1.0:
        points_n3 = points_n3.clone()
        points_n3[:, 0] /= (W - 1)
        points_n3[:, 1] /= (H - 1)
        points_n3[:, 2] /= (D - 1)
    
    return sample_intensities_from_volume(points_n3, volume, scale)
