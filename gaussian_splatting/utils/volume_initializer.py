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
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Tuple, Optional
import torch.nn.functional as F
from pathlib import Path

from scene.gaussian_model import GaussianModel

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

    # Print mask stats for debugging
    print(
        f"Mask stats: min={mask_flat.min().item():.4f}, max={mask_flat.max().item():.4f}, "
        f"mean={mask_flat.mean().item():.4f}, nonzero={(mask_flat > 0).sum().item()}"
    )

    # Sample points based on mask values
    if mask_flat.unique().numel() <= 2:  # Binary mask
        valid_idx = torch.nonzero(mask_flat > 0).squeeze(1)
        if len(valid_idx) == 0:
            raise ValueError("No valid points found in mask")

        print(
            f"Binary mask detected: {len(valid_idx)} valid points of {len(mask_flat)} total"
        )

        # Random sampling from valid points
        if len(valid_idx) > n_points:
            selected_idx = valid_idx[torch.randperm(len(valid_idx))[:n_points]]
        else:
            # If we have fewer valid points than requested, duplicate some
            print(
                f"Warning: Only {len(valid_idx)} valid points, fewer than requested {n_points}"
            )
            repeats_needed = (n_points + len(valid_idx) - 1) // len(valid_idx)
            repeated_idx = valid_idx.repeat(repeats_needed)
            selected_idx = repeated_idx[:n_points]

        points = coords_flat[selected_idx]
        opacities = torch.ones(len(points), 1, device=device)

    else:  # Continuous mask - sample proportional to values
        print(
            f"Continuous mask detected: values range [{mask_flat.min().item():.4f}, {mask_flat.max().item():.4f}]"
        )

        # Add small epsilon and add more weight to positive values to ensure good coverage
        mask_weighted = mask_flat.clone()
        mask_weighted[mask_weighted > 0] = (
            mask_weighted[mask_weighted > 0] + 0.2
        )  # Boost positive values

        probs = mask_weighted + 1e-6
        probs = probs / probs.sum()  # Normalize to probability distribution

        # Sample point indices according to mask values
        selected_idx = torch.multinomial(probs, n_points, replacement=True)
        points = coords_flat[selected_idx]

        # Use mask values as initial opacities, but ensure a minimum value
        raw_opacities = mask_flat[selected_idx].unsqueeze(1)
        opacities = torch.clamp(
            raw_opacities, min=0.3
        )  # Minimum opacity for visibility

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
        points: Points in normalized volume space [0,1]^3 (shape [N, 3])
        volume_transform: Optional 4x4 transform matrix
        scene_bounds: Optional (min, max) scene bounds to scale into

    Returns:
        Points in world space (shape [N, 3])
    """
    device = points.device

    if scene_bounds is not None:
        min_bound, max_bound = scene_bounds
        # Ensure bounds are on same device as points
        min_bound = min_bound.to(device)
        max_bound = max_bound.to(device)
        scale = max_bound - min_bound
        points = points * scale + min_bound

    if volume_transform is not None:
        # Ensure transform is on same device as points
        volume_transform = volume_transform.to(device)
        # Add homogeneous coordinate
        points_h = torch.cat([points, torch.ones(len(points), 1, device=device)], dim=1)

        # Transform
        points = (volume_transform @ points_h.T).T[:, :3]

    return points


def initialize_gaussians(
    model: GaussianModel,
    n_points: int = 5000,
    volume_transform: Optional[Tensor] = None,
    scene_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    volume_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    **kwargs
):
    """
    Initialize a Gaussian model from a volume or mask.

    Args:
        model: Gaussian model to initialize
        n_points: Number of points to sample
        volume_transform: Optional 4x4 transform matrix
        scene_bounds: Optional (min, max) scene bounds
        volume_path: Optional path to volume file
        mask_path: Optional path to mask file
        **kwargs: Additional args for initialize_from_volume
    """
    # Get points in volume space
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points, scales, opacities = initialize_from_volume(
        mask_path if mask_path else volume_path, n_points, device=device, **kwargs
    )

    # Transform to world space
    points = transform_points_to_world(points, volume_transform, scene_bounds)

    # Update model parameters
    # Get shapes and device
    num_points = points.shape[0]  # points is [N, 3]
    device = points.device

    # Initialize all model tensors with proper nn.Parameters
    model._xyz = nn.Parameter(points.T.contiguous().requires_grad_(True))  # Convert [N, 3] -> [3, N]
    model._scaling = nn.Parameter(torch.log(scales).contiguous().requires_grad_(True))  # [N, 3], model expects log-scales
    model._opacity = nn.Parameter(torch.log(opacities).contiguous().requires_grad_(True))  # [N, 1], model expects log-opacity

    # Initialize rotation quaternions to identity
    rotations = torch.zeros((num_points, 4), device=device)
    rotations[..., 0] = 1  # w=1, x=y=z=0 for identity rotation
    model._rotation = nn.Parameter(rotations.contiguous().requires_grad_(True))

    # Initialize max 2D radii
    model.max_radii2D = torch.zeros(num_points, device=device)

    # Initialize SH features if needed
    if model.max_sh_degree > 0:
        model.num_sh_channels = (model.max_sh_degree + 1) ** 2 * 3
        model._features_dc = nn.Parameter(
            torch.full((num_points, 3), 0.5, device=device).contiguous().requires_grad_(True)
        )  # Mid-gray
        model._features_rest = nn.Parameter(
            torch.zeros(num_points, model.num_sh_channels - 3, device=device).contiguous().requires_grad_(True)
        )

    return model
