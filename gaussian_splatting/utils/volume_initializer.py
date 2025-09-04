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
from gaussian_splatting.data.volume_loader import VolumeLoader
from gaussian_splatting.utils.intensity_sampler import (
    sample_intensities_from_volume,
    update_opacities,
    update_intensities,
)

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


def _setup_model_parameters(
    model: GaussianModel,
    points: Tensor,
    scales: Tensor,
    opacities: Tensor,
    opacity_values: Optional[Tensor] = None,
) -> None:
    """
    Set up core model parameters (positions, scales, rotations, opacities).

    Args:
        model: The Gaussian model to initialize
        points: Point positions [N, 3]
        scales: Scale values [N, 3]
        opacities: Default opacity values [N, 1]
        opacity_values: Optional volume-derived opacity values [N, 1]
    """
    # Get shapes and device
    num_points = points.shape[0]
    device = points.device

    # Initialize all model tensors with proper nn.Parameters
    model._xyz = nn.Parameter(
        points.T.contiguous().requires_grad_(True)
    )  # Convert [N, 3] -> [3, N]
    model._scaling = nn.Parameter(
        torch.log(scales).contiguous().requires_grad_(True)
    )  # [N, 3], model expects log-scales

    # Initialize opacity based on whether we're using volume-based opacity or not
    if opacity_values is not None:
        # Store non-learnable opacity values from the mask
        model.opacities = opacity_values
        # Also keep the _opacity parameter but without gradients (for backward compatibility)
        model._opacity = nn.Parameter(
            torch.log(opacities).detach().contiguous().requires_grad_(False)
        )
        print("Using non-learnable opacities from mask")
    else:
        # Use traditional learnable opacity parameters
        model._opacity = nn.Parameter(
            torch.log(opacities).contiguous().requires_grad_(True)
        )

    # Initialize rotation quaternions to identity
    rotations = torch.zeros((num_points, 4), device=device)
    rotations[..., 0] = 1  # w=1, x=y=z=0 for identity rotation
    model._rotation = nn.Parameter(rotations.contiguous().requires_grad_(True))

    # Initialize max 2D radii
    model.max_radii2D = torch.zeros(num_points, device=device)


def _setup_feature_tensors(
    model: GaussianModel,
    intensities: Tensor,
    volume_min: float,
    volume_max: float,
) -> None:
    """
    Set up feature tensors based on intensity values.

    Args:
        model: The Gaussian model to initialize
        intensities: Intensity values [N, 1]
        volume_min: Global minimum intensity value
        volume_max: Global maximum intensity value
    """
    num_points = intensities.shape[0]
    device = intensities.device

    # Store intensity values and volume range (not learnable parameters)
    model.intensities = intensities
    model.volume_min = volume_min
    model.volume_max = volume_max

    print(f"Initialized {num_points} Gaussians with intensity values")
    print(f"Stored volume min/max values: [{volume_min:.4f}, {volume_max:.4f}]")

    # Map intensity values to spherical harmonic coefficients
    normalized_intensities = model._map_intensities_to_sh_coefficients(
        intensities, volume_min, volume_max
    )

    # Check if we have valid intensity values
    if torch.allclose(normalized_intensities, torch.zeros_like(normalized_intensities)):
        print(
            "Warning: Using default mid-gray intensities. Check if volume sampling worked correctly."
        )

    # Initialize feature tensors based on SH degree
    if model.max_sh_degree > 0:
        # For RGB training, use full SH feature tensors
        model.num_sh_channels = (model.max_sh_degree + 1) ** 2 * 3

        # Create RGB features by repeating intensity for all channels (grayscale)
        model._features_dc = nn.Parameter(
            torch.cat([normalized_intensities] * 3, dim=1)  # Repeat for R, G, B
            .contiguous()
            .requires_grad_(True)
        )

        # Initialize higher-order SH coefficients to zero
        model._features_rest = nn.Parameter(
            torch.zeros(num_points, model.num_sh_channels - 3, device=device)
            .contiguous()
            .requires_grad_(True)
        )
    else:
        # For volume-only training with no SH
        # Expand intensities to RGB and reshape for DC features
        intensity_tensor = normalized_intensities.expand(-1, 3)  # [N, 3]
        intensity_tensor = intensity_tensor.unsqueeze(1)  # [N, 1, 3]

        print(
            f"Creating feature_dc from normalized intensities: shape {intensity_tensor.shape}, "
            f"range [{intensity_tensor.min().item():.4f}, {intensity_tensor.max().item():.4f}]"
        )

        if num_points > 0:
            print(
                f"First 5 RGB values: {intensity_tensor[:min(5, num_points), 0, :].cpu().numpy()}"
            )

        model._features_dc = nn.Parameter(
            intensity_tensor.contiguous().requires_grad_(True)
        )

        # Create empty rest features
        model._features_rest = nn.Parameter(
            torch.zeros((num_points, 0, 3), device=device)
            .contiguous()
            .requires_grad_(True)
        )


def _is_valid_sampling(intensities: Tensor) -> bool:
    """
    Check if sampled intensities have valid range.

    Args:
        intensities: Sampled intensity values

    Returns:
        bool: True if intensities are valid, False otherwise
    """
    # Check common failure cases
    if (
        intensities.max() <= intensities.min()
        or torch.allclose(intensities, torch.full_like(intensities, 0.5))
        or (intensities.max() - intensities.min()) < 1e-4
    ):
        return False
    return True


def _sample_fallback_intensities(
    points: Tensor, volume: Tensor, device: torch.device
) -> Tuple[Tensor, float, float]:
    """
    Fallback method for sampling intensities directly from volume.

    Args:
        points: Point positions in normalized [0,1] coordinates
        volume: Volume tensor
        device: Torch device

    Returns:
        Tuple of:
            - Sampled intensities
            - Volume min value
            - Volume max value
    """
    D, H, W = volume.shape
    # Convert normalized points to indices
    point_indices = (points * torch.tensor([W - 1, H - 1, D - 1], device=device)).long()
    point_indices = torch.clamp(
        point_indices,
        min=torch.tensor([0, 0, 0], device=device),
        max=torch.tensor([W - 1, H - 1, D - 1], device=device),
    )

    # Get intensity values at nearest voxels
    x, y, z = point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]
    direct_intensities = volume[z, y, x].unsqueeze(1)

    print(
        f"Raw intensity range: [{direct_intensities.min().item():.4f}, {direct_intensities.max().item():.4f}]"
    )

    # If direct sampling didn't work, try nonzero sampling
    if direct_intensities.max() <= 1e-4:
        print("Direct sampling failed, sampling from nonzero regions...")
        nonzero = torch.nonzero(volume > 1e-4, as_tuple=False)
        if len(nonzero) > 0:
            # Sample random points from nonzero regions
            indices = torch.randint(0, len(nonzero), (len(points),), device=device)
            sampled_points = nonzero[indices]
            # Get intensity values
            sampled_intensities = volume[
                sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
            ]
            direct_intensities = sampled_intensities.unsqueeze(1)

    # Get global min/max
    volume_min = float(volume.min().item())
    volume_max = float(volume.max().item())

    print(
        f"Updated intensity range: [{direct_intensities.min().item():.4f}, {direct_intensities.max().item():.4f}]"
    )
    print(f"Updated volume range: [{volume_min:.4f}, {volume_max:.4f}]")

    return direct_intensities, volume_min, volume_max


def initialize_gaussians(
    model: GaussianModel,
    n_points: int = 5000,
    volume_transform: Optional[Tensor] = None,
    scene_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    volume_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    **kwargs,
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

    # Sample intensity values from the volume if available
    intensities = None
    mask_volume = None
    volume_min = 0.0
    volume_max = 1.0
    loader = VolumeLoader(device=device)

    # Load and sample intensities from volume if provided
    if volume_path:
        # Load volume for intensity sampling
        print(f"Loading intensity volume from: {volume_path}")
        volume = loader.load_volume(volume_path)

        # Sample intensities using the utility function
        print("Sampling intensity values from volume...")
        intensities, volume_min, volume_max = update_intensities(
            points, volume, scales, normalize=False
        )

        # Check if sampling was successful
        if not _is_valid_sampling(intensities):
            print(
                "Warning: Invalid intensity range detected. Trying alternative sampling..."
            )

            # Try direct sampling at nearest voxels
            intensities, volume_min, volume_max = _sample_fallback_intensities(
                points, volume, device
            )

        print(f"Final volume global range: [{volume_min:.4f}, {volume_max:.4f}]")
    else:
        # Default mid-gray if no volume is provided
        intensities = torch.full((points.shape[0], 1), 0.5, device=device)

    # Load mask for opacity sampling if available
    opacity_values = None
    if mask_path:
        # Load mask for opacity sampling (use the same mask used for point sampling)
        mask_volume = loader.load_volume(mask_path)

        # Sample opacity values from the mask
        print("Sampling opacity values from mask...")
        opacity_values, mask_min, mask_max = update_opacities(
            points, mask_volume, scales
        )
        print(
            f"Opacity range: [{opacity_values.min().item():.4f}, {opacity_values.max().item():.4f}]"
        )
        print(f"Mask global range: [{mask_min:.4f}, {mask_max:.4f}]")

    # Transform to world space
    points = transform_points_to_world(points, volume_transform, scene_bounds)

    # Set up model parameters and feature tensors
    _setup_model_parameters(model, points, scales, opacities, opacity_values)
    _setup_feature_tensors(model, intensities, volume_min, volume_max)

    return model
