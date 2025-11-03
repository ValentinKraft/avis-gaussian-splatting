# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Initialize Gaussian points from volume data for 3D Gaussian Splatting.
"""

import math
import heapq
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scene.gaussian_model import GaussianModel
from gaussian_splatting.data.volume_loader import VolumeLoader
from gaussian_splatting.utils.intensity_sampler import (
    sample_intensities_from_volume,
    update_opacities,
    update_intensities,
)
from gaussian_splatting.utils.orientation_field import (
    default_origin_and_spacing,
    random_quat_perturb,
)

if TYPE_CHECKING:
    from gaussian_splatting.utils.volume_supervisor import VolumeSupervisor

def _compute_distance_field(mask: Tensor, threshold: float = 0.1) -> Tensor:
    """Approximate Euclidean distance transform using a weighted grid Dijkstra."""
    mask_cpu = mask.detach().float().cpu()
    D, H, W = mask_cpu.shape
    outside = mask_cpu <= threshold
    outside_np = outside.numpy()

    if torch.all(~outside):
        # Entire volume is foreground; return zeros to avoid NaNs
        return torch.zeros_like(mask, dtype=torch.float32)

    dist = np.full((D, H, W), np.inf, dtype=np.float32)
    visited = np.zeros((D, H, W), dtype=bool)

    heap: List[Tuple[float, int, int, int]] = []
    outside_idx = np.argwhere(outside_np)
    for z, y, x in outside_idx:
        dist[z, y, x] = 0.0
        heapq.heappush(heap, (0.0, int(z), int(y), int(x)))

    offsets: List[Tuple[float, int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                offsets.append((cost, dz, dy, dx))

    while heap:
        current_dist, z, y, x = heapq.heappop(heap)
        if visited[z, y, x]:
            continue
        visited[z, y, x] = True

        for cost, dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if nz < 0 or nz >= D or ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            new_dist = current_dist + cost
            if new_dist < dist[nz, ny, nx]:
                dist[nz, ny, nx] = new_dist
                heapq.heappush(heap, (new_dist, nz, ny, nx))

    dist[outside_np] = 0.0
    return torch.from_numpy(dist).to(mask.device)


def _hash_indices(coords: Tensor, grid_size: Tuple[int, int, int]) -> Tensor:
    """Hash integer voxel coordinates for uniqueness filtering."""
    W = grid_size[2]
    H = grid_size[1]
    stride_y = W + 1
    stride_z = (H + 1) * stride_y
    return coords[:, 2] * stride_z + coords[:, 1] * stride_y + coords[:, 0]


def initialize_from_volume(
    mask_path: str,
    n_points: int = 5000,
    noise_std: float = 0.01,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample Gaussian seeds with distance-weighted importance and spacing control."""

    from gaussian_splatting.data.volume_loader import VolumeLoader

    loader = VolumeLoader(device=device)
    sampling_volume = loader.load_volume(mask_path)
    sampling_volume = sampling_volume.to(device=device, dtype=torch.float32)

    D, H, W = sampling_volume.shape
    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    coords = torch.stack([x, y, z], dim=-1).float()
    coords_flat = coords.reshape(-1, 3)
    volume_flat = sampling_volume.reshape(-1)

    positive_vals = volume_flat[volume_flat > 0]
    if positive_vals.numel() == 0:
        raise ValueError("Sampling volume contains no positive entries.")

    threshold = float(positive_vals.mean().item() * 0.3)
    distance_field = _compute_distance_field(sampling_volume, threshold=threshold)
    distance_flat = distance_field.reshape(-1)

    weights = (volume_flat.clamp_min(0.0) + 1e-6) * (distance_flat + 1e-4)
    weights_sum = weights.sum()
    if weights_sum <= 0.0:
        weights = torch.full_like(weights, 1.0 / weights.numel())
    else:
        weights = weights / weights_sum

    oversample = max(n_points * 4, n_points + 32)
    sampled_idx = torch.multinomial(weights, oversample, replacement=True)
    sampled_coords = coords_flat[sampled_idx]
    sampled_dist = distance_flat[sampled_idx]
    sampled_vals = volume_flat[sampled_idx]

    min_spacing_vox = 1.0
    cell_coords = torch.floor(sampled_coords / min_spacing_vox).long()
    cell_keys = _hash_indices(cell_coords, (D, H, W))
    unique_idx_np = np.unique(cell_keys.cpu().numpy(), return_index=True)[1]
    unique_idx = torch.from_numpy(unique_idx_np).to(device=device, dtype=torch.long)
    unique_idx, _ = torch.sort(unique_idx)

    sampled_coords = sampled_coords[unique_idx]
    sampled_dist = sampled_dist[unique_idx]
    sampled_vals = sampled_vals[unique_idx]

    extra_idx = None
    if sampled_coords.shape[0] < n_points:
        deficit = n_points - sampled_coords.shape[0]
        extra_idx = torch.multinomial(weights, deficit, replacement=True)
        sampled_coords = torch.cat([sampled_coords, coords_flat[extra_idx]], dim=0)
        sampled_dist = torch.cat([sampled_dist, distance_flat[extra_idx]], dim=0)
        sampled_vals = torch.cat([sampled_vals, volume_flat[extra_idx]], dim=0)

    sampled_coords = sampled_coords[:n_points]
    sampled_dist = sampled_dist[:n_points]
    sampled_vals = sampled_vals[:n_points]

    jitter = ((torch.rand_like(sampled_coords) - 0.5) * 0.5)
    jittered = sampled_coords + jitter
    jittered[:, 0].clamp_(0, W - 1)
    jittered[:, 1].clamp_(0, H - 1)
    jittered[:, 2].clamp_(0, D - 1)

    scale_den = torch.tensor([W - 1, H - 1, D - 1], device=device).clamp_min(1)
    points = jittered / scale_den

    _, voxel_size = default_origin_and_spacing((D, H, W), device)
    voxel_sizes_xyz = voxel_size
    dist_norm = sampled_dist / sampled_dist.max().clamp_min(1e-3)
    scale_min = voxel_sizes_xyz * 0.5
    scale_max = voxel_sizes_xyz * 2.5
    scales = scale_min + dist_norm.unsqueeze(1) * (scale_max - scale_min)

    val_min = float(positive_vals.min().item())
    val_max = float(positive_vals.max().item())
    if val_max > val_min:
        norm_vals = (sampled_vals - val_min) / (val_max - val_min)
    else:
        norm_vals = torch.ones_like(sampled_vals)
    opacities = norm_vals.clamp(0.1, 1.0).unsqueeze(1)

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
    initial_rotations: Optional[Tensor] = None,
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
    model._initial_scaling = (
        torch.log(scales).clone().detach()
    )  # Store initial scales for max size constraint

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

    # Initialize rotation quaternions
    if initial_rotations is not None and initial_rotations.numel() != 0:
        rotations = initial_rotations.to(device)
        rotations = rotations / (rotations.norm(dim=1, keepdim=True) + 1e-8)
    else:
        rotations = torch.zeros((num_points, 4), device=device)
        rotations[..., 0] = 1  # Identity quaternion
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
    model.intensities = intensities.detach().contiguous()
    model.intensities.requires_grad = False
    model.volume_min = volume_min
    model.volume_max = volume_max

    print(f"Initialized {num_points} Gaussians with intensity values")
    print(f"Stored volume min/max values: [{volume_min:.4f}, {volume_max:.4f}]")

    if getattr(model, "intensity_mode", "learned") in {
        "sampled",
        "sampled_mean_covered",
    }:
        # Disable SH features; rely entirely on sampled intensities
        model._features_dc = torch.zeros((num_points, 0, 3), device=device)
        model._features_rest = torch.zeros((num_points, 0, 3), device=device)
        return

    # Map intensity values to spherical harmonic coefficients for learnable color modes
    normalized_intensities = model._map_intensities_to_sh_coefficients(
        intensities, volume_min, volume_max
    )

    if torch.allclose(normalized_intensities, torch.zeros_like(normalized_intensities)):
        print(
            "Warning: Using default mid-gray intensities. Check if volume sampling worked correctly."
        )

    if model.max_sh_degree > 0:
        model.num_sh_channels = (model.max_sh_degree + 1) ** 2 * 3
        model._features_dc = nn.Parameter(
            torch.cat([normalized_intensities] * 3, dim=1)
            .contiguous()
            .requires_grad_(True)
        )
        model._features_rest = nn.Parameter(
            torch.zeros(num_points, model.num_sh_channels - 3, device=device)
            .contiguous()
            .requires_grad_(True)
        )
    else:
        intensity_tensor = normalized_intensities.expand(-1, 3).unsqueeze(1)
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
    orientation_helper: Optional["VolumeSupervisor"] = None,
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
        global_min = float(volume.min().item())
        global_max = float(volume.max().item())
        normalize_samples = getattr(model, "intensity_mode", "learned") in {
            "sampled",
            "sampled_mean_covered",
        }

        # Sample intensities using the utility function
        print("Sampling intensity values from volume...")
        intensities, volume_min, volume_max = update_intensities(
            points,
            volume,
            scales,
            normalize=normalize_samples,
            min_val=global_min if normalize_samples else None,
            max_val=global_max if normalize_samples else None,
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

            if normalize_samples:
                denom = max(global_max - global_min, 1e-8)
                if denom <= 1e-8:
                    intensities = torch.full_like(intensities, 0.5)
                else:
                    intensities = (intensities - global_min) / denom
                    intensities = intensities.clamp_(0.0, 1.0)
                volume_min = global_min
                volume_max = global_max
        elif normalize_samples:
            volume_min = global_min
            volume_max = global_max

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

    initial_rotations = None
    orientation_field = None
    fallback_count = 0
    if orientation_helper is not None:
        quats, fallback_count = orientation_helper.get_quat_for_points(points)
        initial_rotations = quats.detach()
        orientation_field = orientation_helper.export_orientation_field()
        print(
            f"Orientation initialized for {quats.shape[0]} points "
            f"(fallback {fallback_count})."
        )
    else:
        identity = torch.zeros(points.shape[0], 4, device=points.device)
        identity[:, 0] = 1.0
        initial_rotations = random_quat_perturb(identity, deg=2.0)
        fallback_count = points.shape[0]
        print(f"Orientation initialized without field (fallback {fallback_count}).")

    # Set up model parameters and feature tensors
    _setup_model_parameters(
        model,
        points,
        scales,
        opacities,
        opacity_values,
        initial_rotations,
    )
    _setup_feature_tensors(model, intensities, volume_min, volume_max)

    # Cache orientation data for densification if available
    model.orientation_field = orientation_field

    return model
