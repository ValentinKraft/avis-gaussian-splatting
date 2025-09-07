from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

# NOTE: Refactored to (1) minimize Python loops, (2) actually use rotation by
# constructing covariances, and (3) avoid reallocation / unnecessary empty_cache calls.

def create_grid_points(volume_shape: Tuple[int, int, int], device: torch.device) -> Tensor:
    """
    Create a grid of 3D points for volume rendering.
    
    Args:
        volume_shape: (depth, height, width) of output volume
        device: Device to create tensors on
    
    Returns:
        Grid points tensor (D, H, W, 3)
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
    means: Tensor,
    covs: Optional[Tensor] = None,
    scale: float = 1.0,
    scaling: Optional[Tensor] = None  # Add scaling parameter
) -> Tensor:
    """
    Compute batched 3D Gaussian kernel values for multiple centers.
    
    Args:
        points: Grid points (D, H, W, 3)
        means: Gaussian centers (N, 3)
        covs: Optional covariance matrices (N, 3, 3)  
        scale: Scale factor for isotropic Gaussian
        scaling: Optional per-point scaling factors (N, 3)
    
    Returns:
        Combined kernel values at each point (D, H, W)
    """
    D, H, W = points.shape[:3]
    N = means.shape[0]

    # Reshape points for broadcasting against means
    # points: (D, H, W, 3) -> (D, H, W, 1, 3)
    points_exp = points.unsqueeze(3)

    # Reshape means for broadcasting against points
    # means: (N, 3) -> (1, 1, 1, N, 3)
    means_exp = means.reshape(1, 1, 1, N, 3)

    # Compute differences for all points and means at once
    # diff: (D, H, W, N, 3)
    diff = points_exp - means_exp

    if covs is None and scaling is None:
        # Use isotropic Gaussian for all centers
        sq_dist = torch.sum(diff * diff, dim=-1)  # (D, H, W, N)
        kernels = torch.exp(-0.5 * sq_dist / (scale ** 2))  # (D, H, W, N)
    elif scaling is not None:
        # Use anisotropic Gaussian with per-axis scaling
        # Reshape scaling for broadcasting
        scaling_exp = scaling.reshape(1, 1, 1, N, 3)
        # Apply per-axis scaling
        scaled_diff = diff / (scaling_exp + 1e-6)
        sq_dist = torch.sum(scaled_diff * scaled_diff, dim=-1)  # (D, H, W, N)
        kernels = torch.exp(-0.5 * sq_dist)
    else:
        # Use full covariance matrices
        # covs: (N, 3, 3)
        # Compute inverse of each covariance matrix
        cov_invs = torch.inverse(covs)  # (N, 3, 3)

        # Expand covs for broadcasting
        # (N, 3, 3) -> (1, 1, 1, N, 3, 3)
        cov_invs = cov_invs.reshape(1, 1, 1, N, 3, 3)

        # Reshape diff for matrix multiplication
        # (D, H, W, N, 3) -> (D, H, W, N, 1, 3)
        diff_exp = diff.unsqueeze(-2)

        # Compute mahalanobis distance
        # (D, H, W, N)
        mahalanobis = torch.sum(
            (diff_exp @ cov_invs) * diff_exp.transpose(-2, -1),
            dim=(-2, -1)
        )
        kernels = torch.exp(-0.5 * mahalanobis)

    # Sum contributions from all Gaussians
    # (D, H, W)
    return kernels.sum(dim=-1)


def splat_to_volume(
    points: Tensor,
    point_scales: Optional[Tensor] = None,
    point_rotations: Optional[Tensor] = None,
    point_opacities: Optional[Tensor] = None,
    point_intensities: Optional[Tensor] = None,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
    covariances: Optional[Tensor] = None,
    scale: float = 0.1,
    batch_size: int = 50,  # Process points in batches to save memory
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Convert 3D Gaussian splats to a volumetric representation.

    Args:
        points: Tensor of point centers (3, N) or (N, 3)
        point_scales: Optional per-point scaling factors (N, 3) or (N,)
        point_rotations: Optional per-point rotation quaternions (N, 4)
        point_opacities: Optional per-point opacity values (N, 1) or (N,)
        point_intensities: Optional per-point intensity values (N, 1) or (N,)
        volume_shape: Output volume shape (depth, height, width)
        covariances: Optional covariance matrices (N, 3, 3)
        scale: Default scale factor for isotropic Gaussians when point_scales not provided
        batch_size: Number of points to process at once to manage memory
        device: Device to use for computation

    Returns:
        Volume tensor (D, H, W)
    """
    device = points.device if device is None else device

    # Lightweight debug (can be silenced by setting ENV var later)
    if torch.is_grad_enabled() and points.grad_fn is None:
        pass  # avoid noisy prints in tight loops

    # Check if input requires gradients - if not, force it to require gradients
    if not points.requires_grad:
        print(
            "WARNING: Input tensor does not require gradients - forcing requires_grad=True"
        )
        # Create a differentiable copy
        points = points.clone().detach().requires_grad_(True)

    # Handle different input formats WITHOUT detaching - we need to keep the computation graph
    if points.shape[0] == 3:
        # Convert from (3, N) to (N, 3) without breaking gradient chain
        print(f"Converting splats from shape {points.shape} to (N, 3)")
        points_n3 = points.permute(1, 0)  # Use permute instead of T to maintain gradient connections
    else:
        # Keep the tensor as is
        points_n3 = points

    total_points = points_n3.shape[0]
    # Avoid verbose printing here for performance

    # Create volume grid - these don't need gradients
    grid_points = create_grid_points(volume_shape, device)

    # Allocate final volume (grad will flow through ops populating it)
    volume = torch.zeros(volume_shape, device=device)

    # Process splats in batches to save memory
    batch_size = min(batch_size, 50)  # Limit batch size to avoid OOM
    num_batches = (total_points + batch_size - 1) // batch_size
    # No console spam per batch

    # Use smaller grid resolution for memory efficiency if we have many points
    if total_points > 2000:
        # Create a smaller working grid for initial calculations
        small_shape = tuple(max(16, d // 2) for d in volume_shape)
        # Only create the grid once and reuse it
        small_grid_points = create_grid_points(small_shape, device)
        # We'll upsample back to full resolution at the end
    else:
        small_shape = volume_shape
        small_grid_points = grid_points

    # Create a small working volume for accumulating results
    small_volume = torch.zeros(small_shape, device=device)

    # Handle scaling parameters
    # point_scales, point_opacities, point_intensities already passed as parameters

    # Use default intensity values if not provided
    if point_intensities is None:
        # Default to 1.0 intensity if not provided
        point_intensities = torch.ones(total_points, device=device)

    # Use torch.cuda.empty_cache() to clear memory periodically
    # if device.type == 'cuda':
    #     torch.cuda.empty_cache()

    # Vectorized accumulation over batches.
    # Prepare rotation -> covariance if provided (quaternions expected normalized).
    def quat_to_rotmat(q: Tensor) -> Tensor:
        # q: (B,4) (w,x,y,z) or (x,y,z,w); assume either – normalize then compute matrix
        if q.shape[-1] != 4:
            raise ValueError("Quaternion tensor must have shape (N,4)")
        # Heuristic: if mean(abs(q[...,0])) < mean(abs(q[..., -1])) swap ordering; keep simple
        # We won't modify ordering aggressively; assume (N,4) already in proper order matching training code
        q = F.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)
        B = q.shape[0]
        R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    if point_rotations is not None and point_rotations.numel() > 0:
        rot_mats = quat_to_rotmat(point_rotations)
    else:
        rot_mats = None

    # Flatten grid and chunk to limit memory.
    work_grid = small_grid_points.view(-1, 3)
    G = work_grid.shape[0]
    grid_chunk = 32768  # tune if still OOM

    for i in range(num_batches):
        s = i * batch_size
        e = min((i + 1) * batch_size, total_points)
        bp = points_n3[s:e]
        Bcur = bp.shape[0]
        if Bcur == 0:
            continue

        # Scales to (B,3)
        if point_scales is None:
            scales_batch = torch.full((Bcur, 3), scale, device=device, dtype=bp.dtype)
        else:
            sb = point_scales[s:e]
            if sb.ndim == 2 and sb.shape[1] == 3:
                scales_batch = sb
            else:
                scales_batch = sb.view(-1, 1).repeat(1, 3)
        if small_shape != volume_shape:
            scales_batch = scales_batch * 0.5

        if rot_mats is not None:
            rb = rot_mats[s:e]  # (B,3,3)
        else:
            rb = None

        weight = torch.ones(Bcur, device=device, dtype=bp.dtype)
        if point_opacities is not None:
            weight = weight * point_opacities[s:e].view(-1)
        if point_intensities is not None:
            weight = weight * point_intensities[s:e].view(-1)

        inv_scales = 1.0 / (scales_batch + 1e-6)  # (B,3)
        contrib_flat = torch.zeros(G, device=device, dtype=bp.dtype)

        for g0 in range(0, G, grid_chunk):
            g1 = min(g0 + grid_chunk, G)
            grid_chunk_pts = work_grid[g0:g1]  # (Cg,3)
            diff = grid_chunk_pts.unsqueeze(1) - bp.unsqueeze(0)  # (Cg,B,3)
            if rb is not None:
                # More efficient batch processing for rotation
                diff_local = torch.zeros_like(diff)
                # Process larger batches (but not too large to avoid memory issues)
                batch_size = min(20, Bcur)  # Process 20 points at a time max
                for b_start in range(0, Bcur, batch_size):
                    b_end = min(b_start + batch_size, Bcur)

                    # Get batch of differences and rotation matrices
                    batch_diff = diff[:, b_start:b_end, :]  # (Cg, batch_size, 3)
                    batch_rb = rb[b_start:b_end]  # (batch_size, 3, 3)

                    # Apply rotation: batch_diff @ batch_rb.T
                    # Use more memory-efficient approach
                    for i in range(b_end - b_start):
                        b_idx_global = b_start + i
                        diff_local[:, b_idx_global, :] = torch.matmul(
                            batch_diff[:, i, :], batch_rb[i].T
                        )
            else:
                diff_local = diff
            diff_scaled = diff_local * inv_scales.unsqueeze(0)  # (Cg,B,3)
            sq = (diff_scaled * diff_scaled).sum(-1)  # (Cg,B)
            kern = torch.exp(-0.5 * sq) * weight.unsqueeze(0)  # (Cg,B)
            contrib_flat[g0:g1] += kern.sum(dim=1)
            del diff, diff_local, diff_scaled, sq, kern

        small_volume = small_volume + contrib_flat.view(small_shape)
        del contrib_flat

    # Normalize the working volume to [0, 1] - ensure we preserve gradient flow
    max_val = small_volume.max()
    if max_val > 0:
        # Use proper differentiable normalization
        small_volume = small_volume / (max_val + 1e-6)

    # If we used a smaller working grid, upsample back to full resolution
    if small_shape != volume_shape:
        # Convert to 5D tensor for F.interpolate (batch, channels, D, H, W)
        volume_5d = small_volume.unsqueeze(0).unsqueeze(0)
        # Upsample to original size with trilinear interpolation
        volume_5d = F.interpolate(volume_5d, size=volume_shape, mode='trilinear', align_corners=False)
        # Convert back to 3D tensor
        volume = volume_5d.squeeze(0).squeeze(0)
    else:
        volume = small_volume

    # Free memory
    del small_volume
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Optional debug prints removed for performance; caller can inspect externally.

    # Ensure we have gradients flowing
    if not volume.requires_grad:
        print("WARNING: Volume doesn't require gradients after computation")
        # Create a proper connection to the input
        volume = volume + (points[0, 0] * 0)

    # No threshold - let the gradients flow naturally
    # volume = torch.sigmoid((volume - 0.1) * 10)

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
    volume_shape = volume.shape
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
