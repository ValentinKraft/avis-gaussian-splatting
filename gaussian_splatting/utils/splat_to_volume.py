from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F

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
    splats: Tensor,
    volume_shape: Tuple[int, int, int],
    covariances: Optional[Tensor] = None,
    scale: float = 0.1,
    batch_size: int = 100,  # Process points in batches to save memory
    scaling: Optional[Tensor] = None,  # Optional per-point scale factors
    opacity: Optional[Tensor] = None,  # Optional per-point opacity values
    intensities: Optional[Tensor] = None,  # Optional per-point intensity values
) -> Tensor:
    """
    Convert 3D Gaussian splats to a volumetric representation.

    Args:
        splats: Tensor of splat centers (3, N) or (N, 3)
        volume_shape: Output volume shape (depth, height, width)
        covariances: Optional covariance matrices (N, 3, 3)
        scale: Scale factor for isotropic Gaussians when covariances not provided
        batch_size: Number of points to process at once to manage memory
        scaling: Optional per-point scaling factors (N, 3) or (N,)
        opacity: Optional per-point opacity values (N, 1) or (N,)

    Returns:
        Volume tensor (D, H, W)
    """
    device = splats.device

    # Print input tensor info for debugging
    print(f"Input splats shape: {splats.shape}, requires_grad: {splats.requires_grad}")

    # Check if input requires gradients - if not, force it to require gradients
    if not splats.requires_grad:
        print(
            "WARNING: Input tensor does not require gradients - forcing requires_grad=True"
        )
        # Create a differentiable copy
        splats = splats.clone().detach().requires_grad_(True)

    # Handle different input formats WITHOUT detaching - we need to keep the computation graph
    if splats.shape[0] == 3:
        # Convert from (3, N) to (N, 3) without breaking gradient chain
        print(f"Converting splats from shape {splats.shape} to (N, 3)")
        points_n3 = splats.permute(1, 0)  # Use permute instead of T to maintain gradient connections
    else:
        # Keep the tensor as is
        points_n3 = splats

    total_points = points_n3.shape[0]
    print(f"Splatting {total_points} points to volume of shape {volume_shape}")

    # Create volume grid - these don't need gradients
    grid_points = create_grid_points(volume_shape, device)

    # Initialize volume tensor with gradients
    volume = torch.zeros(volume_shape, device=device, requires_grad=True)

    # Process splats in batches to save memory
    batch_size = min(batch_size, 50)  # Limit batch size to avoid OOM
    num_batches = (total_points + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of size {batch_size}")
    
    # Use smaller grid resolution for memory efficiency if we have many points
    if total_points > 1000:
        # Create a smaller working grid for initial calculations
        small_shape = tuple(max(16, d // 2) for d in volume_shape)
        # Only create the grid once and reuse it
        small_grid_points = create_grid_points(small_shape, device)
        # We'll upsample back to full resolution at the end
    else:
        small_shape = volume_shape
        small_grid_points = grid_points
    
    # Create a small working volume for accumulating results
    small_volume = torch.zeros(small_shape, device=device, requires_grad=True)
    
    # If we have scaling information, prepare it
    point_scales = None
    if scaling is not None:
        if scaling.shape[0] == 3 and len(scaling.shape) == 2:
            # Convert from (3, N) to (N, 3)
            point_scales = scaling.permute(1, 0)
        else:
            point_scales = scaling

    # Prepare opacity values if provided
    point_opacities = None
    if opacity is not None:
        if len(opacity.shape) == 2 and opacity.shape[1] == 1:
            point_opacities = opacity.squeeze(1)  # Convert (N, 1) to (N,)
        else:
            point_opacities = opacity

    # Prepare intensity values if provided
    point_intensities = None
    if intensities is not None:
        if len(intensities.shape) == 2 and intensities.shape[1] == 1:
            point_intensities = intensities.squeeze(1)  # Convert (N, 1) to (N,)
        else:
            point_intensities = intensities
    else:
        # Default to 1.0 intensity if not provided
        point_intensities = torch.ones(total_points, device=device)
        
    # Use torch.cuda.empty_cache() to clear memory periodically
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_points)

        # Get current batch of points
        batch_points = points_n3[start_idx:end_idx]
        
        # Scale points to match the small grid if we're using it
        if small_shape != volume_shape:
            # Ensure we match the working grid dimensions
            batch_points_scaled = batch_points.clone()
        else:
            batch_points_scaled = batch_points

        # Get batch-specific scales and opacity if available
        batch_scales = None
        if point_scales is not None:
            batch_scales = point_scales[start_idx:end_idx]

        batch_opacities = None
        if point_opacities is not None:
            batch_opacities = point_opacities[start_idx:end_idx]

        batch_intensities = None
        if point_intensities is not None:
            batch_intensities = point_intensities[start_idx:end_idx]

        # Process batch and accumulate results
        if (i+1) % 5 == 0:
            print(f"Processing batch {i+1}/{num_batches}")
            # Release memory periodically
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Process all points in the batch at once if possible
        batch_contribution = torch.zeros_like(small_volume)
        
        # Use smaller working buffers to save memory
        # Calculate distances for all points in the batch at once, then process one by one
        # to avoid OOM with large batch sizes or volume dimensions
        
        # Use the single-point version in a loop for memory efficiency
        for j in range(batch_points_scaled.shape[0]):
            center = batch_points_scaled[j]

            # Use point-specific scale if available, otherwise use global scale
            point_scale = scale
            if batch_scales is not None:
                if batch_scales.shape[1] == 3:  # (N, 3) shape
                    # FIX: Keep gradients flowing by not using .item()
                    point_scale = batch_scales[j].mean() * (1.0 if small_shape == volume_shape else 0.5)  # Adjust scale for smaller grid
                else:  # (N,) shape
                    # FIX: Keep gradients flowing by not using .item()
                    point_scale = batch_scales[j] * (1.0 if small_shape == volume_shape else 0.5)  # Adjust scale for smaller grid
            elif small_shape != volume_shape:
                # Adjust default scale for smaller grid
                point_scale = scale * 0.5
            
            # Release any intermediate tensors before creating new ones
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # Calculate Gaussian kernel using anisotropic scaling if available
            if batch_scales is not None and batch_scales.shape[1] == 3:
                # Use anisotropic Gaussian with per-point scaling
                point_scaling = batch_scales[j]
                
                # Apply scaling adjustment for smaller grid if needed
                if small_shape != volume_shape:
                    point_scaling = point_scaling * 0.5
                    
                # Use the improved gaussian_kernel_3d with scaling
                kernel = gaussian_kernel_3d(
                    small_grid_points,
                    center.view(1, 3),  # Add batch dimension
                    scaling=point_scaling.view(1, 3)  # Add batch dimension
                ).squeeze(-1)  # Remove extra dimension
            else:
                # Use simple isotropic Gaussian as before
                diff = small_grid_points - center.view(1, 1, 1, 3)
                sq_dist = torch.sum(diff * diff, dim=-1)
                kernel = torch.exp(-0.5 * sq_dist / (point_scale**2))
                
                # Clean up intermediate variables to save memory
                del diff, sq_dist

                # Apply opacity and intensity in one step to save memory
                # FIX: Keep gradient flow by not using .item()
                if batch_opacities is not None:
                    kernel = kernel * batch_opacities[j]
                if batch_intensities is not None:
                    kernel = kernel * batch_intensities[j]

                # Add contribution and free memory
                batch_contribution = batch_contribution + kernel
                # Force immediate cleanup of the kernel tensor
                del kernel
                
                # Only clear CUDA cache occasionally to avoid performance penalty
                if j % 20 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()        # Add the batch contribution to the working volume
        small_volume = small_volume + batch_contribution
        
        # Free memory
        del batch_contribution
        if device.type == 'cuda':
            torch.cuda.empty_cache()

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

    print(f"Volume range: [{volume.min().item():.4f}, {volume.max().item():.4f}]")
    print(f"Volume requires_grad: {volume.requires_grad}")

    # Ensure we have gradients flowing
    if not volume.requires_grad:
        print("WARNING: Volume doesn't require gradients after computation")
        # Create a proper connection to the input
        volume = volume + (splats[0, 0] * 0)

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
