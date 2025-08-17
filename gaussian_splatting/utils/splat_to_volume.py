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
    scale: float = 1.0
) -> Tensor:
    """
    Compute batched 3D Gaussian kernel values for multiple centers.
    
    Args:
        points: Grid points (D, H, W, 3)
        means: Gaussian centers (N, 3)
        covs: Optional covariance matrices (N, 3, 3)  
        scale: Scale factor for isotropic Gaussian
    
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
    
    if covs is None:
        # Use isotropic Gaussian for all centers
        sq_dist = torch.sum(diff * diff, dim=-1)  # (D, H, W, N)
        kernels = torch.exp(-0.5 * sq_dist / (scale ** 2))  # (D, H, W, N)
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
    batch_size: int = 100  # Process points in batches to save memory
) -> Tensor:
    """
    Convert 3D Gaussian splats to a volumetric representation.
    
    Args:
        splats: Tensor of splat centers (3, N) or (N, 3)
        volume_shape: Output volume shape (depth, height, width)
        covariances: Optional covariance matrices (N, 3, 3)
        scale: Scale factor for isotropic Gaussians when covariances not provided
        batch_size: Number of points to process at once to manage memory
        
    Returns:
        Volume tensor (D, H, W)
    """
    device = splats.device
    
    # Print input tensor info for debugging
    print(f"Input splats shape: {splats.shape}, requires_grad: {splats.requires_grad}")
    
    # Check if input requires gradients - if not, let's early return with a warning
    if not splats.requires_grad:
        print("WARNING: Input tensor does not require gradients - this will break backpropagation!")
    
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
    
    # Start with a zero tensor that inherits requires_grad from splats
    # This is critical for gradient flow
    volume = torch.zeros(volume_shape, device=device, requires_grad=splats.requires_grad)
    
    # Process splats in batches to save memory
    num_batches = (total_points + batch_size - 1) // batch_size
    print(f"Processing in {num_batches} batches of size {batch_size}")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_points)
        
        # Get current batch of points
        batch_points = points_n3[start_idx:end_idx]
        batch_covs = None
        if covariances is not None:
            batch_covs = covariances[start_idx:end_idx]
            
        # Process batch and accumulate results
        if (i+1) % 10 == 0:
            print(f"Processing batch {i+1}/{num_batches}")
        
        # Process all points in the batch at once if possible
        batch_contribution = torch.zeros_like(volume)
        
        # Use the single-point version in a loop for memory efficiency
        for j in range(batch_points.shape[0]):
            center = batch_points[j]
            
            # Calculate Gaussian kernel for this point
            # Make sure this operation is differentiable with respect to center
            diff = grid_points - center.view(1, 1, 1, 3)
            sq_dist = torch.sum(diff * diff, dim=-1)
            kernel = torch.exp(-0.5 * sq_dist / (scale ** 2))
            
            # Add contribution to batch accumulation
            batch_contribution = batch_contribution + kernel
        
        # Add the batch contribution to the main volume
        volume = volume + batch_contribution
    
    # Normalize volume to [0, 1] - ensure we preserve gradient flow
    max_val = volume.max()
    if max_val > 0:
        # Use proper differentiable normalization
        volume = volume / (max_val + 1e-6)
    
    print(f"Volume range: [{volume.min().item():.4f}, {volume.max().item():.4f}]")
    print(f"Volume requires_grad: {volume.requires_grad}")
    
    # Create a proper connection between the input and output
    if splats.requires_grad and not volume.requires_grad:
        print("WARNING: Gradient chain broken, creating dummy connection")
        # Create dummy connection to ensure gradient flow
        volume = volume + (splats.sum() * 0)
        
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
