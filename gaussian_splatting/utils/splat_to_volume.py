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
    scale: float = 0.1
) -> Tensor:
    """
    Convert 3D Gaussian splats to a volumetric representation.
    
    Args:
        splats: Tensor of splat centers (3, N) or (N, 3)
        volume_shape: Output volume shape (depth, height, width)
        covariances: Optional covariance matrices (N, 3, 3)
        scale: Scale factor for isotropic Gaussians when covariances not provided
        
    Returns:
        Volume tensor (D, H, W)
    """
    device = splats.device
    
    # Handle different input formats
    if splats.shape[0] == 3:
        # Convert from (3, N) to (N, 3)
        print(f"Converting splats from shape {splats.shape} to (N, 3)")
        splats = splats.T
    
    print(f"Splatting {splats.shape[0]} points to volume of shape {volume_shape}")
    
    # Create volume grid
    points = create_grid_points(volume_shape, device)
    
    # Process all splats at once
    volume = gaussian_kernel_3d(points, splats, covariances, scale)
    
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
