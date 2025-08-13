# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.

"""
Tests for volume supervision components in 3D Gaussian Splatting.
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import os

from gaussian_splatting.losses.volume_loss import VolumeLoss, DiceLoss, TverskyLoss
from gaussian_splatting.utils.splat_to_volume import splat_to_volume, gaussian_kernel_3d
from gaussian_splatting.data.volume_loader import VolumeLoader
from gaussian_splatting.utils.volume_supervisor import VolumeSupervisor

def create_synthetic_volume(shape=(32, 32, 32)):
    """Create a synthetic volume with a sphere."""
    center = np.array(shape) / 2
    radius = min(shape) / 4
    
    x, y, z = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]), 
        np.arange(shape[2])
    )
    volume = np.zeros(shape)
    
    # Create sphere
    dist = np.sqrt(
        (x - center[0])**2 + 
        (y - center[1])**2 + 
        (z - center[2])**2
    )
    volume[dist <= radius] = 1.0
    
    return torch.from_numpy(volume).float()

def test_volume_loss():
    """Test volume loss functions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = create_synthetic_volume().to(device)
    target = create_synthetic_volume().to(device)
    
    # Test MSE Loss
    mse_loss = VolumeLoss('mse')
    loss = mse_loss(pred, target)
    assert loss.item() < 1e-6
    
    # Test Dice Loss
    dice_loss = VolumeLoss('dice')
    loss = dice_loss(pred, target)
    assert loss.item() < 1e-6
    
    # Test with offset prediction
    offset_pred = torch.roll(pred, shifts=1, dims=0)
    loss = dice_loss(offset_pred, target)
    assert loss.item() > 0.1

def test_splat_to_volume():
    """Test conversion of Gaussian splats to volume."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    volume_shape = (32, 32, 32)
    
    # Create synthetic splats
    splats = torch.tensor([
        [0.5, 0.5, 0.5],  # Center
        [0.25, 0.25, 0.25],  # Corner
        [0.75, 0.75, 0.75]  # Opposite corner
    ], device=device)
    
    # Test volume creation
    volume = splat_to_volume(splats, volume_shape)
    assert volume.shape == volume_shape
    assert volume.min() >= 0.0
    assert volume.max() <= 1.0
    
    # Test gradient flow
    volume.sum().backward()
    assert splats.grad is not None
    assert not torch.isnan(splats.grad).any()

def test_volume_loader():
    """Test volume data loading and preprocessing."""
    shape = (32, 32, 32)
    loader = VolumeLoader(shape)
    
    # Create temporary test volume
    volume = create_synthetic_volume(shape)
    path = 'test_volume.npy'
    np.save(path, volume.numpy())
    
    try:
        # Test loading
        loaded = loader.load_volume(path)
        assert loaded.shape == shape
        assert torch.allclose(loaded, volume, atol=1e-6)
        
        # Test normalization
        assert loaded.min() >= 0.0
        assert loaded.max() <= 1.0
        
    finally:
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

def test_end_to_end():
    """Test end-to-end volume supervision."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shape = (32, 32, 32)
    
    # Create test volume
    volume = create_synthetic_volume(shape)
    path = 'test_volume.npy'
    np.save(path, volume.numpy())
    
    try:
        # Initialize supervisor
        supervisor = VolumeSupervisor(
            volume_path=path,
            volume_shape=shape,
            loss_type='dice'
        )
        
        # Create mock gaussian model
        class MockGaussians:
            def __init__(self):
                self.xyz = torch.rand(100, 3, device=device)
                self.scaling = torch.ones(100, 3, 3, device=device)
            
            @property
            def get_xyz(self):
                return self.xyz
                
            @property
            def get_scaling(self):
                return self.scaling
        
        gaussians = MockGaussians()
        
        # Test loss computation
        loss, metrics = supervisor.compute_loss(gaussians)
        assert not torch.isnan(loss)
        assert 'volume_loss' in metrics
        assert 'dice_score' in metrics
        
        # Test gradient flow
        loss.backward()
        assert gaussians.xyz.grad is not None
        assert not torch.isnan(gaussians.xyz.grad).any()
        
    finally:
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

if __name__ == '__main__':
    pytest.main([__file__])
