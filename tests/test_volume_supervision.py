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
from scene.gaussian_model import GaussianModel
from train import _maybe_reset_opacity

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


def _seed_gaussian_model(n_points: int, device: torch.device) -> GaussianModel:
    """Build a minimal GaussianModel instance with valid tensors."""
    model = GaussianModel(0)
    xyz = torch.rand(3, n_points, device=device)
    scaling = torch.zeros(n_points, 3, device=device)
    rotation = torch.zeros(n_points, 4, device=device)
    rotation[:, 0] = 1.0
    opacity = torch.zeros(n_points, 1, device=device)

    model._xyz = torch.nn.Parameter(xyz)
    model._scaling = torch.nn.Parameter(scaling)
    model._rotation = torch.nn.Parameter(rotation)
    model._opacity = torch.nn.Parameter(opacity)
    model.max_radii2D = torch.zeros(n_points, device=device)
    model.xyz_gradient_accum = torch.zeros(n_points, 1, device=device)
    model.denom = torch.ones(n_points, 1, device=device)
    model.intensities = torch.empty(0, device=device)
    model.opacities = torch.empty(0, device=device)
    model._initial_scaling = scaling.detach().clone()
    model._initial_xyz = xyz.detach().clone()
    model.set_intensity_mode("sampled")
    return model


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
    ], device=device, requires_grad=True)
    
    # Test volume creation
    volume = splat_to_volume(splats, volume_shape=volume_shape)
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

    # Create test volume + mask
    volume = create_synthetic_volume(shape)
    mask = (volume > 0.5).float()
    path = 'test_volume.npy'
    mask_path = 'test_mask.npy'
    np.save(path, volume.numpy())
    np.save(mask_path, mask.numpy())

    try:
        # Initialize supervisor
        supervisor = VolumeSupervisor(
            volume_path=path,
            volume_shape=shape,
            mask_path=mask_path,
            loss_type='mse',
            supervision_target='mask',
            intensity_update_interval=1,
        )

        gaussians = _seed_gaussian_model(100, device)

        # Test loss computation
        loss, metrics, _ = supervisor.compute_loss(gaussians)
        assert not torch.isnan(loss)
        assert 'volume_loss' in metrics
        assert 'dice_score' in metrics

        # Test gradient flow
        loss.backward()
        assert gaussians._xyz.grad is not None
        assert not torch.isnan(gaussians._xyz.grad).any()

    finally:
        # Cleanup
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(mask_path):
            os.remove(mask_path)


def test_volume_supervisor_populates_mask_opacity(tmp_path):
    """Ensure mask-driven opacity buffers override learnable opacities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (8, 8, 8)
    volume = torch.linspace(0.0, 1.0, steps=np.prod(shape), dtype=torch.float32).view(
        shape
    )
    mask = torch.zeros(shape, dtype=torch.float32)
    mask[2:6, 2:6, 2:6] = 0.75

    volume_path = tmp_path / "vol.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(volume_path, volume.numpy())
    np.save(mask_path, mask.numpy())

    supervisor = VolumeSupervisor(
        volume_path=str(volume_path),
        volume_shape=shape,
        mask_path=str(mask_path),
        intensity_update_interval=1,
    )

    gaussians = _seed_gaussian_model(8, device)
    loss, metrics, _ = supervisor.compute_loss(gaussians)
    assert torch.isfinite(loss)
    assert metrics
    assert gaussians.opacities.shape[0] == gaussians.get_opacity.shape[0]
    assert torch.allclose(gaussians.get_opacity, gaussians.opacities)
    assert gaussians.opacities.min() >= 0.0
    assert gaussians.opacities.max() <= 1.0


class _DummyGaussians:
    def __init__(self, mask_active: bool):
        self._mask_active = mask_active
        self.reset_calls = 0
        self.opacities = torch.empty(0)

    def _mask_opacity_active(self):
        return self._mask_active

    def reset_opacity(self):
        self.reset_calls += 1


def test_maybe_reset_opacity_skips_mask_buffer():
    dummy = _DummyGaussians(mask_active=True)
    triggered = _maybe_reset_opacity(dummy, iteration=10, interval=5)
    assert not triggered
    assert dummy.reset_calls == 0


def test_maybe_reset_opacity_triggers_for_learned():
    dummy = _DummyGaussians(mask_active=False)
    triggered = _maybe_reset_opacity(dummy, iteration=10, interval=5)
    assert triggered
    assert dummy.reset_calls == 1


def test_maybe_reset_opacity_respects_interval():
    dummy = _DummyGaussians(mask_active=False)
    triggered = _maybe_reset_opacity(dummy, iteration=9, interval=5)
    assert not triggered
    assert dummy.reset_calls == 0

    triggered = _maybe_reset_opacity(dummy, iteration=10, interval=0)
    assert not triggered
    assert dummy.reset_calls == 0


if __name__ == '__main__':
    pytest.main([__file__])
