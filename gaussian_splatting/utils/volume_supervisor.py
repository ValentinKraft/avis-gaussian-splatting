#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Volume supervision manager for 3D Gaussian Splatting.
Handles volume loading, loss computation, and optimization tracking.
"""

import torch
from torch import Tensor
from typing import Optional, Dict, Tuple

from gaussian_splatting.losses.volume_loss import VolumeLoss
from gaussian_splatting.utils.splat_to_volume import splat_to_volume
from gaussian_splatting.data.volume_loader import VolumeLoader

class VolumeSupervisor:
    """
    Manages volume supervision during training:
    - Loads and preprocesses ground truth volumes
    - Computes volume supervision loss
    - Tracks metrics and optimization progress
    """

    def __init__(self,
                 volume_path: str,
                 volume_shape: Tuple[int, int, int] = (64, 64, 64),
                 loss_type: str = 'dice',
                 loss_weight: float = 1.0,
                 device: torch.device = torch.device('cuda')):
        """
        Args:
            volume_path: Path to ground truth volume
            volume_shape: Target shape for volume optimization
            loss_type: Type of volume loss ('mse', 'dice', 'tversky', 'kl')
            loss_weight: Weight for volume loss term
            device: Device to use for computations
        """
        self.device = device
        self.volume_shape = volume_shape
        self.loss_weight = loss_weight

        # Initialize volume loader and loss
        self.loader = VolumeLoader(volume_shape, device)
        self.criterion = VolumeLoss(loss_type, loss_weight)

        # Load ground truth volume
        self.volume_gt = self.loader.load_volume(volume_path)

        # Initialize metrics tracking
        self.metrics = {
            'volume_loss': 0.0,
            'dice_score': 0.0,
        }

    def compute_loss(self, gaussians) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute volume supervision loss for current gaussians.
        
        Args:
            gaussians: Current gaussian model
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        # Check if xyz requires gradients
        xyz = gaussians.get_xyz
        print(f"xyz requires_grad: {xyz.requires_grad}")
        print(f"xyz shape: {xyz.shape}")

        # Get scaling and opacity values
        scaling = gaussians.get_scaling
        opacity = gaussians.get_opacity

        # Check if intensity values need updating
        if (
            not hasattr(gaussians, "intensities")
            or gaussians.intensities.shape[0] != xyz.shape[1]
        ):
            # Initialize with default intensity
            gaussians.intensities = (
                torch.ones((xyz.shape[1], 1), device=xyz.device) * 0.5
            )

        # Update intensity values if reference volume is available
        if gaussians.reference_volume is None:
            # Store the reference volume for future updates
            gaussians.reference_volume = self.volume_gt

        # Update intensity values every 10 iterations or when they're not initialized
        iteration = getattr(self, "iteration", 0)
        if iteration % 10 == 0 or gaussians.intensities[0, 0] == 0.5:
            gaussians.update_intensities(self.volume_gt)
        self.iteration = iteration + 1

        # Convert gaussians to volume using intensity values
        volume_pred = splat_to_volume(
            xyz,  # Will be transposed in splat_to_volume
            self.volume_shape,
            None,  # Skip covariances for now
            scale=0.05,
            scaling=scaling,
            opacity=opacity,
            intensities=gaussians.intensities,
        )

        # Verify volume_pred has gradients
        print(f"volume_pred requires_grad: {volume_pred.requires_grad}")

        # Store predicted volume for visualization (use clone to avoid breaking gradient chain)
        self.volume_pred = volume_pred.detach().clone()

        # Compute loss - make sure both tensors are on the same device
        self.volume_gt = self.volume_gt.to(volume_pred.device)
        loss = self.criterion(volume_pred, self.volume_gt)

        # Verify loss has gradients
        print(f"loss requires_grad: {loss.requires_grad}")

        # Update metrics
        with torch.no_grad():
            self.metrics['volume_loss'] = loss.item()
            if self.criterion.loss_type == 'dice':
                dice_score = 1 - loss.item()
                self.metrics['dice_score'] = dice_score

        return loss, self.metrics.copy()

    def log_metrics(self, writer, iteration: int):
        """Log current metrics to tensorboard."""
        if writer is not None:
            for name, value in self.metrics.items():
                writer.add_scalar(f'volume/{name}', value, iteration)

            # Log volume visualizations periodically
            if iteration % 1000 == 0:
                writer.add_image('volume/ground_truth',
                               self.volume_gt[None, None],
                               iteration)
                writer.add_image('volume/prediction',
                               self.volume_pred[None, None],
                               iteration)
