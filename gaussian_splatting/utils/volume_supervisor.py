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

    def __init__(
        self,
        volume_path: str,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        mask_path: Optional[str] = None,
        loss_type: str = "mse",
        loss_weight: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Args:
            volume_path: Path to ground truth volume
            volume_shape: Target shape for volume optimization
            mask_path: Optional path to mask volume for opacity values
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

        # Load mask volume if provided
        self.mask_volume = None
        if mask_path:
            self.mask_volume = self.loader.load_volume(mask_path)
            print(f"Loaded mask volume with range [{self.mask_volume.min().item():.4f}, {self.mask_volume.max().item():.4f}]")

        # Initialize metrics tracking
        self.metrics = {
            'volume_loss': 0.0,
            'dice_score': 0.0,
        }

    def compute_loss(self, gaussians) -> Tuple[Tensor, Dict[str, float], Tensor]:
        """
        Compute volume supervision loss for current gaussians.
        
        Args:
            gaussians: Current gaussian model
            
        Returns:
            Tuple of (loss tensor, metrics dict, volume_gradients)
        """
        # Check if xyz requires gradients
        xyz = gaussians.get_xyz

        # Ensure parameters require gradients
        if not xyz.requires_grad:
            # Enable requires_grad without breaking optimizer reference
            gaussians._xyz.requires_grad_(True)
            xyz = gaussians._xyz

        # Get scaling, rotation, and opacity values
        scaling = gaussians.get_scaling
        rotation = gaussians.get_rotation
        opacity = gaussians.get_opacity

        # Check if intensity values need updating (match number of points = xyz.shape[0])
        if (
            not hasattr(gaussians, "intensities")
            or gaussians.intensities.shape[0] != xyz.shape[0]
        ):
            gaussians.intensities = (
                torch.ones((xyz.shape[0], 1), device=xyz.device) * 0.5
            )

        # Update intensity values if reference volume is available
        if gaussians.reference_volume is None:
            # Store the reference volume for future updates
            gaussians.reference_volume = self.volume_gt

        # Check if we need to initialize/update opacities
        if hasattr(self, 'mask_volume') and self.mask_volume is not None:
            # Initialize opacities buffer if needed (match number of points)
            if (
                not hasattr(gaussians, "opacities")
                or gaussians.opacities.shape[0] != xyz.shape[0]
            ):
                gaussians.opacities = (
                    torch.ones((xyz.shape[0], 1), device=xyz.device) * 0.5
                )

        # Initialize values only once - do not resample during training to preserve gradients
        iteration = getattr(self, "iteration", 0)
        features_initialized = getattr(self, "features_initialized", False)

        # Only update on first iteration when features are not yet initialized
        if not features_initialized:
            # Check if we have a mask for opacity
            if hasattr(self, 'mask_volume') and self.mask_volume is not None:
                # Update both intensities and opacities
                gaussians.update_intensities_and_opacities(self.volume_gt, self.mask_volume)
            else:
                # Just update intensities
                gaussians.update_intensities(self.volume_gt)
            # Mark as initialized to prevent future updates
            self.features_initialized = True
        self.iteration = iteration + 1

        # Convert gaussians to volume using intensity values
        # Use non-learnable opacities if they exist, otherwise use the opacity parameter
        use_opacity = opacity
        if hasattr(gaussians, 'opacities') and gaussians.opacities.numel() > 0:
            use_opacity = gaussians.opacities

        # FIX: Ensure opacity and intensity tensors have correct shape to match number of points
        # Get the number of points from xyz (whether [3, N] or [N, 3])
        n_points = xyz.shape[1] if xyz.shape[0] == 3 else xyz.shape[0]

        # Fix opacity tensor shape if needed
        if use_opacity.shape[0] != n_points:
            # If we have a shape mismatch, broadcast the opacity to all points
            if use_opacity.numel() == 3:  # We have exactly 3 values
                use_opacity = use_opacity.mean() * torch.ones(
                    (n_points, 1),
                    device=use_opacity.device,
                    dtype=use_opacity.dtype,
                    requires_grad=use_opacity.requires_grad,
                )
            else:
                # Otherwise use the first value and broadcast
                use_opacity = use_opacity[0] * torch.ones(
                    (n_points, 1),
                    device=use_opacity.device,
                    dtype=use_opacity.dtype,
                    requires_grad=use_opacity.requires_grad,
                )

        # Fix intensity tensor shape if needed
        use_intensities = gaussians.intensities
        if use_intensities.shape[0] != n_points:
            if use_intensities.numel() == 3:  # We have exactly 3 values
                use_intensities = use_intensities.mean() * torch.ones(
                    (n_points, 1),
                    device=use_intensities.device,
                    dtype=use_intensities.dtype,
                    requires_grad=use_intensities.requires_grad,
                )
            else:
                # Otherwise use the first value and broadcast
                use_intensities = use_intensities[0] * torch.ones(
                    (n_points, 1),
                    device=use_intensities.device,
                    dtype=use_intensities.dtype,
                    requires_grad=use_intensities.requires_grad,
                )

        # Debug tensor shapes is no longer needed

        # Convert gaussians to volume (directly uses parameter tensors for gradient flow)
        volume_pred = splat_to_volume(
            points=xyz,
            point_scales=scaling,
            point_rotations=rotation,
            point_opacities=use_opacity,
            point_intensities=use_intensities,
            volume_shape=self.volume_shape,
            device=xyz.device,
        )

        # Optionally retain grad for debugging
        if getattr(self, "debug", False):
            xyz.retain_grad()

        # Debug if needed
        if hasattr(self, "verbose") and self.verbose:
            print(f"volume_pred requires_grad: {volume_pred.requires_grad}")

        # Store predicted volume for visualization (use clone to avoid breaking gradient chain)
        self.volume_pred = volume_pred.detach().clone()

        # Compute loss - make sure both tensors are on the same device
        self.volume_gt = self.volume_gt.to(volume_pred.device)
        loss = self.criterion(volume_pred, self.volume_gt)

        # Scale loss by weight
        if self.loss_weight != 1.0:
            loss = loss * self.loss_weight

        # Optionally compute gradients of loss w.r.t. xyz periodically (for analysis/alignment)
        volume_grads = None
        if hasattr(self, "iteration") and self.iteration % 10 == 0:
            grad_list = torch.autograd.grad(
                loss, xyz, retain_graph=True, allow_unused=True
            )
            volume_grads = grad_list[0]
            self.volume_gradients = volume_grads
        else:
            volume_grads = getattr(self, "volume_gradients", None)

        # Update metrics
        with torch.no_grad():
            self.metrics['volume_loss'] = loss.item()
            if self.criterion.loss_type == 'dice':
                dice_score = 1 - loss.item()
                self.metrics['dice_score'] = dice_score

        # Return both loss and volume gradients for parameter diversity losses
        return loss, self.metrics.copy(), volume_grads

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
