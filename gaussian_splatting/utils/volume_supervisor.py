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
from torch.utils.checkpoint import checkpoint
from gaussian_splatting.utils.splat_to_volume import splat_to_volume
from gaussian_splatting.data.volume_loader import VolumeLoader
from gaussian_splatting.utils.intensity_sampler import (
    update_intensities_and_opacities,
)
from gaussian_splatting.utils.orientation_field import (
    compute_gradient_field,
    default_origin_and_spacing,
    gather_rotation_from_gradient,
    random_quat_perturb,
    rotmat_to_quat,
    world_to_voxel,
)

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
        intensity_update_interval: int = 10,
        dirty_threshold_xyz: float = 1e-3,
        dirty_threshold_scale: float = 5e-3,
        dirty_threshold_rot: float = 8.726646e-3,
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
        self.global_intensity_min = float(self.volume_gt.min().item())
        self.global_intensity_max = float(self.volume_gt.max().item())
        print(
            "Loaded volume intensity range: "
            f"[{self.global_intensity_min:.4f}, {self.global_intensity_max:.4f}]"
        )
        if abs(self.global_intensity_max - self.global_intensity_min) <= 1e-8:
            print(
                "Warning: Volume intensity range is nearly zero; outputs will default to mid-gray."
            )

        # Orientation defaults (no CLI controls for now)
        self.orientation_sigma_grad = 0.8  # Reduced for sharper gradients
        self.orientation_sigma_tensor = 0.0  # No post-smoothing needed
        self.orientation_perturb_deg = 2.0
        self._orientation_grad: Optional[Tensor] = None
        self._orientation_mag: Optional[Tensor] = None

        # Coordinate mapping assumes volume space normalised to [0, 1]
        origin, spacing = default_origin_and_spacing(
            self.volume_gt.shape, self.device
        )
        self.volume_origin = origin
        self.voxel_size = spacing

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
        self._step = 0
        self.last_intensity_update_count = 0
        self.intensity_update_interval = max(1, int(intensity_update_interval))
        self.dirty_threshold_xyz = float(dirty_threshold_xyz)
        self.dirty_threshold_scale = float(dirty_threshold_scale)
        self.dirty_threshold_rot = float(dirty_threshold_rot)

    def _orientation_source(self) -> Tensor:
        """Return the tensor used to derive orientations."""
        # Use the intensity volume for orientation - it has richer gradients
        # than binary/float masks which are mostly uniform
        return self.volume_gt

    def _volume_sampler(self, gaussians, indices: Optional[Tensor]) -> Tensor:
        """Sample mean intensities (and optional opacities) for selected indices."""
        xyz = gaussians.get_xyz
        scaling_full = gaussians.get_scaling
        if indices is None:
            pts = xyz
            scales = scaling_full if scaling_full.numel() > 0 else None
        else:
            idx = indices.long()
            pts = xyz[:, idx] if xyz.shape[0] == 3 else xyz[idx]
            scales = scaling_full[idx] if scaling_full.numel() > 0 else None

        intensities, opacities, v_min, v_max = update_intensities_and_opacities(
            pts,
            self.volume_gt,
            mask=self.mask_volume,
            scale=scales,
            normalize=True,
            min_val=self.global_intensity_min,
            max_val=self.global_intensity_max,
        )

        if indices is None:
            gaussians.volume_min = v_min
            gaussians.volume_max = v_max

        if opacities is not None:
            total = gaussians._xyz.shape[1]
            target_device = opacities.device
            if (
                not hasattr(gaussians, "opacities")
                or gaussians.opacities.shape[0] != total
            ):
                gaussians.opacities = torch.zeros(
                    (total, opacities.shape[1]),
                    device=target_device,
                    dtype=opacities.dtype,
                )
            elif gaussians.opacities.device != target_device:
                gaussians.opacities = gaussians.opacities.to(target_device)

            if indices is None:
                gaussians.opacities[:] = opacities
            else:
                gaussians.opacities[indices.long()] = opacities
            gaussians.opacities.requires_grad = False

        return intensities

    def _ensure_orientation_field(self) -> None:
        """Compute and cache gradient field if needed."""
        if self._orientation_grad is not None:
            return
        source = self._orientation_source().to(self.device)

        # Print source statistics for debugging
        print(
            f"Computing orientation field from intensity volume (range: [{source.min().item():.4f}, {source.max().item():.4f}])"
        )

        grad, mag = compute_gradient_field(
            source,
            sigma_pre=self.orientation_sigma_grad,
            sigma_post=self.orientation_sigma_tensor,
        )
        self._orientation_grad = grad
        self._orientation_mag = mag

        # Print gradient field statistics
        print(
            f"Gradient magnitude range: [{mag.min().item():.6f}, {mag.max().item():.6f}], mean: {mag.mean().item():.6f}"
        )
        print(
            f"Orientation field computed (sigma_pre={self.orientation_sigma_grad}, sigma_post={self.orientation_sigma_tensor})"
        )

    def get_quat_for_points(self, xyz_world: Tensor) -> Tuple[Tensor, int]:
        """Return orientation quaternions and fallback count for points."""
        if xyz_world.numel() == 0:
            print("Warning: Empty point set provided for orientation query.")
            return torch.empty(0, 4, device=self.device), 0

        self._ensure_orientation_field()

        # Debug: Print world coordinate statistics
        print(f"[get_quat_for_points] Processing {xyz_world.shape[0]} points")
        print(
            f"[get_quat_for_points] World coords: x=[{xyz_world[:, 0].min():.4f}, {xyz_world[:, 0].max():.4f}], "
            f"y=[{xyz_world[:, 1].min():.4f}, {xyz_world[:, 1].max():.4f}], "
            f"z=[{xyz_world[:, 2].min():.4f}, {xyz_world[:, 2].max():.4f}]"
        )
        print(
            f"[get_quat_for_points] Origin: {self.volume_origin.tolist()}, Voxel size: {self.voxel_size.tolist()}"
        )

        ijk = world_to_voxel(xyz_world, self.volume_origin, self.voxel_size)
        rotmats, fallback = gather_rotation_from_gradient(
            self._orientation_grad, self._orientation_mag, ijk, eps=1e-6
        )
        quats = rotmat_to_quat(rotmats)
        quats = random_quat_perturb(quats, self.orientation_perturb_deg)
        return quats, int(fallback.sum().item())

    def export_orientation_field(self) -> Dict[str, Tensor]:
        """Expose cached orientation data for reuse by the Gaussian model."""
        self._ensure_orientation_field()
        return {
            "gradient": self._orientation_grad,
            "magnitude": self._orientation_mag,
            "origin": self.volume_origin,
            "voxel_size": self.voxel_size,
            "perturb_deg": torch.tensor(
                self.orientation_perturb_deg, device=self.device
            ),
        }

    def compute_loss(
        self,
        gaussians,
        active_idx: Optional[Tensor] = None,
        total_points: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, float], Tensor]:
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

        self._step += 1
        self.iteration = getattr(self, "iteration", 0) + 1

        n_points = xyz.shape[1] if xyz.shape[0] == 3 else xyz.shape[0]
        intensity_mode = getattr(gaussians, "intensity_mode", "learned")

        use_intensities: Tensor
        if intensity_mode == "sampled":
            gaussians.reference_volume = self.volume_gt
            if self.mask_volume is not None:
                gaussians.reference_mask = self.mask_volume

            needs_resize = (
                not hasattr(gaussians, "intensities")
                or gaussians.intensities.numel() == 0
                or gaussians.intensities.shape[0] != n_points
            )

            update_due = ((self._step - 1) % self.intensity_update_interval) == 0
            dirty_subset = torch.empty(0, dtype=torch.long, device=xyz.device)
            if active_idx is not None and active_idx.numel() > 0:
                dirty_subset = gaussians.dirty_indices(
                    active_idx,
                    self.dirty_threshold_xyz,
                    self.dirty_threshold_scale,
                    self.dirty_threshold_rot,
                )

            indices_for_update: Optional[Tensor]
            if needs_resize:
                indices_for_update = None
            elif dirty_subset.numel() > 0:
                indices_for_update = dirty_subset
            elif update_due:
                indices_for_update = active_idx
            else:
                indices_for_update = None

            if indices_for_update is not None or needs_resize:
                updated = gaussians.update_sampled_intensities(
                    sampler=self._volume_sampler,
                    indices=indices_for_update,
                )
                self.last_intensity_update_count = updated
            else:
                self.last_intensity_update_count = 0

            if (
                not hasattr(gaussians, "intensities")
                or gaussians.intensities.numel() == 0
                or gaussians.intensities.shape[0] != n_points
            ):
                has_prev = (
                    hasattr(gaussians, "intensities")
                    and gaussians.intensities.numel() > 0
                )
                channels = gaussians.intensities.shape[1] if has_prev else 1
                new_buf = torch.full(
                    (n_points, channels),
                    0.5,
                    device=xyz.device,
                    dtype=xyz.dtype,
                )
                if has_prev:
                    copy_count = min(gaussians.intensities.shape[0], n_points)
                    if copy_count > 0:
                        src = (
                            gaussians.intensities.to(xyz.device)
                            if gaussians.intensities.device != xyz.device
                            else gaussians.intensities
                        )
                        new_buf[:copy_count] = src[:copy_count]
                gaussians.intensities = new_buf
            else:
                if gaussians.intensities.device != xyz.device:
                    gaussians.intensities = gaussians.intensities.to(xyz.device)
            gaussians.intensities.requires_grad = False
            gaussians.volume_min = self.global_intensity_min
            gaussians.volume_max = self.global_intensity_max
            use_intensities = gaussians.intensities
            if self._step % 200 == 0 and use_intensities.numel() > 0:
                batch_min = float(use_intensities.min().item())
                batch_max = float(use_intensities.max().item())
                print(
                    (
                        f"[Intensity] global_min={self.global_intensity_min:.4f}, "
                        f"global_max={self.global_intensity_max:.4f}, "
                        f"sampled_batch=[{batch_min:.4f},{batch_max:.4f}]"
                    )
                )
        else:
            self.last_intensity_update_count = 0
            if (
                hasattr(gaussians, "_features_dc")
                and gaussians._features_dc is not None
                and gaussians._features_dc.numel() > 0
            ):
                use_intensities = torch.sigmoid(
                    gaussians._features_dc[:, 0, :].mean(dim=1, keepdim=True)
                )
            else:
                if (
                    not hasattr(gaussians, "intensities")
                    or gaussians.intensities.numel() == 0
                    or gaussians.intensities.shape[0] != n_points
                ):
                    gaussians.intensities = torch.full(
                        (n_points, 1),
                        0.5,
                        device=xyz.device,
                        dtype=xyz.dtype,
                    )
                gaussians.intensities.requires_grad = False
                use_intensities = gaussians.intensities

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

        if total_points is None:
            total_points = n_points

        if (
            getattr(self, "debug_intensity", False)
            and intensity_mode == "sampled"
            and use_intensities.numel() > 0
        ):
            if not torch.isfinite(use_intensities).all():
                raise AssertionError("Sampled intensities contain non-finite values")
            min_val = float(use_intensities.min().item())
            max_val = float(use_intensities.max().item())
            if min_val < -1e-4 or max_val > 1.0 + 1e-4:
                raise AssertionError(
                    f"Sampled intensities out of [0,1] range: [{min_val:.4f}, {max_val:.4f}]"
                )

        # Fix intensity tensor shape if needed (use_intensities was computed above from features)
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
        def _render(points, scales, rotations, opacities, intensities):
            return splat_to_volume(
                points=points,
                point_scales=scales,
                point_rotations=rotations,
                point_opacities=opacities,
                point_intensities=intensities,
                volume_shape=self.volume_shape,
                device=xyz.device,
                active_idx=active_idx,
            )

        render_inputs = (xyz, scaling, rotation, use_opacity, use_intensities)
        if any(t.requires_grad for t in render_inputs):
            volume_pred = checkpoint(_render, *render_inputs, use_reentrant=False)
        else:
            volume_pred = _render(*render_inputs)

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

        if (
            active_idx is not None
            and self.criterion.loss_type == "mse"
            and total_points is not None
            and active_idx.numel() > 0
            and total_points > active_idx.numel()
        ):
            loss = loss * (total_points / active_idx.numel())

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
