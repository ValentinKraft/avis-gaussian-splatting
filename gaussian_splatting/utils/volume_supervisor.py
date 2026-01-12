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
    sample_mean_covered_voxel_intensities,
    update_intensities_and_opacities,
    update_opacities,
)
from gaussian_splatting.utils.orientation_field import (
    build_structure_field,
    compute_gradient_field,
    default_origin_and_spacing,
    gather_rotation_from_gradient,
    random_quat_perturb,
    rotmat_to_quat,
    sample_structure_field,
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
        volume_downscale_factor: Optional[int] = None,
        mask_path: Optional[str] = None,
        loss_type: str = "dice",
        loss_weight: float = 1.0,
        supervision_target: str = "mask",
        mask_loss_threshold_rel: float = 0.01,
        opacity_gamma: float = 1.0,
        density_scale: float = 1.0,
        outside_mask_weight: float = 0.1,
        device: torch.device = torch.device("cuda"),
        intensity_update_interval: int = 10,
        dirty_threshold_xyz: float = 1e-3,
        dirty_threshold_scale: float = 5e-3,
        dirty_threshold_rot: float = 8.726646e-3,
        verbose: bool = False,
        sampling_padding_mode: str = "border",
    ):
        """
        Args:
            volume_path: Path to ground truth volume
            volume_shape: Target shape for volume optimization
            mask_path: Optional path to mask volume for opacity values
            loss_type: Type of volume loss ('mse', 'dice', 'tversky', 'kl')
            loss_weight: Weight for volume loss term
            device: Device to use for computations
            verbose: When True, print detailed diagnostics during sampling
        """
        self.device = device
        self.volume_shape = volume_shape
        self.loss_weight = loss_weight
        self.verbose = bool(verbose)

        if supervision_target not in {"mask", "ct"}:
            raise ValueError(
                "supervision_target must be one of {'mask','ct'}, got "
                f"{supervision_target!r}."
            )
        self.supervision_target = str(supervision_target)
        self.opacity_gamma = float(opacity_gamma)
        self.density_scale = float(density_scale)
        self.mask_loss_threshold_rel = float(mask_loss_threshold_rel)
        self.outside_mask_weight = float(outside_mask_weight)

        # Training is defined to be mask-driven; without a mask the objective is
        # ill-posed for the intended medical workflows.
        if not mask_path:
            raise ValueError(
                "mask_path is required for volume supervision (training without a mask is not supported)."
            )

        # Loss masking: only voxels above a fraction of mask max contribute.
        # Default matches the medical workflow requirement: 1% of mask max.

        # Initialize volume loader and loss
        # Default behavior (omitted flag) matches downscale_factor=1: keep native resolution.
        downscale_factor = (
            int(volume_downscale_factor)
            if volume_downscale_factor is not None
            else 1
        )
        self.loader = VolumeLoader(
            target_shape=None,
            device=device,
            downscale_factor=downscale_factor,
        )
        # Apply loss_weight once in compute_loss for clarity.
        self.criterion = VolumeLoss(loss_type, 1.0)

        # Load ground truth volume used for supervision/rendering (may be downscaled).
        self.volume_gt = self.loader.load_volume(volume_path)
        # Also keep a full-resolution volume for sampling per-splat intensities/colors.
        # This ensures color is not taken from the downscaled supervision volume.
        self.volume_color = self.volume_gt
        if downscale_factor != 1:
            color_loader = VolumeLoader(
                target_shape=None,
                device=device,
                downscale_factor=1,
            )
            self.volume_color = color_loader.load_volume(volume_path)

        # Always trust the loaded tensor shape for supervision/rendering.
        self.volume_shape = tuple(int(v) for v in self.volume_gt.shape)
        self.global_intensity_min = float(self.volume_color.min().item())
        self.global_intensity_max = float(self.volume_color.max().item())
        if self.verbose:
            print(
                "Loaded color volume intensity range: "
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
        self.structure_sigma = 1.0
        self.structure_mask_threshold = 0.1
        self._structure_quat: Optional[Tensor] = None
        self._structure_vesselness: Optional[Tensor] = None

        # Coordinate mapping assumes volume space normalised to [0, 1]
        origin, spacing = default_origin_and_spacing(
            self.volume_gt.shape, self.device
        )
        self.volume_origin = origin
        self.voxel_size = spacing

        # Load mask volume (required)
        self.mask_volume = self.loader.load_volume(mask_path)
        if self.mask_volume.shape != self.volume_gt.shape:
            raise ValueError(
                "Mask and volume shapes must match after loading. "
                f"volume_shape={tuple(self.volume_gt.shape)}, mask_shape={tuple(self.mask_volume.shape)}"
            )
        if self.verbose:
            print(
                "Loaded mask volume with range "
                f"[{self.mask_volume.min().item():.4f}, {self.mask_volume.max().item():.4f}]"
            )

        # Cache mask-derived data for reuse during loss computation.
        self.mask_max = float(self.mask_volume.max().item())
        self.mask_threshold = float(self.mask_loss_threshold_rel) * self.mask_max
        self.mask_bool = (self.mask_volume > self.mask_threshold).to(device=self.device)
        if not self.mask_bool.any():
            raise RuntimeError(
                "Mask thresholding produced an empty region. "
                f"mask_max={self.mask_max:.6f}, rel_thr={self.mask_loss_threshold_rel:.6f}"
            )

        nz = torch.nonzero(self.mask_bool, as_tuple=False)
        z0 = int(nz[:, 0].min().item())
        z1 = int(nz[:, 0].max().item())
        y0 = int(nz[:, 1].min().item())
        y1 = int(nz[:, 1].max().item())
        x0 = int(nz[:, 2].min().item())
        x1 = int(nz[:, 2].max().item())
        self.mask_bounds = (z0, z1, y0, y1, x0, x1)

        D, H, W = self.volume_shape
        self.roi_shape = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
        denom = torch.tensor(
            [max(W - 1, 1), max(H - 1, 1), max(D - 1, 1)],
            device=self.device,
            dtype=torch.float32,
        )
        bounds_min = torch.tensor([x0, y0, z0], device=self.device, dtype=torch.float32) / denom
        bounds_max = torch.tensor([x1, y1, z1], device=self.device, dtype=torch.float32) / denom
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        # Loosen bounds slightly (3 voxels) to avoid freezing positions at the ROI edge.
        # Padding is expressed in voxel units of the current supervision grid.
        self.roi_pad_vox = 3.0
        pad = self.voxel_size * float(self.roi_pad_vox)
        self.bounds_min_padded = torch.clamp(bounds_min - pad, min=0.0)
        self.bounds_max_padded = torch.clamp(bounds_max + pad, max=1.0)

        # Cache ROI-aligned tensors for per-iteration reuse.
        self._roi_slices = (
            slice(z0, z1 + 1),
            slice(y0, y1 + 1),
            slice(x0, x1 + 1),
        )
        zsl, ysl, xsl = self._roi_slices
        self.volume_gt_roi = self.volume_gt[zsl, ysl, xsl]
        self.mask_volume_roi = self.mask_volume[zsl, ysl, xsl]
        self.mask_bool_roi = self.mask_bool[zsl, ysl, xsl]

        # Initialize metrics tracking
        self.metrics = {
            'volume_loss': 0.0,
            'dice_score': 0.0,
            'outside_mask_loss': 0.0,
        }
        self._step = 0
        self.last_intensity_update_count = 0
        self.intensity_update_interval = max(1, int(intensity_update_interval))
        self.dirty_threshold_xyz = float(dirty_threshold_xyz)
        self.dirty_threshold_scale = float(dirty_threshold_scale)
        self.dirty_threshold_rot = float(dirty_threshold_rot)
        self.sampling_padding_mode = str(sampling_padding_mode)

    def _orientation_source(self) -> Tensor:
        """Return the tensor used to derive orientations."""
        # Use the intensity volume for orientation - it has richer gradients
        # than binary/float masks which are mostly uniform
        return self.volume_color

    def _volume_sampler(self, gaussians, indices: Optional[Tensor]) -> Tensor:
        """Sample mean intensities (and optional opacities) for selected indices."""
        xyz = gaussians.get_xyz
        scaling_full = gaussians.get_scaling
        if indices is None:
            pts = xyz
            scales = scaling_full if scaling_full.numel() > 0 else None
            idx_tensor = None
        else:
            idx = indices.long()
            pts = xyz[:, idx] if xyz.shape[0] == 3 else xyz[idx]
            if scaling_full.numel() > 0:
                if (
                    scaling_full.dim() == 2
                    and scaling_full.shape[0] == 3
                    and scaling_full.shape[1] != 3
                ):
                    scales = scaling_full[:, idx]
                else:
                    scales = scaling_full[idx]
            else:
                scales = None
            idx_tensor = idx

        intensity_mode = getattr(gaussians, "intensity_mode", "learned")
        coverage_mask: Optional[Tensor] = None

        intensities, opacities, v_min, v_max = update_intensities_and_opacities(
            pts,
            self.volume_color,
            mask=self.mask_volume,
            scale=scales,
            normalize=True,
            min_val=self.global_intensity_min,
            max_val=self.global_intensity_max,
            padding_mode=self.sampling_padding_mode,
        )

        if (
            self.mask_volume is not None
            and opacities is not None
            and scales is not None
            and scales.numel() > 0
        ):
            large_mask_global = gaussians.large_splat_mask(
                getattr(gaussians, "intensity_large_splat_threshold", 0.0)
            ).to(device=pts.device)
            if indices is None:
                coverage_mask = large_mask_global
            else:
                coverage_mask = large_mask_global[indices.long()]

            if coverage_mask is not None and coverage_mask.any():
                refined, _, _ = sample_mean_covered_voxel_intensities(
                    pts,
                    self.mask_volume,
                    scales,
                    self.volume_origin,
                    self.voxel_size,
                    radius_scale=getattr(gaussians, "mean_covered_radius", 2.5),
                    coverage_mask=coverage_mask,
                    normalize=False,
                    min_val=0.0,
                    max_val=1.0,
                    padding_mode=self.sampling_padding_mode,
                )
                opacities = opacities.clone()
                opacities[coverage_mask] = refined[coverage_mask]

        if indices is None:
            gaussians.volume_min = v_min
            gaussians.volume_max = v_max

        if opacities is not None:
            if self.opacity_gamma != 1.0:
                opacities = opacities.clamp(0.0, 1.0).pow(self.opacity_gamma)

            if xyz.dim() == 2 and xyz.shape[0] == 3:
                total = xyz.shape[1]
            else:
                total = xyz.shape[0]
            cols = opacities.shape[1] if opacities.dim() == 2 else 1
            target_device = opacities.device
            opacity_buf = gaussians.ensure_opacity_buffer(
                total,
                cols,
                device=target_device,
                dtype=opacities.dtype,
            )
            gaussians.opacities = opacity_buf

            if idx_tensor is None:
                opacity_buf.copy_(opacities)
            else:
                opacity_buf[idx_tensor] = opacities
            opacity_buf.requires_grad = False

        return intensities

    def _ensure_orientation_field(self) -> None:
        """Compute and cache gradient field if needed."""
        if self._orientation_grad is not None:
            return
        source = self._orientation_source().to(self.device)

        # Print source statistics for debugging
        if self.verbose:
            print(
                "Computing orientation field from intensity volume "
                f"(range: [{source.min().item():.4f}, {source.max().item():.4f}])"
            )

        grad, mag = compute_gradient_field(
            source,
            sigma_pre=self.orientation_sigma_grad,
            sigma_post=self.orientation_sigma_tensor,
        )
        self._orientation_grad = grad
        self._orientation_mag = mag

        # Print gradient field statistics
        if self.verbose:
            print(
                "Gradient magnitude range: "
                f"[{mag.min().item():.6f}, {mag.max().item():.6f}], mean: {mag.mean().item():.6f}"
            )
            print(
                "Orientation field computed "
                f"(sigma_pre={self.orientation_sigma_grad}, sigma_post={self.orientation_sigma_tensor})"
            )

    def _ensure_structure_field(self) -> None:
        """Compute quaternion/vesselness fields when a mask is available."""
        if self._structure_quat is not None or self.mask_volume is None:
            return

        quat_field, vessel_field = build_structure_field(
            self.mask_volume,
            mask_threshold=self.structure_mask_threshold,
            sigma_pre=self.structure_sigma,
        )
        self._structure_quat = quat_field
        self._structure_vesselness = vessel_field

    def get_quat_for_points(self, xyz_world: Tensor) -> Tuple[Tensor, int]:
        """Return orientation quaternions and fallback count for points."""
        if xyz_world.numel() == 0:
            print("Warning: Empty point set provided for orientation query.")
            return torch.empty(0, 4, device=self.device), 0

        self._ensure_orientation_field()

        # Debug: Print world coordinate statistics
        if self.verbose:
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
        self._ensure_structure_field()
        payload = {
            "gradient": self._orientation_grad,
            "magnitude": self._orientation_mag,
            "origin": self.volume_origin,
            "voxel_size": self.voxel_size,
            "perturb_deg": torch.tensor(
                self.orientation_perturb_deg, device=self.device
            ),
        }

        if self._structure_quat is not None and self._structure_vesselness is not None:
            payload["structure_quat"] = self._structure_quat
            payload["structure_vesselness"] = self._structure_vesselness
        return payload

    def get_structure_for_points(self, xyz_world: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample Hessian-based orientation quaternions and vesselness values at points."""
        if xyz_world.numel() == 0:
            empty = torch.empty(0, 1, device=self.device)
            return torch.empty(0, 4, device=self.device), empty

        self._ensure_structure_field()
        if self._structure_quat is None or self._structure_vesselness is None:
            empty = torch.zeros(xyz_world.shape[0], 1, device=self.device)
            identity = torch.zeros(xyz_world.shape[0], 4, device=self.device)
            identity[:, 0] = 1.0
            return identity, empty

        ijk = world_to_voxel(xyz_world, self.volume_origin, self.voxel_size)
        return sample_structure_field(
            self._structure_quat, self._structure_vesselness, ijk
        )

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
        if intensity_mode in {"sampled", "sampled_mean_covered"}:
            gaussians.reference_volume = self.volume_color
            if self.mask_volume is not None:
                gaussians.reference_mask = self.mask_volume

            needs_resize = (
                not hasattr(gaussians, "intensities")
                or gaussians.intensities.numel() == 0
                or gaussians.intensities.shape[0] != n_points
            )

            is_mean_mode = intensity_mode == "sampled_mean_covered"
            interval = (
                getattr(gaussians, "mean_covered_interval", 1)
                if is_mean_mode
                else self.intensity_update_interval
            )
            interval = max(int(interval), 1)
            update_due = ((self._step - 1) % interval) == 0

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
            else:
                if is_mean_mode:
                    large_mask = gaussians.large_splat_mask(
                        getattr(gaussians, "intensity_large_splat_threshold", 0.0)
                    )
                    large_mask = large_mask.to(device=xyz.device)
                    if active_idx is not None and active_idx.numel() > 0:
                        active_idx_long = active_idx.long().to(device=xyz.device)
                        subset_mask = large_mask[active_idx_long]
                        candidate = active_idx_long[subset_mask]
                        if candidate.numel() == 0:
                            candidate = torch.nonzero(large_mask, as_tuple=False).view(
                                -1
                            )
                    else:
                        candidate = torch.nonzero(large_mask, as_tuple=False).view(-1)

                    if dirty_subset.numel() > 0:
                        indices_for_update = dirty_subset
                    elif update_due and candidate.numel() > 0:
                        if getattr(self, "debug_intensity", False):
                            total_large = int(large_mask.sum().item())
                            print(
                                f"[Intensity] mean-covered refresh updating {int(candidate.numel())}"
                                f" large splats (total tracked: {total_large})."
                            )
                        indices_for_update = candidate
                    elif update_due:
                        indices_for_update = active_idx
                    else:
                        indices_for_update = None
                else:
                    if dirty_subset.numel() > 0:
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

            has_prev = (
                hasattr(gaussians, "intensities")
                and isinstance(gaussians.intensities, torch.Tensor)
                and gaussians.intensities.numel() > 0
            )
            channels = gaussians.intensities.shape[1] if has_prev else 1
            dtype = gaussians.intensities.dtype if has_prev else xyz.dtype
            use_intensities = gaussians.ensure_intensity_buffer(
                n_points,
                channels,
                device=xyz.device,
                dtype=dtype,
                fill_value=0.5,
            )
            gaussians.intensities = use_intensities
            gaussians.intensities.requires_grad = False
            gaussians.volume_min = self.global_intensity_min
            gaussians.volume_max = self.global_intensity_max
            use_intensities = gaussians.intensities
            if (
                self.verbose
                and self._step % 200 == 0
                and use_intensities.numel() > 0
            ):
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
                    gaussians.ensure_intensity_buffer(
                        n_points,
                        1,
                        device=xyz.device,
                        dtype=xyz.dtype,
                        fill_value=0.5,
                    )
                gaussians.intensities.requires_grad = False
                use_intensities = gaussians.intensities

        # Convert gaussians to volume using intensity values (or density for mask supervision)
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
            and intensity_mode in {"sampled", "sampled_mean_covered"}
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

        # Compute loss only inside the mask. Also compute the tight ROI bounding
        # box of the mask and render only that subvolume for speed.
        roi_shape = self.roi_shape
        bounds_min = self.bounds_min.to(xyz.device)
        bounds_max = self.bounds_max.to(xyz.device)

        render_mode = "density" if self.supervision_target == "mask" else "intensity"

        # Convert gaussians to volume ROI (directly uses parameter tensors for gradient flow)
        def _render(points, scales, rotations, opacities, intensities):
            return splat_to_volume(
                points=points,
                point_scales=scales,
                point_rotations=rotations,
                point_opacities=opacities,
                point_intensities=(intensities if render_mode == "intensity" else None),
                volume_shape=roi_shape,
                device=xyz.device,
                active_idx=active_idx,
                grid_bounds=(bounds_min, bounds_max),
                render_mode=render_mode,
                density_scale=float(getattr(self, "density_scale", 1.0)),
            )

        render_inputs = (xyz, scaling, rotation, use_opacity, use_intensities)
        if any(t.requires_grad for t in render_inputs):
            volume_pred_roi = checkpoint(_render, *render_inputs, use_reentrant=False)
        else:
            volume_pred_roi = _render(*render_inputs)

        # Optionally retain grad for debugging
        if getattr(self, "debug", False):
            xyz.retain_grad()

        # Debug if needed
        if hasattr(self, "verbose") and self.verbose:
            print(f"volume_pred requires_grad: {volume_pred_roi.requires_grad}")

        # Slice targets/mask to ROI for loss.
        mask_roi = self.mask_bool_roi.to(device=volume_pred_roi.device)
        if self.supervision_target == "mask":
            target_roi = self.mask_volume_roi.to(device=volume_pred_roi.device)
        else:
            target_roi = self.volume_gt_roi.to(device=volume_pred_roi.device)

        # Store predicted volume for visualization only when needed.
        # (Loss is still computed on the cropped ROI for speed.)
        if getattr(self, "iteration", 0) % 1000 == 0:
            z0, z1, y0, y1, x0, x1 = self.mask_bounds
            full_pred = torch.zeros(
                self.volume_shape,
                device=volume_pred_roi.device,
                dtype=volume_pred_roi.dtype,
            )
            full_pred[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = volume_pred_roi
            self.volume_pred = full_pred.detach().clone()

        main_loss = None
        if self.criterion.loss_type == "mse":
            pred_vals = volume_pred_roi[mask_roi]
            tgt_vals = target_roi[mask_roi]
            diff = pred_vals - tgt_vals
            main_loss = (diff * diff).mean()
        else:
            # For non-MSE objectives, masking by zeroing outside-mask voxels
            # restricts the loss support to the ROI.
            masked_pred = volume_pred_roi * mask_roi.to(dtype=volume_pred_roi.dtype)
            masked_tgt = target_roi * mask_roi.to(dtype=target_roi.dtype)
            main_loss = self.criterion(masked_pred, masked_tgt)

        loss = main_loss

        outside_loss = None
        outside_weight = float(getattr(self, "outside_mask_weight", 0.0))
        if (
            outside_weight > 0.0
            and self.supervision_target == "mask"
        ):
            outside_roi = ~mask_roi
            if outside_roi.any():
                outside_vals = volume_pred_roi[outside_roi]
                outside_loss = (outside_vals * outside_vals).mean()
                loss = loss + outside_weight * outside_loss

        unweighted_loss = loss

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
            self.metrics["volume_loss"] = float(loss.item())
            self.metrics["volume_loss_unweighted"] = float(unweighted_loss.item())
            if outside_loss is not None:
                self.metrics["outside_mask_loss"] = float(outside_loss.item())
            else:
                self.metrics["outside_mask_loss"] = 0.0
            if self.criterion.loss_type == 'dice':
                dice_score = 1.0 - float(main_loss.item())
                self.metrics["dice_score"] = float(dice_score)

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
