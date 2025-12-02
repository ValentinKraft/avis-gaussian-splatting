#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from collections import deque
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from typing import Optional, Dict, Tuple, List, Union, Any

from gaussian_splatting.utils.orientation_field import (
    gather_rotation_from_gradient,
    random_quat_perturb,
    rotmat_to_quat,
    world_to_grid,
    world_to_voxel,
)

try:
    from gaussian_rasterization import SparseGaussianAdam
except:
    pass


# Define constants at the module level for better maintainability
SH_C0 = 0.28209479177387814  # Value of Y_0^0 (first spherical harmonic)
SH_SCALE = 1.77  # Approximate 1/(2*SH_C0)


class GaussianModel:
    """
    Represents a 3D Gaussian Splatting model with trainable parameters.
    Supports both RGB and volume-only training modes.
    """

    def __init__(self, sh_degree: int, optimizer_type: str = "default"):
        """
        Initialize a new Gaussian model with empty tensors.

        Args:
            sh_degree: Maximum spherical harmonics degree
            optimizer_type: Type of optimizer ("default" or "adam_as_sgd")
        """
        # Core parameters
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.optimizer_type = optimizer_type

        # Trainable parameters (initialized as empty tensors)
        self._xyz = torch.empty(0)  # Point positions [3, N] or [N, 3]
        self._scaling = torch.empty(0)  # Log-scale parameters [N, 3]
        self._rotation = torch.empty(0)  # Rotation quaternions [N, 4]
        self._opacity = torch.empty(0)  # Log-opacity values [N, 1]
        self._features_dc = torch.empty(0)  # DC features (0th order SH) [N, 1, 3]
        self._features_rest = torch.empty(0)  # Higher-order SH features [N, ?, 3]

        # Store initial scaling values for maximum size constraint
        self._initial_xyz = torch.empty(0)  # Frozen xyz reference [3, N]
        self._initial_scaling = torch.empty(0)  # Initial log-scale values [N, 3]
        self.max_scale_factor = 2.0  # Maximum allowed scale compared to initial scale
        self.max_position_displacement_scale = 2.0  # Max displacement multiplier

        # Runtime state
        self.max_radii2D = torch.empty(0)  # Maximum 2D radii for each point
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # Volume-based rendering attributes
        self.intensities = torch.empty(0)  # Raw intensity values [N, 1]
        self.opacities = torch.empty(0)  # Raw opacity values [N, 1]
        self.volume_min = 0.0  # Global minimum intensity value
        self.volume_max = 1.0  # Global maximum intensity value
        self.reference_volume = None  # Reference intensity volume
        self.reference_mask = None  # Reference opacity mask

        # Orientation metadata populated during initialization for densification reuse
        self.orientation_field = None
        self.orientation_fallback_stats = {"clone": 0, "split": 0, "hole_fill": 0}

        # Intensity handling mode and cached parameter snapshots
        self.intensity_mode = "learned"
        self._prev_xyz = None
        self._prev_scaling = None
        self._prev_rotation = None
        self.intensity_color_divisor = 1.0
        self.intensity_large_splat_threshold = 0.03
        self.mean_covered_radius = 2.5
        self.mean_covered_interval = 10

        # --- Adaptive densification tracking ---
        self._scale_history = deque(maxlen=32)
        self._position_history = deque(maxlen=32)
        self._density_cache = None
        self._coverage_state = None
        self._adaptive_lr_state = {
            "scale_boost_active": 0,
            "scale_lr_multiplier": 1.0,
            "xyz_lr_multiplier": 1.0,
            "cooldown": 0,
        }
        self._latest_iteration = 0
        self._scale_boost_window = 16
        self._scale_stall_epsilon = 7e-5
        self._scale_boost_factor = 1.20
        self._scale_boost_duration = 6
        self._scale_cooldown = 40
        self._max_scale_factor_base = self.max_scale_factor
        self.low_density_threshold = 4.0
        self.target_coverage = 0.78
        self.density_radius_factor = 3.0
        self._density_cap = 64.0
        self.density_update_interval = 10
        self.dynamics_log_interval = 50
        self.last_densify_counts = {"split": 0, "clone": 0, "hole_fill": 0}
        self.training_dynamics_log = []
        self._last_density_iteration = -1
        self._low_density_mask = None
        self._hole_fill_fraction = 0.05
        self._max_memory_bytes = None
        self.vessel_axial_scale = 2.2
        self.vessel_radial_scale = 0.55
        self.structure_gradient_boost = 0.0
        self.structure_gradient_exponent = 1.0
        self.structure_gradient_threshold = 0.1
        self._base_scaling_lr = None
        self._base_xyz_lr = None
        self._base_rotation_lr = None
        self.position_stall_threshold = 3.0e-4
        self.xyz_boost_factor = 1.15
        self._xyz_boost_active = 0
        self._xyz_boost_duration = 5
        self.densify_grad_percentile = 0.85
        self.scaling_constraint_warmup_iters = 0
        self.scaling_constraint_relaxation = 1.0
        self.early_stats_window = 256
        self._early_iteration_log: List[Dict[str, float]] = []

        # Set up activation functions
        self._setup_activation_functions()

    def _setup_activation_functions(self):
        """Set up activation functions for model parameters."""

        # Function to build covariance matrices from scaling and rotation
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm * scaling_modifier

        # Assign activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def _verify_gradient_requirements(self):
        """
        Verify that all parameters have requires_grad=True.
        This helps catch issues where parameters are not properly set up for optimization.
        """
        # Check core parameters
        if not self._xyz.requires_grad:
            print(
                "WARNING: Position parameters do not require gradients. Setting requires_grad=True."
            )
            self._xyz.requires_grad_(True)

        if self._scaling.numel() > 0 and not self._scaling.requires_grad:
            print(
                "WARNING: Scaling parameters do not require gradients. Setting requires_grad=True."
            )
            self._scaling.requires_grad_(True)

        if self._rotation.numel() > 0 and not self._rotation.requires_grad:
            print(
                "WARNING: Rotation parameters do not require gradients. Setting requires_grad=True."
            )
            self._rotation.requires_grad_(True)

        if (
            self._opacity.numel() > 0
            and not self._opacity.requires_grad
            and not self._mask_opacity_active()
        ):
            print(
                "WARNING: Opacity parameters do not require gradients. Setting requires_grad=True."
            )
            self._opacity.requires_grad_(True)

        # Ensure intensities remain non-learnable when present
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            if isinstance(self.intensities, nn.Parameter):
                self.intensities = self.intensities.detach()
            self.intensities.requires_grad = False

    # ===== Intensity handling utilities =====

    def set_intensity_mode(self, mode: str) -> None:
        """Record which intensity mode is currently active."""
        self.intensity_mode = mode

    def set_intensity_color_divisor(self, divisor: float) -> None:
        """Configure manual brightness divisor for intensity-derived colors."""
        safe_divisor = max(float(divisor), 1e-8)
        if safe_divisor != self.intensity_color_divisor:
            print(
                f"Applying intensity color divisor {safe_divisor:.6f} (was {self.intensity_color_divisor:.6f})."
            )
        self.intensity_color_divisor = safe_divisor

    def _mask_opacity_active(self) -> bool:
        """Return True when a mask-derived opacity buffer is populated."""
        return (
            hasattr(self, "opacities")
            and self.opacities is not None
            and self.opacities.numel() > 0
        )

    def _optimizer_has_group(self, name: str) -> bool:
        """Return True when the optimizer tracks a parameter group with the given name."""
        if self.optimizer is None:
            return False
        return any(group.get("name") == name for group in self.optimizer.param_groups)

    def configure_mean_covered_sampling(
        self,
        *,
        large_splat_threshold: float,
        radius_scale: float,
        update_interval: int,
    ) -> None:
        """
        Set thresholds and cadence for mean-covered-voxel intensity sampling.

        Args:
            large_splat_threshold: Minimum max-axis scale to treat a splat as large.
            radius_scale: Multiplier applied to per-axis scales when gathering voxels.
            update_interval: Iteration cadence used when refreshing cached samples.
        """
        self.intensity_large_splat_threshold = max(float(large_splat_threshold), 0.0)
        self.mean_covered_radius = max(float(radius_scale), 0.0)
        self.mean_covered_interval = max(int(update_interval), 1)

    def large_splat_mask(self, threshold: Optional[float] = None) -> torch.Tensor:
        """Return boolean mask identifying splats exceeding the large-splat threshold."""
        if self._xyz.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self._xyz.device)

        current_threshold = (
            self.intensity_large_splat_threshold
            if threshold is None
            else float(threshold)
        )
        scales = self.get_scaling.detach()
        if scales.dim() != 2:
            raise ValueError("Scaling tensor expected to have shape [N,3] or [3,N].")
        if scales.shape[1] == 3:
            scales_n3 = scales
        elif scales.shape[0] == 3:
            scales_n3 = scales.transpose(0, 1)
        else:
            raise ValueError("Scaling tensor must provide three components per splat.")

        metric = scales_n3.max(dim=1).values
        threshold_val = max(float(current_threshold), 0.0)
        return metric >= threshold_val

    def large_splat_indices(self, threshold: Optional[float] = None) -> torch.Tensor:
        """Return indices of splats considered large under the configured threshold."""
        mask = self.large_splat_mask(threshold)
        if mask.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=mask.device)
        return torch.nonzero(mask, as_tuple=False).view(-1)

    def _ensure_prev_buffers(self) -> None:
        """Ensure previous-parameter buffers exist and match current shapes."""
        if self._xyz.numel() == 0:
            return
        if self._prev_xyz is None or self._prev_xyz.shape != self._xyz.shape:
            self._prev_xyz = self._xyz.detach().clone()
        if (
            self._prev_scaling is None
            or self._prev_scaling.shape != self._scaling.shape
        ):
            self._prev_scaling = self._scaling.detach().clone()
        if (
            self._prev_rotation is None
            or self._prev_rotation.shape != self._rotation.shape
        ):
            self._prev_rotation = self._rotation.detach().clone()

    def _ensure_initial_position_buffer(self) -> None:
        """Ensure the frozen xyz snapshot exists and matches current topology."""
        if self._xyz.numel() == 0:
            self._initial_xyz = torch.empty_like(self._xyz)
            return
        if self._initial_xyz.numel() == 0 or self._initial_xyz.shape != self._xyz.shape:
            self._initial_xyz = self._xyz.detach().clone()

    @torch.no_grad()
    def _reset_prev_buffers(self) -> None:
        """Fully refresh snapshot buffers after topology changes."""
        if self._xyz.numel() == 0:
            self._prev_xyz = None
            self._prev_scaling = None
            self._prev_rotation = None
            return
        self._prev_xyz = self._xyz.detach().clone()
        self._prev_scaling = self._scaling.detach().clone()
        self._prev_rotation = self._rotation.detach().clone()

    @torch.no_grad()
    def snapshot_params_for_dirty_check(
        self, indices: Optional[torch.Tensor] = None
    ) -> None:
        """Store current xyz/scaling/rotation values for future dirty checks."""
        self._ensure_prev_buffers()
        if indices is None or self._xyz.numel() == 0:
            if self._prev_xyz is not None:
                self._prev_xyz.copy_(self._xyz.detach())
                self._prev_scaling.copy_(self._scaling.detach())
                self._prev_rotation.copy_(self._rotation.detach())
            return

        idx = indices.long().unique()
        if idx.numel() == 0:
            return
        self._prev_xyz[:, idx] = self._xyz[:, idx].detach()
        self._prev_scaling[idx] = self._scaling[idx].detach()
        self._prev_rotation[idx] = self._rotation[idx].detach()

    @torch.no_grad()
    def dirty_indices(
        self,
        indices: Optional[torch.Tensor],
        thr_xyz: float,
        thr_log_scale: float,
        thr_rot_rad: float,
    ) -> torch.Tensor:
        """Return subset indices whose parameters changed beyond thresholds."""
        if self._xyz.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self._xyz.device)

        self._ensure_prev_buffers()
        device = self._xyz.device
        if indices is None:
            idx = torch.arange(self._xyz.shape[1], device=device)
        else:
            idx = indices.long().unique()

        if idx.numel() == 0:
            return idx

        # Position delta in world units
        delta_xyz = (self._xyz[:, idx] - self._prev_xyz[:, idx]).norm(dim=0)
        dirty_xyz = delta_xyz > thr_xyz

        # Scaling deltas in log-space
        delta_scale = (self._scaling[idx] - self._prev_scaling[idx]).norm(dim=1)
        dirty_scale = delta_scale > thr_log_scale

        # Rotation delta via quaternion angle
        q_curr = self._rotation[idx]
        q_prev = self._prev_rotation[idx]
        dots = (q_curr * q_prev).sum(dim=1).clamp(-1.0, 1.0).abs()
        angles = 2.0 * torch.arccos(dots)
        dirty_rot = angles > thr_rot_rad

        mask = dirty_xyz | dirty_scale | dirty_rot
        return idx[mask]

    @torch.no_grad()
    def update_sampled_intensities(
        self,
        sampler,
        indices: Optional[torch.Tensor] = None,
    ) -> int:
        """Refresh cached intensities for the requested subset of splats."""
        if self.intensity_mode not in {"sampled", "sampled_mean_covered"}:
            return 0

        if self._xyz.numel() == 0:
            return 0

        device = self._xyz.device
        if indices is None:
            idx = torch.arange(self._xyz.shape[1], device=device)
        else:
            idx = indices.long().unique()

        if idx.numel() == 0:
            return 0

        sampled = sampler(self, idx if indices is not None else None)
        if sampled is None or sampled.numel() == 0:
            return 0

        sampled = sampled.detach()
        if sampled.dim() != 2:
            raise ValueError("Sampler must return [N, C] intensity values.")

        total_points = self._xyz.shape[1]
        channels = sampled.shape[1]
        needs_realloc = (
            not hasattr(self, "intensities")
            or self.intensities is None
            or self.intensities.numel() == 0
            or self.intensities.shape[0] != total_points
            or self.intensities.shape[1] != channels
        )

        if needs_realloc:
            new_buf = torch.zeros(
                (total_points, channels),
                device=device,
                dtype=sampled.dtype,
            )
            if (
                hasattr(self, "intensities")
                and self.intensities is not None
                and self.intensities.numel() > 0
            ):
                copy_rows = min(self.intensities.shape[0], total_points)
                copy_cols = min(self.intensities.shape[1], channels)
                new_buf[:copy_rows, :copy_cols] = self.intensities[
                    :copy_rows, :copy_cols
                ]
            self.intensities = new_buf

        self.intensities[idx] = sampled
        self.snapshot_params_for_dirty_check(idx)
        return idx.numel()

    @property
    def get_xyz(self) -> torch.Tensor:
        """Expose raw xyz tensor for compatibility with legacy call-sites."""
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        """Get point scaling parameters (converted from log-space) with maximum size constraint."""
        if self._scaling.numel() == 0:
            return self._scaling

        log_scales = self._scaling
        if (
            self._initial_scaling is not None
            and self._initial_scaling.numel() == self._scaling.numel()
        ):
            max_log_scaling = self._initial_scaling + torch.log(
                torch.as_tensor(
                    self.max_scale_factor,
                    device=self._scaling.device,
                    dtype=self._scaling.dtype,
                )
            )
            log_scales = torch.minimum(log_scales, max_log_scaling)

        return self.scaling_activation(log_scales)

    @property
    def get_rotation(self) -> torch.Tensor:
        """Get normalized rotation quaternions."""
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self) -> torch.Tensor:
        """Get point opacity values (converted from log-space)."""
        if self._mask_opacity_active():
            return self.opacities
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        """Get combined features (DC and rest)."""
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_features_dc(self) -> torch.Tensor:
        """Get DC features (0th order spherical harmonics)."""
        return self._features_dc

    @property
    def get_features_rest(self) -> torch.Tensor:
        """Get higher-order features."""
        return self._features_rest

    def get_covariance(self, scaling_modifier: float = 1) -> torch.Tensor:
        """
        Compute covariance matrices from scaling and rotation parameters.

        Args:
            scaling_modifier: Multiplier for scaling values

        Returns:
            Covariance matrices
        """
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def enforce_scaling_constraint(self, iteration: Optional[int] = None) -> None:
        """Clamp log-scale values with optional warmup relaxation."""
        if self._initial_scaling.numel() == 0 or self._scaling.numel() == 0:
            return

        with torch.no_grad():
            if iteration is None:
                iteration = self._latest_iteration
            warmup_iters = max(int(self.scaling_constraint_warmup_iters), 0)
            relax = max(float(self.scaling_constraint_relaxation), 1.0)
            warmup_multiplier = 1.0
            if warmup_iters > 0:
                progress = min(max(float(iteration), 0.0) / warmup_iters, 1.0)
                warmup_multiplier = 1.0 + (relax - 1.0) * (1.0 - progress)
            effective_factor = float(self.max_scale_factor * warmup_multiplier)
            max_log_scaling = self._initial_scaling + torch.log(
                torch.as_tensor(
                    effective_factor,
                    device=self._scaling.device,
                    dtype=self._scaling.dtype,
                )
            )
            self._scaling.copy_(torch.minimum(self._scaling, max_log_scaling))
            if (
                self._prev_scaling is not None
                and self._prev_scaling.shape == self._scaling.shape
            ):
                self._prev_scaling.copy_(self._scaling.detach())

    def enforce_position_displacement_constraint(self) -> None:
        """Clamp point displacement relative to the stored initialization positions."""
        if (
            self._xyz.numel() == 0
            or self._initial_xyz.numel() == 0
            or self.max_position_displacement_scale <= 0.0
        ):
            return

        if self._initial_xyz.shape != self._xyz.shape:
            self._ensure_initial_position_buffer()
            if self._initial_xyz.shape != self._xyz.shape:
                return

        with torch.no_grad():
            scales = self.get_scaling
            if scales.numel() == 0:
                return
            if scales.shape[1] == 3:
                scale_axes = scales
            elif scales.shape[0] == 3:
                scale_axes = scales.transpose(0, 1)
            else:
                raise ValueError(
                    "Scaling tensor must provide three components per Gaussian."
                )

            max_axis = torch.max(scale_axes, dim=1).values
            allowed = max_axis * float(self.max_position_displacement_scale)
            delta = self._xyz - self._initial_xyz
            delta_norm = torch.linalg.norm(delta, dim=0)
            mask = delta_norm > allowed
            if not mask.any():
                return

            safe_allowed = allowed[mask].clamp_min(0.0)
            norm = delta_norm[mask].clamp_min(1e-12)
            scale = safe_allowed / norm
            delta[:, mask] = delta[:, mask] * scale
            self._xyz[:, mask] = self._initial_xyz[:, mask] + delta[:, mask]
            if self._prev_xyz is not None and self._prev_xyz.shape == self._xyz.shape:
                self._prev_xyz[:, mask] = self._xyz[:, mask].detach()

    def oneupSHdegree(self) -> None:
        """Increase spherical harmonics degree by one when below the maximum."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ===== Core model functions =====

    def capture(self) -> tuple:
        """
        Capture the current state of the model for saving.

        Returns:
            Tuple containing all model parameters and state
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.intensities,
            self.opacities,
            self._initial_scaling,
            self._initial_xyz,
        )

    def restore(self, model_args: tuple, training_args: Any):
        """
        Restore the model from saved state.

        Args:
            model_args: Tuple containing model parameters from capture()
            training_args: Training arguments for initializing optimizer
        """
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            *extra_args,
        ) = model_args

        device_xyz = (
            self._xyz.device
            if isinstance(self._xyz, torch.Tensor) and self._xyz.numel() > 0
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        device_scale = (
            self._scaling.device
            if isinstance(self._scaling, torch.Tensor) and self._scaling.numel() > 0
            else device_xyz
        )

        extras: list[Any] = list(extra_args)
        while len(extras) < 4:
            extras.append(None)

        self.intensities = (
            extras[0]
            if isinstance(extras[0], torch.Tensor)
            else torch.empty(0, device=device_xyz)
        )
        self.opacities = (
            extras[1]
            if isinstance(extras[1], torch.Tensor)
            else torch.empty(0, device=device_xyz)
        )
        self._initial_scaling = (
            extras[2]
            if isinstance(extras[2], torch.Tensor)
            else torch.empty(0, device=device_scale)
        )
        self._initial_xyz = (
            extras[3]
            if isinstance(extras[3], torch.Tensor)
            else torch.empty(0, device=device_xyz)
        )

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self._ensure_initial_position_buffer()

    def training_setup(self, training_args: Any):
        """
        Set up optimizer and learning rate schedules for training.

        Args:
            training_args: Training arguments for optimizer setup
        """
        self.percent_dense = training_args.percent_dense

        # --- Configure adaptive densification params ---
        self.dynamics_log_interval = getattr(
            training_args, "dynamics_log_interval", self.dynamics_log_interval
        )
        self.density_update_interval = getattr(
            training_args, "density_update_interval", self.density_update_interval
        )
        self._scale_boost_window = getattr(
            training_args, "scale_boost_window", self._scale_boost_window
        )
        self._scale_stall_epsilon = getattr(
            training_args, "scale_stall_epsilon", self._scale_stall_epsilon
        )
        self._scale_boost_factor = getattr(
            training_args, "scale_boost_factor", self._scale_boost_factor
        )
        self._scale_boost_duration = getattr(
            training_args, "scale_boost_duration", self._scale_boost_duration
        )
        self._scale_cooldown = getattr(
            training_args, "scale_cooldown", self._scale_cooldown
        )
        self.low_density_threshold = getattr(
            training_args, "low_density_threshold", self.low_density_threshold
        )
        self.target_coverage = getattr(
            training_args, "target_coverage", self.target_coverage
        )
        self.density_radius_factor = getattr(
            training_args, "density_radius_factor", self.density_radius_factor
        )
        self._density_cap = getattr(training_args, "density_cap", self._density_cap)
        self._hole_fill_fraction = getattr(
            training_args, "hole_fill_fraction", self._hole_fill_fraction
        )
        self.vessel_axial_scale = getattr(
            training_args, "vessel_axial_scale", self.vessel_axial_scale
        )
        self.vessel_radial_scale = getattr(
            training_args, "vessel_radial_scale", self.vessel_radial_scale
        )
        self.structure_gradient_boost = getattr(
            training_args,
            "structure_gradient_boost",
            self.structure_gradient_boost,
        )
        self.structure_gradient_exponent = max(
            0.5,
            getattr(
                training_args,
                "structure_gradient_exponent",
                self.structure_gradient_exponent,
            ),
        )
        self.structure_gradient_threshold = max(
            0.0,
            getattr(
                training_args,
                "structure_gradient_threshold",
                self.structure_gradient_threshold,
            ),
        )
        self._max_memory_bytes = getattr(
            training_args, "densify_memory_budget_bytes", self._max_memory_bytes
        )
        self.densify_grad_percentile = getattr(
            training_args, "densify_grad_percentile", self.densify_grad_percentile
        )

        self._scale_history.clear()
        self._position_history.clear()
        self.training_dynamics_log.clear()
        self._early_iteration_log.clear()
        self._adaptive_lr_state.update(
            {
                "scale_boost_active": 0,
                "scale_lr_multiplier": 1.0,
                "xyz_lr_multiplier": 1.0,
                "cooldown": 0,
            }
        )
        self._base_scaling_lr = None
        self._base_xyz_lr = None
        self._base_rotation_lr = None
        self._xyz_boost_active = 0
        self._latest_iteration = 0
        self.scaling_constraint_warmup_iters = getattr(
            training_args,
            "scaling_constraint_warmup_iters",
            self.scaling_constraint_warmup_iters,
        )
        self.scaling_constraint_relaxation = max(
            1.0,
            getattr(
                training_args,
                "scaling_constraint_relaxation",
                self.scaling_constraint_relaxation,
            ),
        )
        self.early_stats_window = max(
            0,
            getattr(training_args, "early_stats_window", self.early_stats_window),
        )
        self._max_scale_factor_base = self.max_scale_factor
        self._density_cache = None
        self._coverage_state = None
        self._last_density_iteration = -1
        # _xyz has shape [3, N], so use shape[1] to get number of points
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[1], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[1], 1), device="cuda")
        self._reset_prev_buffers()

        # Initialize empty optimizer parameters list
        optimizer_params = self._create_optimizer_param_groups(training_args)

        # Create the optimizer
        self.optimizer = torch.optim.Adam(optimizer_params, lr=0.0, eps=1e-15)
        if self.optimizer_type == "adam_as_sgd":
            self.optimizer = SparseGaussianAdam(optimizer_params, lr=0.0, eps=1e-15)

        # Set up position learning rate schedule
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def _create_optimizer_param_groups(
        self, training_args: Any
    ) -> List[Dict[str, Any]]:
        """
        Create parameter groups for the optimizer.

        Args:
            training_args: Training arguments with learning rates

        Returns:
            List of parameter dictionaries for optimizer
        """
        param_groups = []

        allow_feature_params = getattr(self, "intensity_mode", "learned") not in {
            "sampled",
            "sampled_mean_covered",
        }

        # Only add feature parameters if enabled and they exist
        if (
            allow_feature_params
            and self._features_dc is not None
            and self._features_dc.numel() > 0
        ):
            param_groups.append(
                {
                    "params": [self._features_dc],
                    "lr": training_args.feature_lr,
                    "name": "f_dc",
                }
            )

        if (
            allow_feature_params
            and self._features_rest is not None
            and self._features_rest.numel() > 0
        ):
            param_groups.append(
                {
                    "params": [self._features_rest],
                    "lr": training_args.feature_lr / 20.0,
                    "name": "f_rest",
                }
            )

        # Add position, scaling and rotation parameters
        param_groups.extend(
            [
                {
                    "params": [self._xyz],
                    "lr": training_args.position_lr_init * self.spatial_lr_scale,
                    "name": "xyz",
                },
                {
                    "params": [self._scaling],
                    "lr": training_args.scaling_lr,
                    "name": "scaling",
                },
                {
                    "params": [self._rotation],
                    "lr": training_args.rotation_lr,
                    "name": "rotation",
                },
            ]
        )

        if not self._mask_opacity_active() and self._opacity.numel() > 0:
            param_groups.append(
                {
                    "params": [self._opacity],
                    "lr": training_args.opacity_lr,
                    "name": "opacity",
                }
            )

        return param_groups

    def update_learning_rate(self, iteration: int) -> float:
        """
        Update learning rates based on current iteration.

        Args:
            iteration: Current training iteration

        Returns:
            Current position learning rate
        """
        # --- Apply adaptive learning rate scheduling ---
        self._latest_iteration = iteration
        xyz_group = None
        scaling_group = None
        rotation_group = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                xyz_group = param_group
            elif param_group["name"] == "scaling":
                scaling_group = param_group
            elif param_group["name"] == "rotation":
                rotation_group = param_group

        current_lr = 0.0
        if xyz_group is not None:
            base_lr = self.xyz_scheduler_args(iteration)
            current_lr = self._apply_adaptive_learning_rates(
                iteration, xyz_group, scaling_group, rotation_group, base_lr
            )

        return current_lr

    # === Adaptive learning-rate and density utilities ===

    def _apply_adaptive_learning_rates(
        self,
        iteration: int,
        xyz_group: Dict[str, Any],
        scaling_group: Optional[Dict[str, Any]],
        rotation_group: Optional[Dict[str, Any]],
        base_lr: float,
    ) -> float:
        """Adjust xyz/scaling learning rates based on recent dynamics."""
        if self._xyz.numel() == 0:
            xyz_group["lr"] = base_lr
            if scaling_group is not None:
                scaling_group["lr"] = scaling_group["lr"]
            if rotation_group is not None and self._base_rotation_lr is not None:
                rotation_group["lr"] = self._base_rotation_lr
            return base_lr

        if self._base_xyz_lr is None:
            self._base_xyz_lr = base_lr
        if scaling_group is not None and self._base_scaling_lr is None:
            self._base_scaling_lr = scaling_group["lr"]
        if rotation_group is not None and self._base_rotation_lr is None:
            self._base_rotation_lr = rotation_group["lr"]

        scale_delta, position_change = self._record_training_iteration_stats(iteration)
        density_info = self._maybe_update_density_cache(iteration)
        low_density_mask = (
            density_info.get("low_density_mask") if density_info else None
        )
        low_density_fraction = (
            float(low_density_mask.float().mean().item())
            if low_density_mask is not None and low_density_mask.numel() > 0
            else 0.0
        )

        state = self._adaptive_lr_state
        if state["cooldown"] > 0:
            state["cooldown"] -= 1

        scale_stalled = False
        if len(self._scale_history) >= self._scale_boost_window:
            recent = [
                abs(entry["delta"])
                for entry in list(self._scale_history)[-self._scale_boost_window :]
            ]
            scale_stalled = float(np.mean(recent)) < self._scale_stall_epsilon

        if (
            scale_stalled
            and low_density_fraction > 0.02
            and state["scale_boost_active"] == 0
            and state["cooldown"] == 0
        ):
            state["scale_boost_active"] = self._scale_boost_duration
            state["scale_lr_multiplier"] = self._scale_boost_factor
            state["cooldown"] = self._scale_cooldown

        if state["scale_boost_active"] > 0:
            state["scale_boost_active"] -= 1
            scale_multiplier = state.get("scale_lr_multiplier", 1.0)
        else:
            scale_multiplier = 1.0
            state["scale_lr_multiplier"] = 1.0

        if scaling_group is not None and self._base_scaling_lr is not None:
            scaling_group["lr"] = float(self._base_scaling_lr * scale_multiplier)

        if (
            scale_multiplier > 1.0
            and low_density_mask is not None
            and low_density_mask.any()
        ):
            self.max_scale_factor = self._max_scale_factor_base * scale_multiplier
        else:
            self.max_scale_factor = self._max_scale_factor_base

        # XYZ warm restart logic based on movement stagnation
        mean_pos_change = position_change
        if len(self._position_history) >= self._scale_boost_window:
            recent_pos = [
                entry["mean"]
                for entry in list(self._position_history)[-self._scale_boost_window :]
            ]
            mean_pos_change = float(np.mean(recent_pos))

        if (
            mean_pos_change < self.position_stall_threshold
            and low_density_fraction > 0.02
            and self._xyz_boost_active == 0
        ):
            self._xyz_boost_active = self._xyz_boost_duration

        xyz_multiplier = self.xyz_boost_factor if self._xyz_boost_active > 0 else 1.0
        if self._xyz_boost_active > 0:
            self._xyz_boost_active -= 1

        xyz_group["lr"] = float(self._base_xyz_lr * xyz_multiplier)

        if rotation_group is not None and self._base_rotation_lr is not None:
            rotation_multiplier = xyz_multiplier if xyz_multiplier > 1.0 else 1.0
            rotation_group["lr"] = float(self._base_rotation_lr * rotation_multiplier)

        if iteration % max(1, self.dynamics_log_interval) == 0:
            self._log_training_dynamics(
                iteration,
                scale_delta,
                mean_pos_change,
                low_density_fraction,
                scale_multiplier,
                xyz_multiplier,
                density_info,
            )

        return xyz_group["lr"]

    def _record_training_iteration_stats(self, iteration: int) -> Tuple[float, float]:
        """Track recent scaling and position deltas for adaptive logic."""
        if self._xyz.numel() == 0:
            return 0.0, 0.0

        with torch.no_grad():
            scales = self.get_scaling
            mean_scale = float(scales.mean().item())
            prev_scale_mean = (
                self._scale_history[-1]["mean"] if self._scale_history else mean_scale
            )
            delta_scale = mean_scale - prev_scale_mean
            self._scale_history.append(
                {"iter": iteration, "mean": mean_scale, "delta": delta_scale}
            )

            self._ensure_prev_buffers()
            if self._prev_xyz is not None and self._prev_xyz.shape == self._xyz.shape:
                pos_delta = float(
                    torch.norm(self._xyz - self._prev_xyz, dim=0).mean().item()
                )
            else:
                pos_delta = 0.0

            prev_pos_mean = (
                self._position_history[-1]["mean"]
                if self._position_history
                else pos_delta
            )
            self._position_history.append(
                {
                    "iter": iteration,
                    "mean": pos_delta,
                    "delta": pos_delta - prev_pos_mean,
                }
            )

            # Keep previous buffers in sync for next iteration
            if self._prev_xyz is not None:
                self._prev_xyz.copy_(self._xyz.detach())
            if self._prev_scaling is not None:
                self._prev_scaling.copy_(self._scaling.detach())
            if self._prev_rotation is not None:
                self._prev_rotation.copy_(self._rotation.detach())

            self._maybe_record_early_iteration_stats(
                iteration, scales, pos_delta, delta_scale
            )

        return delta_scale, pos_delta

    def _maybe_record_early_iteration_stats(
        self,
        iteration: int,
        scales: torch.Tensor,
        position_delta: float,
        scale_delta: float,
    ) -> None:
        """Capture lightweight stats for the earliest iterations for debugging."""
        window = max(int(self.early_stats_window), 0)
        if window == 0 or iteration > window:
            return

        stats: Dict[str, float] = {
            "iter": float(iteration),
            "scale_mean": float(scales.mean().item()),
            "scale_std": float(scales.std().item()),
            "scale_delta": float(scale_delta),
            "position_delta": float(position_delta),
        }

        rotation = self.get_rotation
        if rotation.numel() > 0:
            with torch.no_grad():
                w = rotation[:, 0].clamp(-1.0, 1.0)
                angles = 2.0 * torch.arccos(w.abs())
                stats["rotation_angle_mean"] = float(angles.mean().item())
                stats["rotation_angle_std"] = float(angles.std().item())

        self._early_iteration_log.append(stats)

    def get_early_iteration_stats(self) -> List[Dict[str, float]]:
        """Return the collected early-iteration statistics."""
        return list(self._early_iteration_log)

    def latest_early_iteration_stats(self) -> Optional[Dict[str, float]]:
        """Return the most recent early-iteration record, if available."""
        if not self._early_iteration_log:
            return None
        return self._early_iteration_log[-1]

    def _maybe_update_density_cache(
        self, iteration: Optional[int]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Refresh cached density metrics on a fixed cadence."""
        if self._xyz.numel() == 0:
            return None

        if iteration is None:
            force_refresh = True
        else:
            force_refresh = False

        if (
            not force_refresh
            and self._density_cache is not None
            and iteration - self._last_density_iteration < self.density_update_interval
        ):
            return self._density_cache

        if iteration is not None:
            self._last_density_iteration = iteration
        else:
            self._last_density_iteration = max(self._last_density_iteration, 0) + 1

        xyz = self.get_xyz
        scales = self.get_scaling
        mean_scale = float(scales.mean().item())
        radius = max(mean_scale * self.density_radius_factor, 1e-5)

        density = gaussian_compute_local_density(xyz, radius, self._density_cap)
        low_density_mask = density < self.low_density_threshold

        coverage = gaussian_compute_coverage_grid(xyz)
        hole_mask = coverage["occupancy"].view(-1) == 0
        hole_voxels = int(hole_mask.sum().item())
        total_voxels = int(coverage["occupancy"].numel())
        coverage_ratio = 1.0 - (hole_voxels / max(total_voxels, 1))
        self._low_density_mask = low_density_mask
        self._coverage_state = coverage

        self._density_cache = {
            "density": density,
            "radius": torch.tensor(radius, device=xyz.device),
            "low_density_mask": low_density_mask,
            "coverage_ratio": torch.tensor(coverage_ratio, device=xyz.device),
            "hole_voxels": torch.tensor(hole_voxels, device=xyz.device),
        }
        return self._density_cache

    def _log_training_dynamics(
        self,
        iteration: int,
        scale_delta: float,
        position_change: float,
        low_density_fraction: float,
        scale_multiplier: float,
        xyz_multiplier: float,
        density_info: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        """Append a summarized snapshot for later inspection."""
        entry: Dict[str, float] = {
            "iter": float(iteration),
            "scale_delta": float(scale_delta),
            "position_change": float(position_change),
            "low_density_fraction": float(low_density_fraction),
            "scale_lr_multiplier": float(scale_multiplier),
            "xyz_lr_multiplier": float(xyz_multiplier),
        }
        entry["split_added"] = float(self.last_densify_counts.get("split", 0))
        entry["clone_added"] = float(self.last_densify_counts.get("clone", 0))
        entry["hole_fill_added"] = float(self.last_densify_counts.get("hole_fill", 0))
        if density_info is not None:
            density_vals = density_info["density"].detach()
            if density_vals.numel() > 0:
                entry["density_p10"] = float(torch.quantile(density_vals, 0.1).item())
                entry["density_p50"] = float(torch.quantile(density_vals, 0.5).item())
            entry["hole_voxel_count"] = float(density_info["hole_voxels"].item())
            entry["coverage_ratio"] = float(density_info["coverage_ratio"].item())

        self.training_dynamics_log.append(entry)

    def summarize_training_dynamics(self) -> List[Dict[str, float]]:
        """Return the collected adaptive statistics."""
        return list(self.training_dynamics_log)

    # ===== Initialization methods =====

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        cam_infos: int,
        spatial_lr_scale: float,
        source_path: str = None,
    ):
        """
        Initialize Gaussian model from point cloud data.

        Args:
            pcd: Point cloud with positions and colors
            cam_infos: Number of camera views
            spatial_lr_scale: Scale factor for position learning rate
            source_path: Optional source data path
        """
        self.spatial_lr_scale = spatial_lr_scale

        # Convert point cloud data to tensors
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        num_points = fused_point_cloud.shape[0]
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # Initialize features tensor
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at initialization: {num_points}")

        # Calculate initial scales based on point distances
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Initialize rotations to identity quaternions
        rots = torch.zeros((num_points, 4), device="cuda")
        rots[:, 0] = 1  # w=1, x,y,z=0 for identity rotation

        # Initialize opacity values
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((num_points, 1), dtype=torch.float, device="cuda")
        )

        # Create parameter tensors
        xyz_param = fused_point_cloud.transpose(0, 1).contiguous()
        self._xyz = nn.Parameter(xyz_param.requires_grad_(True))
        self._initial_xyz = self._xyz.detach().clone()
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._initial_scaling = (
            scales.clone().detach()
        )  # Store initial scales for max size constraint
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[1]), device="cuda")

        # Optionally override rotations if source data is provided
        if source_path:
            self._override_rotations(source_path)

    def _override_rotations(self, source_path: str):
        """
        Override model rotations with data from source files.

        Args:
            source_path: Path to source data directory
        """
        # Load scaling and rotation data
        source_path = os.path.join(source_path, "sparse", "0")
        scales_np = np.load(os.path.join(source_path, "scalings.npy"))
        scales_np = np.clip(scales_np, -4.0, -0.01)
        rots_np = np.load(os.path.join(source_path, "rotations.npy"))

        print(f"Loaded scaling shape: {scales_np.shape}")
        print(f"Loaded rotation shape: {rots_np.shape}")

        # Replace parameters with loaded data
        self._scaling = nn.Parameter(
            torch.tensor(scales_np, dtype=torch.float32, device="cuda").requires_grad_(
                True
            )
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots_np, dtype=torch.float32, device="cuda").requires_grad_(
                True
            )
        )

    def load_ply(self, path: str, use_train_test_exp: bool = False):
        """
        Load a Gaussian model from a PLY file.

        Args:
            path: Path to the PLY file
            use_train_test_exp: Whether to use expected dataset size for training/testing
        """
        plydata = PlyData.read(path)

        # Extract xyz coordinates
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        if use_train_test_exp:
            # The expected rows is 29060
            xyz = xyz[:29060, :]

        # Extract features_dc
        features_dc = self._extract_ply_attributes(plydata, "f_dc_", use_train_test_exp)
        if len(features_dc) > 0:
            features_dc = features_dc.reshape(-1, 1, 3)  # Reshape to [N, 1, 3]

        # Extract features_rest
        features_rest = self._extract_ply_attributes(
            plydata, "f_rest_", use_train_test_exp
        )
        if len(features_rest) > 0:
            num_rest_feats = features_rest.shape[1] // 3
            features_rest = features_rest.reshape(-1, num_rest_feats, 3)

        # Extract opacity, scale, and rotation
        opacity = np.asarray(plydata.elements[0]["opacity"]).reshape(-1, 1)
        if use_train_test_exp:
            opacity = opacity[:29060, :]

        scale = self._extract_ply_attributes(plydata, "scale_", use_train_test_exp)
        assert scale.shape[1] == 3, "Expected scale to have 3 components"

        rot = self._extract_ply_attributes(plydata, "rot_", use_train_test_exp)
        assert rot.shape[1] == 4, "Expected rotation to have 4 components"

        # Create parameter tensors from loaded data
        xyz_tensor = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._xyz = nn.Parameter(
            xyz_tensor.transpose(0, 1).contiguous().requires_grad_(True)
        )
        self._initial_xyz = self._xyz.detach().clone()

        # Create features_dc tensor
        if len(features_dc) > 0:
            self._features_dc = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
        else:
            self._features_dc = nn.Parameter(
                torch.zeros(
                    (xyz.shape[0], 1, 3), dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )

        # Create features_rest tensor
        if len(features_rest) > 0:
            self._features_rest = nn.Parameter(
                torch.tensor(features_rest, dtype=torch.float, device="cuda")
                .contiguous()
                .requires_grad_(True)
            )
        else:
            self._features_rest = nn.Parameter(
                torch.zeros(
                    (xyz.shape[0], 0, 3), dtype=torch.float, device="cuda"
                ).requires_grad_(True)
            )

        # Create other parameter tensors
        self._opacity = nn.Parameter(
            torch.tensor(opacity, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scale, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._initial_scaling = self._scaling.detach().clone()
        self._rotation = nn.Parameter(
            torch.tensor(rot, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def _extract_ply_attributes(
        self, plydata: PlyData, prefix: str, use_train_test_exp: bool
    ) -> np.ndarray:
        """
        Extract attributes with a common prefix from PLY data.

        Args:
            plydata: Loaded PLY data
            prefix: Attribute prefix to search for
            use_train_test_exp: Whether to limit to expected dataset size

        Returns:
            Array of extracted attributes
        """
        attributes = []
        i = 0
        while True:
            key = f"{prefix}{i}"
            if key in plydata.elements[0].data.dtype.names:
                attributes.append(np.asarray(plydata.elements[0][key]))
                i += 1
            else:
                break

        if len(attributes) > 0:
            attributes = np.stack(attributes, axis=1)
            if use_train_test_exp:
                # The expected rows is 29060
                attributes = attributes[:29060, :]

        return attributes if len(attributes) > 0 else np.array([])

    # ===== Export functions =====

    def _map_intensities_to_sh_coefficients(
        self,
        intensity_values: torch.Tensor,
        volume_min: Optional[float] = None,
        volume_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Maps intensity values to spherical harmonic coefficients range for proper visualization.

        Args:
            intensity_values: Raw intensity values from volume
            volume_min: Global minimum value for normalization
            volume_max: Global maximum value for normalization

        Returns:
            Intensity values mapped to proper SH coefficient range
        """
        # Make a copy to avoid modifying the input
        intensity_tensor = intensity_values.clone()

        # Use provided min/max or compute from the tensor
        if volume_min is None:
            volume_min = intensity_tensor.min()
        if volume_max is None:
            volume_max = intensity_tensor.max()

        # Normalize to [0,1] range if possible
        if volume_max > volume_min:
            intensity_tensor = (intensity_tensor - volume_min) / (
                volume_max - volume_min
            )
            intensity_tensor = intensity_tensor.clamp_(0.0, 1.0)

            # Map normalized [0,1] intensities to spherical harmonic coefficient range
            intensity_tensor = intensity_tensor * 2.0 - 1.0  # Map [0,1] to [-1,1]
            intensity_tensor = (
                intensity_tensor * SH_SCALE
            )  # Map [-1,1] to [-SH_SCALE, SH_SCALE]

        return intensity_tensor

    def _prepare_colors_for_ply(self, num_points: int) -> np.ndarray:
        """
        Prepares color values for PLY file export.

        Args:
            num_points: Number of points in the model

        Returns:
            Array of color values in appropriate format for PLY export
        """
        if getattr(self, "intensity_mode", "learned") in {
            "sampled",
            "sampled_mean_covered",
        }:
            return self._create_colors_from_intensities(num_points)

        # Check if we have valid feature tensors
        if (
            self._features_dc is not None
            and self._features_dc.numel() > 0
            and torch.sum(torch.abs(self._features_dc)) > 0
        ):
            print("Using provided features for volume rendering.")
            features_tensor = self._features_dc.detach()
            # print(
            #     f"Features DC shape before transpose: {features_tensor.shape}, "
            #     f"range: [{features_tensor.min().item():.4f}, {features_tensor.max().item():.4f}]"
            # )
            features_tensor = features_tensor.transpose(
                1, 2
            )  # Change from [N, 1, 3] to [N, 3, 1]
            # print(f"Features DC shape after transpose: {features_tensor.shape}")
            features_tensor = features_tensor.flatten(start_dim=1)  # Change to [N, 3]
            # print(f"Features DC shape after flatten: {features_tensor.shape}")
            f_dc = features_tensor.contiguous().cpu().numpy()

            # Check for zero values in f_dc, which indicates an issue
            if np.allclose(f_dc, 0.0):
                print(
                    "Warning: f_dc values are all zeros! Using intensity values instead."
                )
                f_dc = self._create_colors_from_intensities(num_points)
        else:
            # Create colors from intensity values
            f_dc = self._create_colors_from_intensities(num_points)

        # print(
        #     f"Final f_dc shape: {f_dc.shape}, range: [{f_dc.min():.4f}, {f_dc.max():.4f}]"
        # )
        print(f"RGB value examples (from features): {f_dc[:5]}")

        return f_dc

    def _create_colors_from_intensities(self, num_points: int) -> np.ndarray:
        """
        Creates color values from intensity values for PLY export.

        Args:
            num_points: Number of points in the model

        Returns:
            Array of color values derived from intensities
        """
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            print("Creating colors from intensities.")
            raw_tensor = self.intensities.detach().cpu()
            intensity_values = raw_tensor.view(-1).numpy()
            print(
                f"Raw intensity shape: {tuple(raw_tensor.shape)}, "
                f"range: [{intensity_values.min():.4f}, {intensity_values.max():.4f}]"
            )

            if (
                hasattr(self, "volume_min")
                and hasattr(self, "volume_max")
                and self.volume_max > self.volume_min
            ):
                denom = max(self.volume_max - self.volume_min, 1e-8)
                normalized = (intensity_values - self.volume_min) / denom
                print(
                    f"Applying cached global min/max [{self.volume_min:.4f}, {self.volume_max:.4f}]"
                )
            else:
                local_min = float(intensity_values.min())
                local_max = float(intensity_values.max())
                if local_max > local_min:
                    normalized = (intensity_values - local_min) / (
                        local_max - local_min
                    )
                else:
                    normalized = np.full_like(intensity_values, 0.5, dtype=np.float32)
                print(
                    f"Fallback normalization range [{local_min:.4f}, {local_max:.4f}]"
                )

            normalized = np.clip(normalized, 0.0, 1.0)
            divisor = getattr(self, "intensity_color_divisor", 1.0)
            scaled = normalized / np.abs(divisor)
            rgb_values = np.round(np.clip(scaled, 0.0, 255.0)).astype(np.float32)
            f_dc = np.repeat(rgb_values[:, None], 3, axis=1)
        else:
            # Default to mid-gray if no intensities available
            print("Could not find intensity values, using default mid-gray.")
            f_dc = np.ones((num_points, 3), dtype=np.float32) * 128.0

        return f_dc

    def construct_list_of_attributes(self) -> List[str]:
        """
        Construct list of attribute names for PLY export.

        Returns:
            List of attribute names
        """
        attributes = ["x", "y", "z", "nx", "ny", "nz"]

        # Handle features for volume-only training (might be empty tensors)
        if self._features_dc is not None and self._features_dc.numel() > 0:
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                attributes.append(f"f_dc_{i}")
        else:
            # Add dummy DC features for volume-only model
            for i in range(3):  # RGB channels
                attributes.append(f"f_dc_{i}")

        if self._features_rest is not None and self._features_rest.numel() > 0:
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                attributes.append(f"f_rest_{i}")

        # Add remaining attributes
        attributes.append("opacity")

        if self._scaling.numel() > 0:
            for i in range(self._scaling.shape[1]):
                attributes.append(f"scale_{i}")

        if self._rotation.numel() > 0:
            for i in range(self._rotation.shape[1]):
                attributes.append(f"rot_{i}")

        return attributes

    def save_ply(self, path: str):
        """
        Save the Gaussian model to a PLY file.

        Args:
            path: Path to save the PLY file
        """
        mkdir_p(os.path.dirname(path))

        # Get the number of points
        if self._xyz.shape[0] == 3:  # Shape is [3, N]
            num_points = self._xyz.shape[1]
            xyz = self._xyz.detach().cpu().numpy().T  # Convert to [N, 3]
        else:  # Shape is already [N, 3]
            num_points = self._xyz.shape[0]
            xyz = self._xyz.detach().cpu().numpy()

        # Create normals
        normals = np.zeros_like(xyz)

        # Prepare color values using our helper method
        print(
            f"Feature tensors: _features_dc shape: "
            f"{self._features_dc.shape if self._features_dc is not None else 'None'}, "
            f"numel: {self._features_dc.numel() if self._features_dc is not None else 0}"
        )

        # Use the refactored helper method to get colors
        f_dc = self._prepare_colors_for_ply(num_points)

        # Get rest features if available
        if self._features_rest is not None and self._features_rest.numel() > 0:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            # Create empty features array for volume-only model
            f_rest = np.zeros((num_points, 0))

        # Handle other attributes
        opacity_tensor = self.get_opacity
        if opacity_tensor.numel() == 0:
            opacity_tensor = torch.ones((num_points, 1), device=self._xyz.device)
        opacities = opacity_tensor.detach().cpu().numpy()
        if opacities.shape[0] != num_points:
            # Ensure opacity has shape [N, 1]
            opacities = np.ones((num_points, 1))

        scale = self._scaling.detach().cpu().numpy()
        if scale.shape[0] != num_points:
            # Ensure scale has shape [N, 3]
            scale = np.ones((num_points, 3)) * 0.01

        rotation = self._rotation.detach().cpu().numpy()
        if rotation.shape[0] != num_points:
            # Ensure rotation has shape [N, 4]
            rotation = np.zeros((num_points, 4))
            rotation[:, 0] = 1  # Identity rotation

        # Create PLY file
        self._create_ply_file(
            path, xyz, normals, f_dc, f_rest, opacities, scale, rotation
        )

    def _create_ply_file(
        self,
        path: str,
        xyz: np.ndarray,
        normals: np.ndarray,
        f_dc: np.ndarray,
        f_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
    ):
        """
        Create a PLY file with the given attributes.

        Args:
            path: Output file path
            xyz: Point positions
            normals: Point normals
            f_dc: DC feature values
            f_rest: Rest feature values
            opacities: Opacity values
            scale: Scale values
            rotation: Rotation quaternions
        """
        num_points = xyz.shape[0]
        attributes_list = self.construct_list_of_attributes()
        dtype_full = [(attribute, 'f4') for attribute in attributes_list]

        # Create combined attributes array
        elements = np.empty(num_points, dtype=dtype_full)

        # Safely concatenate all attributes
        all_attributes = []
        all_attributes.append(xyz)          # [N, 3]
        all_attributes.append(normals)      # [N, 3]
        all_attributes.append(f_dc)         # [N, 3]
        if f_rest.shape[1] > 0:
            all_attributes.append(f_rest)  # [N, F-3]
        all_attributes.append(opacities)    # [N, 1]
        all_attributes.append(scale)        # [N, 3]
        all_attributes.append(rotation)     # [N, 4]

        attributes = np.concatenate(all_attributes, axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_sequence(
        self, output_dir: str, iteration: int, prefix: str = "gaussians"
    ) -> str:
        """Write the current Gaussian set to a numbered PLY inside `ply_sequence`."""
        ply_dir = os.path.join(output_dir, "ply_sequence")
        mkdir_p(ply_dir)

        path = os.path.join(ply_dir, f"{prefix}_{iteration:06d}.ply")
        self.save_ply(path)

        print(f"[ITER {iteration}] Saved model as PLY: {path}")
        return path

    # ===== Optimization and densification methods =====

    def reset_opacity(self):
        """Reset all opacity values to a small initial value."""
        if self._mask_opacity_active():
            if self.opacities.numel() > 0:
                self.opacities.zero_()
            return

        if self._opacity.numel() == 0:
            return

        opacities_new = self.inverse_opacity_activation(
            torch.ones_like(self._opacity) * 0.01
        )

        if self._optimizer_has_group("opacity"):
            optimizable_tensors = self.replace_tensor_to_optimizer(
                opacities_new, "opacity"
            )
            self._opacity = optimizable_tensors["opacity"]
        else:
            self._opacity = nn.Parameter(
                opacities_new.detach().clone().requires_grad_(True)
            )

    def replace_tensor_to_optimizer(
        self, tensor: torch.Tensor, name: str
    ) -> Dict[str, torch.nn.Parameter]:
        """
        Replace a tensor in the optimizer state.

        Args:
            tensor: New tensor value
            name: Name of the parameter group

        Returns:
            Dictionary with updated parameter
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0])

                # Create new state if it doesn't exist
                if stored_state is None:
                    stored_state = {
                        "exp_avg": torch.zeros_like(tensor),
                        "exp_avg_sq": torch.zeros_like(tensor),
                    }
                else:
                    # Update existing state
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    # Remove old parameter from state
                    del self.optimizer.state[group["params"][0]]

                # Replace parameter
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor) -> Dict[str, torch.nn.Parameter]:
        """
        Update optimizer state to match pruned tensors.

        Args:
            mask: Boolean mask for points to keep

        Returns:
            Dictionary with updated parameters
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0])

            # Special handling for xyz which has shape [3, N] instead of [N, ...]
            if group["name"] == "xyz":
                # For xyz with shape [3, N], index along dimension 1
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][:, mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:, mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][:, mask].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][:, mask].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # For other parameters with shape [N, ...], index along dimension 0
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][mask].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][mask].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask: torch.Tensor):
        """
        Remove points from the model based on a mask.

        Args:
            mask: Boolean mask for points to remove (True = remove)
        """
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # Update model parameters
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors.get("f_dc", self._features_dc)
        self._features_rest = optimizable_tensors.get("f_rest", self._features_rest)
        if "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        else:
            pruned_opacity = self._opacity[valid_points_mask]
            self._opacity = nn.Parameter(
                pruned_opacity.detach()
                .clone()
                .requires_grad_(not self._mask_opacity_active())
            )
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Update auxiliary tensors
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # Update volume-specific attributes
        if (
            hasattr(self, "intensities")
            and self.intensities is not None
            and self.intensities.numel() > 0
        ):
            # Check if intensities size matches the current point count
            current_point_count = valid_points_mask.shape[0]
            if self.intensities.shape[0] == current_point_count:
                self.intensities = self.intensities[valid_points_mask]
            else:
                # Size mismatch - recreate intensities to match current points
                print(
                    f"Warning: intensities size mismatch. Expected {current_point_count}, got {self.intensities.shape[0]}. Recreating."
                )
                remaining_point_count = valid_points_mask.sum().item()
                self.intensities = torch.zeros(
                    (remaining_point_count, 1), device="cuda"
                )

        if (
            hasattr(self, "opacities")
            and self.opacities is not None
            and self.opacities.numel() > 0
        ):
            # Check if opacities size matches the current point count
            current_point_count = valid_points_mask.shape[0]
            if self.opacities.shape[0] == current_point_count:
                self.opacities = self.opacities[valid_points_mask]
            else:
                # Size mismatch - recreate opacities to match current points
                print(
                    f"Warning: opacities size mismatch. Expected {current_point_count}, got {self.opacities.shape[0]}. Recreating."
                )
                remaining_point_count = valid_points_mask.sum().item()
                self.opacities = torch.zeros((remaining_point_count, 1), device="cuda")

        # Update initial scaling for max constraint
        if hasattr(self, "_initial_scaling") and self._initial_scaling.numel() > 0:
            self._initial_scaling = self._initial_scaling[valid_points_mask]
        if hasattr(self, "_initial_xyz") and self._initial_xyz.numel() > 0:
            self._initial_xyz = self._initial_xyz[:, valid_points_mask]
        self._reset_prev_buffers()

    def cat_tensors_to_optimizer(
        self, tensors_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.nn.Parameter]:
        """
        Add new tensors to existing ones in optimizer state.

        Args:
            tensors_dict: Dictionary of tensors to add

        Returns:
            Dictionary of updated parameters
        """
        optimizable_tensors = {}
        # SAFETY: All concatenations create new nn.Parameter objects; optimizer state for
        # existing portion is preserved, zeros initialized for new tail.
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0])

            # Special handling for xyz which has shape [3, N] instead of [N, 3]
            concat_dim = 1 if group["name"] == "xyz" else 0

            if stored_state is not None:
                # Update optimizer state
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                    dim=concat_dim,
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=concat_dim,
                )

                # Replace parameter in optimizer
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=concat_dim
                    ).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                # No state to update, just concatenate
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=concat_dim
                    ).requires_grad_(True)
                )

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_tmp_radii: torch.Tensor,
    ):
        """
        Add new points to the model after densification.

        Args:
            new_xyz: New point positions
            new_features_dc: New DC features
            new_features_rest: New rest features
            new_opacities: New opacity values
            new_scaling: New scaling values
            new_rotation: New rotation values
            new_tmp_radii: New temporary radii (unused)
        """
        # Prepare dictionary of new tensors - only include those that are in the optimizer
        new_tensors = {
            "xyz": new_xyz,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        # Only add features if they are not None (they are in the optimizer)
        if new_features_dc is not None:
            new_tensors["f_dc"] = new_features_dc

        if new_features_rest is not None:
            new_tensors["f_rest"] = new_features_rest

        # Add new tensors to model parameters
        optimizable_tensors = self.cat_tensors_to_optimizer(new_tensors)
        # Mark that parameter topology changed (can be used by outer training loop if needed)
        self._param_topology_changed = True
        self._xyz = optimizable_tensors["xyz"]
        # Only retrieve features if they were added
        self._features_dc = optimizable_tensors.get("f_dc", self._features_dc)
        self._features_rest = optimizable_tensors.get("f_rest", self._features_rest)
        if "opacity" in optimizable_tensors:
            self._opacity = optimizable_tensors["opacity"]
        else:
            concatenated = torch.cat((self._opacity, new_opacities), dim=0)
            self._opacity = nn.Parameter(
                concatenated.detach()
                .clone()
                .requires_grad_(not self._mask_opacity_active())
            )
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Reset auxiliary tensors (_xyz has shape [3, N], use shape[1])
        device = self._xyz.device if self._xyz.numel() > 0 else torch.device("cpu")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[1], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[1], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[1]), device=device)

        # Update volume-specific attributes
        if hasattr(self, "intensities") and self.intensities is not None:
            if self.intensities.numel() > 0:
                # new_xyz has shape [3, M], so shape[1] gives number of new points
                new_intensities = torch.zeros(
                    (new_xyz.shape[1], 1), device=self.intensities.device
                )
                self.intensities = torch.cat([self.intensities, new_intensities], dim=0)
            else:
                # If intensities is empty, create it to match the current point count
                current_point_count = self.get_xyz.shape[1]
                self.intensities = torch.zeros((current_point_count, 1), device=device)

        if hasattr(self, "opacities") and self.opacities is not None:
            if self.opacities.numel() > 0:
                # new_xyz has shape [3, M], so shape[1] gives number of new points
                new_opacities_vol = torch.zeros(
                    (new_xyz.shape[1], 1), device=self.opacities.device
                )
                self.opacities = torch.cat([self.opacities, new_opacities_vol], dim=0)
            else:
                # If opacities is empty, create it to match the current point count
                current_point_count = self.get_xyz.shape[1]
                self.opacities = torch.zeros((current_point_count, 1), device=device)

        # Update initial scaling for new points (for max scaling constraint)
        if hasattr(self, "_initial_scaling") and self._initial_scaling.numel() > 0:
            # New points inherit their initial scaling values from their current scaling
            new_initial_scaling = new_scaling.clone().detach()
            self._initial_scaling = torch.cat(
                [self._initial_scaling, new_initial_scaling], dim=0
            )
        if hasattr(self, "_initial_xyz") and self._initial_xyz.numel() > 0:
            self._initial_xyz = torch.cat(
                [self._initial_xyz, new_xyz.detach().clone()], dim=1
            )
        else:
            self._initial_xyz = self._xyz.detach().clone()
        self._reset_prev_buffers()

    def _current_parameter_bytes(self) -> float:
        """Estimate current parameter memory footprint in bytes."""
        total = 0
        tensors = [
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._features_dc,
            self._features_rest,
        ]
        for tensor in tensors:
            if tensor is not None and tensor.numel() > 0:
                total += tensor.element_size() * tensor.numel()
        return float(total)

    def _memory_budget_allows(self, new_points: int) -> bool:
        """Check whether spawning new points would exceed configured memory."""
        if self._max_memory_bytes is None or new_points <= 0:
            return True
        current_points = self._xyz.shape[1] if self._xyz.numel() > 0 else 0
        if current_points == 0:
            return True
        total_bytes = self._current_parameter_bytes()
        bytes_per_point = total_bytes / max(current_points, 1)
        projected = total_bytes + bytes_per_point * new_points
        return projected <= float(self._max_memory_bytes)

    def _sample_orientation_quats(
        self,
        xyz: torch.Tensor,
        fallback_quats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return quaternions for new points using cached orientation field."""
        if fallback_quats is not None and fallback_quats.numel() > 0:
            device = fallback_quats.device
            dtype = fallback_quats.dtype
        elif xyz.numel() > 0:
            device = xyz.device
            dtype = xyz.dtype
        elif self._rotation.numel() > 0:
            device = self._rotation.device
            dtype = self._rotation.dtype
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        coords = xyz
        if coords.dim() == 2 and coords.shape[0] == 3:
            coords = coords.transpose(0, 1)
        if coords.numel() == 0:
            empty = torch.empty(0, 4, device=device, dtype=dtype)
            mask = torch.empty(0, dtype=torch.bool, device=device)
            return empty, mask
        if coords.shape[1] != 3:
            raise ValueError(
                f"Orientation sampling expects xyz shaped [*, 3]; got {tuple(coords.shape)}."
            )

        coords = coords.contiguous()
        count = coords.shape[0]

        if fallback_quats is None or fallback_quats.numel() == 0:
            fallback_quats = torch.zeros(count, 4, device=device, dtype=dtype)
            fallback_quats[:, 0] = 1.0
        else:
            fallback_quats = fallback_quats.to(device=device, dtype=dtype)
            if fallback_quats.shape[0] not in {1, count}:
                raise ValueError(
                    "Fallback quaternions must have either 1 or N entries to match xyz."
                )
            if fallback_quats.shape[0] == 1 and count > 1:
                fallback_quats = fallback_quats.expand(count, -1)
            fallback_quats = torch.nn.functional.normalize(fallback_quats, dim=1)

        field = getattr(self, "orientation_field", None)
        has_field = (
            isinstance(field, dict)
            and field.get("gradient") is not None
            and field.get("magnitude") is not None
            and field["gradient"].numel() > 0
        )
        if not has_field:
            quats = random_quat_perturb(fallback_quats, 2.0)
            mask = torch.ones(count, dtype=torch.bool, device=device)
            quats = torch.nn.functional.normalize(quats, dim=1)
            return quats, mask

        grad_field = field["gradient"]
        if grad_field.device != device:
            grad_field = grad_field.to(device)
            field["gradient"] = grad_field
        mag_field = field["magnitude"]
        if mag_field.device != device:
            mag_field = mag_field.to(device)
            field["magnitude"] = mag_field
        origin = field["origin"]
        if origin.device != device:
            origin = origin.to(device)
            field["origin"] = origin
        voxel = field["voxel_size"]
        if voxel.device != device:
            voxel = voxel.to(device)
            field["voxel_size"] = voxel
        deg_tensor = field.get("perturb_deg")
        if deg_tensor is None:
            deg = 2.0
        else:
            deg_tensor = deg_tensor.to(device)
            field["perturb_deg"] = deg_tensor
            deg = float(deg_tensor.item())

        ijk = world_to_voxel(coords, origin, voxel)
        rotmats, fallback_mask = gather_rotation_from_gradient(
            grad_field, mag_field, ijk
        )
        quats = rotmat_to_quat(rotmats)
        if deg > 0.0:
            quats = random_quat_perturb(quats, deg)
        if fallback_mask.any():
            repl = fallback_quats[fallback_mask]
            if deg > 0.0:
                repl = random_quat_perturb(repl, deg)
            quats[fallback_mask] = repl
        quats = torch.nn.functional.normalize(quats, dim=1)
        return quats, fallback_mask

    def _structure_strength_from_field(
        self, xyz: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Sample normalized gradient magnitudes for given xyz positions."""
        field = getattr(self, "orientation_field", None)
        if not isinstance(field, dict):
            return None
        mag_field = field.get("magnitude")
        origin = field.get("origin")
        voxel = field.get("voxel_size")
        if mag_field is None or origin is None or voxel is None:
            return None

        coords = xyz
        if coords.dim() == 2 and coords.shape[0] == 3:
            coords = coords.transpose(0, 1).contiguous()
        elif coords.dim() != 2 or coords.shape[1] != 3:
            coords = coords.view(-1, 3)

        if coords.numel() == 0:
            return None

        device = mag_field.device
        grid = world_to_grid(
            coords.to(device=device),
            origin.to(device=device),
            voxel.to(device=device),
            mag_field.shape,
        )
        grid = grid.view(1, -1, 1, 1, 3)
        mag_tensor = mag_field.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            mag_tensor,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(-1)
        strength = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        if strength.numel() == 0:
            return None
        max_val = strength.max().clamp_min(1e-6)
        strength = (strength / max_val).clamp(0.0, 1.0)
        target_device = xyz.device if xyz.numel() > 0 else device
        return strength.to(device=target_device)

    def _structure_boost_factors(
        self, strength: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Return per-point LR/gradient boost factors derived from structure strength."""
        if strength is None or self.structure_gradient_boost <= 0.0:
            return None
        boost = 1.0 + strength * self.structure_gradient_boost
        if self.structure_gradient_threshold > 0.0:
            weak_mask = strength < self.structure_gradient_threshold
            if weak_mask.any():
                damped = 1.0 + strength * (self.structure_gradient_boost * 0.25)
                boost = torch.where(weak_mask, damped, boost)
        return boost

    @torch.no_grad()
    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
        structure_strength: Optional[torch.Tensor] = None,
    ):
        """
        Split large Gaussians that have high gradients.

        Args:
            grads: XYZ gradients
            grad_threshold: Minimum gradient magnitude for splitting
            scene_extent: Scene size for scaling reference
            N: Number of new points per split
        """
        xyz = self.get_xyz
        device = xyz.device
        n_init_points = xyz.shape[1]
        if n_init_points == 0:
            return

        # Extract points that satisfy the gradient condition
        grad_metric = torch.zeros(
            (n_init_points), device=grads.device, dtype=grads.dtype
        )
        grad_metric[: grads.shape[0]] = grads.squeeze()
        boost = self._structure_boost_factors(structure_strength)
        if boost is not None and boost.numel() == grad_metric.numel():
            grad_metric = grad_metric * boost.to(device=grad_metric.device)
        selected_pts_mask = torch.where(grad_metric >= grad_threshold, True, False)

        # Filter by scale criteria
        scales = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values > self.percent_dense * scene_extent,
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.min(scales, dim=1).values > 0.0
        )

        if (
            self._low_density_mask is not None
            and self._low_density_mask.numel() == selected_pts_mask.numel()
        ):
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, self._low_density_mask
            )

        if not selected_pts_mask.any():
            return

        # Create new points
        parent_scaling = scales[selected_pts_mask]
        stds = parent_scaling.repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=stds.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        # Apply rotation and add to original positions
        selected_xyz = xyz[:, selected_pts_mask].T  # [M, 3]
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(
            -1
        ) + selected_xyz.repeat(N, 1)
        # Transpose new_xyz back to [3, N*M] to match _xyz shape
        new_xyz = new_xyz.T.contiguous()

        # Create scaled-down versions of other attributes
        # Use smaller children so splits refine structure instead of propagating blobs
        anisotropic = parent_scaling.repeat(N, 1)
        anisotropic[:, :2] *= self.vessel_radial_scale
        anisotropic[:, 2] *= self.vessel_axial_scale
        # Radial axes shrink more aggressively; axial shrinks moderately
        shrink_radial = 2.0 * max(float(N), 1.0)
        shrink_axial = max(float(N) ** 0.5, 1.0)
        anisotropic[:, :2] = anisotropic[:, :2] / shrink_radial
        anisotropic[:, 2] = anisotropic[:, 2] / shrink_axial
        anisotropic = anisotropic.clamp_min(1e-6)
        new_scaling = self.scaling_inverse_activation(anisotropic)
        parent_quats = self.get_rotation[selected_pts_mask].detach()
        fallback_quats = parent_quats.repeat(N, 1)
        new_rotation, fallback_mask = self._sample_orientation_quats(
            new_xyz, fallback_quats
        )
        fallback_used = (
            int(fallback_mask.sum().item()) if fallback_mask.numel() > 0 else 0
        )
        if fallback_used > 0:
            self.orientation_fallback_stats["split"] += fallback_used
        num_new_points = N * selected_pts_mask.sum().item()
        net_new_points = max(N - 1, 0) * selected_pts_mask.sum().item()
        if not self._memory_budget_allows(net_new_points):
            self.last_densify_counts["split"] = 0
            return
        self.last_densify_counts["split"] = net_new_points

        # Handle features - create new features matching the shape of existing ones
        # Check if f_dc is in the optimizer (not just if it has elements)
        has_f_dc = any(group["name"] == "f_dc" for group in self.optimizer.param_groups)
        if has_f_dc and self._features_dc is not None:
            if self._features_dc.numel() > 0:
                new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
            else:
                feature_device = self._features_dc.device
                new_features_dc = torch.zeros(
                    (num_new_points, 1, 3), device=feature_device
                )
        else:
            new_features_dc = None

        # Handle rest features - check if there are actual SH features (shape[1] > 0)
        has_f_rest = any(
            group["name"] == "f_rest" for group in self.optimizer.param_groups
        )
        if has_f_rest and self._features_rest is not None:
            if self._features_rest.numel() > 0 and self._features_rest.shape[1] > 0:
                new_features_rest = self._features_rest[selected_pts_mask].repeat(
                    N, 1, 1
                )
            else:
                # Features exist in optimizer but are empty - create empty features for new points
                feature_device = self._features_rest.device
                new_features_rest = torch.zeros(
                    (num_new_points, 0, 3), device=feature_device
                )
        else:
            new_features_rest = None

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = torch.zeros(num_new_points, device=device)

        # Add new points to the model
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

        # Create pruning filter to remove the original points that were split
        # After densification, we now have original_points + N * split_points
        current_num_points = self.get_xyz.shape[1]
        prune_filter = torch.zeros(current_num_points, dtype=torch.bool, device=device)

        # Mark original split points for removal
        # The original split points are at the beginning, new points are at the end
        num_split_points = selected_pts_mask.sum().item()
        original_split_indices = torch.where(selected_pts_mask)[0]
        prune_filter[original_split_indices] = True

        self.prune_points(prune_filter)

    @torch.no_grad()
    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        structure_strength: Optional[torch.Tensor] = None,
    ):
        """
        Clone small Gaussians that have high gradients.

        Args:
            grads: XYZ gradients
            grad_threshold: Minimum gradient magnitude for cloning
            scene_extent: Scene size for scaling reference
        """
        # Extract points that satisfy the gradient condition
        grad_metric = torch.norm(grads, dim=-1)
        boost = self._structure_boost_factors(structure_strength)
        if boost is not None and boost.numel() == grad_metric.numel():
            grad_metric = grad_metric * boost.to(device=grad_metric.device)
        selected_pts_mask = torch.where(grad_metric >= grad_threshold, True, False)

        # Filter by scale criteria
        scales = self.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scales, dim=1).values <= self.percent_dense * scene_extent,
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.min(scales, dim=1).values > 0.0
        )

        self._clone_by_mask(selected_pts_mask, reason="clone")

    @torch.no_grad()
    def _clone_by_mask(self, selected_pts_mask: torch.Tensor, reason: str) -> int:
        """Clone points specified by mask; returns number of clones added."""
        if selected_pts_mask is None or selected_pts_mask.numel() == 0:
            return 0
        selected_pts_mask = selected_pts_mask.bool()
        if not selected_pts_mask.any():
            return 0

        xyz = self.get_xyz
        device = xyz.device
        num_new_points = selected_pts_mask.sum().item()
        if not self._memory_budget_allows(num_new_points):
            return 0

        new_xyz = xyz[:, selected_pts_mask]

        has_f_dc = any(group["name"] == "f_dc" for group in self.optimizer.param_groups)
        if has_f_dc and self._features_dc is not None:
            if self._features_dc.numel() > 0:
                new_features_dc = self._features_dc[selected_pts_mask]
            else:
                feature_device = self._features_dc.device
                new_features_dc = torch.zeros(
                    (num_new_points, 1, 3), device=feature_device
                )
        else:
            new_features_dc = None

        has_f_rest = any(
            group["name"] == "f_rest" for group in self.optimizer.param_groups
        )
        if has_f_rest and self._features_rest is not None:
            if self._features_rest.numel() > 0 and self._features_rest.shape[1] > 0:
                new_features_rest = self._features_rest[selected_pts_mask]
            else:
                feature_device = self._features_rest.device
                new_features_rest = torch.zeros(
                    (num_new_points, 0, 3), device=feature_device
                )
        else:
            new_features_rest = None

        new_opacities = self._opacity[selected_pts_mask]
        # Slightly shrink cloned scales so clones act as refinements, not duplicate blobs
        parent_scaling = self.get_scaling[selected_pts_mask]
        shrink_factor = 0.8
        shrunk_scaling = (parent_scaling * shrink_factor).clamp_min(1e-6)
        new_scaling = self.scaling_inverse_activation(shrunk_scaling)
        parent_quats = self.get_rotation[selected_pts_mask].detach()
        new_rotation, fallback_mask = self._sample_orientation_quats(
            new_xyz, parent_quats
        )
        fallback_used = (
            int(fallback_mask.sum().item()) if fallback_mask.numel() > 0 else 0
        )
        if fallback_used > 0:
            self.orientation_fallback_stats[reason] = (
                self.orientation_fallback_stats.get(reason, 0) + fallback_used
            )
        new_tmp_radii = torch.zeros(new_xyz.shape[1], device=device)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )
        self.last_densify_counts[reason] = num_new_points
        return num_new_points

    @torch.no_grad()
    def _trigger_hole_fill(self, target_count: int) -> int:
        """Clone additional splats in sparse regions to fill coverage holes."""
        if (
            target_count <= 0
            or self._low_density_mask is None
            or self._low_density_mask.numel() == 0
        ):
            return 0

        candidates = torch.nonzero(self._low_density_mask, as_tuple=False).view(-1)
        if candidates.numel() == 0:
            return 0

        fill_count = min(target_count, candidates.numel())
        perm = torch.randperm(candidates.numel(), device=candidates.device)[:fill_count]
        selected = candidates[perm]
        mask = torch.zeros(
            self.get_xyz.shape[1], dtype=torch.bool, device=self._xyz.device
        )
        mask[selected] = True
        return self._clone_by_mask(mask, reason="hole_fill")

    @torch.no_grad()
    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: float,
        radii: torch.Tensor,
    ):
        """
        Perform densification (splitting and cloning) and pruning in one step.

        Args:
            max_grad: Threshold for gradient-based densification
            min_opacity: Minimum opacity for a point to keep
            extent: Scene extent for scale reference
            max_screen_size: Maximum allowed screen-space size
            radii: Point radii in screen space
        """
        # Reset last densify counts for this iteration snapshot
        self.last_densify_counts = {"split": 0, "clone": 0, "hole_fill": 0}

        xyz = self.get_xyz
        structure_strength = None
        if self.structure_gradient_boost > 0.0:
            structure_strength = self._structure_strength_from_field(xyz)
            if (
                structure_strength is not None
                and self.structure_gradient_exponent != 1.0
            ):
                structure_strength = structure_strength.pow(
                    self.structure_gradient_exponent
                )

        # Calculate normalized gradients
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grad_norm = torch.norm(grads, dim=-1, keepdim=False)
        valid_grad = grad_norm[torch.isfinite(grad_norm)]
        if valid_grad.numel() > 0:
            adaptive_threshold = torch.quantile(
                valid_grad, min(max(self.densify_grad_percentile, 0.0), 1.0)
            ).item()
            grad_threshold = max(adaptive_threshold, float(max_grad))
        else:
            grad_threshold = float(max_grad)

        # Update density cache to guide densification heuristics
        density_info = self._maybe_update_density_cache(None)

        # Perform densification
        self.densify_and_clone(grads, grad_threshold, extent, structure_strength)
        self.densify_and_split(
            grads,
            grad_threshold,
            extent,
            N=2,
            structure_strength=structure_strength,
        )

        # Perform targeted hole filling when coverage is poor
        hole_added = 0
        if density_info is not None:
            coverage_ratio = float(density_info["coverage_ratio"].item())
            if coverage_ratio < self.target_coverage:
                desired = int(self._hole_fill_fraction * self._xyz.shape[1])
                hole_added = self._trigger_hole_fill(desired)

        if hole_added > 0:
            self.last_densify_counts["hole_fill"] = hole_added

        # Determine points to prune
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        # Prune points and reset tracking tensors
        self.prune_points(prune_mask)
        # _xyz has shape [3, N], so use shape[1] to get number of points
        device = self._xyz.device if self._xyz.numel() > 0 else grads.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[1], 1), device=device)
        self.denom = torch.zeros((self.get_xyz.shape[1], 1), device=device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[1]), device=device)

    def add_densification_stats(
        self, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor
    ):
        """
        Update densification statistics based on viewspace gradients.

        Args:
            viewspace_point_tensor: Points with gradients in view space
            update_filter: Boolean mask for points to update
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    # ===== Volume-based update methods =====

    def update_intensities(self, volume: torch.Tensor):
        """
        Update intensity values for all Gaussians based on their current positions.
        This should be called when positions change significantly.

        Args:
            volume: Reference volume with intensity values
        """
        if volume is None:
            return

        self.reference_volume = volume
        from gaussian_splatting.utils.intensity_sampler import update_intensities

        normalize_samples = getattr(self, "intensity_mode", "learned") in {
            "sampled",
            "sampled_mean_covered",
        }
        global_min = float(volume.min().item()) if normalize_samples else None
        global_max = float(volume.max().item()) if normalize_samples else None
        scales = self.get_scaling if self._scaling.numel() > 0 else None

        with torch.no_grad():
            intensities, vol_min, vol_max = update_intensities(
                self.get_xyz,
                volume,
                scales,
                normalize=normalize_samples,
                min_val=global_min,
                max_val=global_max,
            )

        self.intensities = intensities.detach()
        self.intensities.requires_grad = False
        self.volume_min = vol_min
        self.volume_max = vol_max

        if (
            self._features_dc is not None
            and self._features_dc.numel() > 0
            and self.intensities.numel() > 0
        ):
            sh_vals = (
                self._map_intensities_to_sh_coefficients(
                    self.intensities, self.volume_min, self.volume_max
                )
                .expand(-1, 3)
                .unsqueeze(1)
            )
            if sh_vals.shape == self._features_dc.shape:
                self._features_dc.data.copy_(sh_vals)
            else:
                self._features_dc = torch.nn.Parameter(
                    sh_vals.detach().clone(), requires_grad=True
                )

        print(
            f"Updated intensities: range [{self.intensities.min().item():.4f}, {self.intensities.max().item():.4f}]"
        )

    def update_intensities_and_opacities(
        self, volume: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """
        Update both intensity and opacity values for all Gaussians based on their current positions.
        This should be called when positions or scales change significantly.

        Args:
            volume: Reference volume with intensity values
            mask: Optional reference mask volume with opacity values [0,1]
        """
        if volume is None:
            return

        self.reference_volume = volume
        if mask is not None:
            self.reference_mask = mask

        from gaussian_splatting.utils.intensity_sampler import (
            update_intensities_and_opacities as sample_intensity_and_opacity,
        )

        normalize_samples = getattr(self, "intensity_mode", "learned") in {
            "sampled",
            "sampled_mean_covered",
        }
        global_min = float(volume.min().item()) if normalize_samples else None
        global_max = float(volume.max().item()) if normalize_samples else None
        scales = self.get_scaling if self._scaling.numel() > 0 else None

        with torch.no_grad():
            intensities, opacities, volume_min, volume_max = (
                sample_intensity_and_opacity(
                    self.get_xyz,
                    volume,
                    mask=mask,
                    scale=scales,
                    normalize=normalize_samples,
                    min_val=global_min,
                    max_val=global_max,
                )
            )

        self.intensities = intensities.detach()
        self.intensities.requires_grad = False
        self.volume_min = volume_min
        self.volume_max = volume_max

        if opacities is not None:
            self.opacities = opacities.detach()
            self.opacities.requires_grad = False

        if (
            self._features_dc is not None
            and self._features_dc.numel() > 0
            and self.intensities.numel() > 0
        ):
            sh_vals = (
                self._map_intensities_to_sh_coefficients(
                    self.intensities, volume_min, volume_max
                )
                .expand(-1, 3)
                .unsqueeze(1)
            )
            if sh_vals.shape == self._features_dc.shape:
                self._features_dc.data.copy_(sh_vals)
            else:
                self._features_dc = torch.nn.Parameter(
                    sh_vals.detach().clone(), requires_grad=True
                )

        print(
            f"Updated intensities: range [{self.intensities.min().item():.4f}, {self.intensities.max().item():.4f}]"
        )
        if opacities is not None:
            print(
                f"Updated opacities: range [{self.opacities.min().item():.4f}, {self.opacities.max().item():.4f}]"
            )


# === Gaussian utility helpers =================================================


def gaussian_compute_local_density(
    xyz: torch.Tensor,
    radius: float,
    density_cap: float,
) -> torch.Tensor:
    """Approximate per-point density via inverse mean neighbor distance."""
    if xyz.numel() == 0:
        return torch.empty(0, device=xyz.device if xyz.is_cuda else torch.device("cpu"))

    pts = xyz.transpose(0, 1).contiguous()
    mean_dist2 = torch.clamp(distCUDA2(pts), min=1e-12)
    mean_dist = torch.sqrt(mean_dist2)
    density = (radius / (mean_dist + 1e-6)) ** 3
    return density.clamp(max=density_cap)


def gaussian_compute_coverage_grid(
    xyz: torch.Tensor,
    resolution: int = 32,
) -> Dict[str, torch.Tensor]:
    """Compute coarse occupancy grid for coverage diagnostics."""
    if xyz.numel() == 0:
        device = xyz.device if xyz.is_cuda else torch.device("cpu")
        occupancy = torch.zeros((resolution, resolution, resolution), device=device)
        return {
            "occupancy": occupancy,
            "bounds_min": torch.zeros(3, device=device),
            "bounds_size": torch.ones(3, device=device),
        }

    device = xyz.device
    mins = xyz.min(dim=1)[0]
    maxs = xyz.max(dim=1)[0]
    extent = (maxs - mins).clamp_min(1e-5)

    normalized = (xyz - mins.unsqueeze(1)) / extent.unsqueeze(1)
    normalized = normalized.clamp(0.0, 0.999999)
    idx = (normalized * resolution).long()

    occupancy = torch.zeros((resolution, resolution, resolution), device=device)
    lin_idx = idx[0] * resolution * resolution + idx[1] * resolution + idx[2]
    occupancy.view(-1).index_put_(
        (lin_idx,), torch.ones_like(lin_idx, dtype=occupancy.dtype), accumulate=True
    )
    occupancy = occupancy.clamp_max(1.0)

    return {
        "occupancy": occupancy,
        "bounds_min": mins,
        "bounds_size": extent,
    }
