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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from typing import Optional, Dict, Tuple, List, Union, Any

try:
    from gaussian_rasterization import SparseGaussianAdam
except:
    pass


# Define constants at the module level for better maintainability
SH_C0 = 0.28209479177387814  # Value of Y_0^0 (first spherical harmonic)
SH_SCALE = 1.77  # Approximate 1/(2*SH_C0)
INTENSITY_SCALE = 50.0  # Scale factor to improve visibility (only for debugging)


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

        if self._opacity.numel() > 0 and not self._opacity.requires_grad:
            print(
                "WARNING: Opacity parameters do not require gradients. Setting requires_grad=True."
            )
            self._opacity.requires_grad_(True)

        # Check intensity values for volume rendering
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            if not isinstance(self.intensities, nn.Parameter):
                print(
                    "WARNING: Intensity values are not nn.Parameter. Converting to nn.Parameter."
                )
                self.intensities = nn.Parameter(self.intensities)
            if not self.intensities.requires_grad:
                print(
                    "WARNING: Intensity values do not require gradients. Setting requires_grad=True."
                )
                self.intensities.requires_grad_(True)

    # ===== Properties for accessing model parameters =====

    @property
    def get_xyz(self) -> torch.Tensor:
        """Get point positions."""
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        """Get point scaling parameters (converted from log-space)."""
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """Get normalized rotation quaternions."""
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self) -> torch.Tensor:
        """Get point opacity values (converted from log-space)."""
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

        # Restore intensity values if available
        if len(extra_args) > 0:
            self.intensities = extra_args[0]
        else:
            self.intensities = torch.empty(0)

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def training_setup(self, training_args: Any):
        """
        Set up optimizer and learning rate schedules for training.

        Args:
            training_args: Training arguments for optimizer setup
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

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

        # Only add feature parameters if they exist (for volume-only training they might be empty)
        if self._features_dc is not None and self._features_dc.numel() > 0:
            param_groups.append(
                {
                    "params": [self._features_dc],
                    "lr": training_args.feature_lr,
                    "name": "f_dc",
                }
            )

        if self._features_rest is not None and self._features_rest.numel() > 0:
            param_groups.append(
                {
                    "params": [self._features_rest],
                    "lr": training_args.feature_lr / 20.0,
                    "name": "f_rest",
                }
            )

        # Add position, opacity, scaling and rotation parameters
        param_groups.extend(
            [
                {
                    "params": [self._xyz],
                    "lr": training_args.position_lr_init * self.spatial_lr_scale,
                    "name": "xyz",
                },
                {
                    "params": [self._opacity],
                    "lr": training_args.opacity_lr,
                    "name": "opacity",
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

        return param_groups

    def update_learning_rate(self, iteration: int) -> float:
        """
        Update learning rates based on current iteration.

        Args:
            iteration: Current training iteration

        Returns:
            Current position learning rate
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
        return 0.0

    def oneupSHdegree(self):
        """Increase spherical harmonics degree by one, if below maximum."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

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
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        # Initialize features tensor
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at initialization: {fused_point_cloud.shape[0]}")

        # Calculate initial scales based on point distances
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Initialize rotations to identity quaternions
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1  # w=1, x,y,z=0 for identity rotation

        # Initialize opacity values
        opacities = self.inverse_opacity_activation(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        # Create parameter tensors
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

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
        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )

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

            # Map normalized [0,1] intensities to spherical harmonic coefficient range
            intensity_tensor = (
                intensity_tensor * 2.0 * INTENSITY_SCALE - 1.0
            )  # Map [0,1] to [-1,1]
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
        # Check if we have valid feature tensors
        if (
            self._features_dc is not None
            and self._features_dc.numel() > 0
            and torch.sum(torch.abs(self._features_dc)) > 0
        ):
            print("Using provided features for volume rendering.")
            features_tensor = self._features_dc.detach()
            print(
                f"Features DC shape before transpose: {features_tensor.shape}, "
                f"range: [{features_tensor.min().item():.4f}, {features_tensor.max().item():.4f}]"
            )
            features_tensor = features_tensor.transpose(
                1, 2
            )  # Change from [N, 1, 3] to [N, 3, 1]
            print(f"Features DC shape after transpose: {features_tensor.shape}")
            features_tensor = features_tensor.flatten(start_dim=1)  # Change to [N, 3]
            print(f"Features DC shape after flatten: {features_tensor.shape}")
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

        print(
            f"Final f_dc shape: {f_dc.shape}, range: [{f_dc.min():.4f}, {f_dc.max():.4f}]"
        )
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
            # Get raw intensity values
            intensity_values = self.intensities.detach().cpu().numpy()
            print(
                f"Raw intensity shape: {intensity_values.shape}, "
                f"range: [{intensity_values.min():.4f}, {intensity_values.max():.4f}]"
            )

            # Use the class method to map intensities to SH coefficients
            if hasattr(self, "volume_min") and hasattr(self, "volume_max"):
                volume_min = self.volume_min
                volume_max = self.volume_max
                print(
                    f"Normalized intensity using global min/max [{volume_min:.4f}, {volume_max:.4f}]"
                )

                # Convert numpy array to tensor for processing
                intensity_tensor = torch.from_numpy(intensity_values).to(
                    self._xyz.device
                )
                sh_values = (
                    self._map_intensities_to_sh_coefficients(
                        intensity_tensor, volume_min, volume_max
                    )
                    .cpu()
                    .numpy()
                )
            else:
                # Use numpy operations directly if we don't have global min/max
                if intensity_values.max() > intensity_values.min():
                    intensity_values = (intensity_values - intensity_values.min()) / (
                        intensity_values.max() - intensity_values.min()
                    )
                    intensity_values = intensity_values * 2.0 * INTENSITY_SCALE - 1.0
                    sh_values = intensity_values * SH_SCALE
                else:
                    sh_values = intensity_values

            # Create grayscale RGB values
            f_dc = np.zeros((num_points, 3))
            f_dc[:, 0] = sh_values[:, 0]  # Red
            f_dc[:, 1] = sh_values[:, 0]  # Green
            f_dc[:, 2] = sh_values[:, 0]  # Blue
        else:
            # Default to mid-gray if no intensities available
            print("Could not find intensity values, using default mid-gray.")
            f_dc = np.ones((num_points, 3)) * 0.5

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
        opacities = self._opacity.detach().cpu().numpy()
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
        """
        Save the Gaussian model to a PLY file with iteration number.

        Args:
            output_dir: Directory to save the PLY file
            iteration: Current iteration number
            prefix: Prefix for the filename

        Returns:
            Path to the saved PLY file
        """
        # Create the ply_sequence directory if it doesn't exist
        ply_dir = os.path.join(output_dir, "ply_sequence")
        mkdir_p(ply_dir)

        # Create the path to save the PLY file
        path = os.path.join(ply_dir, f"{prefix}_{iteration:06d}.ply")
        self.save_ply(path)

        print(f"[ITER {iteration}] Saved model as PLY: {path}")
        return path

    # ===== Optimization and densification methods =====

    def reset_opacity(self):
        """Reset all opacity values to a small initial value."""
        opacities_new = self.inverse_opacity_activation(
            torch.ones_like(self._opacity) * 0.01
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
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
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
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
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Update auxiliary tensors
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # Update volume-specific attributes
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            self.intensities = self.intensities[valid_points_mask]

        if hasattr(self, "opacities") and self.opacities.numel() > 0:
            self.opacities = self.opacities[valid_points_mask]

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
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0])

            if stored_state is not None:
                # Update optimizer state
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                # Replace parameter in optimizer
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                # No state to update, just concatenate
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
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
        # Prepare dictionary of new tensors
        new_tensors = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        # Add new tensors to model parameters
        optimizable_tensors = self.cat_tensors_to_optimizer(new_tensors)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Reset auxiliary tensors
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Update volume-specific attributes
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            # Create zeros tensor for new points (to be filled in later during a volume query)
            new_intensities = torch.zeros(
                (new_xyz.shape[0], 1), device=self.intensities.device
            )
            self.intensities = torch.cat([self.intensities, new_intensities], dim=0)

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
    ):
        """
        Split large Gaussians that have high gradients.

        Args:
            grads: XYZ gradients
            grad_threshold: Minimum gradient magnitude for splitting
            scene_extent: Scene size for scaling reference
            N: Number of new points per split
        """
        n_init_points = self.get_xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # Filter by scale criteria
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.min(self.get_scaling, dim=1).values > 0.0
        )

        # Create new points
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        # Apply rotation and add to original positions
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        # Create scaled-down versions of other attributes
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)

        # Handle rest features (might be empty tensor)
        if self._features_rest.shape[0] > 0:
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        else:
            new_features_rest = torch.zeros((0, 0, 0), device="cuda")

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = torch.zeros((N * selected_pts_mask.sum()), device="cuda")

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
        prune_filter = torch.cat(
            (
                torch.ones_like(selected_pts_mask),
                torch.zeros_like(selected_pts_mask).repeat(N),
            ),
            dim=0,
        )
        prune_filter = torch.where(
            prune_filter == 1, selected_pts_mask.repeat(N + 1), False
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self, grads: torch.Tensor, grad_threshold: float, scene_extent: float
    ):
        """
        Clone small Gaussians that have high gradients.

        Args:
            grads: XYZ gradients
            grad_threshold: Minimum gradient magnitude for cloning
            scene_extent: Scene size for scaling reference
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Filter by scale criteria
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.min(self.get_scaling, dim=1).values > 0.0
        )

        # Clone selected points
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]

        # Handle rest features (might be empty tensor)
        if self._features_rest.shape[0] > 0:
            new_features_rest = self._features_rest[selected_pts_mask]
        else:
            new_features_rest = torch.zeros((0, 0, 0), device="cuda")

        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = torch.zeros(new_xyz.shape[0], device="cuda")

        # Add cloned points to the model
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_tmp_radii,
        )

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
        # Calculate normalized gradients
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Perform densification
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

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
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

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

        # Store reference volume for future updates
        self.reference_volume = volume

        from gaussian_splatting.utils.intensity_sampler import update_intensities

        # Update intensities based on current positions and scales
        with torch.no_grad():  # No gradients needed for this operation
            self.intensities, _, _ = update_intensities(
                self.get_xyz, volume, self.get_scaling
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

        # Store reference volume for future updates
        self.reference_volume = volume

        from gaussian_splatting.utils.intensity_sampler import (
            update_intensities_and_opacities,
        )

        # Update both intensities and opacities based on current positions and scales
        with torch.no_grad():  # No gradients needed for this operation
            intensities, opacities, volume_min, volume_max = (
                update_intensities_and_opacities(
                    self.get_xyz, volume, mask, self.get_scaling, normalize=False
                )
            )

            self.intensities = intensities

            # Store global min/max values for consistent normalization
            self.volume_min = volume_min
            self.volume_max = volume_max
            print(f"Volume global range: [{volume_min:.4f}, {volume_max:.4f}]")

            if opacities is not None:
                self.opacities = opacities
                print(
                    f"Updated opacities: range [{self.opacities.min().item():.4f}, {self.opacities.max().item():.4f}]"
                )

            # Update features_dc with intensities to ensure proper colors in PLY export
            if self._features_dc is not None:
                # Map intensities to SH coefficient range using our helper method
                sh_intensities = self._map_intensities_to_sh_coefficients(
                    intensities.clone(), volume_min, volume_max
                )

                # Expand to RGB channels and reshape
                sh_intensities = sh_intensities.expand(-1, 3)  # shape [N, 3]
                sh_intensities = sh_intensities.unsqueeze(1)  # shape [N, 1, 3]

                # Replace the existing features_dc with the new intensity-based colors
                self._features_dc.copy_(sh_intensities)
                print(
                    f"Updated features_dc with intensities: range [{sh_intensities.min().item():.4f}, {sh_intensities.max().item():.4f}]"
                )

        print(
            f"Updated intensities: range [{self.intensities.min().item():.4f}, {self.intensities.max().item():.4f}]"
        )
