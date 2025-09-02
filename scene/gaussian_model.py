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

try:
    from gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def _map_intensities_to_sh_coefficients(
        self, intensity_values, volume_min=None, volume_max=None
    ):
        """
        Maps intensity values to spherical harmonic coefficients range for proper visualization.

        Parameters:
        intensity_values (Tensor): Raw intensity values from volume
        volume_min (float, optional): Global minimum value for normalization
        volume_max (float, optional): Global maximum value for normalization

        Returns:
        Tensor: Intensity values mapped to proper SH coefficient range
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
            sh_scale = 3.54  # Approximate 1/0.28209479177387814
            intensity_tensor = intensity_tensor * 2.0 - 1.0  # Map [0,1] to [-1,1]
            intensity_tensor = (
                intensity_tensor * sh_scale
            )  # Map [-1,1] to [-sh_scale, sh_scale]

        return intensity_tensor

    def _prepare_colors_for_ply(self, num_points):
        """
        Prepares color values for PLY file export.

        Parameters:
        num_points (int): Number of points in the model

        Returns:
        numpy.ndarray: Array of color values in appropriate format for PLY export
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
                f"Features DC shape before transpose: {features_tensor.shape}, range: [{features_tensor.min().item():.4f}, {features_tensor.max().item():.4f}]"
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

    def _create_colors_from_intensities(self, num_points):
        """
        Creates color values from intensity values for PLY export.

        Parameters:
        num_points (int): Number of points in the model

        Returns:
        numpy.ndarray: Array of color values derived from intensities
        """
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            print("Creating colors from intensities.")
            # Get raw intensity values
            intensity_values = self.intensities.detach().cpu().numpy()
            print(
                f"Raw intensity shape: {intensity_values.shape}, range: [{intensity_values.min():.4f}, {intensity_values.max():.4f}]"
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
                    sh_scale = 3.54
                    intensity_values = intensity_values * 2.0 - 1.0
                    sh_values = intensity_values * sh_scale
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

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm * scaling_modifier

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        # Store opacity as a regular tensor, not a Parameter that requires gradients
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # Add non-learnable attributes for volume-based rendering
        self.intensities = torch.empty(0)
        self.opacities = torch.empty(0)  # Raw opacity values (not Parameters)
        # Store reference volumes for updates
        self.reference_volume = None
        self.reference_mask = None
        self.setup_functions()

    def capture(self):
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

    def restore(self, model_args, training_args):
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

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, source_path : str = None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self._override_rotations(source_path)

    def _override_rotations(self, source_path : str):
        source_path = os.path.join(source_path, "sparse", "0")
        scales_np = np.load(os.path.join(source_path, "scalings.npy"))
        # scales_np = np.sqrt(scales_np)
        scales_np = np.clip(scales_np, -4.0, -0.01)
        # scales_np *= 2.0
        rots_np = np.load(os.path.join(source_path, "rotations.npy"))

        print("Loaded scaling shape:", scales_np.shape)  # z.B. (100000, 3)
        print("Loaded rotation shape:", rots_np.shape)  # z.B. (100000, 3, 3)

        # Umwandeln in Torch-Tensoren und Parameter
        # print(f"Before: {self._scaling[:20]}")
        self._scaling = nn.Parameter(
           torch.tensor(scales_np, dtype=torch.float32, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots_np, dtype=torch.float32, device="cuda").requires_grad_(True)
        )
        # print(f"After: {self._scaling[:20]}")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Initialize empty optimizer parameters list
        l = []

        # Only add feature parameters if they exist (for volume-only training they might be None)
        if self._features_dc is not None:
            l.append({'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"})

        if self._features_rest is not None:
            l.append({'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"})

        # Add position, scaling and rotation parameters - these are always optimized
        l.append(
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            }
        )
        l.append(
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            }
        )
        l.append(
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            }
        )
        l.append(
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            }
        )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.optimizer_type == "adam_as_sgd":
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        # Handle features for volume-only training (might be empty tensors)
        if self._features_dc is not None and self._features_dc.numel() > 0:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
        else:
            # Add dummy DC features for volume-only model
            for i in range(3):  # RGB channels
                l.append('f_dc_{}'.format(i))

        if self._features_rest is not None and self._features_rest.numel() > 0:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))

        # Don't add intensity as a separate attribute - we'll use it for RGB values

        l.append('opacity')

        if self._scaling.numel() > 0:
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))

        if self._rotation.numel() > 0:
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))

        return l

    def save_ply(self, path):
        """Save the Gaussian model to a PLY file.

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
            f"Feature tensors: _features_dc shape: {self._features_dc.shape if self._features_dc is not None else 'None'}, numel: {self._features_dc.numel() if self._features_dc is not None else 0}"
        )

        # Use the refactored helper method to get colors
        f_dc = self._prepare_colors_for_ply(num_points)

        if self._features_rest is not None and self._features_rest.numel() > 0:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            # Create empty features array for volume-only model
            f_rest = np.zeros((num_points, 0))

        # Get intensity values
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            intensity_values = self.intensities.detach().cpu().numpy()
        else:
            # Default intensity values if not available
            intensity_values = np.ones((num_points, 1)) * 0.5

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

        # Special handling for empty tensors in construct_list_of_attributes
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
            all_attributes.append(f_rest)   # [N, F-3]
        # Don't include intensity_values as a separate attribute - they're already mapped to f_dc
        all_attributes.append(opacities)    # [N, 1]
        all_attributes.append(scale)        # [N, 3]
        all_attributes.append(rotation)     # [N, 4]

        attributes = np.concatenate(all_attributes, axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_sequence(self, output_dir, iteration, prefix="gaussians"):
        """Save the Gaussian model to a PLY file with iteration number.

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

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.ones_like(self._opacity) * 0.01
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
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
        features_dc = []
        i = 0
        while True:
            key = f"f_dc_{i}"
            if key in plydata.elements[0].data.dtype.names:
                features_dc.append(np.asarray(plydata.elements[0][key]))
                i += 1
            else:
                break

        if len(features_dc) > 0:
            features_dc = np.stack(features_dc, axis=1)
            if use_train_test_exp:
                # The expected rows is 29060
                features_dc = features_dc[:29060, :]

            # Reshape features_dc from [N, 3] to [N, 1, 3] for compatibility
            features_dc = features_dc.reshape(-1, 1, 3)

        # Extract features_rest
        features_rest = []
        i = 0
        while True:
            key = f"f_rest_{i}"
            if key in plydata.elements[0].data.dtype.names:
                features_rest.append(np.asarray(plydata.elements[0][key]))
                i += 1
            else:
                break

        if len(features_rest) > 0:
            features_rest = np.stack(features_rest, axis=1)

            if use_train_test_exp:
                # The expected rows is 29060
                features_rest = features_rest[:29060, :]

            # Check if features_rest is divisible by 3 (RGB)
            num_rest_feats = features_rest.shape[1] // 3
            features_rest = features_rest.reshape(-1, num_rest_feats, 3)

        # Extract opacity
        opacity = np.asarray(plydata.elements[0]["opacity"]).reshape(-1, 1)
        if use_train_test_exp:
            opacity = opacity[:29060, :]

        # Extract scale
        scale = []
        i = 0
        while True:
            key = f"scale_{i}"
            if key in plydata.elements[0].data.dtype.names:
                scale.append(np.asarray(plydata.elements[0][key]))
                i += 1
            else:
                break

        assert i == 3, "Expected scale to have 3 components"
        scale = np.stack(scale, axis=1)
        if use_train_test_exp:
            scale = scale[:29060, :]

        # Extract rotation
        rot = []
        i = 0
        while True:
            key = f"rot_{i}"
            if key in plydata.elements[0].data.dtype.names:
                rot.append(np.asarray(plydata.elements[0][key]))
                i += 1
            else:
                break

        assert i == 4, "Expected rotation to have 4 components"
        rot = np.stack(rot, axis=1)
        if use_train_test_exp:
            rot = rot[:29060, :]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

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

    def replace_tensor_to_optimizer(self, tensor, name):
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

    def _prune_optimizer(self, mask):
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

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # Update intensities and opacities for volume-only training
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            self.intensities = self.intensities[valid_points_mask]

        if hasattr(self, "opacities") and self.opacities.numel() > 0:
            self.opacities = self.opacities[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0])
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Densification should update intensities and opacities for volume-only training
        if hasattr(self, "intensities") and self.intensities.numel() > 0:
            # Create zeros tensor for new points (to be filled in later during a volume query)
            new_intensities = torch.zeros(
                (new_xyz.shape[0], 1), device=self.intensities.device
            )
            self.intensities = torch.cat([self.intensities, new_intensities], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.min(self.get_scaling, dim=1).values > 0.0
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = (
            self._features_rest[selected_pts_mask].repeat(N, 1, 1)
            if self._features_rest.shape[0] > 0
            else torch.zeros((0, 0, 0), device="cuda")
        )
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = torch.zeros((N * selected_pts_mask.sum()), device="cuda")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

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

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.min(self.get_scaling, dim=1).values > 0.0
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = (
            self._features_rest[selected_pts_mask]
            if self._features_rest.shape[0] > 0
            else torch.zeros((0, 0, 0), device="cuda")
        )
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = torch.zeros(new_xyz.shape[0], device="cuda")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def update_intensities(self, volume):
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
            self.intensities = update_intensities(
                self.get_xyz, volume, self.get_scaling
            )

        print(
            f"Updated intensities: range [{self.intensities.min().item():.4f}, {self.intensities.max().item():.4f}]"
        )

    def update_intensities_and_opacities(self, volume, mask=None):
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

            # Also update features_dc with intensities to ensure proper colors in PLY export
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
