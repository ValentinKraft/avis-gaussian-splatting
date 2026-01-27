# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Volume data loading and preprocessing for 3D Gaussian Splatting.
"""

import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Optional, Union
from pathlib import Path
try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover
    nib = None
# import SimpleITK as sitk
import torch.nn.functional as F

class VolumeLoader:
    """
    Loader for volumetric data supporting various medical imaging formats.
    Handles loading, optional resampling, and coordinate system alignment.
    """

    def __init__(
                 self,
                 target_shape: Optional[Tuple[int, int, int]] = None,
                 device: torch.device = torch.device('cuda'),
                 downscale_factor: Optional[int] = None):
        """
        Args:
            target_shape: Optional target shape for resampling. If None, keeps original dimensions
            device: Device to load tensors to
            downscale_factor: Optional integer downscale factor applied to the input volume shape.
                When provided and > 1, volumes are resampled to (D//factor, H//factor, W//factor).
                When provided and <= 1, resampling is disabled (native resolution), unless the
                automatic overflow guard triggers.
        """
        self.target_shape = target_shape
        self.device = device
        self.downscale_factor = downscale_factor

    def load_nifti(self, path: Union[str, Path]) -> Tensor:
        """Load a NIfTI volume file."""
        if nib is None:
            raise ModuleNotFoundError(
                "nibabel is required to load NIfTI volumes. Install it or use a .npy input."
            )
        nii = nib.load(str(path))
        volume = torch.from_numpy(nii.get_fdata()).float()
        # nibabel returns arrays in voxel index order (i, j, k) which typically
        # corresponds to (X, Y, Z). This project consistently represents volumes
        # as torch tensors in (D, H, W) = (Z, Y, X) order.
        if volume.dim() == 3:
            volume = volume.permute(2, 1, 0).contiguous()
        return self._process_volume(volume)

    def load_npy(self, path: Union[str, Path]) -> Tensor:
        """Load a NumPy volume file."""
        volume = torch.from_numpy(np.load(str(path))).float()
        return self._process_volume(volume)

    # def load_mhd(self, path: Union[str, Path]) -> Tensor:
    #     """Load a MetaImage (MHD/Raw) volume file."""
    #     img = sitk.ReadImage(str(path))
    #     volume = torch.from_numpy(sitk.GetArrayFromImage(img)).float()
    #     return self._process_volume(volume)

    def load_volume(self, path: Union[str, Path]) -> Tensor:
        """
        Load a volume file based on its extension.
        
        Args:
            path: Path to volume file (.nii, .nii.gz, .npy, .mhd)
            
        Returns:
            Normalized and resampled volume tensor
        """
        path = Path(path)

        if path.suffix in ['.nii', '.gz']:
            volume = self.load_nifti(path)
        elif path.suffix == '.npy':
            volume = self.load_npy(path)
        # elif path.suffix == '.mhd':
        #     volume = self.load_mhd(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return volume

    def _process_volume(self, volume: Tensor) -> Tensor:
        """
        Process loaded volume:
        1. Normalize to [0, 1]
        2. Automatically resample if volume is too large
        3. Optionally resample to target shape
        4. Move to device
        """
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        effective_target_shape = self.target_shape

        # Optional downscale relative to the input volume shape.
        if effective_target_shape is None and self.downscale_factor is not None:
            factor = int(self.downscale_factor)
            if factor > 1:
                D, H, W = volume.shape
                effective_target_shape = (
                    max(1, D // factor),
                    max(1, H // factor),
                    max(1, W // factor),
                )

        # Automatically determine target shape to prevent multinomial overflow
        if effective_target_shape is None:
            # Keep aspect ratio while ensuring total voxels < 2^24
            max_voxels = 2**24 - 1  # Maximum safe number for multinomial
            current_voxels = volume.numel()

            if current_voxels > max_voxels:
                # Calculate scale factor to reduce voxels below threshold
                scale = (max_voxels / current_voxels) ** (1 / 3)
                D, H, W = volume.shape
                effective_target_shape = (
                    max(32, int(D * scale)),
                    max(32, int(H * scale)),
                    max(32, int(W * scale)),
                )
                print(
                    f"Auto-resizing volume from {(D,H,W)} to {effective_target_shape} to prevent overflow"
                )

        # Resample if a target shape is specified
        if effective_target_shape is not None:
            # Add batch and channel dimensions for resampling
            volume = volume.unsqueeze(0).unsqueeze(0)

            # Resample to target shape
            volume = F.interpolate(
                volume,
                size=effective_target_shape,
                mode='trilinear',
                align_corners=True
            )

            # Remove batch and channel dimensions
            volume = volume.squeeze(0).squeeze(0)

        return volume.to(self.device)

    def align_to_space(self, 
                      volume: Tensor,
                      bbox_min: Tensor,
                      bbox_max: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Align volume coordinates to match the Gaussian splat coordinate system.
        
        Args:
            volume: Input volume tensor
            bbox_min: Minimum point of bounding box in world space
            bbox_max: Maximum point of bounding box in world space
            
        Returns:
            Tuple of (aligned volume, coordinate grid)
        """
        # Create normalized coordinate grid
        coords = create_grid_points(volume.shape, volume.device)

        # Scale coordinates to bounding box
        scale = bbox_max - bbox_min
        coords = coords * scale.view(1, 1, 1, 3) + bbox_min.view(1, 1, 1, 3)

        return volume, coords
