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
import nibabel as nib
import SimpleITK as sitk
import torch.nn.functional as F

class VolumeLoader:
    """
    Loader for volumetric data supporting various medical imaging formats.
    Handles loading, resampling, and coordinate system alignment.
    """
    
    def __init__(self, 
                 target_shape: Tuple[int, int, int] = (64, 64, 64),
                 device: torch.device = torch.device('cuda')):
        """
        Args:
            target_shape: Desired output volume shape
            device: Device to load tensors to
        """
        self.target_shape = target_shape
        self.device = device
        
    def load_nifti(self, path: Union[str, Path]) -> Tensor:
        """Load a NIfTI volume file."""
        nii = nib.load(str(path))
        volume = torch.from_numpy(nii.get_fdata()).float()
        return self._process_volume(volume)
    
    def load_npy(self, path: Union[str, Path]) -> Tensor:
        """Load a NumPy volume file."""
        volume = torch.from_numpy(np.load(str(path))).float()
        return self._process_volume(volume)
        
    def load_mhd(self, path: Union[str, Path]) -> Tensor:
        """Load a MetaImage (MHD/Raw) volume file."""
        img = sitk.ReadImage(str(path))
        volume = torch.from_numpy(sitk.GetArrayFromImage(img)).float()
        return self._process_volume(volume)
    
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
        elif path.suffix == '.mhd':
            volume = self.load_mhd(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        return volume
    
    def _process_volume(self, volume: Tensor) -> Tensor:
        """
        Process loaded volume:
        1. Normalize to [0, 1]
        2. Resample to target shape
        3. Move to device
        """
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Add batch and channel dimensions for resampling
        volume = volume.unsqueeze(0).unsqueeze(0)
        
        # Resample to target shape
        volume = F.interpolate(
            volume,
            size=self.target_shape,
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
