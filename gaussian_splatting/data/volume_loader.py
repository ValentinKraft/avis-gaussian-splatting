"""
VolumeLoader: Loads 3D volumes (.nii, .npy, .mhd) and resamples to target shape.
Follows PyTorch conventions and 3DGS style.
"""
import os
import numpy as np
import torch
from typing import Tuple, Optional
try:
    import nibabel as nib
except ImportError:
    nib = None

class VolumeLoader:
    def __init__(self, data_dir: str, target_shape: Tuple[int, int, int] = (64, 64, 64)):
        self.data_dir = data_dir
        self.target_shape = target_shape
    def load(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.data_dir, filename)
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            if nib is None:
                raise ImportError('nibabel required for .nii files')
            vol = nib.load(path).get_fdata()
        elif filename.endswith('.npy'):
            vol = np.load(path)
        else:
            raise ValueError('Unsupported file type')
        vol = self._resample(vol, self.target_shape)
        vol = torch.from_numpy(vol).float()
        return vol
    def _resample(self, vol: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        from scipy.ndimage import zoom
        factors = [s / float(vol.shape[i]) for i, s in enumerate(shape)]
        return zoom(vol, factors, order=1)
