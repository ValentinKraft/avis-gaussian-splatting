# 🎯 Project: Volume-Based 3D Gaussian Splatting Extension

## 📝 Overview

Extend the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) framework to optimize Gaussian splats directly from volumetric data instead of 2D images. This enables direct training from 3D medical imaging data (CT, MRI) or segmentation masks.

## 🔑 Key Components

### 1. Volume Data Loading and Processing (`gaussian_splatting/data/volume_loader.py`)
```python
class VolumeLoader:
    def load_volume(self, path: str) -> torch.Tensor:
        """Load and preprocess volume data.
        
        - Supports .nii, .npy, .mhd formats
        - Auto-resizes to prevent multinomial overflow
        - Normalizes to [0,1] range
        """
```

### 2. Volume Supervision Loss (`gaussian_splatting/losses/volume_loss.py`)
```python
class VolumeLoss:
    def __init__(self, loss_type='dice'):
        """
        Args:
            loss_type: One of ['mse', 'dice', 'tversky', 'kl']
        """
        
    def forward(self, volume_pred, volume_gt):
        """Compare voxelized splats with ground truth volume"""
```

### 3. Splat Voxelization (`gaussian_splatting/utils/splat_to_volume.py`)
```python
def splat_to_volume(
    splats, 
    volume_shape: Tuple[int, int, int],
    gaussian_scale: float = 1.0
) -> torch.Tensor:
    """Rasterize Gaussians into a 3D grid
    
    Args:
        splats: Gaussian model with positions, scales, rotations
        volume_shape: Output volume dimensions (D,H,W)
        gaussian_scale: Global scale factor for Gaussian kernels
    """
```

### 4. Volume-Based Initialization (`gaussian_splatting/utils/volume_initializer.py`)

Key initialization components:

```python
def initialize_from_volume(
    mask_path: str,
    n_points: int = 5000,
    noise_std: float = 0.01, 
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize Gaussian points by sampling from a volume/mask
    
    Returns:
        points: [N, 3] tensor of positions 
        scales: [N, 3] tensor of scales
        opacities: [N, 1] tensor of alpha values
    """

def initialize_gaussians(
    model,
    n_points: int,
    volume_transform: Optional[torch.Tensor] = None, 
    scene_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    volume_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    **kwargs
):
    """Initialize full Gaussian model from volume data
    
    - Samples points from volume/mask
    - Transforms to world space
    - Initializes model tensors with correct shapes
    """
```

## ⚠️ Critical Implementation Details 

### 1. Point Initialization
- Sample point coordinates using torch.multinomial on flattened volume
- Add small noise (std=0.01) to positions
- Scale coordinates to [0,1] range
- Transform to world space if scene_bounds/transform provided

### 2. Tensor Shape Management
- Points tensor: [3, N] after transposing for model
- Scales tensor: [N, 3] for XYZ scales per point
- Opacities tensor: [N, 1] for alpha values
- Rotations tensor: [N, 4] for quaternions
- Device consistency critical throughout

### 3. Model Parameter Initialization
```python
# Initialize with correct shapes immediately
model._xyz = points.T.clone()  # [3, N]
model._scaling = scales.clone()  # [N, 3] 
model._opacity = opacities.clone()  # [N, 1]
model._rotation = torch.zeros((num_points, 4), device=device)
model._rotation[..., 0] = 1  # Identity rotation
model.max_radii2D = torch.zeros(num_points, device=device)
```

### 4. Volume Loading Safety
- Auto-resize large volumes to prevent multinomial overflow
- Example: (512,512,286) → (310,310,173)
- Maintain aspect ratio during resizing
- Use float32 precision for masks/volumes

## 🧪 Testing & Validation

### 1. Test Data
```bash
# Test with synthetic data first
python train.py \
  --volume_supervision \
  --model_path output \
  --init_from_mask \
  --mask_path _test-data_/vesselmask-float.nii.gz \
  --volume_path _test-data_/volume.nii.gz
```

### 2. Common Issues & Solutions
1. Tensor Shape Mismatches
   - Ensure consistent [N,3] vs [3,N] handling
   - Use .T (transpose) at correct points
   
2. Device Placement
   - Set device early in initialization
   - Use .to(device) consistently
   
3. Memory Management  
   - Auto-resize large volumes
   - Use appropriate point counts
   - Monitor VRAM usage

4. Initialization Order
   - Configure device first
   - Sample points/scales/opacities
   - Transform to world space
   - Initialize model tensors

## 🎛️ Configuration Options

```python
# Key Parameters
n_points = 5000  # Number of Gaussian points
noise_std = 0.01  # Position noise during sampling
base_scale = 0.01 * (5000 / n_points) ** (1/3)  # Scale based on density
```

## 📈 Expected Results

1. Initialization should complete without tensor shape/device errors
2. Point distribution should match volume density
3. Scales should be appropriate for point density
4. Transformations should preserve volume structure

## 🗺️ Implementation Path

1. First implement basic volume loading
2. Add point sampling from masks
3. Handle tensor shapes carefully
4. Add world space transforms
5. Implement volume loss
6. Add volume supervision to training

## 🧮 Mathematical Background

### Volume Sampling
Points sampled proportional to volume intensity:
```python
# Add small epsilon to ensure some probability everywhere
probs = mask_flat + 1e-6  
probs = probs / probs.sum()
selected_idx = torch.multinomial(probs, n_points, replacement=True)
```

### Scale Computation
Scales based on point cloud density:
```python
base_scale = 0.01 * (5000 / n_points) ** (1/3)
scales = torch.ones(len(points), 3, device=device) * base_scale
```

### World Space Transform
```python
def transform_points_to_world(points, transform=None, bounds=None):
    """
    points: [N,3] in [0,1]³
    transform: Optional [4,4] matrix
    bounds: Optional (min,max) tensors
    """
```

## 🔍 Key Files to Modify

1. `train.py`
   - Add volume supervision options
   - Initialize from volume/mask
   
2. `gaussian_splatting/utils/volume_initializer.py`
   - Point sampling
   - Model initialization
   
3. `gaussian_splatting/data/volume_loader.py`
   - Volume loading
   - Auto-resizing
   
4. `gaussian_splatting/losses/volume_loss.py`
   - Volume loss computation
   - Loss type implementations
