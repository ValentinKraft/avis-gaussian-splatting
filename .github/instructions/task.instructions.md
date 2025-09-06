# 🎯 Project: Volume-Based 3D Gaussian Splatting Extension

## 📝 Overview

This project extends the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) framework to optimize Gaussian splats directly from volumetric data instead of 2D images. This advancement enables direct training from 3D medical imaging data (CT, MRI) or segmentation masks without requiring 2D views.

### Justification and Benefits

- **Direct 3D Representation**: Traditional 3D Gaussian Splatting requires multi-view images, which are often unavailable for medical data
- **Medical Application**: Enables working with CT, MRI, and other volumetric medical data formats
- **Parameter Diversity**: Implements specialized techniques to ensure proper scaling and rotation diversity in 3D space
- **Computational Efficiency**: Faster optimization compared to using artificial multi-view rendering for volumetric data
- **Pipeline Integration**: Seamlessly integrates with existing medical imaging workflows

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

### 5. Parameter Diversity Losses (`gaussian_splatting/losses/parameter_diversity_loss.py`)
```python
class ScaleDiversityLoss(nn.Module):
    """Encourages variation in scale parameters"""
    def forward(self, scales, progress_ratio=1.0):
        # Compute variance across scale dimensions to encourage anisotropy
        # Apply orthogonality loss to prevent uniform scaling
        # Use target range loss to maintain reasonable scale values
```

```python
class RotationDiversityLoss(nn.Module):
    """Encourages non-identity rotations"""
    def forward(self, rotations, progress_ratio=1.0):
        # Apply quaternion dispersion loss to avoid identity rotations
        # Use rotation entropy loss to maximize distribution entropy
        # Optional principal direction alignment with volume gradients
```

### 6. Parameter Monitoring (`utils/parameter_monitoring.py`)
```python
class ParameterMonitor:
    """Tracks parameter statistics during training"""
    def track_parameters(self, model):
        # Record parameter statistics for position, scale, rotation
        # Generate visualization plots for parameter evolution
        # Save final reports and statistics to output folder
```

## 🛠️ Current Pipeline Implementation

1. **Initialization**:
   - Load volume and/or mask data from disk
   - Sample points based on volume/mask intensity distribution
   - Initialize Gaussian parameters with appropriate defaults
   - Set up optimization parameters and loss functions

2. **Training Loop**:
   - Forward pass: Voxelize current Gaussians into a volume
   - Compute losses: Volume supervision + parameter diversity losses
   - Backward pass: Compute gradients for all parameters
   - Direct parameter updates: Apply small perturbations periodically
   - Update parameters via optimizer or direct manipulation
   - Monitor and visualize parameter evolution

3. **Output Generation**:
   - Export trained Gaussian model as PLY files
   - Generate parameter statistics and evolution plots
   - Provide performance metrics (timing, memory usage)

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
1. **Gradient Flow Issues**:
   - Ensure all operations maintain gradient flow
   - Check for `.item()` or `.detach()` calls that break the chain
   - Monitor gradient magnitudes during backpropagation
   
2. **Parameter Update Problems**:
   - Use parameter diversity losses to encourage changes
   - Apply direct parameter updates when gradients fail
   - Increase learning rates for scaling and rotation parameters
   
3. **Memory Management**:  
   - Auto-resize large volumes to prevent OOM errors
   - Use appropriate point counts based on available VRAM
   - Batch process large point sets during voxelization

4. **Tensor Shape Consistency**:
   - Be aware of [N,3] vs [3,N] conventions in different parts
   - Ensure proper transposition when passing between components
   - Validate tensor shapes before and after key operations

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

### Parameter Diversity Loss Functions
```python
# Scale Diversity: Encourage anisotropic scaling
def compute_scale_diversity_loss(scales):
    # Variance across dimensions should be high
    scale_var = torch.var(scales, dim=1).mean()
    return -scale_var  # Negative to maximize variance
    
# Rotation Diversity: Encourage non-identity rotations
def compute_rotation_diversity_loss(rotations):
    # Identity quaternion is [1,0,0,0]
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotations.device)
    # Distance from identity should be high
    dist_from_identity = 1.0 - torch.abs(torch.sum(rotations * identity, dim=1))
    return -dist_from_identity.mean()  # Negative to maximize distance
```

## ✅ Implementation Status

### Completed Features
1. **Volume Data Loading**:
   - Support for multiple volume formats (.nii, .mhd, .npy)
   - Auto-resizing of volumes to prevent memory overflow
   - Proper normalization and preprocessing

2. **Volume-Based Initialization**:
   - Sampling points based on volume/mask intensity
   - Setting initial parameters based on point cloud density
   - Efficient initialization with proper device management

3. **Volume Supervision**:
   - Multiple loss functions for volume supervision
   - Efficient batch processing for voxelization
   - Proper handling of gradient flow

4. **Parameter Diversity**:
   - Scale diversity losses to encourage anisotropic scaling
   - Rotation diversity losses to avoid identity rotations
   - Parameter monitoring and visualization during training

### Current Limitations and Ongoing Work
1. **Gradient Flow Issues**:
   - Gradients sometimes fail to flow to all parameters
   - Current workaround uses direct parameter updates
   - Need to identify and fix root causes in computation graph

2. **Performance Optimization**:
   - Large volumes still require significant memory
   - Voxelization process is computationally expensive
   - Opportunity for further batching and optimization

3. **Evaluation Metrics**:
   - Need more comprehensive evaluation of resulting models
   - Comparison with traditional 3DGS on medical data
   - Integration with downstream medical visualization tools

## 🔍 Key Files Modified

1. `train.py`
   - Added volume supervision options
   - Implemented parameter diversity losses
   - Added parameter monitoring and direct updates
   
2. `gaussian_splatting/utils/volume_initializer.py`
   - Implemented point sampling from volumes/masks
   - Created efficient parameter initialization
   - Refactored for better code organization and reuse
   
3. `gaussian_splatting/data/volume_loader.py`
   - Added support for multiple volume formats
   - Implemented auto-resizing and preprocessing
   - Added safety checks and error handling
   
4. `gaussian_splatting/losses/volume_loss.py`
   - Implemented multiple volume loss functions
   - Ensured gradient flow throughout the computation
   - Added support for different comparison metrics

5. `gaussian_splatting/losses/parameter_diversity_loss.py`
   - Added scale diversity loss components
   - Implemented rotation diversity loss
   - Created combined parameter diversity loss

6. `utils/parameter_monitoring.py`
   - Created parameter tracking and visualization
   - Added statistics collection and reporting
   - Implemented diagnostic tools for optimization
