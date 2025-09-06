# Parameter Diversity Losses for 3D Gaussian Splatting

This implementation adds specialized loss functions to encourage parameter diversity during 3D Gaussian Splatting optimization. It focuses on improving the training of scaling and rotation parameters in volumetric settings.

## Key Components

### 1. Volumetric Loss (gaussian_splatting/losses/volume_loss.py)
- Computes the difference between rendered splats and ground-truth CT volume
- Supports multiple loss types: MSE, Dice, Tversky, KL-Divergence
- Optimizes the positions of the splats through gradient flow

### 2. Scale Diversity Loss (gaussian_splatting/losses/parameter_diversity_loss.py)
- `Variance Loss`: Encourages variation across the three scale dimensions
- `Orthogonality Loss`: Penalizes scales if all three dimensions are similar
- `Target Range Loss`: Pushes scales toward a desired range

### 3. Rotation Diversity Loss (gaussian_splatting/losses/parameter_diversity_loss.py)
- `Quaternion Dispersion Loss`: Encourages rotations to deviate from identity
- `Rotation Entropy Loss`: Maximizes the entropy of the rotation distribution
- `Principal Direction Loss`: Aligns main axes with volume gradients

### 4. Parameter Update Tracking (utils/parameter_update_tracking.py)
- Monitors the magnitude of parameter changes during optimization
- Helps detect stagnation in parameter updates
- Provides real-time feedback on training effectiveness

## Implementation Details

### Gradient Flow
- All parameters have `requires_grad=True` ensured through model._verify_gradient_requirements()
- Volume rendering pipeline maintains gradient connections for proper backpropagation
- Losses are fully compatible with torch.autograd

### Training Integration
- Total loss computed as weighted sum: total_loss = vol_loss + λ₁·scale_loss + λ₂·rot_loss
- Weights increase gradually during training through progress_ratio
- Parameter updates are tracked and logged to validate optimization

### Parameter Activations
- Scales remain positive through scaling_activation = torch.exp
- Quaternions are normalized after updates through rotation_activation = torch.nn.functional.normalize

## Usage

To train with parameter diversity losses:

```bash
python train.py --volume_supervision --model_path output/your_model \
    --init_from_mask --mask_path path/to/mask.nii.gz \
    --volume_path path/to/volume.nii.gz \
    --save_ply_every 10 --iterations 2000 --init_n_points 1000
```

## Monitoring

The implementation provides extensive logging:

- Console output shows parameter update magnitudes
- TensorBoard logs track loss components and parameter statistics
- PLY sequence shows geometric evolution of the Gaussians
