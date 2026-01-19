# Implementation Summary: Maximum Scaling Constraint & Densification

**Date**: October 7, 2025  
**Goal**: Implement maximum splat size constraint (2x initial) and enable densification/pruning

## Changes Implemented

### 1. Maximum Scaling Constraint (2x Initial Size)

#### Files Modified:
- `scene/gaussian_model.py`
- `gaussian_splatting/utils/volume_initializer.py`
- `train.py`

#### Implementation Details:

**A. GaussianModel Initialization (`scene/gaussian_model.py`)**
```python
# Added to __init__:
self._initial_scaling = torch.empty(0)  # Store initial log-scale values [N, 3]
self.max_scale_factor = 2.0  # Maximum allowed scale = 2x initial scale
```

**B. Modified get_scaling Property**
The `get_scaling` property now enforces the constraint when returning scaling values:
```python
@property
def get_scaling(self) -> torch.Tensor:
    """Get point scaling parameters with maximum size constraint."""
    if self._initial_scaling.numel() > 0:
        # Clamp log-scale to ensure exp(log_scale) <= 2.0 * exp(initial_log_scale)
        max_log_scaling = self._initial_scaling + torch.log(torch.tensor(2.0))
        clamped_scaling = torch.min(self._scaling, max_log_scaling)
        return self.scaling_activation(clamped_scaling)
    else:
        return self.scaling_activation(self._scaling)
```

**C. Added enforce_scaling_constraint() Method**
Called after optimizer steps to hard-clamp the parameter values:
```python
def enforce_scaling_constraint(self):
    """Enforce maximum scaling constraint by clamping scaling parameters."""
    if self._initial_scaling.numel() > 0:
        with torch.no_grad():
            max_log_scaling = self._initial_scaling + torch.log(
                torch.tensor(self.max_scale_factor, device=self._scaling.device)
            )
            self._scaling.data = torch.min(self._scaling.data, max_log_scaling)
```

**D. Store Initial Scales at Initialization**
- In `create_from_pcd()`: Added `self._initial_scaling = scales.clone().detach()`
- In `volume_initializer.py`: Added `model._initial_scaling = torch.log(scales).clone().detach()`

**E. Updated Capture/Restore for Checkpointing**
- `capture()` now returns `_initial_scaling` as last element
- `restore()` now properly restores `_initial_scaling` from saved checkpoints

### 2. Densification and Pruning

#### Files Modified:
- `train.py`

#### Implementation Details:

**A. Added Gradient Accumulation**
For volume-based training (which doesn't have viewspace gradients), we use position gradients:
```python
# Accumulate gradients for densification
if gaussians._xyz.grad is not None:
    xyz_grad_norm = torch.norm(gaussians._xyz.grad, dim=0, keepdim=True).T
    gaussians.xyz_gradient_accum += xyz_grad_norm
    gaussians.denom += 1
```

**B. Periodic Densification & Pruning**
```python
if iteration >= opt.densify_from_iter and iteration <= opt.densify_until_iter:
    if iteration % opt.densification_interval == 0:
        gaussians.densify_and_prune(
            max_grad=opt.densify_grad_threshold,  # Default: 0.0002
            min_opacity=0.005,  # Lower threshold for volume training
            extent=dataset.cameras_extent,
            max_screen_size=None,  # No screen size limit for volume
            radii=None  # No radii for volume training
        )
```

**C. Default Densification Settings** (from `arguments/__init__.py`):
- `densification_interval = 100` (every 100 iterations)
- `densify_from_iter = 500` (start at iteration 500)
- `densify_until_iter = 15_000` (stop at iteration 15,000)
- `densify_grad_threshold = 0.0002` (gradient threshold)
- `opacity_reset_interval = 3000` (reset opacity every 3000 iterations)

## Why Parameters Weren't Changing Before

### Issues Identified:

1. **No Densification**: The training loop had no densification/pruning code
   - Gaussians couldn't split when gradients were high
   - No cloning of under-represented regions
   - No pruning of low-opacity points
   - Result: Static point cloud that couldn't adapt

2. **Possible Learning Rate Issues**: Need to verify learning rates weren't set to zero
   - Check if any command-line args override defaults
   - Verify optimizer param groups are set up correctly

3. **Short Training Duration**: 300 iterations is very short
   - Densification doesn't even start until iteration 500
   - Typical training: 15,000-30,000 iterations
   - For testing: use at least 1,000-2,000 iterations

## How to Verify the Implementation

### Test 1: Run Scaling Constraint Test
```bash
python test_scaling_constraint.py
```
Expected: All tests pass, constraint enforced correctly

### Test 2: Run Densification Test
```bash
python test_densification.py
```
Expected:
- Learning rates are non-zero
- Parameters update after optimizer steps
- Densification can split/clone points
- Scaling constraint is enforced

### Test 3: Run Short Training with Densification
```bash
python train.py --model_path output/densification-test \
    --mask_path _test-data_/vesselmask-float.nii.gz \
    --volume_path _test-data_/volume.nii.gz \
    --iterations 1500 \
    --save_ply_every 100 \
    --volume_loss_type mse \
    --init_n_points 500
```

Expected outputs:
- "[ITER 500] Densification: XXX points" (first densification)
- "[ITER 600] Densification: YYY points" (second densification)
- Point count should change (increase or decrease based on gradients)
- Scaling values should vary but stay ≤ 2x initial
- Rotation quaternions should change significantly
- Loss should decrease more rapidly with adaptive density

## Configuration Recommendations

### For Volume-Based Training:

1. **Iteration Count**: Use 5,000-15,000 iterations minimum
2. **Initial Points**: Start with 500-2,000 points (not 1,000)
3. **Densification Start**: Keep at 500 or lower to 300
4. **Learning Rates**: 
   - Position: 0.00016 → 0.0000016 (default is good)
   - Scaling: 0.005 (default is good)
   - Rotation: 0.001 (default is good)
5. **Densify Until**: Set to 80% of total iterations
6. **Gradient Threshold**: Start with 0.0002, may increase to 0.001 for volume

### Customization Options:

```bash
# Custom densification settings
python train.py ... \
  --densify_from_iter 300 \
  --densify_until_iter 1200 \
  --densify_grad_threshold 0.0005 \
  --densification_interval 50 \
  --opacity_reset_interval 1000
```

## Expected Behavior After Fix

### During Training:

1. **Iterations 1-500**: Initial optimization with fixed point count
   - Scaling and rotation should start changing
   - Loss should decrease
   - Point count: constant

2. **Iterations 500-1500**: Active densification period
   - Point count changes every 100 iterations
   - High-gradient areas get split
   - Low-opacity points get pruned
   - Scaling constrained to ≤ 2x initial
   - More dynamic optimization

3. **Iterations 1500+**: Refinement only
   - No more densification
   - Pure parameter optimization
   - Converge to final solution

### In PLY Files:

- Early iterations (< 500): All points similar size
- Mid iterations (500-1500): Varying point sizes, some at 2x initial
- Late iterations: Optimized distribution with size variations

## Summary

**What was broken:**
- No densification/pruning in training loop
- Splats couldn't adapt to improve quality
- Point cloud was completely static

**What was fixed:**
- ✅ Added maximum 2x scaling constraint
- ✅ Implemented gradient accumulation for volume training
- ✅ Added densification & pruning every 100 iterations
- ✅ Constraint enforcement after each optimizer step
- ✅ Proper checkpoint save/restore for initial scales

**Next steps:**
1. Run test scripts to verify implementation
2. Train for 1500+ iterations to see densification in action
3. Compare PLY files before/after densification
4. Adjust hyperparameters based on results
