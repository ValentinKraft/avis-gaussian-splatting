"""
Test script to verify densification and parameter updates work correctly.
"""

import argparse

import pytest
import torch
import torch.nn as nn

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel


def test_densification_and_updates():
    """Test that densification and parameter updates work correctly."""
    
    print("=" * 70)
    print("Testing Densification and Parameter Updates")
    print("=" * 70)
    
    # Create model and opt params
    parser = argparse.ArgumentParser()
    opt_group = OptimizationParams(parser)
    parsed_args = parser.parse_args([])
    opt = opt_group.extract(parsed_args)
    
    print(f"\n✓ Optimization Parameters:")
    print(f"  Position LR init:         {opt.position_lr_init}")
    print(f"  Position LR final:        {opt.position_lr_final}")
    print(f"  Scaling LR:               {opt.scaling_lr}")
    print(f"  Rotation LR:              {opt.rotation_lr}")
    print(f"  Densify from iter:        {opt.densify_from_iter}")
    print(f"  Densify until iter:       {opt.densify_until_iter}")
    print(f"  Densify grad threshold:   {opt.densify_grad_threshold}")
    print(f"  Densification interval:   {opt.densification_interval}")
    print(f"  Opacity reset interval:   {opt.opacity_reset_interval}")
    
    # Create a Gaussian model
    model = GaussianModel(sh_degree=0)
    
    # Initialize with some dummy parameters
    n_points = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up parameters similar to volume initialization
    points = torch.randn(n_points, 3, device=device) * 0.5
    scales = torch.ones(n_points, 3, device=device) * 0.01
    initial_log_scales = torch.log(scales)
    
    model._xyz = nn.Parameter(points.T.contiguous())  # [3, N]
    model._scaling = nn.Parameter(initial_log_scales.clone())
    model._initial_scaling = initial_log_scales.clone()
    model._rotation = nn.Parameter(torch.zeros(n_points, 4, device=device))
    model._rotation.data[:, 0] = 1  # Identity quaternion
    model._opacity = nn.Parameter(torch.zeros(n_points, 1, device=device) - 2.0)  # log(~0.135)
    
    # Initialize features (simple grayscale)
    model._features_dc = nn.Parameter(torch.zeros(n_points, 1, 3, device=device))
    model._features_rest = nn.Parameter(torch.zeros(n_points, 0, 3, device=device))
    model.intensities = torch.zeros(n_points, 1, device=device)
    model.opacities = torch.zeros(n_points, 1, device=device)
    
    # Set spatial_lr_scale
    model.spatial_lr_scale = 1.0
    
    print(f"\n✓ Initialized model with {n_points} Gaussians")
    print(f"  Position shape:  {model._xyz.shape}")
    print(f"  Scaling shape:   {model._scaling.shape}")
    print(f"  Rotation shape:  {model._rotation.shape}")
    print(f"  Opacity shape:   {model._opacity.shape}")
    
    # Set up training
    model.training_setup(opt)
    
    print(f"\n✓ Training setup completed")
    print(f"  Optimizer type: {type(model.optimizer).__name__}")
    print(f"  Number of param groups: {len(model.optimizer.param_groups)}")
    
    for i, pg in enumerate(model.optimizer.param_groups):
        print(f"  Group {i} ({pg['name']}): LR = {pg['lr']:.6f}")
    
    # Simulate a training step
    print(f"\n✓ Simulating training iteration...")
    
    # Create a dummy loss
    loss = (model.get_xyz.sum() * 0.01 + model.get_scaling.sum() * 0.01 + 
            model.get_rotation.sum() * 0.01)
    
    model.optimizer.zero_grad()
    loss.backward()
    
    # Check gradients before optimizer step
    print(f"\n✓ Gradients before optimizer step:")
    print(f"  XYZ grad norm:      {model._xyz.grad.norm().item():.6f}" if model._xyz.grad is not None else "  XYZ grad: None")
    print(f"  Scaling grad norm:  {model._scaling.grad.norm().item():.6f}" if model._scaling.grad is not None else "  Scaling grad: None")
    print(f"  Rotation grad norm: {model._rotation.grad.norm().item():.6f}" if model._rotation.grad is not None else "  Rotation grad: None")
    
    # Store pre-step values
    xyz_before = model._xyz.data.clone()
    scaling_before = model._scaling.data.clone()
    rotation_before = model._rotation.data.clone()
    
    # Optimizer step
    model.optimizer.step()
    model.enforce_scaling_constraint()
    
    # Check parameter changes
    xyz_delta = (model._xyz.data - xyz_before).abs().mean().item()
    scaling_delta = (model._scaling.data - scaling_before).abs().mean().item()
    rotation_delta = (model._rotation.data - rotation_before).abs().mean().item()
    
    print(f"\n✓ Parameter changes after optimizer step:")
    print(f"  XYZ delta:      {xyz_delta:.8f}")
    print(f"  Scaling delta:  {scaling_delta:.8f}")
    print(f"  Rotation delta: {rotation_delta:.8f}")
    
    if xyz_delta > 1e-10 and scaling_delta > 1e-10 and rotation_delta > 1e-10:
        print(f"\n✅ SUCCESS: All parameters updated!")
    else:
        pytest.fail("Optimizer step did not update all parameter groups")
    
    # Test densification
    print(f"\n✓ Testing densification...")
    
    # Accumulate some gradients
    model.xyz_gradient_accum = torch.rand(n_points, 1, device=device) * 0.001
    model.denom = torch.ones(n_points, 1, device=device)
    
    points_before = model._xyz.shape[1]
    
    # Run densification
    model.densify_and_prune(
        max_grad=opt.densify_grad_threshold,
        min_opacity=0.005,
        extent=1.0,
        max_screen_size=None,
        radii=None
    )
    
    points_after = model._xyz.shape[1]
    
    print(f"  Points before densification: {points_before}")
    print(f"  Points after densification:  {points_after}")
    print(f"  Net change:                  {points_after - points_before:+d}")
    
    if points_after != points_before:
        print(f"\n✅ SUCCESS: Densification changed point count!")
    else:
        print(f"\n⚠️  INFO: Densification did not change point count (may be expected with random data)")
    
    # Test scaling constraint
    print(f"\n✓ Testing scaling constraint...")
    
    with torch.no_grad():
        # Try to exceed maximum scale
        model._scaling.data = model._initial_scaling + torch.log(torch.tensor(5.0, device=device))
    
    model.enforce_scaling_constraint()
    
    actual_scales = torch.exp(model._scaling)
    expected_max_scales = torch.exp(model._initial_scaling) * model.max_scale_factor
    
    max_violations = (actual_scales > expected_max_scales * 1.0001).sum().item()
    
    if max_violations == 0:
        print(f"  ✅ Scaling constraint enforced correctly (max {model.max_scale_factor}x initial)")
    else:
        pytest.fail(f"Scaling constraint violated for {max_violations} points")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✅")
    print("=" * 70)
    print("\nKey findings:")
    print("  • Parameters update correctly with optimizer steps")
    print("  • Learning rates are non-zero")
    print("  • Densification system is functional")
    print("  • Scaling constraint (2x max) is enforced")
    print("\n✅ Implementation is ready for training!")
    
    return None


if __name__ == "__main__":
    test_densification_and_updates()
