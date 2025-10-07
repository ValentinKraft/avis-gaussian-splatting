"""
Test script to verify maximum scaling constraint implementation.
"""

import torch
import torch.nn as nn
from scene.gaussian_model import GaussianModel


def test_scaling_constraint():
    """Test that scaling constraint is properly enforced."""
    
    print("=" * 60)
    print("Testing Maximum Scaling Constraint (2x initial size)")
    print("=" * 60)
    
    # Create a simple Gaussian model
    model = GaussianModel(sh_degree=0)
    
    # Verify the max_scale_factor attribute exists
    assert hasattr(model, 'max_scale_factor'), "Model missing max_scale_factor attribute"
    assert model.max_scale_factor == 2.0, f"Expected max_scale_factor=2.0, got {model.max_scale_factor}"
    print(f"✓ max_scale_factor set to: {model.max_scale_factor}")
    
    # Create some dummy parameters
    n_points = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize scales to 0.01 (in real space, log-scale ~ -4.6)
    initial_scales = torch.ones(n_points, 3, device=device) * 0.01
    initial_log_scales = torch.log(initial_scales)
    
    # Set up model parameters
    model._xyz = nn.Parameter(torch.randn(3, n_points, device=device))
    model._scaling = nn.Parameter(initial_log_scales.clone())
    model._initial_scaling = initial_log_scales.clone()
    model._rotation = nn.Parameter(torch.zeros(n_points, 4, device=device))
    model._rotation.data[:, 0] = 1  # Identity quaternion
    model._opacity = nn.Parameter(torch.zeros(n_points, 1, device=device))
    
    print(f"✓ Initialized {n_points} Gaussians")
    print(f"  Initial scale (real): {initial_scales[0, 0].item():.6f}")
    print(f"  Initial scale (log):  {initial_log_scales[0, 0].item():.6f}")
    
    # Simulate optimizer trying to increase scales beyond 2x
    with torch.no_grad():
        # Try to set scales to 5x the initial value (should be clamped to 2x)
        model._scaling.data = initial_log_scales + torch.log(torch.tensor(5.0, device=device))
    
    print(f"\n✓ Attempted to increase scales to 5x initial")
    print(f"  New scale (log): {model._scaling[0, 0].item():.6f}")
    
    # Enforce constraint
    model.enforce_scaling_constraint()
    
    print(f"\n✓ Applied enforce_scaling_constraint()")
    print(f"  Constrained scale (log): {model._scaling[0, 0].item():.6f}")
    
    # Verify the constraint
    actual_scales = torch.exp(model._scaling)
    expected_max_scales = initial_scales * model.max_scale_factor
    
    print(f"\n✓ Verification:")
    print(f"  Actual scale:     {actual_scales[0, 0].item():.6f}")
    print(f"  Expected max:     {expected_max_scales[0, 0].item():.6f}")
    print(f"  Initial scale:    {initial_scales[0, 0].item():.6f}")
    print(f"  Scale multiplier: {(actual_scales[0, 0] / initial_scales[0, 0]).item():.2f}x")
    
    # Check all points
    max_violations = (actual_scales > expected_max_scales * 1.0001).sum().item()  # Small tolerance for float precision
    
    if max_violations == 0:
        print(f"\n✅ SUCCESS: All {n_points} Gaussians respect the 2x scaling constraint!")
    else:
        print(f"\n❌ FAILURE: {max_violations} Gaussians exceed 2x constraint!")
        return False
    
    # Test get_scaling property
    print(f"\n✓ Testing get_scaling property with constraint...")
    scaling_from_property = model.get_scaling
    print(f"  Scale from property: {scaling_from_property[0, 0].item():.6f}")
    print(f"  Expected max:        {expected_max_scales[0, 0].item():.6f}")
    
    # Verify property also respects constraint
    if (scaling_from_property <= expected_max_scales * 1.0001).all():
        print(f"\n✅ SUCCESS: get_scaling property also respects constraint!")
    else:
        print(f"\n❌ FAILURE: get_scaling property violates constraint!")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_scaling_constraint()
    exit(0 if success else 1)
