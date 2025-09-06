import torch
import unittest

from gaussian_splatting.losses.parameter_diversity_loss import (
    ScaleDiversityLoss,
    RotationDiversityLoss,
    compute_parameter_diversity_losses
)


class TestParameterDiversityLoss(unittest.TestCase):
    """Test parameter diversity loss functions"""

    def test_scale_diversity_loss(self):
        """Test scale diversity loss calculation"""
        # Create a batch of uniform scales
        uniform_scales = torch.ones(10, 3)  # All dimensions are 1.0
        
        # Create a batch of diverse scales
        diverse_scales = torch.ones(10, 3)
        diverse_scales[:, 0] = 0.5  # First dimension is 0.5
        diverse_scales[:, 2] = 2.0  # Third dimension is 2.0
        
        # Initialize loss
        loss_fn = ScaleDiversityLoss(
            variance_weight=1.0,
            orthogonality_weight=1.0,
            target_range_weight=1.0
        )
        
        # Compute losses
        uniform_losses = loss_fn(uniform_scales)
        diverse_losses = loss_fn(diverse_scales)
        
        # Variance loss should be lower (better) for diverse scales
        self.assertLess(diverse_losses["variance"].item(), uniform_losses["variance"].item())
        
        # Orthogonality loss should be lower (better) for diverse scales
        self.assertLess(diverse_losses["orthogonality"].item(), uniform_losses["orthogonality"].item())
        
        # Overall loss should be lower for diverse scales
        self.assertLess(diverse_losses["total"].item(), uniform_losses["total"].item())
        
    def test_rotation_diversity_loss(self):
        """Test rotation diversity loss calculation"""
        # Create a batch of identity rotations
        identity_rotations = torch.zeros(10, 4)
        identity_rotations[:, 0] = 1.0  # [1, 0, 0, 0] = identity quaternion
        
        # Create a batch of diverse rotations (90 degrees around different axes)
        diverse_rotations = torch.zeros(10, 4)
        # First half: 90 deg around X axis [0.7071, 0.7071, 0, 0]
        diverse_rotations[:5, 0] = 0.7071
        diverse_rotations[:5, 1] = 0.7071
        # Second half: 90 deg around Y axis [0.7071, 0, 0.7071, 0]
        diverse_rotations[5:, 0] = 0.7071
        diverse_rotations[5:, 2] = 0.7071
        
        # Initialize loss
        loss_fn = RotationDiversityLoss(
            dispersion_weight=1.0,
            entropy_weight=1.0
        )
        
        # Compute losses
        identity_losses = loss_fn(identity_rotations)
        diverse_losses = loss_fn(diverse_rotations)
        
        # Dispersion loss should be lower (better) for diverse rotations
        self.assertLess(diverse_losses["dispersion"].item(), identity_losses["dispersion"].item())
        
        # Entropy loss should be lower (better) for diverse rotations
        self.assertLess(diverse_losses["entropy"].item(), identity_losses["entropy"].item())
        
        # Overall loss should be lower for diverse rotations
        self.assertLess(diverse_losses["total"].item(), identity_losses["total"].item())
        
    def test_alignment_loss_with_gradients(self):
        """Test alignment loss with volume gradients"""
        # Create rotation batch
        rotations = torch.zeros(10, 4)
        rotations[:, 0] = 0.7071
        rotations[:, 1] = 0.7071  # 90 degrees around X
        
        # Create volume gradients in Z direction
        volume_grads = torch.zeros(10, 3)
        volume_grads[:, 2] = 1.0  # Point upward along Z
        
        # Loss with alignment
        loss_fn = RotationDiversityLoss(alignment_weight=1.0)
        losses = loss_fn(rotations, volume_grads)
        
        # Check that alignment loss is non-zero
        self.assertGreater(abs(losses["alignment"].item()), 0)


if __name__ == "__main__":
    unittest.main()
