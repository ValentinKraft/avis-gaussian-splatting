import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class ParameterMonitor:
    """
    Class for monitoring Gaussian parameter statistics during training.
    Tracks changes in scaling, rotation, position, opacity and other parameters.
    """
    def __init__(self, output_dir):
        """
        Initialize the parameter monitor.
        
        Args:
            output_dir: Directory to save statistics and plots
        """
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "parameter_stats"), exist_ok=True)
        
        self.stats = {
            "iteration": [],
            "scaling_mean": [],
            "scaling_std": [],
            "scaling_min": [],
            "scaling_max": [],
            "rotation_mean_angle": [],
            "rotation_std_angle": [],
            "opacity_mean": [],
            "opacity_std": [],
            "position_delta_mean": [],
            "position_delta_std": []
        }
        
        # Store initial positions for tracking movement
        self.initial_positions = None
        self.last_positions = None

    def update(self, iteration, gaussian_model):
        """
        Update statistics with current Gaussian parameters.
        
        Args:
            iteration: Current training iteration
            gaussian_model: The Gaussian model with parameters to monitor
        """
        with torch.no_grad():
            # Store iteration
            self.stats["iteration"].append(iteration)
            
            # Get scaling statistics
            scaling = gaussian_model.get_scaling
            scaling_np = scaling.detach().cpu().numpy()
            self.stats["scaling_mean"].append(np.mean(scaling_np))
            self.stats["scaling_std"].append(np.std(scaling_np))
            self.stats["scaling_min"].append(np.min(scaling_np))
            self.stats["scaling_max"].append(np.max(scaling_np))
            
            # Get rotation statistics (convert to angles)
            rot = gaussian_model.get_rotation
            # Calculate rotation angle from quaternions
            # For simplicity, we calculate approximate "angle" from quaternion magnitudes
            rot_np = rot.detach().cpu().numpy()
            angles = 2 * np.arccos(np.abs(rot_np[:, 0]))  # Simple approximation from w component
            self.stats["rotation_mean_angle"].append(np.mean(angles))
            self.stats["rotation_std_angle"].append(np.std(angles))
            
            # Get opacity statistics
            opacity = gaussian_model.get_opacity
            opacity_np = opacity.detach().cpu().numpy()
            self.stats["opacity_mean"].append(np.mean(opacity_np))
            self.stats["opacity_std"].append(np.std(opacity_np))
            
            # Track position changes
            positions = gaussian_model.get_xyz
            if self.initial_positions is None:
                self.initial_positions = positions.detach().clone()
                self.last_positions = positions.detach().clone()
                position_delta = torch.zeros_like(positions)
            else:
                position_delta = positions - self.last_positions
                self.last_positions = positions.detach().clone()
                
            position_delta_np = position_delta.detach().cpu().numpy()
            self.stats["position_delta_mean"].append(np.mean(np.abs(position_delta_np)))
            self.stats["position_delta_std"].append(np.std(np.abs(position_delta_np)))
            
            # Save stats periodically
            if iteration % 1000 == 0 or iteration == 1:
                self.save_stats()

    def save_stats(self):
        """
        Save current statistics to disk and generate plots.
        """
        # Save raw statistics as numpy arrays
        stats_path = os.path.join(self.output_dir, "parameter_stats", "stats.npz")
        np.savez(stats_path, **{k: np.array(v) for k, v in self.stats.items()})
        
        # Generate plots
        self._plot_parameter_evolution("scaling", 
                                      ["scaling_mean", "scaling_std", "scaling_min", "scaling_max"],
                                      "Scaling Parameter Evolution")
        
        self._plot_parameter_evolution("rotation", 
                                      ["rotation_mean_angle", "rotation_std_angle"],
                                      "Rotation Parameter Evolution")
        
        self._plot_parameter_evolution("opacity", 
                                      ["opacity_mean", "opacity_std"],
                                      "Opacity Parameter Evolution")
        
        self._plot_parameter_evolution("position_delta", 
                                      ["position_delta_mean", "position_delta_std"],
                                      "Position Change Magnitude")

    def _plot_parameter_evolution(self, name, stat_keys, title):
        """
        Generate and save a plot for parameter evolution.
        
        Args:
            name: Name of parameter (used for filename)
            stat_keys: List of keys from self.stats to plot
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        iterations = self.stats["iteration"]
        
        for key in stat_keys:
            if key in self.stats and len(self.stats[key]) == len(iterations):
                plt.plot(iterations, self.stats[key], label=key.replace("_", " "))
        
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "parameter_stats", f"{name}_evolution.png"), dpi=150)
        plt.close()

def add_parameter_regularization_loss(gaussian_model, scaling_weight=0.001, rotation_weight=0.001):
    """
    Compute regularization loss to encourage diversity in scaling and rotation.
    
    Args:
        gaussian_model: The Gaussian model with parameters
        scaling_weight: Weight for scaling regularization
        rotation_weight: Weight for rotation regularization
        
    Returns:
        Regularization loss (scalar tensor)
    """
    # Get scaling parameters
    scaling = gaussian_model.get_scaling
    
    # Encourage non-uniform scaling (anisotropy)
    # For each point, compute variance across its 3 scaling values
    scaling_reshaped = scaling.view(-1, 3)
    scaling_var_per_point = torch.var(scaling_reshaped, dim=1)
    # We want to maximize variance, so we minimize negative variance
    scaling_reg = -torch.mean(scaling_var_per_point)
    
    # Get rotation parameters
    rotation = gaussian_model.get_rotation
    
    # Encourage diverse rotations by penalizing similarity to identity rotation
    # Identity quaternion is (1, 0, 0, 0)
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotation.device)
    # Compute dot product with identity quaternion
    # When dot is close to 1, rotation is close to identity
    dot_products = torch.sum(rotation * identity_quat, dim=1)
    # We want to minimize dot product with identity (encourage rotation)
    rotation_reg = torch.mean(dot_products * dot_products)
    
    # Combine regularization terms
    reg_loss = scaling_weight * scaling_reg + rotation_weight * rotation_reg
    
    return reg_loss
