"""
Parameter monitoring utility for Gaussian Splatting training.
Tracks and visualizes parameter changes during optimization.
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time

class ParameterMonitor:
    """
    Monitors and tracks parameter changes during training.
    Provides visualization and statistics for scaling, rotation, and position changes.
    """
    
    def __init__(self, output_path: str, log_interval: int = 10):
        """
        Initialize parameter monitor.
        
        Args:
            output_path: Directory to save visualizations and logs
            log_interval: How often to log parameter statistics (iterations)
        """
        self.output_path = output_path
        self.log_interval = log_interval
        os.makedirs(os.path.join(output_path, "parameter_stats"), exist_ok=True)
        
        # History tracking for parameters
        self.iterations = []
        self.scaling_history = {
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "x": [],
            "y": [],
            "z": []
        }
        self.rotation_history = {
            "mean": [],
            "std": [],
            "w": [],
            "x": [],
            "y": [],
            "z": []
        }
        self.position_history = {
            "delta_mean": [],
            "delta_max": []
        }
        
        # Track previous positions for measuring change
        self.prev_positions = None
        
        # For timing
        self.start_time = time.time()
    
    def update(self, 
               iteration: int, 
               model_xyz: torch.Tensor, 
               scaling: torch.Tensor, 
               rotation: torch.Tensor) -> Dict[str, float]:
        """
        Update parameter statistics and track changes.
        
        Args:
            iteration: Current training iteration
            model_xyz: Current point positions [3, N] or [N, 3]
            scaling: Current scaling parameters [N, 3]
            rotation: Current rotation quaternions [N, 4]
            
        Returns:
            Dictionary of current statistics
        """
        # Skip if not a logging iteration
        if iteration % self.log_interval != 0:
            return {}
            
        # Track iteration
        self.iterations.append(iteration)
        
        # Make sure tensors are detached and on CPU
        model_xyz = model_xyz.detach().cpu()
        scaling = scaling.detach().cpu()
        rotation = rotation.detach().cpu()
        
        # Handle position change tracking
        current_xyz = model_xyz.clone()
        if model_xyz.shape[0] == 3:  # [3, N] format
            current_xyz = current_xyz.permute(1, 0)  # Convert to [N, 3]
            
        if self.prev_positions is not None:
            # Only compare points that exist in both tensors
            min_points = min(self.prev_positions.shape[0], current_xyz.shape[0])
            position_delta = torch.norm(current_xyz[:min_points] - self.prev_positions[:min_points], dim=1)
            self.position_history["delta_mean"].append(position_delta.mean().item())
            self.position_history["delta_max"].append(position_delta.max().item())
        
        self.prev_positions = current_xyz.clone()
        
        # Track scaling statistics
        self.scaling_history["mean"].append(scaling.mean().item())
        self.scaling_history["std"].append(scaling.std().item())
        self.scaling_history["min"].append(scaling.min().item())
        self.scaling_history["max"].append(scaling.max().item())
        
        # Track per-axis scaling
        if scaling.shape[1] == 3:
            self.scaling_history["x"].append(scaling[:, 0].mean().item())
            self.scaling_history["y"].append(scaling[:, 1].mean().item())
            self.scaling_history["z"].append(scaling[:, 2].mean().item())
        
        # Track rotation statistics
        self.rotation_history["mean"].append(rotation.mean().item())
        self.rotation_history["std"].append(rotation.std().item())
        
        # Track quaternion components
        if rotation.shape[1] == 4:
            self.rotation_history["w"].append(rotation[:, 0].mean().item())
            self.rotation_history["x"].append(rotation[:, 1].mean().item())
            self.rotation_history["y"].append(rotation[:, 2].mean().item())
            self.rotation_history["z"].append(rotation[:, 3].mean().item())
            
        # Create visualization if we have enough data
        if len(self.iterations) >= 3:
            self._create_visualizations(iteration)
        
        # Return current statistics
        current_stats = {
            "scaling_mean": self.scaling_history["mean"][-1],
            "scaling_std": self.scaling_history["std"][-1],
            "rotation_mean": self.rotation_history["mean"][-1],
            "rotation_std": self.rotation_history["std"][-1],
        }
        
        # Add position change stats if available
        if len(self.position_history["delta_mean"]) > 0:
            current_stats["xyz_delta"] = self.position_history["delta_mean"][-1]
            
        # Calculate rate of change over last few iterations
        if len(self.scaling_history["mean"]) >= 3:
            scale_change = (self.scaling_history["mean"][-1] - self.scaling_history["mean"][-3])
            current_stats["scale_change_rate"] = scale_change
            
            rot_change = (self.rotation_history["std"][-1] - self.rotation_history["std"][-3])
            current_stats["rot_change_rate"] = rot_change
        
        return current_stats
    
    def _create_visualizations(self, iteration: int):
        """
        Create visualizations of parameter statistics.
        
        Args:
            iteration: Current iteration number
        """
        # Path for saving the visualization
        viz_path = os.path.join(self.output_path, "parameter_stats", f"params_{iteration:06d}.png")
        
        # Create a 2x2 subplot
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot scaling statistics
        axs[0, 0].plot(self.iterations, self.scaling_history["mean"], label="Mean")
        axs[0, 0].fill_between(
            self.iterations,
            np.array(self.scaling_history["mean"]) - np.array(self.scaling_history["std"]),
            np.array(self.scaling_history["mean"]) + np.array(self.scaling_history["std"]),
            alpha=0.3
        )
        axs[0, 0].set_title("Scaling Parameters")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Scale Value")
        axs[0, 0].legend()
        
        # Plot per-axis scaling
        if len(self.scaling_history["x"]) > 0:
            axs[0, 1].plot(self.iterations, self.scaling_history["x"], label="X Scale")
            axs[0, 1].plot(self.iterations, self.scaling_history["y"], label="Y Scale")
            axs[0, 1].plot(self.iterations, self.scaling_history["z"], label="Z Scale")
            axs[0, 1].set_title("Per-Axis Scaling")
            axs[0, 1].set_xlabel("Iteration")
            axs[0, 1].set_ylabel("Scale Value")
            axs[0, 1].legend()
        
        # Plot rotation statistics
        axs[1, 0].plot(self.iterations, self.rotation_history["mean"], label="Mean")
        axs[1, 0].plot(self.iterations, self.rotation_history["std"], label="Std Dev")
        axs[1, 0].set_title("Rotation Parameters")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Rotation Value")
        axs[1, 0].legend()
        
        # Plot quaternion components
        if len(self.rotation_history["w"]) > 0:
            axs[1, 1].plot(self.iterations, self.rotation_history["w"], label="W")
            axs[1, 1].plot(self.iterations, self.rotation_history["x"], label="X")
            axs[1, 1].plot(self.iterations, self.rotation_history["y"], label="Y")
            axs[1, 1].plot(self.iterations, self.rotation_history["z"], label="Z")
            axs[1, 1].set_title("Quaternion Components")
            axs[1, 1].set_xlabel("Iteration")
            axs[1, 1].set_ylabel("Component Value")
            axs[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(viz_path)
        plt.close(fig)
        
        # Also plot position changes if available
        if len(self.position_history["delta_mean"]) >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(
                self.iterations[1:], 
                self.position_history["delta_mean"], 
                label="Mean Position Change"
            )
            ax.plot(
                self.iterations[1:], 
                self.position_history["delta_max"], 
                label="Max Position Change"
            )
            ax.set_title("Position Changes Between Iterations")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Change Magnitude")
            ax.legend()
            
            pos_viz_path = os.path.join(
                self.output_path, 
                "parameter_stats", 
                f"position_changes_{iteration:06d}.png"
            )
            plt.tight_layout()
            plt.savefig(pos_viz_path)
            plt.close(fig)

    def final_report(self):
        """Generate a final report of parameter changes over training."""
        report_path = os.path.join(self.output_path, "parameter_stats", "final_report.txt")
        
        with open(report_path, "w") as f:
            runtime = time.time() - self.start_time
            f.write(f"Training completed in {runtime:.2f} seconds\n")
            f.write(f"Total iterations: {len(self.iterations)}\n\n")
            
            if len(self.scaling_history["mean"]) > 0:
                initial_scale = self.scaling_history["mean"][0]
                final_scale = self.scaling_history["mean"][-1]
                f.write(f"Scaling parameters:\n")
                f.write(f"  Initial mean: {initial_scale:.6f}\n")
                f.write(f"  Final mean: {final_scale:.6f}\n")
                f.write(f"  Change: {final_scale - initial_scale:.6f}\n")
                f.write(f"  Relative change: {(final_scale - initial_scale)/initial_scale:.2%}\n\n")
            
            if len(self.rotation_history["std"]) > 0:
                initial_rot_std = self.rotation_history["std"][0]
                final_rot_std = self.rotation_history["std"][-1]
                f.write(f"Rotation parameters:\n")
                f.write(f"  Initial std dev: {initial_rot_std:.6f}\n")
                f.write(f"  Final std dev: {final_rot_std:.6f}\n")
                f.write(f"  Change: {final_rot_std - initial_rot_std:.6f}\n\n")
            
            if len(self.position_history["delta_mean"]) > 0:
                total_pos_change = sum(self.position_history["delta_mean"])
                f.write(f"Position changes:\n")
                f.write(f"  Total accumulated change: {total_pos_change:.6f}\n")
                f.write(f"  Average change per step: {total_pos_change/len(self.position_history['delta_mean']):.6f}\n")

def add_parameter_regularization_loss(
    model,
    loss: torch.Tensor,
    scale_diversity_weight: float = 0.001,
    rotation_diversity_weight: float = 0.001
) -> torch.Tensor:
    """
    Add regularization loss terms to encourage diversity in scaling and rotation.
    
    Args:
        model: GaussianModel instance
        loss: Current loss value
        scale_diversity_weight: Weight for scale diversity term
        rotation_diversity_weight: Weight for rotation diversity term
        
    Returns:
        Modified loss with regularization terms
    """
    modified_loss = loss.clone()
    
    # Encourage diverse scaling along different axes (non-uniform scaling)
    if hasattr(model, "_scaling") and model._scaling is not None and model._scaling.numel() > 0:
        scaling = model.get_scaling  # Get actual (non-log) scaling
        
        # Penalize similar scaling along different axes
        if scaling.shape[1] == 3:  # If we have per-axis scaling
            scale_similarity = torch.abs(scaling[:, 0] - scaling[:, 1]) + \
                              torch.abs(scaling[:, 1] - scaling[:, 2]) + \
                              torch.abs(scaling[:, 0] - scaling[:, 2])
            
            # We want to maximize differences, so we minimize the negative
            scale_reg_loss = -scale_similarity.mean() * scale_diversity_weight
            modified_loss = modified_loss + scale_reg_loss
    
    # Encourage non-identity rotations
    if hasattr(model, "_rotation") and model._rotation is not None and model._rotation.numel() > 0:
        # Identity quaternion is [1, 0, 0, 0]
        # We want to encourage deviation from identity
        rot = model.get_rotation
        if rot.shape[1] == 4:  # If we have quaternion rotations
            identity_distance = torch.abs(rot[:, 0] - 1.0) + \
                               torch.norm(rot[:, 1:], dim=1)
                               
            # We want to maximize distance from identity, so minimize negative
            rot_reg_loss = -identity_distance.mean() * rotation_diversity_weight
            modified_loss = modified_loss + rot_reg_loss
    
    return modified_loss
