#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import sys
import numpy as np
from scene.gaussian_model import GaussianModel
from scene.volume_scene import VolumeScene
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_splatting.utils.parameter_monitor import (
    ParameterMonitor,
)
from utils.parameter_monitoring import add_parameter_regularization_loss
from torch.cuda.amp import autocast, GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)

    # Set default SH degree and create gaussians
    gaussians = GaussianModel(
        0, opt.optimizer_type
    )  # Use degree 0 for volume-only training

    # Initialize parameter monitoring with increased log interval for better performance
    parameter_monitor = ParameterMonitor(
        args.model_path, log_interval=50
    )  # Changed from default 10 to 50

    # Initialize parameter update tracker
    from utils.parameter_update_tracking import ParameterUpdateTracker

    update_tracker = ParameterUpdateTracker()

    # Create scene for volume-based training
    scene = VolumeScene(args, gaussians)

    # Initialize from segmentation mask if requested
    if args.init_from_mask:
        if not args.mask_path:
            sys.exit("Error: --mask_path required when using --init_from_mask")

        from gaussian_splatting.utils.volume_initializer import initialize_gaussians

        # Load volume transform if provided
        volume_transform = None
        if args.volume_transform:
            volume_transform = (
                torch.from_numpy(np.load(args.volume_transform)).float().cuda()
            )

        # Get scene bounds for scaling
        scene_bounds = None
        if dataset.cameras_extent is not None:
            scene_bounds = (
                torch.tensor([-dataset.cameras_extent], device="cuda"),
                torch.tensor([dataset.cameras_extent], device="cuda"),
            )

        # Initialize gaussians by sampling from mask
        initialize_gaussians(
            model=gaussians,  # Make the argument explicit
            mask_path=args.mask_path,
            volume_path=args.volume_path,  # Added volume_path for proper intensity sampling
            n_points=args.init_n_points,
            volume_transform=volume_transform,
            scene_bounds=scene_bounds,
            noise_std=(
                args.position_noise
                if hasattr(args, "position_noise")
                else opt.position_noise
            ),
        )

        # Set spatial_lr_scale after volume initialization
        # This is critical for position learning rate to be non-zero
        if hasattr(dataset, "cameras_extent") and dataset.cameras_extent is not None:
            gaussians.spatial_lr_scale = dataset.cameras_extent
        else:
            gaussians.spatial_lr_scale = 1.0  # Default fallback

        print(f"Set spatial_lr_scale to {gaussians.spatial_lr_scale}")

    # Initialize volume supervision if enabled
    volume_supervisor = None
    if args.volume_supervision and args.volume_path:
        from gaussian_splatting.utils.volume_supervisor import VolumeSupervisor

        volume_supervisor = VolumeSupervisor(
            volume_path=args.volume_path,
            volume_shape=tuple(
                args.volume_shape if hasattr(args, "volume_shape") else opt.volume_shape
            ),
            # Pass mask path if available - use the same mask for both initialization and opacity
            mask_path=args.mask_path if hasattr(args, "mask_path") else None,
            loss_type=(
                args.volume_loss_type
                if hasattr(args, "volume_loss_type")
                else opt.volume_loss_type
            ),
            loss_weight=(
                args.volume_loss_weight
                if hasattr(args, "volume_loss_weight")
                else opt.volume_loss_weight
            ),
        )

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

        # Initialize intensity values if they don't exist
        # if not hasattr(gaussians, "intensities") or gaussians.intensities.numel() == 0:
        #     num_points = gaussians._xyz.shape[1]
        #     gaussians.intensities = torch.ones((num_points, 1), device="cuda") * 0.5
        #     print(f"Initialized {num_points} intensity values to 0.5")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Initialize mixed precision training
    scaler = GradScaler()
    use_amp = (
        not args.disable_mixed_precision
    )  # Use mixed precision unless explicitly disabled
    if use_amp:
        print("Using mixed precision training for better performance")
    else:
        print("Mixed precision training disabled")

    # Initialize tracking variables
    ema_loss_for_log = 0.0
    ema_vol_loss_for_log = 0.0
    param_stats = {"scale_change_rate": 0.0, "rot_change_rate": 0.0}

    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="#### Training progress ####"
    )
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Skip GUI network in volume-only mode

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        gaussians.optimizer.zero_grad()

        # No SH updates needed for volume-only training

        # Initialize total loss
        loss = 0.0
        # Volume supervision loss
        volume_loss = 0.0

        if volume_supervisor is not None:
            # Compute the volume loss and get volume gradients for parameter diversity loss
            with autocast(enabled=use_amp):
                vol_loss, vol_metrics, vol_gradients = volume_supervisor.compute_loss(
                    gaussians
                )

                # CRITICAL: Don't call item() on the loss until after backward() is called!
                # This would break the computation graph
                loss = vol_loss

            # Store value for logging only
            volume_loss = vol_loss.detach().item()

            # Configure loss weights based on training stage
            # Use stronger weights for parameter diversity to force gradient flow
            # Temporarily increased for testing
            scale_diversity_weight = 0.05
            rotation_diversity_weight = 0.05

            # Add parameter diversity losses to encourage scaling and rotation diversity
            # This includes scale diversity, orthogonality, target range, quaternion dispersion,
            # rotation entropy, and principal direction alignment
            # reg_loss = add_parameter_regularization_loss(
            #     model=gaussians,
            #     loss=loss,
            #     scale_diversity_weight=scale_diversity_weight,
            #     rotation_diversity_weight=rotation_diversity_weight,
            #     scale_variance_weight=0.2,
            #     target_range_weight=0.2,
            #     dispersion_weight=0.2,
            #     alignment_weight=0.2,
            #     volume_gradients=vol_gradients,
            # )
            # loss = reg_loss  # Use the regularized loss

            # Track parameter statistics for monitoring (only on every 50th iteration)
            if iteration % 50 == 0:
                new_stats = parameter_monitor.update(
                    iteration,
                    gaussians._xyz,
                    gaussians.get_scaling,
                    gaussians.get_rotation,
                    loss=loss.item(),
                    volume_loss=vol_loss.item() if vol_loss is not None else None,
                    reg_loss=None,  # No regularization loss yet
                )
                # Update our param_stats dictionary with new values
                if new_stats:
                    param_stats.update(new_stats)

            # Log volume metrics
            if tb_writer and iteration % 10 == 0:
                for name, value in vol_metrics.items():
                    tb_writer.add_scalar(f"volume/{name}", value, iteration)

                # Also log diversity loss components
                tb_writer.add_scalar(
                    "diversity/scale_weight", scale_diversity_weight, iteration
                )
                tb_writer.add_scalar(
                    "diversity/rotation_weight", rotation_diversity_weight, iteration
                )
                # Log regularization loss
                # tb_writer.add_scalar("loss/regularization", reg_loss.item(), iteration)

        # Make sure the loss requires gradients before calling backward
        if loss.requires_grad:
            # Make sure parameters require gradients BEFORE calling backward
            # Ensure core parameters keep identity as nn.Parameter (never reassign tensor objects)
            for name in ["_xyz", "_scaling", "_rotation"]:
                p = getattr(gaussians, name, None)
                if p is None or not isinstance(p, torch.nn.Parameter):
                    # Skip silently if absent – initialization should have created them
                    continue
                if not p.requires_grad:
                    print(
                        f"WARNING: {name} had requires_grad=False – enabling in-place."
                    )
                    p.requires_grad_(True)

            # Use gradient scaler for mixed precision training
            scaler.scale(loss).backward()

            # Debug: Save pre-step values to verify updates happen
            if iteration <= 5 or iteration % 500 == 0:
                pre_xyz_mean = gaussians._xyz.mean().item()
                pre_scaling_mean = gaussians._scaling.mean().item()

            # Debug gradients less frequently to improve performance
            if iteration % 50 == 0:  # Reduced from every 10th to every 50th iteration
                # Check if gradients exist and what their magnitudes are
                with torch.no_grad():  # Ensure we don't track unnecessary operations
                    if gaussians._xyz.grad is not None:
                        print(f"XYZ grad norm: {gaussians._xyz.grad.norm().item():.6f}")

                    if gaussians._scaling.grad is not None:
                        print(
                            f"Scaling grad norm: {gaussians._scaling.grad.norm().item():.6f}"
                        )

                    if gaussians._rotation.grad is not None:
                        print(
                            f"Rotation grad norm: {gaussians._rotation.grad.norm().item():.6f}"
                        )

            # Clip gradients to prevent numerical instability
            if iteration > 1:
                clip_candidates = []

                if gaussians._xyz.grad is not None:
                    clip_candidates.append(gaussians._xyz)

                if gaussians._scaling.grad is not None:
                    clip_candidates.append(gaussians._scaling)

                if gaussians._rotation.grad is not None:
                    clip_candidates.append(gaussians._rotation)

                if (
                    hasattr(gaussians, "_features_dc")
                    and gaussians._features_dc is not None
                    and gaussians._features_dc.requires_grad
                    and gaussians._features_dc.grad is not None
                ):
                    clip_candidates.append(gaussians._features_dc)

                if clip_candidates:
                    torch.nn.utils.clip_grad_norm_(clip_candidates, max_norm=10.0)

            # Apply optimizer step with gradient scaler
            scaler.step(gaussians.optimizer)
            scaler.update()

            # Debug: Check if scaler skipped the step (happens when gradients are inf/nan)
            if iteration <= 10 or iteration % 500 == 0:
                scale = scaler.get_scale()
                print(f"[ITER {iteration}] Scaler scale: {scale:.2f}")

                # Check if parameters actually changed
                if iteration <= 5 or iteration % 500 == 0:
                    post_xyz_mean = gaussians._xyz.mean().item()
                    post_scaling_mean = gaussians._scaling.mean().item()
                    xyz_change = abs(post_xyz_mean - pre_xyz_mean)
                    scaling_change = abs(post_scaling_mean - pre_scaling_mean)
                    print(f"  XYZ mean change: {xyz_change:.10f}")
                    print(f"  Scaling mean change: {scaling_change:.10f}")

            # Verify learning rates on first few iterations
            if iteration <= 3:
                print(f"\n[ITER {iteration}] Learning Rates:")
                for i, group in enumerate(gaussians.optimizer.param_groups):
                    print(f"  {group['name']:15s}: {group['lr']:.8f}")

                # Also print spatial_lr_scale for debugging
                print(f"  spatial_lr_scale: {gaussians.spatial_lr_scale:.8f}")

            # Enforce maximum scaling constraint (2x initial size)
            gaussians.enforce_scaling_constraint()

            # Adaptive density control for volume-based training
            with torch.no_grad():
                # For volume-based training, use position gradients instead of viewspace gradients
                if (
                    iteration >= opt.densify_from_iter
                    and iteration <= opt.densify_until_iter
                ):
                    # Accumulate gradients for densification (every iteration during densification period)
                    if gaussians._xyz.grad is not None:
                        # _xyz has shape [3, N], so grad also has shape [3, N]
                        # Compute norm across the 3D dimension (dim=0) to get magnitude per point
                        # Result shape: [N]
                        xyz_grad_norm = torch.norm(
                            gaussians._xyz.grad, dim=0, keepdim=False
                        )
                        # Reshape to [N, 1] to match xyz_gradient_accum shape
                        xyz_grad_norm = xyz_grad_norm.unsqueeze(1)
                        gaussians.xyz_gradient_accum += xyz_grad_norm
                        gaussians.denom += 1

                    # Perform densification and pruning at intervals
                    if iteration % opt.densification_interval == 0:
                        # Get scene extent (use a default if not available)
                        extent = (
                            dataset.cameras_extent
                            if hasattr(dataset, "cameras_extent")
                            else 1.0
                        )

                        # Perform densification and pruning
                        gaussians.densify_and_prune(
                            max_grad=opt.densify_grad_threshold,
                            min_opacity=0.005,  # Lower threshold for volume-based training
                            extent=extent,
                            max_screen_size=None,  # No screen size limit for volume training
                            radii=None,  # No radii for volume training
                        )

                        # Log densification
                        if iteration % 100 == 0 or iteration == opt.densify_from_iter:
                            print(
                                f"\n[ITER {iteration}] Densification: {gaussians._xyz.shape[1]} points"
                            )

            # Debug gradients less frequently to improve performance
            if iteration % 50 == 0:  # Reduced from every 10th to every 50th iteration
                # Check if gradients exist and what their magnitudes are
                with torch.no_grad():  # Ensure we don't track unnecessary operations
                    if gaussians._xyz.grad is not None:
                        print(f"XYZ grad norm: {gaussians._xyz.grad.norm().item():.6f}")

                    if gaussians._scaling.grad is not None:
                        print(
                            f"Scaling grad norm: {gaussians._scaling.grad.norm().item():.6f}"
                        )

                    if gaussians._rotation.grad is not None:
                        print(
                            f"Rotation grad norm: {gaussians._rotation.grad.norm().item():.6f}"
                        )

            # Add a manual gradient perturbation if gradients are zero
            # This is a drastic measure to force parameter updates
            # REMOVE direct random parameter perturbations (they break true gradient-based optimization)
        else:
            # This should no longer happen with the fixed gradient chain
            print("WARNING: Loss does not require gradients! Check gradient chain.")
            dummy_loss = (gaussians._xyz.sum() * 0) + loss
            dummy_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            # Update EMA loss values
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_vol_loss_for_log = 0.4 * volume_loss + 0.6 * ema_vol_loss_for_log

            # Get scaling and rotation stats for logging
            with torch.no_grad():
                scaling = gaussians.get_scaling
                scaling_mean = scaling.mean().item()
                scaling_std = scaling.std().item()

                rotation = gaussians.get_rotation
                # Simple approximation of rotation "magnitude"
                rotation_magnitude = torch.norm(rotation[:, 1:], dim=1).mean().item()

                # Calculate changes from previous iterations
                scale_change = param_stats.get("scale_change_rate", 0.0)
                rot_change = param_stats.get("rot_change_rate", 0.0)

            # Update progress bar every iteration
            postfix = {
                "Loss": f"{ema_loss_for_log:.{5}f}",
                "Vol": f"{ema_vol_loss_for_log:.{5}f}",
                "Scale": f"{scaling_mean:.{3}f}±{scaling_std:.{3}f}",
                "Rot": f"{rotation_magnitude:.{3}f}",
                "Δs": f"{scale_change:.{3}f}",
                "Δr": f"{rot_change:.{3}f}",
            }
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)  # Update by 1 each iteration
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar("loss/total", loss.item(), iteration)
                tb_writer.add_scalar("loss/volume", volume_loss, iteration)
                tb_writer.add_scalar(
                    "timing/iter_ms", iter_start.elapsed_time(iter_end), iteration
                )
                tb_writer.add_scalar("model/points", gaussians._xyz.shape[1], iteration)

                # Log parameter statistics
                tb_writer.add_scalar("parameters/scaling_mean", scaling_mean, iteration)
                tb_writer.add_scalar("parameters/scaling_std", scaling_std, iteration) 
                tb_writer.add_scalar("parameters/rotation_magnitude", rotation_magnitude, iteration)

                # Track parameter updates to verify optimization is working
                update_metrics = update_tracker.update(gaussians)
                for name, value in update_metrics.items():
                    tb_writer.add_scalar(f"updates/{name}", value, iteration)

                # Log parameter update metrics to the console periodically
                if iteration % 100 == 0:
                    xyz_delta = update_metrics.get("xyz_delta_avg", 0)
                    scale_delta = update_metrics.get("scaling_delta_avg", 0)
                    rot_delta = update_metrics.get("rotation_delta_avg", 0)
                    print(
                        f"\n[ITER {iteration}] Parameter updates - XYZ: {xyz_delta:.5f}, Scale: {scale_delta:.5f}, Rot: {rot_delta:.5f}"
                    )

            # Save PLY file at specified iterations
            save_ply_every = (
                args.save_ply_every if hasattr(args, "save_ply_every") else 1
            )
            if (
                iteration % save_ply_every == 0
                or iteration == 1
                or iteration == opt.iterations
            ):
                ply_output_dir = os.path.join(args.model_path, "ply_sequence")
                prefix = (
                    args.ply_output_prefix
                    if hasattr(args, "ply_output_prefix")
                    else "gaussians"
                )
                ply_output_path = gaussians.save_ply_sequence(
                    ply_output_dir, iteration, prefix
                )

                # Log PLY saving every 100 iterations to avoid console spam
                if (
                    iteration % 100 == 0
                    or iteration == 1
                    or iteration == opt.iterations
                ):
                    print(f"\n[ITER {iteration}] Saved model as PLY: {ply_output_path}")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Reset opacity periodically
            if iteration % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.enforce_scaling_constraint()  # Enforce maximum scaling constraint
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Generate parameter report on last iteration
            if iteration == opt.iterations:
                # Force a final parameter update for the report
                final_stats = parameter_monitor.update(
                    iteration,
                    gaussians._xyz,
                    gaussians.get_scaling,
                    gaussians.get_rotation,
                    force=True,  # Force update regardless of log interval
                    loss=loss.item(),
                    volume_loss=vol_loss.item() if vol_loss is not None else None,
                    reg_loss=None  # No regularization loss yet
                )
                parameter_monitor.final_report()
                print(
                    "\nParameter monitoring report saved to:",
                    os.path.join(args.model_path, "parameter_stats"),
                )


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, loss, elapsed):
    """Simple training report for volume-based training"""
    if tb_writer:
        tb_writer.add_scalar("train/loss", loss.item(), iteration)
        tb_writer.add_scalar("train/iter_time_ms", elapsed, iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # Core parameter groups
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Volume supervision arguments
    volume_group = parser.add_argument_group("Volume Supervision")
    volume_group.add_argument(
        "--volume_supervision",
        action="store_true",
        help="Enable volume supervision loss",
    )
    volume_group.add_argument(
        "--volume_path",
        type=str,
        default="",
        help="Path to ground truth volume file (.nii, .npy, .mhd)",
    )
    volume_group.add_argument(
        "--volume_loss_type",
        type=str,
        default="mse",
        choices=["mse", "dice", "tversky", "kl"],
        help="Type of volume supervision loss",
    )
    volume_group.add_argument(
        "--volume_loss_weight",
        type=float,
        default=1.0,
        help="Weight for volume supervision loss",
    )
    volume_group.add_argument(
        "--volume_shape",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Target shape for volume supervision (D, H, W)",
    )

    # Volume initialization arguments
    init_group = parser.add_argument_group("Volume Initialization")
    init_group.add_argument(
        "--init_from_mask",
        action="store_true",
        help="Initialize Gaussian points by sampling from segmentation mask",
    )
    init_group.add_argument(
        "--mask_path",
        type=str,
        default="",
        help="Path to segmentation mask file (.nii, .npy, .mhd)",
    )
    init_group.add_argument(
        "--volume_transform",
        type=str,
        default="",
        help="Path to 4x4 transform matrix for volume alignment (.npy)",
    )
    init_group.add_argument(
        "--init_n_points",
        type=int,
        default=5000,
        help="Number of points to sample from mask",
    )
    init_group.add_argument(
        "--position_noise",
        type=float,
        default=0.01,
        help="Standard deviation of position noise for initialization",
    )

    # PLY saving options
    ply_group = parser.add_argument_group("PLY Export Options")
    ply_group.add_argument(
        "--save_ply_every",
        type=int,
        default=1,
        help="Save PLY file every N iterations (default: 1 = every iteration)",
    )
    ply_group.add_argument(
        "--ply_output_prefix",
        type=str,
        default="gaussians",
        help="Prefix for PLY filenames",
    )

    # Other arguments
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--disable_mixed_precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Create dummy dataset for volume-only mode
    if not args.source_path and args.init_from_mask:

        class DummyDataset:
            def __init__(self, model_path):
                self.cameras_extent = 1.0  # Default extent
                self.white_background = False
                self.model_path = model_path
                self.source_path = ""
                self.sh_degree = 0

        dataset = DummyDataset(args.model_path)
    else:
        dataset = lp.extract(args)

    # Start training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset,
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,  # Pass the full arguments
    )

    # All done
    print("\nTraining complete.")
