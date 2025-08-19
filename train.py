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
            n_points=args.init_n_points,
            volume_transform=volume_transform,
            scene_bounds=scene_bounds,
            noise_std=(
                args.position_noise
                if hasattr(args, "position_noise")
                else opt.position_noise
            ),
        )

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
        if not hasattr(gaussians, "intensities") or gaussians.intensities.numel() == 0:
            num_points = gaussians._xyz.shape[1]
            gaussians.intensities = torch.ones((num_points, 1), device="cuda") * 0.5
            print(f"Initialized {num_points} intensity values to 0.5")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # Initialize viewpoints based on supervision type
    # viewpoint_stack = scene.getTrainCameras().copy() if opt.rgb_supervision else []
    # viewpoint_indices = list(range(len(viewpoint_stack))) if opt.rgb_supervision else []

    # Initialize tracking variables
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_vol_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # Skip GUI network in volume-only mode

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # No SH updates needed for volume-only training

        # Initialize total loss
        loss = 0.0

        # Volume supervision loss
        volume_loss = 0.0
        if volume_supervisor is not None:
            # Ensure model has proper requires_grad set
            print(f"Before compute_loss: xyz requires_grad: {gaussians._xyz.requires_grad}")

            # Compute the volume loss
            vol_loss, vol_metrics = volume_supervisor.compute_loss(gaussians)
            volume_loss = vol_loss.item()
            loss = vol_loss

            # Check the loss gradient status
            print(f"After compute_loss: loss requires_grad: {vol_loss.requires_grad}, loss: {vol_loss.item()}")

            # Log volume metrics
            if tb_writer and iteration % 10 == 0:
                for name, value in vol_metrics.items():
                    tb_writer.add_scalar(f"volume/{name}", value, iteration)

        # Make sure the loss requires gradients before calling backward
        if loss.requires_grad:
            print("Loss requires gradients, calling backward...")
            loss.backward()
        else:
            print("ERROR: Loss does not require gradients! Check gradient chain.")
            # Try to create a dummy loss with gradients
            dummy_loss = (gaussians._xyz.sum() * 0) + loss
            print(f"Created dummy loss with requires_grad: {dummy_loss.requires_grad}")
            dummy_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_vol_loss_for_log = 0.4 * volume_loss + 0.6 * ema_vol_loss_for_log

            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Vol": f"{ema_vol_loss_for_log:.{7}f}",
                }
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
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
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


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
        default="dice",
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
