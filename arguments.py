"""Unified, well-documented CLI surface for the volume-only trainer."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
import os
import sys


class GroupParams(SimpleNamespace):
    """Simple namespace used for legacy compatibility in training/render scripts."""


class ParamGroup:
    """Tracks which flags belong to a parameter block so `.extract` can copy them."""

    def __init__(self) -> None:
        self._fields: list[str] = []

    def _register(self, name: str) -> None:
        self._fields.append(name)

    def extract(self, args: Namespace) -> GroupParams:
        params = GroupParams()
        for field in self._fields:
            setattr(params, field, getattr(args, field))
        return params


class ModelParams(ParamGroup):
    """Filesystem inputs and data-shaping configuration exposed to end users."""

    def __init__(self, parser: ArgumentParser, sentinel: bool = False) -> None:
        super().__init__()

        core = parser.add_argument_group("Core Volume Inputs")

        # Where checkpoints and exported PLYs live.
        core.add_argument(
            "--model_path",
            type=str,
            required=not sentinel,
            help="Output directory that stores checkpoints, logs, and PLY exports.",
        )
        self._register("model_path")

        # Supervision volume (nii/npy/mhd) resampled during training.
        core.add_argument(
            "--volume_path",
            type=str,
            required=True,
            help="Path to the CT/MR volume that acts as the optimization target.",
        )
        self._register("volume_path")

        # Segmentation mask for initializing Gaussian seeds.
        core.add_argument(
            "--mask_path",
            type=str,
            required=True,
            help="Binary/probability mask used to sample initial Gaussians.",
        )
        self._register("mask_path")

        # Output resolution for voxelized supervision.
        core.add_argument(
            "--volume_shape",
            type=int,
            nargs=3,
            default=[64, 64, 64],
            metavar=("D", "H", "W"),
            help="Voxel grid resolution used when loading and supervising the volume.",
        )
        self._register("volume_shape")

        # Optional 4x4 transform to align volume/mask with world coordinates.
        core.add_argument(
            "--volume_transform",
            type=str,
            default="",
            help="Numpy 4x4 transform bringing volume voxels into the training frame.",
        )
        self._register("volume_transform")

        # Loss family used when comparing rendered splats to the target volume.
        core.add_argument(
            "--volume_loss_type",
            type=str,
            default="dice",
            choices=["mse", "dice", "tversky", "kl"],
            help="Volume-domain loss to optimize (dice/mse/tversky/kl).",
        )
        self._register("volume_loss_type")

        # Scalar weight applied to the chosen volume loss.
        core.add_argument(
            "--volume_loss_weight",
            type=float,
            default=1.0,
            help="Weight applied to the chosen volume supervision loss.",
        )
        self._register("volume_loss_weight")

        # Number of Gaussians sampled from the mask.
        core.add_argument(
            "--init_n_points",
            type=int,
            default=5000,
            help="How many Gaussians to sample from the mask during initialization.",
        )
        self._register("init_n_points")

        # Random jitter applied to initial Gaussian positions.
        core.add_argument(
            "--position_noise",
            type=float,
            default=0.01,
            help="Std-dev of positional noise added to mask samples at init time.",
        )
        self._register("position_noise")

        legacy = parser.add_argument_group("Legacy RGB Inputs (kept for compatibility)")

        # COLMAP/SfM dataset directory.
        legacy.add_argument(
            "--source_path",
            type=str,
            default="",
            help="Directory with RGB assets (ignored for pure volume training).",
        )
        self._register("source_path")

        # Relative image folder.
        legacy.add_argument(
            "--images",
            type=str,
            default="images",
            help="Image folder name inside --source_path (RGB workflows only).",
        )
        self._register("images")

        # Relative depth folder.
        legacy.add_argument(
            "--depths",
            type=str,
            default="",
            help="Depth folder name inside --source_path (RGB workflows only).",
        )
        self._register("depths")

        # Target render resolution.
        legacy.add_argument(
            "--resolution",
            type=int,
            default=-1,
            help="Image resolution override for RGB renders (ignored for volumes).",
        )
        self._register("resolution")

        # Spherical-harmonics degree for RGB appearance.
        legacy.add_argument(
            "--sh_degree",
            type=int,
            default=3,
            help="Maximum SH degree used when RGB rendering is enabled.",
        )
        self._register("sh_degree")

        # Background color toggle for legacy renders.
        legacy.add_argument(
            "--white_background",
            action="store_true",
            help="Render RGB frames on white instead of black background.",
        )
        self._register("white_background")

        # Device used to stage datasets before training.
        legacy.add_argument(
            "--data_device",
            type=str,
            default="cuda",
            help="Device used to host dataset tensors (cuda or cpu).",
        )
        self._register("data_device")

        # Toggles train/test exposure split (RGB feature, retained for parity).
        legacy.add_argument(
            "--train_test_exp",
            action="store_true",
            help="Enable separate train/test exposure parameters (RGB only).",
        )
        self._register("train_test_exp")

        # Evaluation-only flag.
        legacy.add_argument(
            "--eval",
            action="store_true",
            help="Skip training and run evaluation only (legacy behavior).",
        )
        self._register("eval")

    def extract(self, args: Namespace) -> GroupParams:  # type: ignore[override]
        params = super().extract(args)
        params.source_path = os.path.abspath(params.source_path)
        return params


class OptimizationParams(ParamGroup):
    """Learning schedules, densification knobs, and regularization weights."""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__()

        schedule = parser.add_argument_group("Training Schedule & Optimizer")

        # Total number of optimization iterations.
        schedule.add_argument(
            "--iterations",
            type=int,
            default=30_000,
            help="Total number of gradient steps to run before finishing training.",
        )
        self._register("iterations")

        # Choice of optimizer backend.
        schedule.add_argument(
            "--optimizer_type",
            type=str,
            default="default",
            choices=["default", "sparse_adam"],
            help="Optimizer backend (use 'sparse_adam' only when supported).",
        )
        self._register("optimizer_type")

        # Random background blending for RGB rendering.
        schedule.add_argument(
            "--random_background",
            action="store_true",
            help="Enable random background colors for RGB supervision.",
        )
        self._register("random_background")

        lr = parser.add_argument_group("Learning-Rate Schedule")

        # Initial center learning rate.
        lr.add_argument(
            "--position_lr_init",
            type=float,
            default=0.00016,
            help="Initial learning rate for Gaussian centers.",
        )
        self._register("position_lr_init")

        # Final center learning rate.
        lr.add_argument(
            "--position_lr_final",
            type=float,
            default=0.0000016,
            help="Final learning rate for Gaussian centers.",
        )
        self._register("position_lr_final")

        # Delay multiplier for the LR ramp.
        lr.add_argument(
            "--position_lr_delay_mult",
            type=float,
            default=0.01,
            help="Multiplier that delays how fast the position LR ramps up.",
        )
        self._register("position_lr_delay_mult")

        # Steps used to transition between LR endpoints.
        lr.add_argument(
            "--position_lr_max_steps",
            type=int,
            default=30_000,
            help="Iterations over which the position LR decays from init to final.",
        )
        self._register("position_lr_max_steps")

        # Appearance-feature LR.
        lr.add_argument(
            "--feature_lr",
            type=float,
            default=0.0025,
            help="Learning rate for SH feature coefficients.",
        )
        self._register("feature_lr")

        # Opacity LR.
        lr.add_argument(
            "--opacity_lr",
            type=float,
            default=0.025,
            help="Learning rate applied to Gaussian opacity logits.",
        )
        self._register("opacity_lr")

        # Scale LR.
        lr.add_argument(
            "--scaling_lr",
            type=float,
            default=0.005,
            help="Learning rate driving Gaussian scale updates.",
        )
        self._register("scaling_lr")

        # Rotation LR.
        lr.add_argument(
            "--rotation_lr",
            type=float,
            default=0.001,
            help="Learning rate for quaternion rotations.",
        )
        self._register("rotation_lr")

        # Exposure LR init (legacy RGB).
        lr.add_argument(
            "--exposure_lr_init",
            type=float,
            default=0.01,
            help="Initial exposure LR (legacy RGB feature).",
        )
        self._register("exposure_lr_init")

        # Exposure LR final (legacy RGB).
        lr.add_argument(
            "--exposure_lr_final",
            type=float,
            default=0.001,
            help="Final exposure LR after scheduling (legacy RGB feature).",
        )
        self._register("exposure_lr_final")

        # Delay before exposure LR decay.
        lr.add_argument(
            "--exposure_lr_delay_steps",
            type=int,
            default=0,
            help="Iterations before exposure LR decay starts (legacy RGB feature).",
        )
        self._register("exposure_lr_delay_steps")

        # Exposure LR decay multiplier.
        lr.add_argument(
            "--exposure_lr_delay_mult",
            type=float,
            default=0.0,
            help="Multiplier applied to exposure LR after the delay window.",
        )
        self._register("exposure_lr_delay_mult")

        densify = parser.add_argument_group("Densification & Regularization")

        # Dense fraction (legacy compatibility).
        densify.add_argument(
            "--percent_dense",
            type=float,
            default=0.01,
            help="Fraction of Gaussians kept dense (legacy RGB compatibility).",
        )
        self._register("percent_dense")

        # DSSIM weight when RGB supervision is active.
        densify.add_argument(
            "--lambda_dssim",
            type=float,
            default=0.2,
            help="DSSIM loss weight (RGB workflows).",
        )
        self._register("lambda_dssim")

        # Interval between densification passes.
        densify.add_argument(
            "--densification_interval",
            type=int,
            default=100,
            help="Iterations between densification/splitting passes.",
        )
        self._register("densification_interval")

        # Opacity reset cadence.
        densify.add_argument(
            "--opacity_reset_interval",
            type=int,
            default=3000,
            help="Iterations between global opacity resets.",
        )
        self._register("opacity_reset_interval")

        # Iteration at which densification starts.
        densify.add_argument(
            "--densify_from_iter",
            type=int,
            default=500,
            help="Iteration to begin spawning/splitting Gaussians.",
        )
        self._register("densify_from_iter")

        # Iteration after which densification stops.
        densify.add_argument(
            "--densify_until_iter",
            type=int,
            default=15_000,
            help="Iteration after which densification/pruning stops.",
        )
        self._register("densify_until_iter")

        # Gradient magnitude threshold for densification.
        densify.add_argument(
            "--densify_grad_threshold",
            type=float,
            default=0.0002,
            help="Gradient energy required for a Gaussian to be split.",
        )
        self._register("densify_grad_threshold")

        # Depth L1 initial weight (RGB compatibility).
        densify.add_argument(
            "--depth_l1_weight_init",
            type=float,
            default=1.0,
            help="Initial depth-L1 weight (legacy RGB feature).",
        )
        self._register("depth_l1_weight_init")

        # Depth L1 final weight (RGB compatibility).
        densify.add_argument(
            "--depth_l1_weight_final",
            type=float,
            default=0.01,
            help="Final depth-L1 weight after scheduling (legacy RGB feature).",
        )
        self._register("depth_l1_weight_final")

        intensity = parser.add_argument_group("Intensity & Appearance Controls")

        # How grayscale intensities are produced.
        intensity.add_argument(
            "--intensity_mode",
            type=str,
            default="sampled",
            choices=["sampled", "learned", "sampled_mean_covered"],
            help="Strategy for assigning per-Gaussian intensity values.",
        )
        self._register("intensity_mode")

        # Interval between intensity statistic updates.
        intensity.add_argument(
            "--intensity_update_interval",
            type=int,
            default=10,
            help="Iterations between intensity statistic updates.",
        )
        self._register("intensity_update_interval")

        # Brightness divisor for intensity-to-color mapping.
        intensity.add_argument(
            "--intensity_color_divisor",
            type=float,
            default=1.0,
            help="Divisor applied when mapping intensities to pseudo-RGB colors.",
        )
        self._register("intensity_color_divisor")

        # Threshold for classifying large splats.
        intensity.add_argument(
            "--intensity_large_splat_threshold",
            type=float,
            default=0.03,
            help='Radius threshold used to treat splats as "large" during sampling.',
        )
        self._register("intensity_large_splat_threshold")

        # Radius multiplier for mean-covered sampling.
        intensity.add_argument(
            "--intensity_mean_cover_radius",
            type=float,
            default=2.5,
            help="Neighborhood radius multiplier for mean-covered sampling.",
        )
        self._register("intensity_mean_cover_radius")

        # Interval for recomputing mean-covered intensities.
        intensity.add_argument(
            "--intensity_mean_cover_interval",
            type=int,
            default=20,
            help="Iterations between mean-covered intensity updates.",
        )
        self._register("intensity_mean_cover_interval")

        constraint = parser.add_argument_group("Scale Constraints & Diagnostics")

        # Optional L2 penalty on absolute scales.
        constraint.add_argument(
            "--scale_l2_weight",
            type=float,
            default=0.005,
            help="Weight for the L2 penalty applied to physical Gaussian scales.",
        )
        self._register("scale_l2_weight")

        # Max growth relative to the initial scale.
        constraint.add_argument(
            "--max_scale_factor",
            type=float,
            default=3.0,
            help="Cap on how much a Gaussian scale may grow vs. initialization.",
        )
        self._register("max_scale_factor")

        # Warmup iterations for the scaling constraint.
        constraint.add_argument(
            "--scaling_constraint_warmup_iters",
            type=int,
            default=1500,
            help="Iterations over which the scale constraint tightens to its final value.",
        )
        self._register("scaling_constraint_warmup_iters")

        # Initial relaxation multiplier for the constraint.
        constraint.add_argument(
            "--scaling_constraint_relaxation",
            type=float,
            default=3.0,
            help="Initial relaxation multiplier applied to the scale constraint.",
        )
        self._register("scaling_constraint_relaxation")

        # Window for logging early Gaussian statistics.
        constraint.add_argument(
            "--early_stats_window",
            type=int,
            default=256,
            help="Number of early iterations that log detailed Gaussian statistics.",
        )
        self._register("early_stats_window")

        diversity = parser.add_argument_group("Parameter Diversity Warmup")

        # Duration of the diversity warmup phase.
        diversity.add_argument(
            "--diversity_warmup_iterations",
            type=int,
            default=2000,
            help="Iterations to keep diversity losses enabled (0 disables).",
        )
        self._register("diversity_warmup_iterations")

        # Logging cadence for diversity diagnostics.
        diversity.add_argument(
            "--diversity_log_interval",
            type=int,
            default=25,
            help="Iterations between diversity diagnostic prints.",
        )
        self._register("diversity_log_interval")

        # Base weight for the scale diversity loss.
        diversity.add_argument(
            "--diversity_scale_weight",
            type=float,
            default=0.05,
            help="Overall strength of the scale diversity loss.",
        )
        self._register("diversity_scale_weight")

        # Base weight for the rotation diversity loss.
        diversity.add_argument(
            "--diversity_rotation_weight",
            type=float,
            default=0.05,
            help="Overall strength of the rotation diversity loss.",
        )
        self._register("diversity_rotation_weight")

        # Variance component weight for scale diversity.
        diversity.add_argument(
            "--diversity_scale_variance_weight",
            type=float,
            default=0.2,
            help="Weight on the per-axis variance component of the scale loss.",
        )
        self._register("diversity_scale_variance_weight")

        # Range penalty weight for scale diversity.
        diversity.add_argument(
            "--diversity_scale_range_weight",
            type=float,
            default=0.2,
            help="Penalty pushing scales toward a desired range.",
        )
        self._register("diversity_scale_range_weight")

        # Target clamp weight for scale diversity.
        diversity.add_argument(
            "--diversity_target_range_weight",
            type=float,
            default=0.2,
            help="Clamp weight reinforcing the preferred scale interval.",
        )
        self._register("diversity_target_range_weight")

        # Rotation entropy weight.
        diversity.add_argument(
            "--diversity_rotation_entropy_weight",
            type=float,
            default=0.2,
            help="Encourages diverse quaternion orientations.",
        )
        self._register("diversity_rotation_entropy_weight")

        # Quaternion dispersion weight.
        diversity.add_argument(
            "--diversity_dispersion_weight",
            type=float,
            default=0.2,
            help="Penalizes quaternions collapsing toward identity.",
        )
        self._register("diversity_dispersion_weight")

        # Alignment weight with gradient-derived directions.
        diversity.add_argument(
            "--diversity_alignment_weight",
            type=float,
            default=0.1,
            help="Aligns Gaussians with local volume gradients when set > 0.",
        )
        self._register("diversity_alignment_weight")


class PipelineParams(ParamGroup):
    """Low-level pipeline toggles retained for debugging/compatibility."""

    def __init__(self, parser: ArgumentParser) -> None:
        super().__init__()
        group = parser.add_argument_group("Pipeline Toggles")

        # Forces SH conversion to happen on the Python side.
        group.add_argument(
            "--convert_SHs_python",
            action="store_true",
            help="Convert spherical harmonics coefficients on CPU/Python for debugging.",
        )
        self._register("convert_SHs_python")

        # Forces covariance computation to happen on CPU/Python.
        group.add_argument(
            "--compute_cov3D_python",
            action="store_true",
            help="Compute 3D covariance matrices on CPU/Python for debugging.",
        )
        self._register("compute_cov3D_python")

        # Enables verbose diagnostics in the rasterization pipeline.
        group.add_argument(
            "--debug",
            action="store_true",
            help="Enable verbose debugging prints inside the rasterizer.",
        )
        self._register("debug")

        # Enables antialiasing when using the legacy RGB rasterizer.
        group.add_argument(
            "--antialiasing",
            action="store_true",
            help="Turn on rasterizer anti-aliasing (legacy RGB workflows).",
        )
        self._register("antialiasing")


def get_combined_args(parser: ArgumentParser) -> Namespace:
    """Merge CLI arguments with a saved cfg_args file when present."""

    cmdline_args = parser.parse_args(sys.argv[1:])
    cfg_contents = "Namespace()"

    try:
        cfg_path = os.path.join(cmdline_args.model_path, "cfg_args")
        print("Looking for config file in", cfg_path)
        with open(cfg_path) as cfg_file:
            print(f"Config file found: {cfg_path}")
            cfg_contents = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found; using command-line arguments only.")

    cfg_args = eval(cfg_contents)
    merged = vars(cfg_args).copy()
    for key, value in vars(cmdline_args).items():
        if value is not None:
            merged[key] = value
    return Namespace(**merged)
