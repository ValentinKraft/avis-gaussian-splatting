#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.

"""
Pipeline parameters for Gaussian Splatting with volume supervision.
"""

from argparse import ArgumentParser, Namespace
from typing import List, Optional
import os
import sys

class GroupParams:
    pass

class OptimizationParams:
    def __init__(self, parser):
        # Original optimization parameters
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        # Volume supervision parameters
        self.rgb_supervision = True
        self.volume_supervision = False
        self.volume_path = ""
        self.volume_loss_type = "dice"
        self.volume_loss_weight = 1.0
        self.volume_shape = [64, 64, 64]

        # Add arguments to parser
        parser.add_argument('--iterations', type=int, default=self.iterations)
        parser.add_argument('--position_lr_init', type=float, default=self.position_lr_init)
        parser.add_argument('--position_lr_final', type=float, default=self.position_lr_final)
        parser.add_argument('--position_lr_delay_mult', type=float, default=self.position_lr_delay_mult)
        parser.add_argument('--position_lr_max_steps', type=int, default=self.position_lr_max_steps)
        parser.add_argument('--feature_lr', type=float, default=self.feature_lr)
        parser.add_argument('--opacity_lr', type=float, default=self.opacity_lr)
        parser.add_argument('--scaling_lr', type=float, default=self.scaling_lr)
        parser.add_argument('--rotation_lr', type=float, default=self.rotation_lr)
        parser.add_argument('--percent_dense', type=float, default=self.percent_dense)
        parser.add_argument('--lambda_dssim', type=float, default=self.lambda_dssim)
        parser.add_argument('--densification_interval', type=int, default=self.densification_interval)
        parser.add_argument('--opacity_reset_interval', type=int, default=self.opacity_reset_interval)
        parser.add_argument('--densify_from_iter', type=int, default=self.densify_from_iter)
        parser.add_argument('--densify_until_iter', type=int, default=self.densify_until_iter)
        parser.add_argument('--densify_grad_threshold', type=float, default=self.densify_grad_threshold)

        # Volume supervision arguments
        parser.add_argument('--rgb_supervision', action='store_true', default=True,
                          help='Enable RGB image supervision')
        parser.add_argument('--volume_supervision', action='store_true', default=False,
                          help='Enable volume supervision loss')
        parser.add_argument('--volume_path', type=str, default="",
                          help='Path to ground truth volume file (.nii, .npy, .mhd)')
        parser.add_argument('--volume_loss_type', type=str, default='dice',
                          choices=['mse', 'dice', 'tversky', 'kl'],
                          help='Type of volume supervision loss')
        parser.add_argument('--volume_loss_weight', type=float, default=1.0,
                          help='Weight for volume supervision loss')
        parser.add_argument('--volume_shape', type=int, nargs=3, default=[64, 64, 64],
                          help='Target shape for volume supervision (D, H, W)')

        # Volume initialization arguments
        parser.add_argument(
            "--init_from_mask",
            action="store_true",
            default=False,
            help="Initialize Gaussian points by sampling from segmentation mask",
        )
        parser.add_argument(
            "--mask_path",
            type=str,
            default="",
            help="Path to segmentation mask file (.nii, .npy, .mhd)",
        )
        parser.add_argument(
            "--volume_transform",
            type=str,
            default="",
            help="Path to 4x4 transform matrix for volume alignment (.npy)",
        )
        parser.add_argument(
            "--init_n_points",
            type=int,
            default=5000,
            help="Number of points to sample from mask",
        )
        parser.add_argument(
            "--position_noise",
            type=float,
            default=0.01,
            help="Standard deviation of position noise for initialization",
        )

    def extract(self, args):
        """Extract parameters from parsed arguments."""
        group = GroupParams()
        for key, value in vars(self).items():
            if key in vars(args):
                setattr(group, key, getattr(args, key))
            else:
                setattr(group, key, value)
        return group

class ModelParams:
    def __init__(self, parser, sentinel=False):
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.sh_degree = 3
        self.white_background = False
        self.data_device = "cuda"

        # Add arguments
        parser.add_argument(
            "--source_path",
            type=str,
            default="",
            help="Path to source directory containing images (optional if using --init_from_mask)",
        )
        parser.add_argument('--model_path', type=str, required=not sentinel,
                          help='Path to save model')
        parser.add_argument(
            "--images",
            type=str,
            default=self.images,
            help="Image folder (only needed with RGB supervision)",
        )
        parser.add_argument(
            "--resolution",
            type=int,
            default=self.resolution,
            help="Resolution of images (only needed with RGB supervision)",
        )
        parser.add_argument('--sh_degree', type=int, default=self.sh_degree,
                          help='Degree of spherical harmonics')
        parser.add_argument('--white_background', action='store_true',
                          help='Render with white background')
        parser.add_argument('--data_device', type=str, default=self.data_device,
                          help='Device to store data')

    def extract(self, args):
        """Extract parameters from parsed arguments."""
        group = GroupParams()
        for key, value in vars(self).items():
            if key in vars(args):
                setattr(group, key, getattr(args, key))
            else:
                setattr(group, key, value)
        return group

class PipelineParams:
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        
        # Add arguments
        parser.add_argument('--convert_SHs_python', action='store_true',
                          help='Convert spherical harmonics in Python')
        parser.add_argument('--compute_cov3D_python', action='store_true',
                          help='Compute 3D covariance in Python')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode')

    def extract(self, args):
        """Extract parameters from parsed arguments."""
        group = GroupParams()
        for key, value in vars(self).items():
            if key in vars(args):
                setattr(group, key, getattr(args, key))
            else:
                setattr(group, key, value)
        return group

def get_combined_args(parser : ArgumentParser):
    """Combine command line arguments with config file."""
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    args_cfgfile = eval(cfgfile_string)
    
    # Merge arguments
    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
