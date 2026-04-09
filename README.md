# Avis Gaussian Splatting (Volume-Supervised)

This repository is a research fork of 3D Gaussian Splatting adapted for **volume-supervised** training on 3D medical-style data (e.g., CT/MR-like volumes). Instead of supervising with multi-view RGB images, training optimizes a 3D Gaussian representation to match a **target volume** inside a **binary/probability ROI mask**.

## Project scope

- **Inputs**: a volumetric target (`--volume_path`) plus a ROI mask (`--mask_path`).
- **Initialization**: seed Gaussians from mask voxels.
- **Supervision**: rasterize Gaussians into a voxel grid and compute loss **inside the ROI**.
- **Targets**: optimize mask/density (`--supervision_target mask`), intensity/CT (`ct`), or both (`joint`).
- **Export**: periodic PLY snapshots with per-splat intensity fields (plus optional AO baked from the mask).
- **Viewing**: a lightweight standalone PLY viewer in `gs_viewer/`.

## Setup

This repo uses a Conda environment defined in `environment.yml` (Python 3.12, CUDA 12.6, PyTorch 2.7.0).

```shell
SET DISTUTILS_USE_SDK=1
conda env create --file environment.yml
conda activate avis_gaussian_splatting
```

Notes:
- Building CUDA extensions requires a working C++ toolchain on Windows (e.g., Visual Studio Build Tools) and a CUDA toolkit compatible with your PyTorch CUDA runtime.
- `environment.yml` installs CUDA extensions as editable packages from `submodules/`.

## Data conventions

- `--volume_path`: CT/MR-like volume (e.g., `.nii/.nii.gz`, `.npy`, `.mhd`).
- `--mask_path`: binary/probability mask used for seeding and ROI definition.
- Volumes are loaded and normalized to **[0, 1]** by the loader (do not assume HU units unless you modify the loader).
- Volume tensor axis convention is **[D, H, W] = [Z, Y, X]**.
- Gaussian positions are treated as normalized **[x, y, z]** in **[0, 1]^3**.

## How to train (volume-supervised)

At minimum, provide `--model_path`, `--volume_path`, and `--mask_path`.

Common knobs:
- `--iterations`: total optimization steps.
- `--init_n_points`: number of initial Gaussians sampled from the mask.
- `--supervision_target {mask,ct,joint}`.
- `--volume_loss_type {mse,dice,tversky,kl}`: loss for the mask/density branch.
- `--ct_loss_type {mse,dice,tversky,kl}`: loss for the CT/intensity branch (ct/joint).
- `--volume_downscale_factor`: downscale applied at load time (1 = native).
- `--volume_render_downscale_factor`: internal working-grid downscale for rasterization (1 = full-res, slower/more VRAM).
- `--max_points_per_iter`: cap how many Gaussians are rendered/updated per iteration.
- `--save_ply_every`: PLY snapshot cadence.
- `--enable_diagnostics`: more detailed monitoring / TensorBoard scalars.

### Example commands

Minimal test dataset (joint supervision, full-res render grid):

```shell
python train.py --model_path _output_/mini-ct-test-50k --mask_path _input_/minimask-float.nii.gz --volume_path _input_/minict.nii.gz --iterations 1000 --init_n_points 50000 --save_ply_every 100 --enable_diagnostics --volume_downscale_factor 1 --volume_render_downscale_factor 1 --disable_volume_overflow_guard --supervision_target joint --volume_loss_type dice --ct_loss_type mse --medical_mode none --volume_storage_dtype fp16
```

Clinical vessel dataset (mask-focused preset via `--medical_mode vessel`):

```shell
python train.py --model_path _output_/vessel-float-test --mask_path _input_/vessel-mask-float.nii.gz --volume_path _input_/volume.nii.gz --iterations 2000 --init_n_points 20000 --save_ply_every 100 --enable_diagnostics --volume_loss_type dice --volume_downscale_factor 2 --medical_mode vessel
```

Clinical liver dataset:

```shell
python train.py --model_path _output_/liver-float-test --mask_path _input_/liver-mask-float.nii.gz --volume_path _input_/ct.nii.gz --iterations 2000 --init_n_points 20000 --save_ply_every 100 --enable_diagnostics --volume_downscale_factor 2
```

Synthetic test dataset (MSE supervision):

```shell
python train.py --model_path _output_/synthetic-float-test --mask_path _input_/synthetic-mask-float.nii.gz --volume_path _input_/synthetic-gradient.nii.gz --iterations 500 --volume_loss_type mse --init_n_points 10000 --save_ply_every 100 --enable_diagnostics
```

---------------------------------------------------

```shell
python train.py `
   --model_path _output_/vshuman-v7-test `
   --mask_path _input_/vshuman_tsmasks3.nii.gz `
   --volume_path _input_/vshuman.nii.gz `
   --iterations 1000 `
   --checkpoint_iterations 800 1200 1600 2000 `
   --init_n_points 200000 `
   --medical_mode none `
   --supervision_target joint `
   --mask_loss_weight 0.2 `
   --ct_loss_weight 1.00 `
   --enable_densification `
   --densify_from_iter 100 `
   --densification_interval 50 `
   --densify_max_new_points 10000 `
   --prune_min_opacity 0.001 `
   --max_points_per_iter 8000 `
   --volume_downscale_factor 1 `
   --volume_render_downscale_factor 1 `
   --disable_volume_overflow_guard `
   --save_ply_every 50 `
   --enable_diagnostics `
   --intensity_mode sampled `
   --opacity_mode sampled
```

Standard CLI command (learned):

```shell
python train.py `
  --model_path _output_/vshuman-v4 `
  --mask_path _input_/vshuman_tsmasks3.nii.gz `
  --volume_path _input_/vshuman.nii.gz `
  --iterations 1000 `
  --checkpoint_iterations 800 1200 1600 2000 `
  --init_n_points 200000 `
  --medical_mode none `
  --supervision_target joint `
  --volume_loss_type dice `
  --ct_loss_type mse `
  --enable_densification `
  --densify_from_iter 100 `
  --densification_interval 50 `
  --densify_max_new_points 10000 `
  --prune_min_opacity 0.002 `
  --max_points_per_iter 8000 `
  --volume_downscale_factor 1 `
  --volume_render_downscale_factor 1 `
  --disable_volume_overflow_guard `
  --save_ply_every 50 `
  --enable_diagnostics `
  --intensity_mode learned `
  --opacity_mode learned
```

```shell
python train.py `
  --model_path _output_/fidelity-hiinit-growth-v2 `
  --mask_path _input_/minimask-binary3.nii.gz `
  --volume_path _input_/minict.nii.gz `
  --iterations 2200 `
  --checkpoint_iterations 800 1200 1600 2200 `
  --init_n_points 120000 `
  --medical_mode none `
  --supervision_target joint `
  --volume_loss_type dice `
  --ct_loss_type mse `
  --mask_loss_weight 1.0 `
  --ct_loss_weight 0.75 `
  --outside_mask_weight 0.15 `
  --intensity_mode sampled_mean_covered `
  --opacity_mode learned `
  --intensity_large_splat_threshold 0.03 `
  --intensity_mean_cover_radius 2.5 `
  --intensity_mean_cover_interval 20 `
  --enable_densification `
  --densify_from_iter 100 `
  --densification_interval 100 `
  --densify_until_iter 1800 `
  --densify_grad_threshold 0.0 `
  --densify_grad_percentile 0.38 `
  --densify_max_new_points 5000 `
  --densify_spawn_jitter_vox 0.08 `
  --densify_vessel_spawn_bias 0.60 `
  --densify_vessel_spawn_power 1.50 `
  --structure_gradient_boost 0.28 `
  --structure_gradient_threshold 0.10 `
  --vessel_axial_scale 1.35 `
  --vessel_radial_scale 0.70 `
  --low_density_threshold 5.0 `
  --target_coverage 0.82 `
  --hole_fill_fraction 0.02 `
  --density_radius_factor 2.0 `
  --density_update_interval 10 `
  --prune_min_opacity 0.002 `
  --opacity_reset_interval 0 `
  --max_points_per_iter 8000 `
  --volume_downscale_factor 1 `
  --volume_render_downscale_factor 1 `
  --disable_volume_overflow_guard `
  --volume_storage_dtype fp16 `
  --init_scale_min_vox 0.3 `
  --init_scale_max_vox 1.5 `
  --min_scale_vox 0.5 `
  --max_scale_vox 8.0 `
  --max_scale_factor 2.5 `
  --percent_dense 0.015 `
  --save_ply_every 100 `
  --enable_diagnostics
```

## Outputs and export

- Training artifacts are written under `--model_path`.
- PLY snapshots are controlled by:
  - `--save_ply_every`: export cadence.
  - `--ply_output_prefix`: filename prefix.
  - `--export_ao`: bake fast ambient occlusion from the mask into exported colors and add an `ao` PLY attribute.
  - `--export_ao_method {isotropic,normal}`, `--export_ao_radius_vox`, `--export_ao_strength`.

## GS Viewer (standalone PLY viewer)

`gs_viewer/` is a minimal viewer for Gaussian PLY models exported by this repo. It supports orbit/pan/zoom controls and a medical-style 1D transfer function that maps a per-splat scalar to color + transparency.

- Install:

```shell
uv pip install -r gs_viewer/requirements.txt
uv pip install -e gs_viewer
```

- Run (from repo root):

```shell
gs-viewer --ply path\to\model.ply
```

- Script alternative:

```shell
python gs_viewer\run_viewer.py --ply path\to\model.ply
```

See `gs_viewer/README.md` for controls and PLY schema notes.

## Intensity & opacity modes

- `--intensity_mode learned` (default): optimize per-splat intensity.
- `--intensity_mode sampled`: sample intensities from the input volume at splat positions.
- `--intensity_mode sampled_mean_covered`: for large splats, use the mean of covered voxels; smaller splats keep cached values.
  - `--intensity_large_splat_threshold`: threshold used to classify splats as “large”.
  - `--intensity_mean_cover_radius`: coverage expansion multiplier.
  - `--intensity_mean_cover_interval`: refresh cadence for large splats.

Opacity is controlled similarly via `--opacity_mode {sampled,learned,sampled_mean_covered}` with `--opacity_update_interval` for refresh cadence in sampled modes.

## Optional: accelerated rasterizer (`sparse_adam`)

This repo supports `--optimizer_type sparse_adam`, but it requires the accelerated rasterizer variant. If it is not installed, `train.py` will exit with an instruction message.

To switch the rasterizer to the accelerated branch:

```shell
uv pip uninstall diff-gaussian-rasterization -y
cd submodules\diff-gaussian-rasterization
git checkout 3dgs_accel
uv pip install -e .
```

Then run training with:

```shell
python train.py ... --optimizer_type sparse_adam
```

## Acknowledgements

- This project builds on the 3D Gaussian Splatting codebase (Kerbl et al., 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

## License

See `LICENSE.md`.
