# Nerfstudio commands
wsl -d Ubuntu-22.04
conda activate nerfstudio

ns-process-data images --data C:\DEV\TESTS\gs\_SCENES_\_scene_ --output-dir C:\DEV\TESTS\gs\_SCENES_\NERFSTUDIO\AHrEZ-800 --colmap-cmd C:\DEV\TESTS\gs\COLMAP\bin\colmap.exe --matching-method exhaustive --camera-type simple_pinhole --num-downscales 0

## Cheap training
ns-train splatfacto --data /mnt/c/DEV/TESTS/gs/_SCENES_/NERFSTUDIO/AHrEZ-200-png --mixed-precision True --pipeline.model.sh-degree 2

## Hard training
ns-train splatfacto --data /mnt/c/DEV/TESTS/gs/_SCENES_/NERFSTUDIO/abdomen --pipeline.model.sh-degree 2 --pipeline.model.stop-split-at 50000 --max-num-iterations 6000 --pipeline.model.densify-grad-thresh 0.0001

## Export
ns-export gaussian-splat --load-config outputs/MINICT/splatfacto/2026-02-28_163704/config.yml --output-dir /mnt/c/DEV/TESTS/gs/_SCENES_

ns-viewer --load-config ...


---------------------------

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

Recommended Base command:

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

## Outputs and export

- Training artifacts are written under `--model_path`.
- PLY snapshots are controlled by:
  - `--save_ply_every`: export cadence.
  - `--ply_output_prefix`: filename prefix.
  - `--export_ao`: bake fast ambient occlusion from the mask into exported colors and add an `ao` PLY attribute.
  - `--export_ao_method {isotropic,normal}`, `--export_ao_radius_vox`, `--export_ao_strength`.

## External PLY masked MSE

Use `evaluate_ply_masked_mse.py` to score an external PLY with the same full-ROI masked-MSE path used during training.

Example using a saved training run to inherit the same volume, mask, raster settings, and full-ROI eval downscale:

If you have a real training output directory:

```shell
python evaluate_ply_masked_mse.py --ply_path path\to\standard_3dgs.ply --training_model_path _output_/your_run
```

If you want to evaluate directly from inputs:
```shell
python evaluate_ply_masked_mse.py --ply_path abdomen_220k.ply --volume_path abdomen1.5.nii.gz --mask_path abdomen1.5_liver-float.nii.gz --target ct
```

If the PLY was exported by this fork, `--intensity_source auto` will prefer the stored `intensity_01` attribute. For a standard 3DGS PLY without `intensity_01`, the evaluator falls back to SH/DC appearance via `f_dc_*`.

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
