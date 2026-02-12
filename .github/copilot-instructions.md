## Project Map
- Volume-supervised trainer entrypoint: `train.py` (uses `arguments.py` + `gaussian_splatting/utils/volume_supervisor.py`).
- Volume I/O + normalization: `gaussian_splatting/data/volume_loader.py` (loads NIfTI/NPY, normalizes to [0,1], NIfTI permute to [D,H,W]=[Z,Y,X]).
- Seed initialization from mask voxels: `gaussian_splatting/utils/volume_initializer.py`.
- Volume rasterization (Gaussians -> voxel grid) for supervision: `gaussian_splatting/utils/splat_to_volume.py`.
- PLY export + baked appearance: `scene/gaussian_model.py` (`save_ply*` and intensity-to-SH/DC mapping).
- Camera/image rendering (legacy 3DGS path): `gaussian_renderer/__init__.py` + `submodules/diff-gaussian-rasterization`.

## Core Data Conventions
- Volume tensors are `[D,H,W] = [Z,Y,X]`.
- Gaussian positions are normalized to `[0,1]^3` and ordered `[x,y,z]` (see `gaussian_splatting/utils/ambient_occlusion.py`).
- `VolumeLoader` always normalizes loaded volumes to `[0,1]` (do not assume HU units unless you change the loader).
- When sampling volumes at Gaussian positions, use `sample_intensities_from_volume(...)` in `gaussian_splatting/utils/intensity_sampler.py` (it expects normalized xyz; it will also auto-normalize if given voxel-index-like coordinates).

## Training & Supervision Flow
- `train.py` constructs `VolumeSupervisor`, loads `volume_gt` + `mask_volume`, thresholds mask to `mask_bool`, and caches ROI slices/bounds.
- `VolumeSupervisor.compute_loss(...)` rasterizes to a working grid via `splat_to_volume(...)` and computes loss only in masked ROI.
- Point-count is capped per iteration by `MAX_POINTS_PER_ITER` in `train.py`; always thread `active_idx` through any per-point updates.
- Render-time performance knobs: `--volume_render_downscale_factor` (working grid) and ROI cropping in `VolumeSupervisor` (mask bounds + padding).

## Appearance / Export Features
- Intensity/opacity strategies are CLI-controlled in `arguments.py` via `--intensity_mode` and `--opacity_mode`.
- Ambient occlusion is export-only: `--export_ao` triggers `compute_ao_volume_from_mask(...)` in `gaussian_splatting/utils/ambient_occlusion.py`, sampled at Gaussian XYZ on export, then applied in `GaussianModel.save_ply(...)` as `f_dc *= (1-strength)+strength*ao`.
- PLY grayscale is stored as SH DC coefficients (see `_create_colors_from_intensities(...)` in `scene/gaussian_model.py`); multiplying `f_dc` affects rendered brightness.

## Common Workflows
- Environment (Windows): `SET DISTUTILS_USE_SDK=1` then `conda env create --file environment.yml`.
- Run volume training (examples live at the top of `README.md`; this fork expects `--mask_path` + `--volume_path`).
- Focused tests: `pytest -q tests/test_ambient_occlusion.py` and `pytest -q tests/test_ply_export_ao.py`.
- Debug hook: set `GS_VALIDATE_SAMPLING=1` to run the trilinear round-trip check inside `gaussian_splatting/utils/intensity_sampler.py`.

## Codebase Patterns (Keep Consistent)
- New CLI flags belong in `arguments.py` (grouped by intent) and must be threaded into `train.py` and/or `VolumeSupervisor`.
- Preserve coordinate/shape conversions explicitly (avoid silent `[z,y,x]` vs `[x,y,z]` swaps).
- Prefer reusing existing helpers (`VolumeLoader`, `sample_intensities_from_volume`, `splat_to_volume`) instead of adding parallel implementations.
- When adding per-point updates in the training loop, handle subsampling: updates should apply only to `active_idx` when present.
