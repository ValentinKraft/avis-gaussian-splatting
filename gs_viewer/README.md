## Standalone Gaussian PLY Viewer

### Goal
- Minimal 3D viewer for Gaussian splatting PLY models exported by this repo.
- Unity-style camera orbit/pan/zoom.
- Medical-style transfer function (1D LUT) mapping a per-splat scalar to color + transparency.

### Install (Windows)
1. Activate your Python environment.
2. Install dependencies:

```shell
uv pip install -r gs_viewer/requirements.txt
```

3. (Optional, recommended) Install the viewer package editable:

```shell
uv pip install -e gs_viewer
```

### Run
- From repo root:

```shell
gs-viewer --ply path\\to\\model.ply
```

- Or without installing the package:

```shell
python gs_viewer\\run_viewer.py --ply path\\to\\model.ply
```

### Controls
- Left mouse drag: orbit
- Right mouse drag: pan
- Middle mouse drag (vertical): zoom
- Mouse wheel: zoom

### Notes
- This viewer expects the GaussianModel PLY schema written by `scene/gaussian_model.py`:
  `x,y,z`, `f_dc_0..2`, `intensity_01`, optional `hu`, `opacity`, `scale_0..2` (log), `rot_0..3` (quat), optional `ao`.
- Transfer-function scalar uses per-splat `intensity_01` when present.
- Backward compatibility: if `intensity_01` is missing, the viewer falls back to decoded SH-DC intensity
  (`rgb = f_dc * SH_C0 + 0.5`).
