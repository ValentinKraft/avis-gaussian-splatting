Standalone Gaussian PLY Viewer

Goal
- Minimal 3D viewer for Gaussian splatting PLY models exported by this repo.
- Unity-style camera orbit/pan/zoom.
- Medical-style Transfer Function (1D LUT) mapping a per-splat scalar to color + transparency.

Install (Windows)
1) Activate your Python environment.
2) Install dependencies:
   uv pip install -r gs_viewer/requirements.txt
3) (Optional, recommended) Install the viewer package editable:
  uv pip install -e gs_viewer

Run
- From repo root:
  gs-viewer --ply path\\to\\model.ply

- Or without installing the package:
  python gs_viewer\\run_viewer.py --ply path\\to\\model.ply

Notes
- This viewer expects the GaussianModel PLY schema written by scene/gaussian_model.py:
  x,y,z, f_dc_0..2, opacity, scale_0..2 (log), rot_0..3 (quat), optional ao.
- Transfer-function scalar uses decoded SH-DC intensity: rgb = f_dc * SH_C0 + 0.5.
