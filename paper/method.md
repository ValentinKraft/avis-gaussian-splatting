## Pipeline

- Load a target volume and a corresponding ROI mask from scientific or medical
  volume formats such as NIfTI, NPY, or MHD.
- Normalize the input volumes to the internal range used by the training
  pipeline and keep axis conventions explicit: volumes are stored as
  [D, H, W] = [Z, Y, X], while Gaussian centers are represented as normalized
  [x, y, z] coordinates in [0, 1]^3.
- Initialize Gaussian centers by sampling voxels from the binary or soft mask,
  so the initial splat distribution is concentrated inside the anatomical or
  structural region of interest.
- Assign each Gaussian learnable parameters for position, anisotropic scale,
  rotation, opacity, and optional scalar intensity / feature values.
- Rasterize the Gaussian set into a voxel grid with a differentiable
  splat-to-volume operator that accumulates anisotropic Gaussian contributions
  into a predicted volume.
- Compute supervision losses between the predicted volume and the reference
  target volume, typically only inside the ROI to reduce background dominance
  and focus optimization on relevant structures.
- Update the Gaussian parameters with gradient-based optimization, while using
  diagnostics, clipping, and optional densification / pruning to keep training
  stable.
- Periodically export the evolving representation as PLY snapshots for external
  visualization, inspection, or downstream rendering.

## Special Features

- Direct support for medical-style 3D data rather than only view-based RGB
  imagery.
- Mask-driven initialization that can use either hard binary masks or soft
  probability volumes.
- ROI-aware supervision, which restricts loss computation to the masked region
  instead of wasting capacity on large empty backgrounds.
- Support for multiple supervision targets: mask / density, scalar intensity,
  or joint optimization of both.
- Flexible voxel-space losses including Dice, Tversky, MSE, and KL, allowing
  the method to cover both sparse segmentation-like targets and continuous
  intensity regression.
- Medical-oriented presets and tuning knobs for organ-like or vessel-like data,
  where sparsity, thin structures, and imbalance require different behavior.
- Optional learned, sampled, or mean-covered intensity and opacity modes,
  allowing the splat attributes to be either optimized directly or refreshed
  from the source volume.
- Densification and pruning adapted to volume supervision, so the Gaussian set
  can grow in high-gradient regions and remove weak splats during training.
- Scale constraints and related stabilization logic to prevent degenerate or
  overly large splats.
- Memory-aware engineering such as downscaled working grids, ROI cropping, and
  chunked accumulation to keep volumetric optimization tractable.
- Export-oriented additions such as PLY sequences, optional ambient occlusion
  baked from the mask, and a lightweight viewer workflow for interactive
  inspection.

## Differences From Standard Gaussian Splatting

- Standard 3D Gaussian Splatting is supervised from multiple 2D images and
  optimized for image-plane rendering quality; this project is supervised
  directly from a 3D target volume.
- Standard 3DGS renders colors along camera rays; this method rasterizes
  Gaussians directly into a voxel grid and compares the result in volume space.
- Standard 3DGS depends on calibrated camera poses; this pipeline does not
  require a camera setup when fitting a single volume.
- Standard 3DGS usually initializes from point clouds or SfM reconstructions;
  this method initializes from mask voxels or volume-derived sampling
  distributions.
- Standard 3DGS focuses on photometric losses and view synthesis metrics; this
  method uses voxel-space objectives such as Dice, Tversky, MSE, and KL.
- Standard 3DGS is naturally oriented toward natural-image appearance;
  this project adds intensity-oriented supervision, medical data conventions,
  and ROI-based reasoning for scientific volumes.
- Standard 3DGS training logic is tied to visibility and screen-space effects;
  this repository introduces volume-specific densification heuristics,
  gradient handling, and stability fixes for voxel-based optimization.
- Standard 3DGS exports a representation mainly for novel-view rendering; this
  project also treats the Gaussian set as an interpretable volumetric proxy for
  scientific visualization and structural analysis.