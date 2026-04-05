## Method Overview

We adapt 3D Gaussian Splatting from multi-view image supervision to direct
volume supervision. Instead of fitting a Gaussian scene to a set of calibrated
RGB images, the method fits a set of anisotropic 3D Gaussians to a target
volume and an associated region-of-interest mask. This design is intended for
medical and scientific data, where supervision naturally lives in voxel space
and where sparse structures, strong class imbalance, and large empty
backgrounds make image-centric training objectives a poor fit.

The pipeline takes as input a target volume $V$ and a binary or soft mask $M$.
Both are loaded from scientific volume formats such as NIfTI, NPY, or MHD and
normalized to the internal training range. Throughout the implementation,
volumes are stored as $[D,H,W] = [Z,Y,X]$, whereas Gaussian centers are
represented as normalized $[x,y,z]$ coordinates in $[0,1]^3$. Starting from a
mask-driven Gaussian initialization, the method optimizes positions, anisotropic
scales, rotations, opacities, and optional per-Gaussian intensity values.
During training, the Gaussian set is rasterized into a voxel grid via a
differentiable splat-to-volume operator, and losses are computed directly in
volume space, primarily inside the mask-defined region of interest.

## Problem Formulation

Let $V \in [0,1]^{D \times H \times W}$ denote the normalized target volume and
let $M \in [0,1]^{D \times H \times W}$ denote the corresponding ROI mask. The
mask may be binary or soft. A thresholded ROI indicator
$B = \mathbb{1}[M > \tau_{\mathrm{roi}}]$ defines the voxels used for the main
supervision terms. The model represents the object of interest as a set of
$N$ Gaussians,

$$
\mathcal{G} = \{g_i\}_{i=1}^{N}, \qquad
g_i = (\mu_i, s_i, q_i, \alpha_i, c_i),
$$

where $\mu_i \in [0,1]^3$ is the center, $s_i \in \mathbb{R}_{+}^3$ is an
anisotropic scale, $q_i$ is a rotation quaternion, $\alpha_i$ is an opacity,
and $c_i$ is an optional scalar intensity parameter. Depending on the selected
training mode, the method predicts either a density-like mask volume, an
intensity volume, or both jointly.

## Gaussian Representation in Volume Space

Each Gaussian is optimized directly in normalized volume coordinates instead of
camera space. This is a crucial departure from standard 3DGS: there is no
camera model, no ray marching through views, and no supervision in image plane.
Instead, the representation is treated as a volumetric proxy whose splats are
accumulated into a target voxel grid. The implementation supports learnable
anisotropic scaling and quaternion rotations, allowing elongated structures
such as vessels or thin boundaries to be modeled more efficiently than with
isotropic kernels.

Besides geometry, the method supports different appearance parameterizations.
Intensity and opacity can either be learned directly or refreshed from the
input data through volume sampling. In particular, the repository supports
sampled and mean-covered modes, where large splats periodically receive updated
intensity or opacity values derived from the source volume or mask. This allows
the method to interpolate between a purely learned representation and a more
data-anchored one.

## Mask-Driven Initialization

Initialization is explicitly mask-centric. Rather than starting from a point
cloud or a structure-from-motion reconstruction, the method samples candidate
voxels from the mask volume and places Gaussian centers inside the selected
foreground cells with sub-voxel jitter. The initialization logic supports both
hard masks and soft masks. In the current repository state, initialization uses
a low default mask threshold so that low-confidence but relevant soft-mask
regions are not discarded prematurely.

Initial scales are defined in voxel units and then mapped into the normalized
volume coordinate system. This produces a controlled initial support size rather
than a degenerate point-like start. Optional deduplication quotas reduce severe
local oversampling during seed generation.

The initialization stage can also inject structural priors. Rotations are
initialized from a gradient-derived orientation field computed from the target
intensity volume. In addition, the mask can be analyzed with Hessian-based
structure cues to estimate vesselness and locally preferred directions. These
signals are used to introduce anisotropy already at initialization time, for
example by elongating splats along vessel-like directions or flattening splats
near boundaries. As a result, the optimizer starts from a representation that
is already biased toward plausible medical morphology instead of having to
discover it from scratch.

## Differentiable Splat-to-Volume Rendering

Given the current Gaussian set, the method renders a prediction directly into a
voxel grid with the same spatial interpretation as the target data. The
renderer constructs a normalized 3D sampling grid and accumulates contributions
from the active Gaussian subset. Depending on the supervision target, the same
core operator is used in two modes:

- Density mode: splat opacities into a predicted mask or density volume.
- Intensity mode: splat opacity-weighted scalar intensities into a predicted
  CT-like or scalar-valued volume.

The implementation supports anisotropic scales and rotations during rendering,
so the predicted volume is not restricted to spherical kernels. To keep memory
usage tractable, the working grid can be rendered at a configurable internal
downscale factor and the Gaussian set can be processed in batches.

## ROI-Aware Supervision

Losses are computed in volume space and restricted primarily to the mask-defined
region of interest. This is important because large medical volumes often
contain extensive empty background that would otherwise dominate optimization.
The repository supports mask supervision, intensity supervision, and joint
supervision.

For mask-focused training, the method compares the predicted density volume to
the mask target. For intensity-focused training, it compares the predicted
intensity volume to a normalized version of the target volume, typically only
inside the ROI. The implemented loss family includes Dice, Tversky, MSE, and
KL, which makes the method usable for both sparse segmentation-like targets and
continuous scalar reconstruction.

At a high level, the joint objective can be written as

$$
\mathcal{L} =
\lambda_{m} \, \mathcal{L}_{m}(\hat{M} \odot B, M \odot B)
+ \lambda_{c} \, \mathcal{L}_{c}(\hat{V} \odot B, \tilde{V} \odot B)
+ \lambda_{o} \, \lVert \hat{M} \odot (1-B) \rVert_2^2,
$$

where $\hat{M}$ is the rendered density prediction, $\hat{V}$ is the rendered
intensity prediction, $\tilde{V}$ is the normalized target volume, and the last
term penalizes density leakage outside the ROI. This outside-mask penalty is a
simple but important stabilizer in sparse settings.

## Optimization and Memory-Aware Training

Training proceeds with gradient-based optimization over Gaussian parameters.
The repository includes several engineering choices that are relevant for a
paper description because they materially affect feasibility and stability.

First, the method can cap the number of Gaussians rendered per iteration. When
this active subset mode is enabled, the training loop cycles fairly through the
full Gaussian set rather than drawing purely random subsets each time. This
reduces memory pressure while preventing systematic starvation of certain points
during optimization and densification.

Second, the implementation supports mixed precision and configurable volume
storage dtypes, which is important for fitting full 3D data on limited GPUs.
Third, the code maintains diagnostic statistics, gradient clipping, scale
constraints, and adaptive learning-rate adjustments that react to stalled scale
growth or low local coverage. These details are implementation-oriented, but
they are part of what makes direct volume optimization practical.

## Topology Adaptation via Densification and Pruning

A central contribution of the repository relative to standard 3DGS practice is
the adaptation of densification and pruning to voxel-space supervision.
Instead of relying on view-dependent visibility heuristics, the method uses
volume-space gradient statistics and local density estimates.

The current implementation distinguishes between two main densification actions.
Small high-gradient splats are cloned, while larger high-gradient splats are
split into smaller offspring. In addition, the method tracks low-density and
low-coverage regions and can trigger targeted hole filling when the current
Gaussian set undersamples the occupied volume. This is especially useful for
thin vessels and fragmented structures, where pure prune-and-grow dynamics tend
to leave gaps.

Structure guidance can further bias topology growth. If structure fields are
available, the system can boost densification in vessel-like regions, scale new
children anisotropically, and preferentially sample hole-fill candidates where
the mask-derived structure signal is strong. Pruning then removes splats with
very low opacity or excessive size, preventing uncontrolled growth.

## Practical Features Relevant to the Method

Although the repository contains several engineering components that may belong
to an implementation section in the final paper, they are still method-relevant
because they shape the usable training regime.

- Soft-mask compatibility: the pipeline treats masks as normalized probability
  volumes rather than assuming hard labels everywhere.
- Multiple supervision targets: the same representation can be optimized for
  density, scalar intensity, or both jointly.
- Multiple intensity and opacity modes: attributes can be either learned or
  periodically sampled from the source data.
- Exportable volumetric proxy: the optimized Gaussian set can be saved as a PLY
  sequence for analysis, inspection, or downstream rendering.
- Optional ambient occlusion at export time: mask-derived AO can be baked into
  exported appearance without changing training.

## Relation to Standard 3D Gaussian Splatting

The method should be understood as a volume-native reformulation of Gaussian
splatting rather than a minor adaptation of the original image-based pipeline.
Standard 3DGS is designed for novel-view synthesis from calibrated images and
optimizes rendered colors along camera rays. In contrast, this repository uses
no camera setup, initializes from mask voxels instead of SfM geometry, renders
directly into a voxel grid, and compares predictions to a 3D target volume.

As a consequence, the supervision, initialization, densification logic, and
stability requirements all change. The resulting representation is not only a
rendering primitive but also an interpretable volumetric proxy for anatomical
or structural data. That perspective is the core methodological shift of this
project and should remain central in the final paper section.