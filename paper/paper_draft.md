## Introduction

- 3D Gaussian Splatting is a highly efficient graphics representation, but the
	original formulation assumes supervision from calibrated multi-view RGB
	images.
- In medical and scientific imaging, the available input is often already a
	3D scalar field, such as a CT, MR, or probability mask volume, so going
	through synthetic rendering views is unnecessary.
- This project reframes Gaussian Splatting as a volume-native optimization
	problem: learn a compact set of anisotropic 3D Gaussians that directly match
	a target volume inside a region of interest.
- The main motivation is to obtain a representation that is both compact and
	practical for visualization, export, and downstream analysis, rather than
	only for novel-view synthesis.
- The key technical challenge is that volume supervision changes nearly every
	part of the standard 3DGS pipeline: initialization, rendering, loss design,
	topology adaptation, and stability constraints.
- Additional practical challenges arise from sparse targets, strong foreground
	versus background imbalance, high memory cost for volumetric rendering, and
	the need to preserve gradient flow through a custom splat-to-volume operator.
- The repository therefore combines a direct voxel-space objective with
	mask-driven initialization, anisotropic / orientation-aware priors, joint
	mask-and-intensity supervision, and memory-aware training engineering.

## Related Work

see relatedwork.md

## Method

see method.md
AND:

- Represent the target object or anatomy as a set of 3D Gaussians with
	normalized centers, anisotropic scales, quaternion rotations, opacities, and
	optional scalar intensity values.
- Use the target volume and the ROI mask as the primary inputs; both are loaded
	from volume formats such as NIfTI, NPY, or MHD and normalized into the
	internal training range.
- Keep the coordinate conventions explicit throughout the pipeline: volumes are
	stored as [D, H, W] = [Z, Y, X], while Gaussian centers live in normalized
	[x, y, z] coordinates in [0, 1]^3.

### Initialization

- Initialize Gaussian seeds directly from the mask rather than from SfM points,
	cameras, or a reconstructed point cloud.
- Threshold the mask to obtain candidate foreground voxels, but fall back to
	nonzero or top-valued voxels if the mask is extremely sparse.
- Sample initialization points with a coarse per-cell quota so that dense mask
	regions do not collapse into severe local duplicates.
- Place seeds continuously inside voxel cells by adding sub-voxel jitter,
	followed by additional bounded noise, then clamp them back into the volume.
- Re-sample invalid jittered seeds if they drift outside the mask support,
	which is especially important for soft masks and thin structures.
- Convert sampled voxel coordinates into normalized [0, 1]^3 positions for the
	Gaussian model.
- Initialize scales in voxel units from a configurable range and convert them
	into the normalized coordinate system, giving each splat a finite initial
	support instead of starting from near-zero size.
- Initialize opacity from mask intensity, with optional gamma shaping, so that
	stronger mask support yields higher initial density.
- Sample per-Gaussian intensity values from the target volume, with additional
	boundary correction and a fallback nearest-voxel strategy when trilinear
	sampling becomes unreliable.
- Initialize rotations from a gradient-derived orientation field of the volume,
	so the splats start with a meaningful local orientation rather than identity
	quaternions everywhere.
- Optionally inject Hessian / vesselness structure cues from the mask to apply
	initialization-time anisotropy, blend structure-derived orientations, and
	elongate splats along vessel-like directions.
- Optionally detect border splats near the mask boundary, align them to the
	local surface normal, and flatten them to better represent thin shells and
	boundary layers.

### Training

- Train the Gaussian set directly against the loaded target volume using a
	VolumeSupervisor that manages the volume tensors, mask tensors, intensity
	ranges, and loss computation.
- Support three supervision modes: mask-only density fitting, CT / intensity
	fitting, or joint optimization of both branches.
- Render the current Gaussian set into a voxel grid using a differentiable
	splat-to-volume operator instead of the standard image-plane renderer.
- Use the same Gaussian parameters for both density and intensity rendering,
	but switch the rendering mode depending on whether the target branch is mask
	or scalar intensity.
- Compute mask loss and CT loss primarily inside the mask-defined support so
	that the optimization focuses on the relevant anatomy rather than empty
	background.
- Add an outside-mask penalty in mask or joint mode to discourage density
	leakage outside the foreground region.
- Support multiple loss families, including Dice, Tversky, MSE, and KL, so the
	method can handle both sparse segmentation targets and continuous-valued
	intensity targets.
- Support learned, sampled, and sampled-mean-covered intensity or opacity
	modes, allowing large splats to refresh attributes from covered voxels while
	smaller splats remain trainable.
- Use Adam-based optimization with mixed precision, gradient clipping, and
	explicit checks that core parameters remain attached to the optimizer graph.
- Cap the number of splats rendered per iteration with a fair active-subset
	scheduler that cycles through the full Gaussian set instead of repeatedly
	sampling a small biased subset.
- Enforce scale and position constraints after optimizer steps to avoid overly
	large splats, uncontrolled displacement, and out-of-bounds points.
- During densification windows, accumulate per-point position-gradient norms,
	then periodically clone, split, or prune Gaussians based on gradient and
	opacity criteria.
- Export intermediate PLY snapshots during training so the evolving structure
	can be inspected outside the optimization loop.

## Implementation Details

- The main training entry point is train.py, which wires together the Gaussian
	model, volume supervisor, initialization logic, optimizer, logging, and PLY
	export.
- Initialization is implemented in gaussian_splatting/utils/volume_initializer.py,
	where mask sampling, intensity sampling, orientation setup, anisotropy, and
	border-aware seeding are handled.
- Volume supervision is implemented in
	gaussian_splatting/utils/volume_supervisor.py, which loads the target volume
	and mask, computes branch-specific losses, and manages attribute refresh for
	sampled intensity and opacity modes.
- Volumetric rasterization is implemented in
	gaussian_splatting/utils/splat_to_volume.py, which builds a normalized 3D
	grid and accumulates anisotropic Gaussian contributions into density or
	intensity volumes.
- The loader supports automatic safety downscaling for very large volumes, plus
	configurable storage dtypes such as fp32, fp16, and bf16 to control memory.
- The rasterizer supports an internal working-grid downscale factor so training
	can use a cheaper render grid than the native supervision resolution when
	point counts become large.
- The implementation uses gradient checkpointing during the render step so the
	forward pass can trade compute for memory during backpropagation.
- Diagnostic tooling includes TensorBoard logging, parameter monitoring,
	parameter update tracking, gradient norm reporting, and optional verbose
	debug output.
- Medical-style presets change initialization density, diversity settings, and
	densification behavior for organ-like or vessel-like data.
- Training outputs include checkpoints, TensorBoard logs, and PLY sequences;
	exported PLY files can optionally include ambient occlusion values derived
	from the mask.

## Discussion

- The main conceptual contribution is that Gaussian Splatting can be treated as
	a direct volumetric proxy rather than only as an image-based rendering model.
- Mask-driven initialization is important because it moves model capacity into
	the foreground from the first iteration instead of forcing the optimizer to
	discover sparse anatomy from scratch.
- Orientation-aware and anisotropy-aware initialization is especially useful
	for elongated structures such as vessels, where isotropic seeds would waste
	representation power.
- Joint mask and CT supervision gives a more expressive target than mask-only
	fitting, since it can preserve both structural occupancy and scalar
	appearance information.
- The project shows that stability engineering is not incidental but central:
	direct volume supervision only works reliably when gradient flow, scale
	control, memory limits, and topology updates are handled carefully.
- The current implementation is still limited by the cost of volumetric
	rasterization, especially for large Gaussian counts and orientation-aware
	kernels.
- Rotation parameters remain harder to supervise than positions and scales,
	suggesting that additional orientation priors or multi-scale objectives may
	be needed in future work.
- The present code base is best described as a strong research prototype: it is
	functionally rich and already useful for experimentation, but broader
	benchmarking and cleaner quantitative validation are still needed.

## Conclusion

- This project establishes a volume-supervised variant of Gaussian Splatting
	that replaces multi-view image supervision with direct optimization against a
	target volume and ROI mask.
- The resulting pipeline combines volume-native initialization, anisotropic and
	orientation-aware Gaussian priors, differentiable splat-to-volume rendering,
	flexible voxel-space losses, and stability-oriented training logic.
- A key practical outcome is that the learned Gaussian field is not only an
	optimization object but also an exportable, interpretable proxy for
	visualization and downstream analysis.
- The most important near-term research directions are faster rasterization,
	better quantitative evaluation, stronger orientation supervision, and broader
	validation on real medical datasets.
