## Motivation

* Fully automated segmentation of structures is becoming increasingly easier and more accurate (TotalSegmentator, SAM, etc.)
* However, efficient and realistic visualization (= volume rendering) in XR remains challenging (speed, quality, etc.)
* Efficient visualization is only possible via streaming, but streaming is error-prone and complex
* “Gaussian Splatting” is a new, extremely high-performance visualization method
* But in the medical field, we already have “ground-truth” volume data --> no need to go through rendering at all!
* Furthermore: special requirements for accuracy in the medical field
* Naive brute-force solution (1 splat per voxel) is not practical/optimal
* Therefore, a specialized Gaussian Splatting solution with optimization makes sense!

## Ideas

- Position the project as a bridge between real-time computer graphics
  representations and volume-native scientific / medical modeling.
- Emphasize that the representation is not only a rendering primitive but also
  a compact structural proxy for anatomy or other volumetric phenomena.
- Frame the method as removing the need for synthetic camera generation when
  only a single 3D volume is available.
- Highlight the potential dual role of the Gaussian field: reconstruction model
  during optimization and interpretable export format after training.
- Present the work as a first step toward hybrid geometric-semantic volumetric
  representations rather than only as a medical imaging application.
- Motivate the method through sparse structures such as vessels, where direct
  voxel supervision and adaptive densification are especially meaningful.
- Use the viewer and PLY export path as evidence that the method is useful for
  qualitative analysis, not just loss minimization.
- Future work idea: Compute ambient occlusion for all splats and bake this into the exported model. This will result in a more realistic rendering. 

## Problems

- Rotation parameters appear weakly constrained under voxel-space supervision,
  especially when the loss is insensitive to local ellipsoid orientation.
- Volumetric rasterization is expensive in memory and runtime, especially when
  using full-resolution working grids and larger Gaussian counts.
- Gradient flow is fragile when cached attributes or detached intermediate
  tensors accidentally break the optimization path.
- Medical-style targets are highly imbalanced, so naive losses can be dominated
  by background voxels and fail to learn thin structures.
- Densification improves flexibility but can introduce instability, including
  sudden topology changes and large gradient spikes.
- Large or dense splats can saturate the predicted volume and reduce the
  quality of gradients for position and orientation refinement.
- Quantitative validation is still limited, which makes it harder to argue for
  performance beyond proof-of-concept status.

## Failure Cases

- Thin vessels or filamentary structures may still be underfit when the number
  of Gaussians is too low or densification is too conservative.
- Very high Gaussian counts can become impractical because the current volume
  rasterization path scales poorly, especially around rotation handling.
- If initialization from the mask is poor or too sparse, optimization can get
  stuck in a bad local configuration and fail to cover the ROI adequately.
- If the ROI is too broad, the loss can spend too much effort on empty space
  and weaken supervision on the relevant anatomy.
- If intensity or opacity updates are overly aggressive, training can become
  noisy or unstable instead of converging smoothly.
- Joint supervision may fail to balance mask fidelity and intensity fidelity if
  the task weights are not tuned well for a given dataset.
- Data with strong anisotropy, unusual spacing, or inconsistent intensity
  scaling may expose assumptions in preprocessing and normalization.

## Design Decisions

- Use direct volume supervision instead of view synthesis because the primary
  input modality is already a 3D scalar field.
- Initialize splats from a mask rather than from SfM points or random samples,
  so the model starts inside the relevant structure.
- Keep Gaussian positions in normalized [0, 1]^3 coordinates while preserving
  explicit conversion rules from voxel indexing to avoid silent axis mistakes.
- Supervise inside an ROI whenever possible to reduce class imbalance and make
  training compute more useful.
- Support multiple loss types because sparse mask targets and continuous
  intensity targets require different optimization behavior.
- Keep intensity and opacity handling flexible, allowing either learned values
  or values sampled from the source volume depending on the experiment.
- Add scale constraints, clipping, and diagnostics because stable optimization
  is a core engineering problem in volume-based Gaussian fitting.
- Adapt densification and pruning to volume-space gradients instead of relying
  purely on the screen-space logic used in standard Gaussian splatting.
- Treat PLY export as part of the method workflow, not just a debugging tool,
  because external visualization is important for scientific interpretation.