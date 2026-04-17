# Pros

## With respect to 3DGS / In general

- Works directly from a 3D volume and ROI mask, so it does not require rendering of multi-view images, SfM preprocessing or any other preprocessing. That is a major advantage over standard 3DGS for medical and scientific data, where the ground truth already exists in voxel space.
- The supervision is aligned with the real task: the model is optimized against the target volume itself rather than against rendered surrogate images. This makes mask fitting, CT/intensity fitting, and joint supervision more natural.
- Initialization is much more task-aware than in standard 3DGS. Seeding from the mask, plus optional orientation and anisotropy priors, puts model capacity directly into the relevant anatomy from the start.
- The representation is useful not only for rendering but also as a compact volumetric proxy that can be exported, inspected, and visualized with intensity-aware splats and optional AO.
- The method can exploit medical structure priors and voxel-space densification rules that are better suited to sparse structures such as vessels and thin boundaries than view-driven 3DGS heuristics.
- The reconstructed gaussian splatting models are more precise and trust-worthy than with standard 3DGS. Since the colors are not view-dependent, the rendering is also more precise.
- SPlats store a intensity / HU value which makes it possible to apply a transfer function like in Volume Rendering. This is in contrast to standard 3DGS, where the models are mostly static.

## With respect to Volume Rendering

- Once optimized, the Gaussian representation is much more compact than a dense volume, which makes storage, streaming, and interactive rendering more efficient.
- Rendering is way more efficient than high-quality ray-marched volume rendering, even suited for weak hardware (Mixed Reality etc.)


# Cons

## With respect to 3DGS / In general

- It loses one of the main strengths of standard 3DGS: native support for novel-view synthesis from photographs. This project is specialized for volume fitting, not for general image-based scene reconstruction.
- Volumetric rasterization is expensive in memory and compute, especially at full resolution and high Gaussian counts. In practice, training can be heavier than standard image-plane 3DGS.
- The method depends on a good mask and on careful hyperparameter tuning. If the mask is poor, too sparse, or too broad, initialization and training quality can degrade significantly.
- The current system is more specialized and less mature than standard 3DGS: it has stronger medical-data assumptions, more custom stabilization logic, and less broad benchmark validation.
- The implementation is not yet optimized and the resulting gaussian splatting models are still quite uniformly sized and point-cloud-like. There is potential for more optimal and adaptive splat parametrization.

## With respect to Volume Rendering

- The visual results are not as precise and realistic as Volume Renderings.
- Volume rendering can visualize the original data immediately, whereas this method first requires an optimization step to fit the Gaussian model.
- The Gaussian model is only an approximation of the volume, so subtle intensity variations and very fine structures may be lost compared to direct rendering of the source voxels.
- Quality depends strongly on initialization, losses, and the number of splats; poor settings can lead to underfitting, oversmoothing, or point-cloud-like artifacts.