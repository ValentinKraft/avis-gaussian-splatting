## Direct volumetric supervision for 3D Gaussian Splatting
We reformulate 3D Gaussian Splatting by replacing conventional multi-view RGB supervision with direct optimization in voxel space using target volumes and region-of-interest masks.

## Differentiable splat-to-volume rasterization.
We introduce a differentiable rasterization method that projects anisotropic Gaussians into a voxel grid, enabling direct optimization of spatial and radiometric parameters in volumetric space.

## Volume-native Gaussian representation with intensity-aware splats.
Each Gaussian explicitly encodes a physical intensity value (e.g., Hounsfield units), enabling a volume-rendering-like representation where transfer functions can be modified post hoc without retraining.

## Volume-driven initialization and coordinate-consistent pipeline.
We propose a volume-native initialization strategy, including mask-based Gaussian seeding and consistent handling of voxel-to-world coordinate transformations.

## Efficient volumetric rendering for resource-constrained devices.
We demonstrate that the proposed representation enables efficient, volume-rendering-like visualization suitable for real-time applications on resource-constrained devices such as mixed reality hardware.