# Direct Volumetric 3D Gaussian Splatting for Medical Image Supervision

## Abstract
We extend 3D Gaussian Splatting from multi-view RGB supervision to direct optimization on volumetric medical data (e.g., CT, MRI, segmentation masks). Our method initializes and trains anisotropic Gaussians directly from scalar or mask volumes without requiring synthetic projections. We introduce: (1) a robust sampling and initialization pipeline from volumetric intensity or mask distributions; (2) differentiable volumetric rasterization (splat-to-volume) incorporating per-Gaussian position, scale, opacity, and quaternion rotation; (3) volume supervision losses (Dice, Tversky, MSE, KL) applied directly in voxel space; (4) parameter diversity regularizers encouraging anisotropy and non-trivial rotations; (5) monitoring and diagnostics for training dynamics; (6) memory- and stability-oriented engineering (auto-resizing, chunked voxelization, adaptive batching). Remaining limitations include gradient attenuation in deep parameter chains, rotation under-utilization, and incomplete quantitative validation. This work positions volumetric 3DGS as an emerging representation for medical model compression, structural abstraction, and radiological visualization.

## 1. Introduction
Neural scene representations increasingly target sparse, view-based inputs. Medical imaging lacks calibrated multi-view photographs but offers rich 3D scalar fields. Bridging 3D Gaussian Splatting (3DGS) to volumetric supervision removes the need for surrogate rendering pipelines. We directly optimize a Gaussian field to approximate anatomical structure encoded in volumetric intensity or mask probability. This reduces data conversion overhead, avoids camera synthesis, and opens paths to hybrid geometric–semantic priors. Challenges addressed: stable initialization from heterogeneous masks, preserving autograd integrity for millions of Gaussian parameters, preventing scale collapse, exposing rotational degrees of freedom, and constraining memory footprint.

## 2. Related Work (Brief)
- NeRF & radiance fields: require calibrated views; unsuitable for single clinical volumes.
- Original 3D Gaussian Splatting: real-time view synthesis; image-plane constraints.
- Volumetric shape extraction (marching cubes, point clouds): lack parametric anisotropic density control.
- Sparse volumetric data structures (octrees / hash grids): complementary; could accelerate our rasterization.
- Parameter diversity & dispersion: echoes entropy and quaternion decorrelation literature.

## 3. Method Overview
We represent a target volume \(V \in \mathbb{R}^{D\times H\times W}\) with \(N\) anisotropic Gaussians: positions \(x_i\), log-scales \(s_i\), unit quaternions \(q_i\), opacity \(\alpha_i\), and optional feature/intensity channel \(f_i\). A differentiable voxelization operator accumulates Gaussian densities into predicted volume \(\hat V\). A supervision loss \(L_{sup}(\hat V, V)\) plus diversity regularizers \(L_{div}\) enforce structural faithfulness and parameter dispersion. Optimization uses Adam on persistent nn.Parameters (no tensor identity breakage).

## 4. Data & Initialization
- Input: intensity volumes or (soft) masks (.nii, .mhd, .npy).
- Safety auto-resize (e.g., 512×512×286 → ≈310×310×173) to stabilize multinomial sampling and fit memory.
- Sampling: multinomial over flattened intensity / mask with epsilon smoothing.
- Coordinate normalization to [0,1]^3; optional world transform (future extension).
- Scales: base density-compensated constant; quaternions start as identity; opacities from mask or intensity-derived range-clamped.

## 5. Differentiable Volumetric Splatting
`splat_to_volume`:
- Chunked accumulation (grid sub-blocks × point batches) to bound memory.
- Rotation via quaternion → rotation matrix (current partial loop fallback for stability; further vectorization needed).
- Approximated anisotropic covariance: rotated diagonal scaling (avoid full inversion / determinant overhead).
- Raw accumulation kept linear (no early sigmoid) to preserve gradient magnitude.
- Engineering: removed frequent cache purges; adaptive batch size heuristics; fallback paths for large N to avoid CUDA einsum failures.

## 6. Loss Functions
**Supervision**: Dice, Tversky (class imbalance), MSE (intensity regression), KL (probabilistic masks).  
**Diversity**:
- Scale diversity: maximize intra-point axis variance; penalize isotropy.
- Rotation diversity: distance from identity + orientation entropy surrogate.
Potential scheduling (not yet implemented) to reduce diversity weight late in convergence.

## 7. Training Dynamics & Monitoring
`ParameterMonitor` logs distribution statistics (min/max/mean/std) and gradient norms for xyz, scales, rotations, opacities. Empirical observations:
- Rotation gradients weak when volume supervision insensitive to local ellipsoid orientation.
- Scale collapse prevented by variance-based penalty.
- xyz gradients can diminish if volume saturates (dense overlapping Gaussians) → suggests occupancy regularization.

## 8. Implementation Integrity
- All core tensors as nn.Parameters (no optimizer state loss by reassignment).
- Shape conventions explicit: internal position [3,N], external sampling [N,3].
- Neighborhood + fallback sampling for sparse mask regions.
- Avoid `.detach()` / `.item()` in gradient-critical paths.
- Chunked processing prevents transient OOM; rotation step remains bottleneck for N≥1000.

## 9. Experimental Status
Current scope: synthetic vascular-like mask + paired intensity volume.  
Tested N: 100–1000 (runtime blow-up beyond due to rotation loops).  
Logged: loss evolution (Dice/MSE), parameter dispersion, gradient norms.  
Missing: baseline comparisons (marching cubes, NeRF-from-slices), ablations (±diversity), runtime scaling curves, memory scaling table.

## 10. Preliminary Findings
- Direct volumetric supervision feasible without view synthesis bridge.
- Diversity losses maintain anisotropy; prevent trivial spherical Gaussian set.
- Rotation under-utilization indicates need for orientation priors (e.g., gradient-aligned frames).
- Performance dominated by rotation transform step (Python-level loop) at higher N.
- Gradient stability improved after eliminating manual perturbation hacks.

## 11. Limitations
1. Limited dataset breadth (single prototype volume).  
2. Performance: rotation application not fully vectorized; scaling poorly with N.  
3. No spatial acceleration (hash grid / octree) → unnecessary dense evaluation.  
4. Opacity model simplistic (no learned transfer function).  
5. Diversity regularizers heuristic; lacks theoretical bound or annealing schedule.  
6. No uncertainty modeling or probabilistic occupancy.  
7. Lacks rigorous quantitative comparison vs established volumetric recon methods.

## 12. Future Work
- CUDA / fused kernel for batched anisotropic splats.
- Structure-tensor–guided rotation initialization (align ellipsoids to local gradients / vessels).
- Hierarchical densification (coarse-to-fine Gaussian introduction). 
- Comparison with NeRF slice-adaptations & sparse conv voxel grids.
- Multi-channel feature Gaussians (semantic + intensity). 
- Uncertainty / Bayesian opacity modeling.
- Public benchmark suite and evaluation scripts.

## 13. Applications
- Compact structural encodings for interactive visualization.  
- Hybrid overlays in radiological viewers (Gaussian proxies + raw slices).  
- Prior field for segmentation refinement (teacher → student distillation).  
- Fast sampling substrate for simulation / registration tasks.

## 14. Ethical & Clinical Considerations
- Data privacy (PHI stripping) remains external requirement.  
- Must validate that compression does not remove diagnostically relevant signals.  
- Clinical deployment requires multi-institution robustness assessment.

## 15. Conclusion
We introduce a direct volumetric supervision pipeline for 3D Gaussian Splatting, removing dependency on multi-view image capture and enabling nascent applications in medical image representation. Although performance and validation gaps remain, the framework establishes a practical foundation for future research into hybrid geometric–semantic volumetric modeling.

## Acknowledgments
Built atop open-source 3D Gaussian Splatting (GraphDeco INRIA). Extensions for volumetric loading, initialization, supervision, and parameter diversity developed in this project.

## References (Placeholders)
[1] Kerbl et al. 3D Gaussian Splatting for Real-Time Radiance Field Rendering.  
[2] Representative NeRF medical adaptations.  
[3] Dice / Tversky loss foundational works.  
[4] Quaternion dispersion / rotation diversity literature.

## Claimed Contributions (Concise)
1. Direct volumetric supervision pipeline (no multi-view precondition).  
2. Robust mask/intensity-driven initialization with safety resizing.  
3. Differentiable voxel accumulation with anisotropic + rotational parameters.  
4. Scale & rotation diversity regularization to avoid degeneracy.  
5. Diagnostics for gradient and parameter health.  
6. Memory-conscious engineering (chunking, auto-resize, parameter integrity safeguards).

## Status Summary
Functional for moderate N (≤1000) with stable gradients; requires vectorization, broader evaluation, and principled rotation priors for publication readiness.
