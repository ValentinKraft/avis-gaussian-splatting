# 3DGS

https://arxiv.org/html/2308.04079 / https://github.com/graphdeco-inria/gaussian-splatting 
Title: 3D Gaussian Splatting for Real-Time Radiance Field Rendering
Comments: Basispaper!

https://arxiv.org/html/2404.11285v2 / https://github.com/KeKsBoTer/cinematic-gaussians 
Title: Application of 3D Gaussian Splatting for Cinematic Anatomy on Consumer Class Devices
Comments: Von Siemens Healthineers - Anwendung von Gaussian SPlatting auf High End Realistic Rendered images via path Tracing --> Sieht ähnlich gut aus aber ist viel schneller --> Gleiche Grundidee wie bei mir

https://nju-3dv.github.io/projects/Relightable3DGaussian/ 
Title: Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing
Comments: Realistic Relighting of GS --> We dont need Volume Rendering / Path Tracing anymore for realistic rendering. If we combine our approach with one of these approaches, we get highly realistic and very fast rendering.

https://arxiv.org/html/2406.02518
Title: DDGS-CT: Direction-Disentangled Gaussian Splatting for Realistic Volume Rendering

https://arxiv.org/html/2505.19175
Title: Triangle Splatting for Real-Time Radiance Field Rendering
Comments: idea: using other primitives for GS

https://arxiv.org/html/2506.20202v1
Title: RaRa Clipper: A Clipper for Gaussian Splatting Based on Ray Tracer and Rasterizer
Comments: Clipping for 3DGS

https://arxiv.org/html/2504.17954
Title: iVR-GS: Inverse Volume Rendering for Explorable Visualization via Editable 3D Gaussian Splatting
Comments: Core Idea: Mimicing Changing Transfer Functions for 3DGS by creating multiple 3DGS models for multiple "thresholds" (a bit like ISO-surfaces?) and combining them together. Quasi like ISO-surfaces for 3DGS.  

https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.70032 
Title: Does 3D Gaussian Splatting Need Accurate Volumetric Rendering?

# Volume Rendering / Realistic Medical Rendering

https://www.researchgate.net/publication/48546966_EWA_volume_splatting
Title: EWA volume splatting
Comments: Basic idea is not new; volume rendering using a splatting approach based on elliptical Gaussian kernels

https://pubmed.ncbi.nlm.nih.gov/38231802/ / https://ieeexplore.ieee.org/document/10403818 
Title: A Clinical User Study Investigating the Benefits of Adaptive Volumetric Illumination Sampling
Comments: Paper written by me. A study that proves that our new method called "AVIS" volume rendering (realistic and fast volume rendering, well suited for mixed reality) is beneficial in the medical domain (surgeons are able to make faster decisions). This emphasizes that realistic volume rendering can have a positive impact in the medical domain but although volume rendering got quite fast, it might still be too slow for large datasets in weak hardware devices (such as AR devices).

R. A. Drebin, L. Carpenter, and P. Hanrahan, “Volume rendering,”Comput. Graph., vol. 22, no. 4, pp. 65–74, Aug. 1988.
Title: "Volume Rendering"
Comments: Volume Rendering basic paper

T. Kroes, F. H. Post, and C. P. Botha, “Exposure render: An interactive photo-realistic volume rendering framework,” PLoS One, vol. 7, no. 7, 2012, Art. no. e38586
Title: Exposure render: An interactive photo-realistic volume rendering framework
Comments: Path Tracer implementation

# Segmentation

- nnUnet
- TotalSegmentator
- SAM

# Mixed Reality

https://pmc.ncbi.nlm.nih.gov/articles/PMC11327191/
Title: Mixed Reality in the Operating Room: A Systematic Review
Comments: Mixed Reality offers significant benefits in the medical domain, but there are still open challenges such as ergonomic issues, limited field of view, and battery autonomy that must be addressed to ensure widespread acceptance.

https://games.jmir.org/2023/1/e41297
Title: Mixed Reality in Modern Surgical and Interventional Practice: Narrative Review of the Literature

# Summary

With 3DGS, we can get similar results to volume rendering, with realistic lighting, clipping / transfer function changes, but significantly faster.