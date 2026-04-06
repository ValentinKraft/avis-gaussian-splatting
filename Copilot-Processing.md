
## User Request Details (2026-04-06, Increase Splat Anisotropy)
- Start implementation of the anisotropy-improvement plan for smoother surfaces, cleaner object boundaries, and more elongated vessel splats.
- Goal: first leverage existing anisotropy machinery, then add only the smallest missing features that materially improve init-time orientation guidance.

## Action Plan (2026-04-06)
1. Expose orientation-field smoothing controls on the CLI.
2. Thread the new settings through `train.py` into `VolumeSupervisor`.
3. Let strong Hessian vessel cues influence initial rotations even when gradient-based orientations are available.
4. Run static validation on the touched files.

## Task Tracker (2026-04-06)
- [x] Add CLI flags for orientation-field smoothing and Hessian-orientation blending in `arguments.py`.
- [x] Thread new orientation controls into `train.py` and `VolumeSupervisor`.
- [x] Add init-time quaternion blending so strong Hessian vessel directions can shape initial rotations in `gaussian_splatting/utils/volume_initializer.py`.
- [x] Validate touched files and derive the first anisotropy-focused command.

## Summary / Current State (2026-04-06)
- `arguments.py` now exposes `orientation_sigma_grad`, `orientation_sigma_tensor`, `orientation_perturb_deg`, and `structure_orientation_strength`.
- `train.py` now forwards the orientation-field smoothing controls into `VolumeSupervisor` and passes `structure_orientation_strength` into initialization.
- `gaussian_splatting/utils/volume_initializer.py` now supports blending Hessian vessel quaternions into initial rotations for strong vesselness regions instead of only stretching scales.
- Smoke-validated with a 1-iteration fp16 training run using border flattening, Hessian-orientation blending, and the new orientation smoothing flags.
- Fixed two fp16 border-path bugs found during validation: Hessian eigendecomposition now promotes to float32 before `torch.linalg.eigh(...)`, and border quaternions are cast back to the destination rotation buffer dtype before assignment.

## User Request Details (2026-04-04, Runtime Fidelity Densification)
- Start implementation of the planned runtime densification changes for volume-supervised fidelity.
- Goal: improve vessel-aware offspring placement/scaling, make active-point subsampling fair for densification stats, and expose the missing runtime controls on the CLI.

## Action Plan (2026-04-04)
1. Expose the missing adaptive densification and structure-guidance flags in the CLI.
2. Replace random active-subset sampling with coverage-based cycling so densification stats are not starved.
3. Make runtime split/clone/hole-fill structure-aware and add spawn jitter / vessel-biased selection.
4. Run static validation and a short smoke run that actually exercises the new densification paths.

## Task Tracker (2026-04-04)
- [x] Add runtime densification CLI flags in `arguments.py`.
- [x] Add fair active-subset coverage state in `train.py`.
- [x] Add structure-aware child scaling, spawn jitter, and weighted hole-fill in `scene/gaussian_model.py`.
- [x] Fix the fp16/fp32 structure-field sampling mismatch found during smoke validation.
- [x] Re-run smoke validation and confirm densification events succeed.

## Summary / Current State (2026-04-04)
- `arguments.py` now exposes the missing runtime densification controls, including low-density / coverage tuning and vessel-aware spawn controls.
- `train.py` now cycles capped active subsets through a persistent shuffled order instead of drawing a fresh random subset each iteration.
- `scene/gaussian_model.py` now uses structure-aware child scaling for split/clone paths, optional spawn jitter, and vessel-weighted hole-fill candidate selection.
- A smoke run with capped active points and aggressive densification completed after fixing a dtype mismatch in `_structure_strength_from_field(...)`.
- Verified smoke outcome: densification fired successfully at iteration 5 and 10 with net +4000 points each time.

## User Request Details (2026-02-24, README Refresh)
- Update the top-level README with up-to-date information for this fork.
- README should describe project scope (volume-supervised training), setup, how to run training, outputs, and the standalone PLY viewer.

## Action Plan (2026-02-24)
1. Confirm current environment/setup requirements and canonical env name from `environment.yml`.
2. Confirm current CLI flag names and meanings for volume/mask supervision from `arguments.py`.
3. Rewrite the top of `README.md` to focus on volume-supervised usage, preserving upstream 3DGS README content below for reference.

## Task Tracker (2026-02-24)
- [x] Verify environment name + key deps from `environment.yml`.
- [x] Verify volume/mask CLI flags from `arguments.py`.
- [x] Update `README.md` with scope + setup + training + outputs + viewer.

## Summary / Current State (2026-02-24)
- `README.md` now starts with a fork-specific overview focused on volume-supervised training (setup, data conventions, example commands, outputs, and the standalone `gs_viewer`).
- The upstream 3DGS README content remains below for reference.

## Update (2026-02-24, Remove Upstream README Content)
- Follow-up request: remove the large upstream 3DGS README block and keep only information relevant to this volume-supervised fork.
- Resolution: replaced `README.md` entirely with a concise fork-only README (volume workflow + setup + training + export + viewer + optional `sparse_adam` note + acknowledgements).

## User Request Details (2026-02-09, Joint Loss / Dual Render)
- User wants a single training run to optimize both a clean mask-derived shape and faithful CT intensities/high detail.
- Implement a joint objective that renders both density (mask supervision) and intensity (CT supervision) and combines losses with configurable weights.

## Action Plan (2026-02-09)
1. Extend CLI and supervisor configuration to support `--supervision_target joint` plus CT-loss type and per-branch weights.
2. Implement dual rendering in `VolumeSupervisor.compute_loss`: density branch for mask loss + intensity branch for CT loss.
3. Normalize CT targets consistently with the intensity normalization range used for per-splat intensities.
4. Add a minimal unit test covering joint supervision.

## Task Tracker (2026-02-09)
- [x] Add CLI args: `--supervision_target joint`, `--ct_loss_type`, `--mask_loss_weight`, `--ct_loss_weight`.
- [x] Thread args through `train.py` into `VolumeSupervisor`.
- [x] Implement dual render + weighted joint loss in `VolumeSupervisor.compute_loss`.
- [x] Add a unit test ensuring joint supervision runs and backpropagates.

## Summary / Current State (2026-02-09)
- Joint supervision is available via `--supervision_target joint`.
- Mask branch uses density rendering and `--volume_loss_type`.
- CT branch uses intensity rendering and `--ct_loss_type`.
- Total loss is `mask_loss_weight * mask_loss + ct_loss_weight * ct_loss` (then scaled by `--volume_loss_weight`).
- CT targets are normalized to the same (mask-bounded) intensity range used when sampling per-splat intensities.

## User Request Details (2026-02-07, Full-Resolution Volume Training)
- User wants volume-supervision training to operate at native resolution (no load-time downsampling and no render-time half-resolution working grid), to improve surface sharpness and retain fine detail.
- Current command used `--volume_downscale_factor 2`; user wants a full-resolution option.

## Action Plan (2026-02-07)
1. Identify all implicit downsampling points: (A) load-time resampling in `VolumeLoader`, (B) render-time working-grid downscale in `splat_to_volume`.
2. Expose explicit CLI controls for both, keeping defaults backward-compatible.
3. Ensure the new flags are threaded through `train.py` into `VolumeSupervisor` and initialization paths.
4. Validate no new syntax/editor errors in touched files.

## Task Tracker (2026-02-07)
- [x] Add CLI flags: `--volume_render_downscale_factor` and `--disable_volume_overflow_guard`.
- [x] Add `enable_overflow_guard` toggle to `VolumeLoader`.
- [x] Parameterize render-time working-grid downscale in `splat_to_volume`.
- [x] Thread new flags through `train.py` → `VolumeSupervisor` and initialization.
- [x] Validate touched files have no new editor-reported errors.

## Summary / Current State (2026-02-07)
- Full-resolution volume loading is now possible by using `--volume_downscale_factor 1` and `--disable_volume_overflow_guard` (disables the loader's auto-resize safety).
- Full-resolution supervision rasterization is now possible by using `--volume_render_downscale_factor 1` (disables the half-resolution working grid used when point count is large).

## User Request Details (2026-01-28, Densification Stats Under Subsampling)
- Fix densification/pruning behavior when training uses active-point subsampling (`MAX_POINTS_PER_ITER`).
- Option A: accumulate densification stats only for points that are both active and visible.
  - Volume-only mode does not have a screen-space visibility concept; treat "visible" as "active" (participated in forward/backward) in that path.
- No hard point-count cap.

## Action Plan (2026-01-28)
1. Ensure densification stats accumulation (`xyz_gradient_accum`/`denom`) updates only active points during subsampled volume training.
2. Harden `densify_and_prune` against `denom==0` causing `Inf` gradients, and compute adaptive thresholds using only valid (updated) points.
3. Run a quick static validation pass (syntax / type checks available via editor tooling).

## Task Tracker (2026-01-28)
- [x] Update volume-mode densification accumulation in `train.py` to increment `xyz_gradient_accum`/`denom` only at `active_idx` (or all points when `active_idx is None`).
- [x] Update `GaussianModel.densify_and_prune` to clamp `denom`, sanitize `Inf`/`NaN`, and compute adaptive threshold on points with `denom>0`.
- [x] Validate there are no new syntax errors in the touched files.

## Summary / Current State (2026-01-28)
- Volume-mode densification accumulation no longer adds `+1` to `denom` for inactive points, preventing gradient dilution under subsampling.
- `densify_and_prune` now ignores never-updated points (`denom==0`) when computing quantile thresholds and avoids `Inf`/`NaN` gradients from divide-by-zero.

## User Request Details (2026-01-26, Learnable Opacity Modes)
- Implement learnable opacity handling analogous to intensity handling.
- Add an opacity mode flag (e.g., `--opacity_mode {sampled,learned,sampled_mean_covered}`) or similar.
- Desired behavior:
	- `sampled`: sample the input mask volume to initialize per-Gaussian opacities (as done today).
	- `learned`: use learnable opacity parameters (optimize during training).
	- `sampled_mean_covered`: use the same mean-covered strategy concept used for intensities, but for opacity.
- Wire the chosen mode into initialization + training.

## User Request Details (2026-01-27, Performance / CPU Overhead)
- Reduce CPU bottlenecks during training (GPU utilization currently shows small spikes).
- Implement practical throttles for progress/TensorBoard logging and avoid unnecessary extra gradient computations.
- Keep behavior the same by default where possible; add opt-in switches when VRAM tradeoffs exist.

## Action Plan (2026-01-26)
1. Add CLI surface for opacity mode selection.
2. Implement opacity mode behavior in initialization.
3. Implement opacity mode behavior during training updates.
4. Add minimal tests/smoke validation hooks.

## Task Tracker (2026-01-26)
### 1) CLI
- [x] Add `--opacity_mode {sampled,learned,sampled_mean_covered}` to `arguments.py` under intensity/appearance controls.
- [x] Decide default (`sampled` for parity with current behavior).
- [x] Update help text to clarify interaction with `--supervision_target mask` and `--opacity_gamma`.
- [x] Add `--opacity_update_interval` for sampled refresh cadence.

### 2) Initialization
- [x] Locate current mask-based opacity sampling path.
- [x] Route initialization based on `opacity_mode`:
	- [x] `sampled`: sample mask at seed positions and populate a non-learnable `opacities` buffer (respecting gamma mapping).
	- [x] `learned`: initialize `_opacity` logits (seeded from mask when available) and keep it trainable.
	- [x] `sampled_mean_covered`: share the mean-covered machinery with intensity (uses existing large-splat thresholds/interval).
- [x] Ensure `_opacity` is an `nn.Parameter` and participates in optimizer when learnable.

### 3) Training Behavior
- [x] `sampled`: refresh opacities from mask regularly via `--opacity_update_interval` and dirty-index checks.
- [x] `learned`: leave opacity learnable and do not overwrite during training.
- [x] `sampled_mean_covered`: refresh large-splat opacities via the same mean-covered knobs used for intensity.

### 4) Validation
- [x] Add/extend a small unit test to ensure learned `opacity_mode` ignores the mask-buffer override.
- [ ] Add a short-run smoke command to confirm the log prints the chosen mode and training runs without CLI errors.


## Summary / Current State
Implemented Option B: New splats are now enforced to spawn inside the mask (mask value >= 0.5).
	- Init-time: `initialize_from_volume()` now resamples any jittered seeds that fall below the mask threshold.
	- CLI/training: `--init_mask_threshold` default is now 0.5 and values < 0.5 are treated as 0.5 when `--mask_path` is provided.
	- Densification: `densification_postfix()` now resamples any newly created points that land outside the reference mask.

## Update (2026-01-19)
- Re-enabled densification/pruning by default via CLI (`--enable_densification` now defaults to true; `--disable_densification` remains the explicit override).
- Medical preset (`medical_mode=organ`) no longer hard-disables densification; it now uses the same gentle schedule as vessel mode when enabled.
- Added [tests/conftest.py](tests/conftest.py) so tests can import top-level modules like `train.py` reliably under pytest.
---
post_title: Copilot Processing Log
author1: GitHub Copilot
post_slug: copilot-processing
microsoft_alias: copilot
featured_image: ''
categories: []
- Running `python train.py --model_path _output_/synth-test --mask_path _input_/synthetic-mask-binary.nii.gz --volume_path _input_/synthetic-gradient.nii.gz --iterations 2000 --init_n_points 5000 --save_ply_every 100` now raises `UnboundLocalError: cannot access local variable 'diagnostics_enabled' where it is not associated with a value` inside `training()` at line 316.
- Goal: make the diagnostics gating logic robust so training proceeds regardless of preset/flag combinations, allowing the synthetic medical test run to succeed without manual code edits.
- Environment: Windows, conda env `avis_gaussian_splatting`, date 2025-12-12; crash occurs before optimization starts.

## User Request Details (2025-12-22)
- User reverted the repository back to the last commit because they did not like the previous set of changes.
- Goal: re-analyze the training loop and determine whether it is mixing coordinate systems (e.g. normalized volume space vs world/camera space).
- Key question: are Gaussian xyz coordinates, volume-grid coordinates, and any scene/camera extent transforms consistently defined for volume supervision?

## User Request Details (2025-12-22, ROI Crop)
- Goal: implement ROI cropping so volume rendering only evaluates the mask bounding box instead of the full volume every iteration.
- Constraints:

## Summary / Current State
- Implemented Option B: New splats are now enforced to spawn inside the mask (mask value >= 0.5).
- Init-time: `initialize_from_volume()` now resamples any jittered seeds that fall below the mask threshold.
- CLI/training: `--init_mask_threshold` default is now 0.5 and values < 0.5 are treated as 0.5 when `--mask_path` is provided.
- Densification: `densification_postfix()` now resamples any newly created points that land outside the reference mask.
- Color sampling: splat intensities/colors are now sampled from the full-resolution input CT volume even when `--volume_downscale_factor` is used for supervision.

## In Progress
- ROI padding/caching + adaptive sparse splat support implementation started (ROI pad set to 3 voxels, ROI tensors cached, sparse support radius driven by max-axis sigma and a 0.5 weight cutoff).

## Change Log
- Init seeding now samples point locations from the full-resolution mask (`downscale_factor=1`) regardless of `--volume_downscale_factor`.
	- Loss must still be computed only over voxels inside the thresholded mask (mask > 1% of mask max).
	- Training should require a mask volume.
	- Coordinates remain normalized volume space ([0,1]^3).

## Action Plan
1. Inspect `train.py` around line 316 to understand how `diagnostics_enabled` is set and why medical presets skip its initialization.
2. Implement a robust initialization path so `diagnostics_enabled` is always defined before use, respecting CLI flags and preset overrides.
3. Validate by re-running the reported training command (or equivalent minimal repro) to ensure the UnboundLocalError is resolved.

## Task Tracker
### Phase 1 – Diagnostics Flag Inspection
- [x] Read the relevant sections of `train.py` (argument parsing, preset configuration, and diagnostics gating) to trace where `diagnostics_enabled` should be defined.
- [x] Confirm how medical presets or CLI defaults could bypass the assignment before line 316 executes.

### Phase 2 – Fix Implementation
- [x] Introduce a safe default or refactor the diagnostics gating logic so `diagnostics_enabled` is defined for every control path before it is read.
- [x] Ensure any helper functions or preset overrides honor the user-provided flag without reintroducing unused diagnostics work.

### Phase 3 – Validation
- [x] Re-run `python train.py ...` (or a quicker equivalent) to verify the training loop starts without crashing. (Superseded by later repo changes; not re-validated here.)
- [x] Update this log with the validation outcome and any follow-up steps.

### ROI Crop – Implementation
- [x] Add ROI bounds plumbing to the renderer (`grid_bounds` in `splat_to_volume`) so a subvolume grid can be generated over an arbitrary [min,max] box.
- [x] Compute ROI bounding box from the thresholded mask inside `VolumeSupervisor.compute_loss`.
- [x] Render only the ROI and slice `volume_gt`/mask to the same ROI before computing the masked loss.
- [x] Preserve expected visualization shape by storing a full-size prediction volume with the ROI inserted into zeros.

### ROI Crop – Validation
- [ ] Run a short smoke test training run and confirm loss decreases and shapes/metrics/logging behave as expected.

## User Request Details (2025-12-22, Optional Resizing)
- Goal: make initial volume/mask resizing optional and provide a CLI flag to downsample by an integer factor (e.g. 2 or 4).
- Desired behavior:
	- When the flag is set to 2/4/...: downsample each axis by that factor during load.
	- When the flag is set to 1 (or 0): keep native resolution (no resampling), unless the overflow safety guard triggers.
	- When the flag is omitted: preserve existing behavior using `--volume_shape`.

## User Request Details (2026-01-04, Mask Supervision)
- Goal: start implementing the redesign to supervise the mask probability volume directly.
- Required points:
  - Supervision target = mask probability (not CT intensity) when selected.
  - A density/alpha render mode (sum of contributions) suitable for mask supervision.
  - Probability-faithful opacity mapping with no forced minimum; optional gamma shaping.

## Action Plan (2026-01-04)
1. Add CLI flags to switch supervision target and configure mask/opacity mapping.
2. Add `render_mode='density'` path in `splat_to_volume` for mask supervision.

## User Request Details (2026-02-11, Copilot Instructions)
- Analyze the repo and generate/update `.github/copilot-instructions.md` to guide AI coding agents.
- Capture the project-specific architecture, workflows, and conventions (esp. volume supervision + Gaussian splatting training flow).
- Merge with any existing agent instruction files if present.
- Keep the instructions concise and actionable (~20-50 lines) and include concrete file references.

## Action Plan (2026-02-11)
1. Discover existing agent rules and current `.github` guidance (if any) and extract repo-specific workflows from README and training scripts.
2. Identify the core architecture paths relevant to daily work (training loop, rendering, volume supervision, AO/intensity/opacity sampling).
3. Write or update `.github/copilot-instructions.md` with concise, codebase-specific instructions (include file references and example commands).
4. Run a quick static check: ensure the new markdown follows repo markdown rules and that referenced paths exist.

## Task Tracker (2026-02-11)
- [x] Locate any existing agent instruction files and collect authoritative workflow commands.
- [x] Summarize big-picture architecture + key data flow for training/rendering.
- [x] Document project-specific conventions (coords/order, tensor shapes, volume vs camera space, CLI flags).
- [x] Create/update `.github/copilot-instructions.md` (20-50 lines, actionable).
- [x] Validate formatting and paths; keep content minimal and factual.

## Summary / Current State (2026-02-11)
- Added `.github/copilot-instructions.md` capturing the volume-supervised training architecture, core data conventions, and the key files to edit for CLI/supervision/sampling/export changes.
- Included concrete workflow commands (conda setup on Windows, training entrypoints via `README.md`, focused `pytest` targets), plus repo-specific gotchas (volume normalization to [0,1], export-only AO, SH-DC grayscale mapping) and a debug hook (`GS_VALIDATE_SAMPLING=1`).
3. Wire mask supervision through `VolumeSupervisor` and initialization.
4. Update tests for API changes and do a quick smoke run.

## Task Tracker (2026-01-04)
- [x] Add `--supervision_target {mask,ct}` and `--mask_loss_threshold_rel`.
- [x] Add `--opacity_gamma` and apply gamma mapping to mask-sampled opacities.
- [x] Remove forced opacity min/max during mask sampling.
- [x] Add density render mode to the volume splatter.
- [x] Route mask supervision to density rendering and mask target in `VolumeSupervisor`.
- [ ] Run a short training smoke test and confirm the new mode learns.

## User Request Details (2026-01-05, Spiky/Missing Center)
- Observed: volume rendering looks broadly correct, but the center of the tubular structure becomes spiky and/or has missing density.
- Context: synthetic run completed (1000 iters) with `--supervision_target mask` and `--opacity_gamma 1.0`.
- Goal: diagnose root cause (renderer vs optimization vs initialization) and fix to produce smooth, contiguous density throughout the structure.

## Project Goal (Current)
- Represent a 3D segmentation probability mask as a smooth volumetric rendering using 3D Gaussians.
- Optimize in normalized volume coordinates ([0,1]^3), with supervision computed only inside the thresholded mask.
- Make CT intensity optional for appearance; primary objective is mask-faithful opacity/density.

## Action Plan (2026-01-05)
1. Reproduce and quantify the artifact with diagnostics (before/after snapshots and basic stats).
2. Isolate whether the artifact comes from (A) rotation handling, (B) scale handling, (C) opacity sampling/mapping, (D) density accumulation/squash, or (E) densification/pruning.
3. Implement the minimal fix once the culprit is confirmed.
4. Add a small regression test or diagnostic assertion so the artifact does not silently return.

## Task Tracker (2026-01-05)
- [ ] Confirm artifact is present in the saved `volume_pred` tensor (not just viewer / colormap).
- [ ] Inspect per-point stats split by axial regions (center vs ends): position histogram, scales (min/mean/max), rotations (norm, stability), opacities (min/mean/max).
- [ ] Run A/B toggles to isolate cause:
	- [ ] Disable rotation usage in volume renderer (treat as identity) and compare.
	- [ ] Disable anisotropy at init (`--anisotropy_strength 0`) and compare.
	- [ ] Clamp max scale (temporary) to see if large splats create spikes.
	- [ ] Try `--opacity_gamma 2.0` and `0.7` to see if opacity mapping causes center dropout.
- [ ] Validate quaternion convention end-to-end (confirm stored quats are (w,x,y,z)); if mismatch, fix rotation-to-matrix conversion.
- [ ] Verify ROI bounds mapping aligns voxel centers (ensure ROI grid covers the correct region without internal seams).
- [ ] Check densification/pruning behavior in volume-only mode (ensure it does not remove points preferentially in the center).
- [ ] Implement fix based on findings (likely candidates: rotation convention bug, anisotropy axis mix-up, or over-aggressive large-scale splats).
- [ ] Add a targeted regression test using a synthetic cylinder mask: ensure density along the cylinder axis is unimodal/continuous (no central void) under `supervision_target=mask`.

## Progress (2026-01-05)
- Simplified volume renderer path for debugging: removed downsample/upsample shortcut and removed gradient checkpointing.
- Added a hard anisotropy cap in the volume splatter to prevent needle-like spikes during rendering/supervision.
- Changed default `--volume_loss_type` to `mse` to better match mask-probability supervision.

## Next Steps (Recommended)
1. Re-run the synthetic training command and compare `current.png` before/after the renderer simplification.
2. If spikes persist, disable anisotropy initialization (`--anisotropy_strength 0`) to isolate whether it is an init/stretch issue.
3. Add a small cylinder regression test once the artifact is resolved.

## User Request Details (2026-01-07, Global Scale Uniformity)
- Observation: interior splats grow very large while boundary splats remain small.
- Goal: keep some size variation, but globally constrain splat sizes to a tighter band.
- Chosen spec: global target band of 1–3 voxels.

## Action Plan (2026-01-07)
1. Make initialization scales global (not distance-field driven), sampling in a 1–3 voxel band.
2. Add a soft training regularizer that penalizes excessive spread in log-scales (in voxel units).
3. Add CLI knobs for the init scale band and spread penalty (default spread penalty off).

## Task Tracker (2026-01-07)
- [x] Add init-scale band args (`--init_scale_min_vox`, `--init_scale_max_vox`).
- [x] Initialize scales globally in the 1–3 voxel band during volume initialization.
- [x] Add global log-scale spread penalty (`--scale_logvar_weight`, `--scale_logvar_warmup_iters`).

## User Request Details (2026-01-07, Speed: Cache ROI Grid)
- Request: training is slow; implement voxel grid caching since volume/ROI are fixed.
- Assumption: volume shape/data and ROI bounds remain constant during training.

## Task Tracker (2026-01-07, Speed)
- [x] Cache `create_grid_points()` outputs keyed by `(shape, bounds, device, dtype)` to avoid per-iteration meshgrid allocations.
- [x] Add sparse splatting path that scatter-adds only within each Gaussian's 3-sigma voxel neighborhood (fallback to dense when splats are large).

## User Request Details (2026-01-08, Movement Constraint)
- Request: limit splat movement (~5 voxels) and keep splats inside the mask.
- Update: user withdrew the constraint and asked to keep existing behavior.

## Task Tracker (2026-01-08)
- [x] Retain the current scale-based `enforce_position_displacement_constraint` behavior with `--position_displacement_scale` (default 1.1).
- [ ] Optional: revisit the default displacement scale if more drift is desired without adding new constraints.

## Progress (2026-01-08)
- [x] Mean-covered intensity mode now reuses the coverage refinement path and de-duplicates sampler logic.
- [x] Mask ROI stats (thresholded mask, bbox, bounds, roi_shape) cached once in `VolumeSupervisor` instead of recomputing each iteration.
- [x] Sampling padding mode is configurable; default switched to `border` to avoid darkening at volume edges.
- [x] Added position bounds clamp to keep splats within the mask ROI each iteration.
 - [x] Relaxed position bounds (padded by ~1.5 voxels) so splats can move while remaining inside the mask ROI.
 - [x] Added displacement warmup and a minimum voxel-based allowance to prevent splats from freezing early.
- [x] Fixed displacement clamp NameError by defining device/dtype before voxel-based min movement.
- [x] Updated default `--position_lr_delay_mult` to 1.0 so xyz starts moving immediately (avoids near-zero early LR).
- [x] Fixed sparse splatting kernel to depend on continuous center positions (restore xyz gradients/motion).


## User Request Details (2026-02-10, Mask-Based Ambient Occlusion Bake)
- Before training, generate an occlusion/AO volume from the input mask volume.
- Compute AO once at startup (not per-iteration) and store it (memory is OK).
- AO should be defined for every non-zero (inside-mask) voxel; if possible, only compute the expensive part on the mask surface.
- AO should be based on a hemisphere neighborhood oriented by the local surface normal (e.g., 3-voxel radius neighborhood).
- For every PLY export, sample AO per Gaussian and multiply it into the exported splat color (baked soft AO look).
- Prefer a fast approximation if it produces good-looking soft AO.

## Action Plan (2026-02-10)
1. Implement fast AO precompute from the loaded mask (`mask_bool`) with radius in voxels and optional surface-only hemisphere refinement.
2. Compute the AO volume once at training startup (after `VolumeSupervisor` loads mask) and keep it in memory.
3. Fix periodic PLY export callsite in `train.py` and sample AO values at Gaussian positions only when exporting.
4. Extend PLY export in `GaussianModel` to accept AO and multiply it into exported SH-DC colors; optionally add an `ao` vertex attribute.
5. Add minimal unit tests for AO volume computation and AO-multiplied export.
6. Run a short smoke train command with `--export_ao` to confirm no runtime errors.

## Task Tracker (2026-02-10)
- [ ] Add `gaussian_splatting/utils/ambient_occlusion.py` with `compute_isotropic_ao_from_mask(...)` (conv3d occupancy-based AO).
- [ ] Add optional surface extraction + gradient-normal hemisphere refinement (surface-only) behind an AO method flag.
- [ ] Add/extend CLI flags for AO method and strength (keep defaults safe and backward-compatible).
- [ ] In `train.py`, compute `ao_volume` once when `--export_ao` is set (after `VolumeSupervisor` init).
- [ ] In `train.py`, repair broken `save_ply_sequence` call and compute `ao_values` only on export iterations.
- [ ] In `scene/gaussian_model.py`, update `save_ply(...)` / `save_ply_sequence(...)` to accept `ao` + `ao_strength` and apply it to `f_dc`.
- [ ] (Optional) Add `ao` to the PLY attributes when provided.
- [ ] Add tests: `tests/test_ambient_occlusion.py` (AO volume shape/range) and `tests/test_ply_export_ao.py` (color multiplied).
- [ ] Run `pytest` for the new tests and run a short training smoke command.


## User Request Details (2026-02-12, Standalone Gaussian Viewer)
- Build a separate Gaussian splatting viewer application in a new subfolder at the repository root, not dependent on the current training/rendering implementation.
- Load and render Gaussian splatting models from PLY files exported by this repo.
- Provide interactive 3D engine-style camera controls (Unity-like orbit/pan/zoom).
- Provide a medical volume-rendering-like Transfer Function / LUT that maps a per-splat scalar to color + transparency.
- Minimum viable viewer focused on inspection of medical Gaussian models (similar spirit to MeVisLab).

## Action Plan (2026-02-12, Standalone Gaussian Viewer)
1. Create a new isolated application folder `gs_viewer/` with its own Python package/entrypoint and dependency list.
2. Implement a robust PLY loader for the repo's `GaussianModel.save_ply()` schema:
	- Required: `x,y,z`, `f_dc_0..2`, `opacity`, `scale_0..2`, `rot_0..3`.
	- Optional: `ao`, `f_rest_*` (ignore for MVP).
	- Detect and reject unsupported PLY schemas (e.g. `red/green/blue` point clouds).
3. Implement camera controls (Unity-style orbit/pan/zoom) and auto-frame on load.
4. Implement GPU rendering of splats:
	- Instanced billboards + fragment Gaussian falloff.
	- Use `sigma = exp(scale_*)` and normalized quaternion (MVP can start isotropic).
	- Use weighted blended OIT for stable transparency without sorting.
5. Implement transfer function / LUT:
	- Decode base RGB from SH-DC: `rgb = f_dc * SH_C0 + 0.5` (clamp [0,1]).
	- Scalar for TF: luminance of decoded RGB.
	- UI editor for a small set of control points; bake to 1D LUT texture.
	- Apply TF to output color + alpha: `finalAlpha = opacity * LUT.a`, `finalRGB = LUT.rgb` (or modulation, if chosen).
6. Add minimal UI/UX:
	- Open PLY, reset camera, TF editor panel, stats (count/FPS).
7. Add docs + a minimal smoke test procedure on Windows.

## Task Tracker (2026-02-12, Standalone Gaussian Viewer)
### 1) Project scaffold
- [x] Create `gs_viewer/` folder structure.
- [x] Add `gs_viewer/README.txt` with install/run steps.
- [x] Add `gs_viewer/pyproject.toml` + `requirements.txt` with pinned deps.

### 2) PLY I/O
- [x] Implement `gs_viewer/src/gs_viewer/ply_loader.py`:
	- [x] Load GaussianModel PLY schema (floats, SH DC, opacity, log-scale, quat).
	- [x] Infer presence of optional `ao`.
	- [x] Validate required fields and give clear errors.
- [x] Add a tiny loader self-check (load and print basic stats, no rendering).

### 3) Rendering core
- [x] Implement window + GL init (GLFW).
- [x] Implement shader pipeline for point-sprite splats (GL_POINTS).
- [x] Upload per-splat attributes to GPU buffers.
- [x] Implement weighted blended OIT framebuffer path.

### 4) Camera
- [x] Implement orbit/pan/zoom camera controls.
- [x] Implement frame-to-bounds on model load.

### 5) Transfer function
- [x] Implement TF control points + LUT baking.
- [x] Upload 1D LUT to GPU and apply in fragment shader.
- [x] Ensure TF affects both color and transparency.

### 6) UI
- [x] Add minimal UI (ImGui): open PLY, reset camera, TF editor, stats.

### 7) Validation
- [ ] Smoke test with a real model from `_output_/.../*.ply`.
- [ ] Confirm intensity decode matches exporter convention (SH DC mapping).
- [ ] Confirm interactive camera + TF updates at runtime.

Validation note (2026-02-12)
- Generated `gs_viewer/_sample/minimal_gaussian.ply` and verified the loader self-check works.
- No `.ply` files were found under `_output_/` in the current workspace snapshot, so a real-model smoke test is pending.


