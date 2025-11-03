
# User Request Details
- Add mean-covered-voxel intensity sampling restricted to large splats.
- Refresh samples on a configurable cadence (every k intensity updates).
- Integrate the option through Gaussian model and training utilities without regressing existing modes.

## Action Plan
1. Introduce configuration hooks for the new sampling mode, large-splat threshold, and refresh cadence.
2. Implement mean-covered-voxel intensity sampling utilities with support for filtering by splat size.
3. Integrate gated sampling into `GaussianModel` intensity updates, caching results between refresh cycles.
4. Update training/volume supervision loops to honor the cadence and mode selection without affecting existing pathways.
5. Add targeted validation to confirm behavior and document any new knobs.

## Task Tracker
### Phase 1: Configuration Hooks
- [x] Expose mode/threshold/cadence flags in `arguments/__init__.py`.
- [x] Thread configuration through `train.py` into the Gaussian model state.

### Phase 2: Sampling Utility
- [x] Add mean-covered-voxel sampler guarded by splat-size filtering in `gaussian_splatting/utils/intensity_sampler.py`.
- [x] Ensure sampler gracefully falls back when coverage data is absent.

### Phase 3: Model Integration
- [x] Gate intensity refresh in `scene/gaussian_model.py` using cadence and cache previous samples.
- [x] Mark large splats for sampling via existing scale or covariance metrics.

### Phase 4: Loop Updates
- [x] Update `gaussian_splatting/utils/volume_supervisor.py` to respect new cadence/mode.
- [x] Keep legacy modes unchanged and confirm scheduling logic remains stable.

### Phase 5: Validation & Docs
- [x] Add sanity test or logging to verify cadence and large-splat filtering.
- [x] Document new options in release notes or README snippet if needed.


## Summary
- Added CLI knobs for mean-covered sampling cadence, threshold, and radius, and
	threaded them through model setup.
- Implemented voxel coverage averaging with graceful fallbacks and wired it
	into the Gaussian sampler, gating updates to large splats only.
- Extended volume supervision scheduling to honour the new cadence and emit
	optional debug metrics.
- Documented the new intensity options in the training guide.


