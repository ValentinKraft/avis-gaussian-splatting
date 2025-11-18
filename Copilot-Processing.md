
# User Request Details
- Activate the diversity warmup (Option A) so scale/rotation regularizers contribute during the initial training window with tunable duration and logging hooks.
- Apply the adaptive scaling/rotation tweaks and Option A GaussianModel refactor without regressing existing training behaviors.
- Run focused validation (py_compile plus a targeted smoke test) after the edits land.

## Action Plan
1. Reinstate the diversity warmup path in `train.py`, exposing configuration knobs and telemetry so the early-iteration regularization is active and observable.
2. Layer in the adaptive scaling/rotation safeguards inside `scene/gaussian_model.py`, keeping existing tensor contracts intact while capturing the requested diagnostics.
3. Carve the GaussianModel refactor into cohesive helper modules per Option A, ensuring the public API and downstream callers remain stable.
4. Execute syntax checks and a constrained training smoke test to confirm the integrated changes behave as expected.

## Task Tracker
### Phase 1 – Diversity Warmup Reactivation
- [x] Audit CLI/arg defaults for diversity warmup controls and add any missing parameters (count, weights, logging cadence).
- [x] Re-enable `add_parameter_regularization_loss` usage during the configured warmup window with clear gating and loss dictionary entries.
- [x] Extend monitoring/logging so warmup contributions are emitted to TensorBoard/stdout for early-iteration visibility.

### Phase 2 – Adaptive Scaling/Rotation Tweaks
- [x] Instrument `gaussian_model.py` with early-iteration stats collectors and relaxed scaling constraints that ramp to full enforcement.
- [x] Tie rotation learning-rate multipliers and xyz boost hooks into the adaptive path without breaking optimizer state.
- [x] Preserve densification/pruning snapshots when new adaptive buffers are introduced.

### Phase 3 – GaussianModel Refactor
- [ ] Define module boundaries (core state, densification, schedulers, IO/volume, diagnostics) and migrate code incrementally.
- [ ] Update imports/call sites to use the new helper modules while maintaining backward-compatible methods.
- [ ] Document the new structure and ensure unit/doc references remain accurate.

### Phase 4 – Validation
- [x] Run `py_compile` (or equivalent) across touched Python files.
- [ ] Launch an abbreviated training smoke test covering the warmup/adaptive flows and record metrics.



