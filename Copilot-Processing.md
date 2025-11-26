# User Request Details
- Remove legacy RGB/point-cloud training paths so the system exclusively supports volume + mask based optimization.
- Delete or refactor CLI flags and code paths that only existed for the classic RGB workflow (e.g., `--volume_supervision`, `--init_from_mask`).
- Update entry points/documentation so volume-driven initialization and supervision are always active defaults.

## Action Plan
1. Audit CLI and training pipelines to identify RGB-only options, dataset loaders, and initialization flows that can be removed or simplified for volume-only use.
2. Refactor argument parsing and defaults so volume supervision and mask-based initialization are always enabled, deleting obsolete flags and conditional branches.
3. Prune unused modules (e.g., SfM point-cloud loaders) or clearly isolate them outside the default path while ensuring remaining code compiles.
4. Run targeted smoke tests (CLI `--help`, short training invocation) to ensure the simplified pipeline works without the removed flags.

## Task Tracker
### Phase 1 – Audit Legacy Paths
- [ ] List CLI flags, dataset hooks, and feature toggles tied exclusively to RGB / SfM pipelines.
- [ ] Map the dependencies of `--volume_supervision` and `--init_from_mask` within `train.py`, `arguments.py`, and helper modules.

### Phase 2 – Refactor & Cleanup
- [ ] Remove/inline the guarded code paths so volume supervision and mask initialization are always enabled, updating docs and defaults accordingly.
- [ ] Excise or quarantine unused modules, imports, and helper functions that were only relevant for the RGB pipeline.

### Phase 3 – Validation
- [ ] Run `python train.py --help` to verify CLI surfaces only relevant options and succeeds.
- [ ] Execute a short training dry-run (e.g., 5 iterations) to confirm the new defaults operate end-to-end.


## Action Plan

1. Inspect the current autograd path (train loop → volume supervisor → `splat_to_volume`) to confirm where `_scaling` disconnects, instrumenting shapes/grad flags as needed.

