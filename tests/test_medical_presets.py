from types import SimpleNamespace

from train import _configure_medical_presets


def _make_args(**overrides) -> SimpleNamespace:
    """Return a namespace that mimics the CLI arguments used by train.py."""

    defaults = {
        "medical_mode": "none",
        "enable_diversity": False,
        "enable_diagnostics": False,
        "init_n_points": 4000,
        "scale_l2_weight": 0.03,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_opt(**overrides) -> SimpleNamespace:
    """Return a namespace with the optimizer-style knobs referenced by the preset."""

    defaults = {
        "iterations": 5000,
        "densify_from_iter": 500,
        "densify_until_iter": 4000,
        "densification_interval": 100,
        "densify_grad_threshold": 2e-4,
        "diversity_warmup_iterations": 1500,
        "diversity_scale_weight": 0.02,
        "diversity_rotation_weight": 0.02,
        "diversity_scale_range_weight": 0.05,
        "diversity_target_range_weight": 0.05,
        "diversity_rotation_entropy_weight": 0.1,
        "diversity_dispersion_weight": 0.1,
        "diversity_alignment_weight": 0.1,
        "scale_l2_weight": 0.03,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_medical_mode_organ_disables_densification() -> None:
    """Organ preset boosts init samples and disables densification/constraints."""

    args = _make_args(medical_mode="organ")
    opt = _make_opt()

    state = _configure_medical_presets(args, opt)

    assert state.active is True
    assert state.densification_enabled is False
    assert state.scale_constraints_enabled is False
    assert opt.densify_from_iter > opt.iterations
    assert opt.diversity_scale_weight == 0.0
    assert opt.diversity_rotation_weight == 0.0
    assert args.init_n_points >= 8000


def test_medical_mode_vessel_uses_gentle_densification() -> None:
    """Vessel preset keeps densification but raises the thresholds and cadence."""

    args = _make_args(medical_mode="vessel")
    opt = _make_opt()

    state = _configure_medical_presets(args, opt)

    assert state.active is True
    assert state.densification_enabled is True
    assert opt.densification_interval >= 200
    assert opt.densify_grad_threshold >= 5e-4
    assert opt.densify_until_iter <= opt.densify_from_iter + 2000
    assert args.init_n_points >= 6000
