import numpy as np
from plyfile import PlyData, PlyElement
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "gs_viewer" / "src"))
from gs_viewer.ply_loader import load_gaussian_model_ply


def _write_test_ply(path, *, include_intensity: bool, include_hu: bool) -> None:
    names = [
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    if include_intensity:
        names.append("intensity_01")
    if include_hu:
        names.append("hu")

    dtype = [(n, "f4") for n in names]
    rows = np.zeros(2, dtype=dtype)

    rows["x"] = np.array([0.0, 1.0], dtype=np.float32)
    rows["y"] = np.array([0.0, 1.0], dtype=np.float32)
    rows["z"] = np.array([0.0, 1.0], dtype=np.float32)

    # Distinct SH-DC values to verify fallback path when explicit intensity is absent.
    rows["f_dc_0"] = np.array([0.0, 0.25], dtype=np.float32)
    rows["f_dc_1"] = np.array([0.0, 0.25], dtype=np.float32)
    rows["f_dc_2"] = np.array([0.0, 0.25], dtype=np.float32)

    rows["opacity"] = np.array([0.5, 0.6], dtype=np.float32)
    rows["scale_0"] = np.array([0.0, 0.0], dtype=np.float32)
    rows["scale_1"] = np.array([0.0, 0.0], dtype=np.float32)
    rows["scale_2"] = np.array([0.0, 0.0], dtype=np.float32)
    rows["rot_0"] = np.array([1.0, 1.0], dtype=np.float32)
    rows["rot_1"] = np.array([0.0, 0.0], dtype=np.float32)
    rows["rot_2"] = np.array([0.0, 0.0], dtype=np.float32)
    rows["rot_3"] = np.array([0.0, 0.0], dtype=np.float32)

    if include_intensity:
        rows["intensity_01"] = np.array([0.2, 0.8], dtype=np.float32)
    if include_hu:
        rows["hu"] = np.array([-500.0, 1500.0], dtype=np.float32)

    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))


def test_loader_prefers_explicit_intensity(tmp_path) -> None:
    ply_path = tmp_path / "with_intensity.ply"
    _write_test_ply(ply_path, include_intensity=True, include_hu=True)

    model = load_gaussian_model_ply(ply_path)
    np.testing.assert_allclose(
        model.intensity01,
        np.array([0.2, 0.8], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert model.hu is not None
    np.testing.assert_allclose(
        model.hu,
        np.array([-500.0, 1500.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_loader_fallback_to_shdc_when_missing_intensity(tmp_path) -> None:
    ply_path = tmp_path / "legacy_no_intensity.ply"
    _write_test_ply(ply_path, include_intensity=False, include_hu=False)

    model = load_gaussian_model_ply(ply_path)
    # Expect two distinct fallback intensities decoded from f_dc.
    assert model.intensity01.shape == (2,)
    assert float(model.intensity01[1]) > float(model.intensity01[0])
