import numpy as np
import torch
from plyfile import PlyData

from scene.gaussian_model import GaussianModel


def test_ply_export_multiplies_fdc_by_ao(tmp_path) -> None:
    gm = GaussianModel(sh_degree=0)
    gm._xyz = torch.tensor(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
        dtype=torch.float32,
    )
    gm.intensities = torch.ones((2, 1), dtype=torch.float32)

    ao = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    out_path = tmp_path / "ao_test.ply"
    gm.save_ply(str(out_path), ao=ao, ao_strength=1.0)

    ply = PlyData.read(str(out_path))
    verts = ply["vertex"].data

    # Export should include an 'ao' attribute when AO is provided.
    assert "ao" in verts.dtype.names
    np.testing.assert_allclose(verts["ao"].astype(np.float32), ao.view(-1).numpy(), rtol=1e-5, atol=1e-6)

    # With identical base intensity, f_dc should scale in proportion to AO.
    f0 = verts["f_dc_0"].astype(np.float32)
    f1 = verts["f_dc_1"].astype(np.float32)
    f2 = verts["f_dc_2"].astype(np.float32)

    np.testing.assert_allclose(f0, f1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(f0, f2, rtol=1e-5, atol=1e-6)

    ratio = float(f0[0] / f0[1])
    assert abs(ratio - 2.0) < 1e-3
