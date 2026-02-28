import numpy as np
import torch
from plyfile import PlyData

from scene.gaussian_model import GaussianModel
from scene.gaussian_model import SH_C0


def test_ply_export_multiplies_rgb_by_ao(tmp_path) -> None:
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

    # f_dc stores SH DC coefficients. Decode RGB and verify AO darkens in RGB space.
    f0 = verts["f_dc_0"].astype(np.float32)
    f1 = verts["f_dc_1"].astype(np.float32)
    f2 = verts["f_dc_2"].astype(np.float32)

    rgb0 = f0 * float(SH_C0) + 0.5
    rgb1 = f1 * float(SH_C0) + 0.5
    rgb2 = f2 * float(SH_C0) + 0.5

    np.testing.assert_allclose(rgb0, rgb1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(rgb0, rgb2, rtol=1e-5, atol=1e-6)

    # Base intensity is 1.0 -> decoded rgb is 1.0 before AO.
    # With ao_strength=1, exported rgb should be multiplied by AO.
    np.testing.assert_allclose(rgb0, ao.view(-1).numpy(), rtol=1e-5, atol=1e-6)


def test_ply_export_applies_voxel_size_to_xyz_and_scale(tmp_path) -> None:
    gm = GaussianModel(sh_degree=0)
    gm._xyz = torch.tensor(
        [
            [0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5],
        ],
        dtype=torch.float32,
    )
    gm._scaling = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    gm._rotation = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    gm._opacity = torch.zeros((2, 1), dtype=torch.float32)
    gm.intensities = torch.full((2, 1), 0.5, dtype=torch.float32)
    gm.voxel_size = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float32)

    out_path = tmp_path / "voxel_scaled_export.ply"
    gm.save_ply(str(out_path))

    ply = PlyData.read(str(out_path))
    verts = ply["vertex"].data

    xyz = np.stack(
        [
            verts["x"].astype(np.float32),
            verts["y"].astype(np.float32),
            verts["z"].astype(np.float32),
        ],
        axis=1,
    )
    expected_xyz = gm._xyz.detach().cpu().numpy() / gm.voxel_size.cpu().numpy().reshape(1, 3)
    np.testing.assert_allclose(xyz, expected_xyz, rtol=1e-5, atol=1e-6)

    exported_log_scale = np.stack(
        [
            verts["scale_0"].astype(np.float32),
            verts["scale_1"].astype(np.float32),
            verts["scale_2"].astype(np.float32),
        ],
        axis=1,
    )
    expected_log_scale = gm._scaling.detach().cpu().numpy() - np.log(
        gm.voxel_size.cpu().numpy().reshape(1, 3)
    )
    np.testing.assert_allclose(exported_log_scale, expected_log_scale, rtol=1e-5, atol=1e-6)
