import torch

from gaussian_splatting.utils.ambient_occlusion import (
    compute_ao_volume_from_mask,
    compute_isotropic_ao_from_mask,
)


def test_isotropic_ao_volume_shape_and_range() -> None:
    mask = torch.zeros((7, 7, 7), dtype=torch.bool)
    mask[1:6, 1:6, 1:6] = True

    ao = compute_isotropic_ao_from_mask(mask, radius_vox=2)

    assert ao.shape == mask.shape
    assert float(ao.min()) >= 0.0
    assert float(ao.max()) <= 1.0

    center = float(ao[3, 3, 3])
    near_surface = float(ao[1, 1, 1])
    assert center < near_surface


def test_normal_method_returns_surface_mask() -> None:
    mask_volume = torch.zeros((7, 7, 7), dtype=torch.float32)
    mask_volume[1:6, 1:6, 1:6] = 1.0
    mask_bool = mask_volume > 0.5

    res = compute_ao_volume_from_mask(
        mask_volume,
        mask_bool,
        radius_vox=2,
        method="normal",
    )

    assert res.ao_volume.shape == mask_volume.shape
    assert res.surface_mask is not None
    assert res.surface_mask.shape == mask_volume.shape
    assert int(res.surface_mask.sum().item()) > 0
