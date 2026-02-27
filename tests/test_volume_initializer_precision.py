import torch

from scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.volume_initializer import (
    _setup_feature_tensors,
    _setup_model_parameters,
)


def test_initializer_promotes_trainable_params_to_fp32() -> None:
    """Initializer should keep trainable Gaussian params in float32."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaussianModel(sh_degree=1)

    n = 16
    points = torch.rand(n, 3, device=device, dtype=torch.float16)
    scales = torch.full((n, 3), 0.01, device=device, dtype=torch.float16)
    opacities = torch.full((n, 1), 0.95, device=device, dtype=torch.float16)
    rotations = torch.zeros(n, 4, device=device, dtype=torch.float16)
    rotations[:, 0] = 1.0

    _setup_model_parameters(
        model=model,
        points=points,
        scales=scales,
        opacities=opacities,
        opacity_values=None,
        initial_rotations=rotations,
    )

    intensities = torch.rand(n, 1, device=device, dtype=torch.float16)
    _setup_feature_tensors(
        model=model,
        intensities=intensities,
        volume_min=0.0,
        volume_max=1.0,
    )

    assert model._xyz.dtype == torch.float32
    assert model._scaling.dtype == torch.float32
    assert model._opacity.dtype == torch.float32
    assert model._rotation.dtype == torch.float32
    assert model._features_dc.dtype == torch.float32
    assert model._features_rest.dtype == torch.float32
