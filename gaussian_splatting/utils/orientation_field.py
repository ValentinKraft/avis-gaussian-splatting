"""Utility functions for orientation initialization from volumetric structure."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_DEFAULT_SIGMA_EPS = 1e-6


def _normalize_index(idx: Tensor, size: int) -> Tensor:
    """Normalize voxel indices to [-1, 1] for grid_sample with align_corners=True."""
    if size <= 1:
        return torch.zeros_like(idx)
    denom = float(size - 1)
    return (idx / denom) * 2.0 - 1.0


def default_origin_and_spacing(
    volume_shape: Tuple[int, int, int],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Return origin and voxel spacing vectors for a normalized [0, 1]^3 volume."""
    dims_dhw = torch.tensor(volume_shape, device=device, dtype=torch.float32)
    dims_xyz = dims_dhw[[2, 1, 0]].clamp_min(1.0)
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    voxel = 1.0 / (dims_xyz - 1.0).clamp_min(1.0)
    return origin, voxel


def _gauss1d_kernel(sigma: float, device: torch.device) -> Tensor:
    """Return a 1D Gaussian kernel normalised to sum 1."""
    if sigma <= _DEFAULT_SIGMA_EPS:
        return torch.tensor([1.0], device=device, dtype=torch.float32)

    radius = max(1, int(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _separable_gaussian_blur3d(volume: Tensor, sigma: float) -> Tensor:
    """Apply separable 3D Gaussian smoothing to a [1, 1, D, H, W] tensor."""
    if sigma <= _DEFAULT_SIGMA_EPS:
        return volume

    kernel = _gauss1d_kernel(sigma, volume.device).to(volume.dtype)
    kx = kernel.view(1, 1, -1, 1, 1)
    ky = kernel.view(1, 1, 1, -1, 1)
    kz = kernel.view(1, 1, 1, 1, -1)

    pad = kernel.numel() // 2
    volume = F.conv3d(volume, kx, padding=(pad, 0, 0))
    volume = F.conv3d(volume, ky, padding=(0, pad, 0))
    volume = F.conv3d(volume, kz, padding=(0, 0, pad))
    return volume


def compute_structure_field(
    volume: Tensor,
    sigma_grad: float = 1.5,
    sigma_tensor: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute per-voxel structure tensor eigenvectors and eigenvalues."""
    if volume.dim() != 3:
        raise ValueError("Volume tensor must have shape [D, H, W].")

    device = volume.device
    dtype = volume.dtype

    with torch.no_grad():
        data = volume.unsqueeze(0).unsqueeze(0)
        data = _separable_gaussian_blur3d(data, sigma_grad)

        kernel_dx = torch.tensor([[[[[-1.0, 0.0, 1.0]]]]], device=device, dtype=dtype)
        kernel_dy = torch.tensor([[[[[-1.0], [0.0], [1.0]]]]], device=device, dtype=dtype)
        kernel_dz = torch.tensor([[[[[-1.0]], [[0.0]], [[1.0]]]]], device=device, dtype=dtype)
        kernel_dx *= 0.5
        kernel_dy *= 0.5
        kernel_dz *= 0.5

        gx = F.conv3d(data, kernel_dx, padding=(0, 0, 1))
        gy = F.conv3d(data, kernel_dy, padding=(0, 1, 0))
        gz = F.conv3d(data, kernel_dz, padding=(1, 0, 0))

        gx = gx.squeeze(0).squeeze(0)
        gy = gy.squeeze(0).squeeze(0)
        gz = gz.squeeze(0).squeeze(0)

        j_xx = gx * gx
        j_xy = gx * gy
        j_xz = gx * gz
        j_yy = gy * gy
        j_yz = gy * gz
        j_zz = gz * gz

        def _blur(component: Tensor) -> Tensor:
            comp = component.unsqueeze(0).unsqueeze(0)
            comp = _separable_gaussian_blur3d(comp, sigma_tensor)
            return comp.squeeze(0).squeeze(0)

        j_xx = _blur(j_xx)
        j_xy = _blur(j_xy)
        j_xz = _blur(j_xz)
        j_yy = _blur(j_yy)
        j_yz = _blur(j_yz)
        j_zz = _blur(j_zz)

        J = torch.stack(
            [
                torch.stack([j_xx, j_xy, j_xz], dim=-1),
                torch.stack([j_xy, j_yy, j_yz], dim=-1),
                torch.stack([j_xz, j_yz, j_zz], dim=-1),
            ],
            dim=-2,
        )

        eigvals, eigvecs = torch.linalg.eigh(J)
        return eigvecs.contiguous(), eigvals.contiguous()


def world_to_voxel(
    xyz_world: Tensor,
    origin_xyz: Tensor,
    voxel_size_xyz: Tensor,
) -> Tensor:
    """Convert world coordinates (x, y, z) to voxel indices (z, y, x)."""
    if xyz_world.dim() != 2 or xyz_world.shape[1] != 3:
        raise ValueError("Expected xyz_world with shape [N, 3].")

    rel = xyz_world - origin_xyz.unsqueeze(0)
    rel = rel / voxel_size_xyz.unsqueeze(0)
    iz = rel[:, 2]
    iy = rel[:, 1]
    ix = rel[:, 0]
    return torch.stack([iz, iy, ix], dim=-1)


def world_to_grid(
    xyz_world: Tensor,
    origin_xyz: Tensor,
    voxel_size_xyz: Tensor,
    volume_shape: Tuple[int, int, int],
) -> Tensor:
    """Convert world coordinates to grid_sample coordinates in [-1, 1]."""
    ijk = world_to_voxel(xyz_world, origin_xyz, voxel_size_xyz)
    D, H, W = volume_shape
    norm_z = _normalize_index(ijk[:, 0], D)
    norm_y = _normalize_index(ijk[:, 1], H)
    norm_x = _normalize_index(ijk[:, 2], W)
    return torch.stack([norm_z, norm_y, norm_x], dim=-1)


def gather_rotation_from_field(
    eigvecs: Tensor,
    eigvals: Tensor,
    ijk: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Sample rotation matrices and fallback mask from structure field."""
    if ijk.numel() == 0:
        return torch.empty(0, 3, 3, device=eigvecs.device), torch.empty(0, dtype=torch.bool, device=eigvecs.device)

    D, H, W = eigvecs.shape[:3]
    idx = ijk.round().long()
    idx[:, 0].clamp_(0, D - 1)
    idx[:, 1].clamp_(0, H - 1)
    idx[:, 2].clamp_(0, W - 1)

    rot = eigvecs[idx[:, 0], idx[:, 1], idx[:, 2]]
    vals = eigvals[idx[:, 0], idx[:, 1], idx[:, 2]]

    fallback = vals.sum(dim=-1) < eps
    if fallback.any():
        print(f"Warning: Rotation fallback for {fallback.sum().item()} points.")
        rot[fallback] = torch.eye(3, device=rot.device, dtype=rot.dtype)

    q, _ = torch.linalg.qr(rot)
    det = torch.det(q)
    neg = det < 0
    if neg.any():
        q[neg, :, 0] = -q[neg, :, 0]
    return q.contiguous(), fallback


def rotmat_to_quat(rot_mats: Tensor) -> Tensor:
    """Convert rotation matrices [N, 3, 3] to unit quaternions [N, 4]."""
    if rot_mats.numel() == 0:
        return torch.empty(0, 4, device=rot_mats.device)

    trace = rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2]
    quats = torch.empty(rot_mats.shape[0], 4, device=rot_mats.device)

    positive = trace > 0.0
    if positive.any():
        t = torch.sqrt(trace[positive] + 1.0) * 2.0
        quats[positive, 0] = 0.25 * t
        quats[positive, 1] = (rot_mats[positive, 2, 1] - rot_mats[positive, 1, 2]) / t
        quats[positive, 2] = (rot_mats[positive, 0, 2] - rot_mats[positive, 2, 0]) / t
        quats[positive, 3] = (rot_mats[positive, 1, 0] - rot_mats[positive, 0, 1]) / t

    remaining = ~positive
    if remaining.any():
        r = rot_mats[remaining]
        diag = torch.stack([r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]], dim=1)
        max_idx = diag.argmax(dim=1)
        q = torch.empty_like(quats[remaining])
        for axis in range(3):
            mask = max_idx == axis
            if not mask.any():
                continue
            i = axis
            j = (axis + 1) % 3
            k = (axis + 2) % 3
            diag_term = r[mask, i, i] - r[mask, j, j] - r[mask, k, k] + 1.0
            s = torch.sqrt(diag_term.clamp_min(1e-8)) * 2.0
            q[mask, i + 1] = 0.25 * s
            q[mask, 0] = (r[mask, k, j] - r[mask, j, k]) / s
            q[mask, j + 1] = (r[mask, j, i] + r[mask, i, j]) / s
            q[mask, k + 1] = (r[mask, k, i] + r[mask, i, k]) / s
        quats[remaining] = q

    quats = quats / (quats.norm(dim=1, keepdim=True) + 1e-8)
    return quats


def random_quat_perturb(quats: Tensor, deg: float = 2.0) -> Tensor:
    """Apply a small random axis-angle perturbation to quaternions."""
    if quats.numel() == 0 or deg <= 0.0:
        return quats

    device = quats.device
    dtype = quats.dtype
    angle = (torch.rand(quats.shape[0], 1, device=device, dtype=dtype) - 0.5) * 2.0
    angle *= torch.deg2rad(torch.tensor(deg, device=device, dtype=dtype))

    axis = torch.randn(quats.shape[0], 3, device=device, dtype=dtype)
    axis /= axis.norm(dim=1, keepdim=True).clamp_min(1e-8)

    half = angle * 0.5
    sin_half = torch.sin(half)
    dq = torch.cat([torch.cos(half), axis * sin_half], dim=1)

    w1, x1, y1, z1 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    w2, x2, y2, z2 = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )
    out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
    return out
