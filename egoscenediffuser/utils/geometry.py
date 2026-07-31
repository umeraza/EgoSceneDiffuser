from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert Zhou et al. 6D rotations to 3x3 matrices."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_rotation_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = pred.transpose(-1, -2) @ target
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def finite_difference(x: torch.Tensor, order: int = 1, dim: int = 1) -> torch.Tensor:
    for _ in range(order):
        x = torch.diff(x, dim=dim)
    return x


def sample_scene_tokens(points: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic uniform subsampling over the point index."""
    n = points.shape[1]
    if n == count:
        return points
    if n < count:
        repeats = (count + n - 1) // n
        return points.repeat(1, repeats, 1)[:, :count]
    idx = torch.linspace(0, n - 1, count, device=points.device).long()
    return points.index_select(1, idx)
