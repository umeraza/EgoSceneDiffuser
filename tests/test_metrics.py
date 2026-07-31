import torch

from egoscenediffuser.metrics.motion import ade_mm, contact_scores, fde_mm, mpjpe_mm, uncertainty_metrics


def test_perfect_motion_metrics():
    motion = torch.zeros(2, 5, 3, 3)
    assert mpjpe_mm(motion, motion).item() == 0.0
    assert ade_mm(motion, motion, 2).item() == 0.0
    assert fde_mm(motion, motion).item() == 0.0


def test_contact_f1_perfect():
    labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    logits = torch.where(labels > 0, torch.tensor(10.0), torch.tensor(-10.0))
    metrics = contact_scores(logits, labels)
    assert metrics["contact_f1"].item() > 0.999


def test_uncertainty_metrics_finite():
    target = torch.zeros(1, 4, 2, 3)
    pred = torch.ones_like(target) * 0.1
    sigma = torch.linspace(0.1, 1.0, pred.numel()).reshape_as(pred)
    assert all(torch.isfinite(value) for value in uncertainty_metrics(sigma, pred, target).values())
