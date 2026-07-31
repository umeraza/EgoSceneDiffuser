from __future__ import annotations

import math

import torch

from egoscenediffuser.utils.geometry import geodesic_rotation_error, rotation_6d_to_matrix


def _mean_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1).mean()


def mpjpe_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1000.0 * _mean_distance(pred, target)


def subset_mpjpe_mm(pred: torch.Tensor, target: torch.Tensor, joint_ids: list[int]) -> torch.Tensor:
    ids = [min(max(int(i), 0), pred.shape[2] - 1) for i in joint_ids]
    return mpjpe_mm(pred[:, :, ids], target[:, :, ids])


def ade_mm(pred: torch.Tensor, target: torch.Tensor, observed_frames: int) -> torch.Tensor:
    if observed_frames >= pred.shape[1]:
        return pred.new_tensor(float("nan"))
    return mpjpe_mm(pred[:, observed_frames:], target[:, observed_frames:])


def fde_mm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1000.0 * _mean_distance(pred[:, -1], target[:, -1])


def observed_drift_mm(pred: torch.Tensor, target: torch.Tensor, observed_frames: int) -> torch.Tensor:
    return mpjpe_mm(pred[:, :observed_frames], target[:, :observed_frames])


def transition_error_mm(pred: torch.Tensor, target: torch.Tensor, observed_frames: int) -> torch.Tensor:
    if observed_frames <= 0 or observed_frames >= pred.shape[1]:
        return pred.new_tensor(float("nan"))
    pred_transition = pred[:, observed_frames] - pred[:, observed_frames - 1]
    true_transition = target[:, observed_frames] - target[:, observed_frames - 1]
    return 1000.0 * _mean_distance(pred_transition, true_transition)




def velocity_error_mm_frame(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    return 1000.0 * _mean_distance(torch.diff(pred, dim=1), torch.diff(target, dim=1))


def mpjre_deg_from_6d(pred_rotation6d: torch.Tensor, target_rotation6d: torch.Tensor) -> torch.Tensor:
    pred_matrix = rotation_6d_to_matrix(pred_rotation6d)
    target_matrix = rotation_6d_to_matrix(target_rotation6d)
    return geodesic_rotation_error(pred_matrix, target_matrix).mean() * (180.0 / math.pi)


def head_hand_alignment_mm(pred: torch.Tensor, sparse_positions: torch.Tensor, joint_ids: list[int]) -> torch.Tensor:
    obs = min(pred.shape[1], sparse_positions.shape[1])
    ids = [min(max(int(i), 0), pred.shape[2] - 1) for i in joint_ids[:3]]
    return 1000.0 * _mean_distance(pred[:, :obs, ids], sparse_positions[:, :obs, :len(ids)])


def acceleration_error_mm(pred: torch.Tensor, target: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    if pred.shape[1] < 3:
        return pred.new_zeros(())
    pred_acc = torch.diff(pred, n=2, dim=1)
    true_acc = torch.diff(target, n=2, dim=1)
    return 1000.0 * _mean_distance(pred_acc, true_acc)


def contact_scores(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = labels >= 0.5
    tp = (prediction & truth).sum().float()
    fp = (prediction & ~truth).sum().float()
    fn = (~prediction & truth).sum().float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (prediction == truth).float().mean()
    return {"contact_accuracy": accuracy, "contact_precision": precision, "contact_recall": recall, "contact_f1": f1}


def physical_metrics(
    pred: torch.Tensor,
    contacts: torch.Tensor,
    contact_joint_ids: list[int],
    ground_height: float,
    fps: float,
) -> dict[str, torch.Tensor]:
    ids = [min(max(int(i), 0), pred.shape[2] - 1) for i in contact_joint_ids]
    feet = pred[:, :, ids]
    ground_depth = torch.relu(ground_height - pred[..., 1])
    penetration_depth = 1000.0 * ground_depth.mean()
    penetration_rate = (ground_depth > 0).float().mean()
    velocity = torch.zeros_like(feet)
    velocity[:, 1:] = (feet[:, 1:] - feet[:, :-1]) * fps
    speed = torch.linalg.norm(velocity, dim=-1)
    skate = (speed * contacts).sum() / contacts.sum().clamp_min(1.0)
    return {
        "ground_penetration_mm": penetration_depth,
        "ground_penetration_rate": penetration_rate,
        "skate_cm_s": skate * 100.0,
    }


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).sum() / torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-12))


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x.flatten())
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(order.numel(), device=x.device, dtype=torch.float32)
    return ranks


def uncertainty_metrics(sigma: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    error = torch.linalg.norm(pred - target, dim=-1)
    scalar_sigma = sigma.square().mean(dim=-1).sqrt().clamp_min(1e-6)
    pearson = pearson_correlation(scalar_sigma, error)
    spearman = pearson_correlation(_rank(scalar_sigma), _rank(error))
    nll = (error.square() / (2 * scalar_sigma.square()) + 1.5 * torch.log(scalar_sigma.square())).mean()
    return {"uncertainty_pearson": pearson, "uncertainty_spearman": spearman, "uncertainty_nll": nll}
