from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from egoscenediffuser.utils.geometry import finite_difference


def prior_loss(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor, uncertainty_weight: float) -> dict[str, torch.Tensor]:
    reconstruction = F.mse_loss(mean, target)
    variance = sigma.square().clamp_min(1e-8)
    heteroscedastic = ((mean - target).square() / variance + torch.log(variance)).mean()
    return {
        "prior_reconstruction": reconstruction,
        "prior_uncertainty": heteroscedastic,
        "prior": reconstruction + uncertainty_weight * heteroscedastic,
    }


def motion_losses(prediction: torch.Tensor, target: torch.Tensor, observed_frames: int) -> dict[str, torch.Tensor]:
    task = F.mse_loss(prediction, target)
    observed = F.mse_loss(prediction[:, :observed_frames], target[:, :observed_frames])
    if observed_frames < target.shape[1]:
        future = F.mse_loss(prediction[:, observed_frames:], target[:, observed_frames:])
    else:
        future = prediction.new_zeros(())
    return {"task": task, "observed_consistency": observed, "future": future}


def contact_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def nearest_scene_distance(motion: torch.Tensor, scene_points: torch.Tensor) -> torch.Tensor:
    # Chunking avoids materializing [B,T,J,N,3] for the full default sequence.
    outputs = []
    for frame_chunk in motion.split(16, dim=1):
        distances = torch.cdist(frame_chunk.flatten(1, 2), scene_points)
        nearest = distances.amin(dim=-1).reshape(frame_chunk.shape[:-1])
        outputs.append(nearest)
    return torch.cat(outputs, dim=1)


def physical_losses(
    prediction: torch.Tensor,
    contact_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    cfg: Any,
) -> dict[str, torch.Tensor]:
    ground = float(cfg.loss.ground_height_m)
    margin = float(cfg.loss.collision_margin_m)
    contact_ids = [min(int(i), prediction.shape[2] - 1) for i in cfg.data.contact_joints]
    device_ids = [int(cfg.data.head_joint), *map(int, cfg.data.hand_joints)]
    device_ids = [min(i, prediction.shape[2] - 1) for i in device_ids[:3]]
    feet = prediction[:, :, contact_ids]
    probabilities = torch.sigmoid(contact_logits)

    velocity = torch.zeros_like(feet)
    velocity[:, 1:] = feet[:, 1:] - feet[:, :-1]
    foot_contact = (probabilities[..., None] * velocity.square()).mean()
    foot_height = (probabilities * (feet[..., 1] - ground).abs()).mean()
    ground_penetration = torch.relu(ground - prediction[..., 1]).mean()

    sparse_positions = batch.get("sparse_positions")
    if sparse_positions is None:
        head_hand = prediction.new_zeros(())
    else:
        obs = min(sparse_positions.shape[1], prediction.shape[1])
        predicted_devices = prediction[:, :obs, device_ids]
        head_hand = F.mse_loss(predicted_devices, sparse_positions[:, :obs, : len(device_ids)])

    if prediction.shape[1] >= 3:
        temporal = finite_difference(prediction, order=2, dim=1).square().mean()
    else:
        temporal = prediction.new_zeros(())

    scene = batch.get("scene_points")
    if scene is None:
        collision = prediction.new_zeros(())
    else:
        nearest = nearest_scene_distance(prediction, scene)
        collision = torch.relu(margin - nearest).square().mean()

    return {
        "collision": collision,
        "foot_contact": foot_contact,
        "foot_height": foot_height,
        "ground_penetration": ground_penetration,
        "head_hand_alignment": head_hand,
        "temporal_smoothness": temporal,
    }
