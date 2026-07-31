from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .motion import (
    acceleration_error_mm, ade_mm, contact_scores, fde_mm, head_hand_alignment_mm, mpjpe_mm,
    mpjre_deg_from_6d, observed_drift_mm, physical_metrics, subset_mpjpe_mm, transition_error_mm,
    uncertainty_metrics, velocity_error_mm_frame,
)


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, values: dict[str, torch.Tensor | float]) -> None:
        for key, value in values.items():
            scalar = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            if scalar == scalar:
                self.sums[key] += scalar
                self.counts[key] += 1

    def compute(self) -> dict[str, float]:
        return {key: self.sums[key] / max(self.counts[key], 1) for key in sorted(self.sums)}


def evaluate_batch(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: Any) -> dict[str, torch.Tensor]:
    pred, target = outputs["motion"], batch["motion"]
    obs = int(cfg.data.observed_frames)
    metrics = {
        "mpjpe_mm": mpjpe_mm(pred[..., :3], target[..., :3]),
        "lower_mpjpe_mm": subset_mpjpe_mm(pred[..., :3], target[..., :3], list(cfg.data.lower_body_joints)),
        "upper_mpjpe_mm": subset_mpjpe_mm(pred[..., :3], target[..., :3], [i for i in range(pred.shape[2]) if i not in set(cfg.data.lower_body_joints)]),
        "contact_joint_mpjpe_mm": subset_mpjpe_mm(pred[..., :3], target[..., :3], list(cfg.data.contact_joints)),
        "root_translation_error_mm": mpjpe_mm(pred[:, :, :1, :3], target[:, :, :1, :3]),
        "ade_mm": ade_mm(pred[..., :3], target[..., :3], obs),
        "fde_mm": fde_mm(pred[..., :3], target[..., :3]),
        "observed_drift_mm": observed_drift_mm(pred[..., :3], target[..., :3], obs),
        "transition_error_mm": transition_error_mm(pred[..., :3], target[..., :3], obs),
        "velocity_error_mm_frame": velocity_error_mm_frame(pred[..., :3], target[..., :3]),
        "acceleration_error_mm_frame2": acceleration_error_mm(pred[..., :3], target[..., :3], float(cfg.data.fps)),
    }
    if "sparse_positions" in batch:
        metrics["head_hand_alignment_mm"] = head_hand_alignment_mm(
            pred[..., :3], batch["sparse_positions"], [cfg.data.head_joint, *cfg.data.hand_joints]
        )
    if pred.shape[-1] >= 9:
        metrics["mpjre_deg"] = mpjre_deg_from_6d(pred[..., 3:9], target[..., 3:9])
    metrics.update(contact_scores(outputs["contact_logits"], batch["contacts"]))
    metrics.update(
        physical_metrics(
            pred[..., :3], batch["contacts"], list(cfg.data.contact_joints), float(cfg.loss.ground_height_m), float(cfg.data.fps)
        )
    )
    metrics.update(uncertainty_metrics(outputs["prior_sigma"][..., :3], pred[..., :3], target[..., :3]))
    for horizon in cfg.evaluation.future_horizons:
        horizon = int(horizon)
        end = min(obs + horizon, pred.shape[1])
        if end > obs:
            metrics[f"mpjpe_h{horizon}_mm"] = mpjpe_mm(pred[:, obs:end, :, :3], target[:, obs:end, :, :3])
    return metrics
