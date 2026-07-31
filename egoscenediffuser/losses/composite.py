from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .objectives import contact_loss, motion_losses, physical_losses, prior_loss


class CompositeLoss:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        stage: str = "stage3",
    ) -> dict[str, torch.Tensor]:
        losses = prior_loss(
            outputs["prior_mean"], outputs["prior_sigma"], batch["motion"], float(self.cfg.loss.uncertainty)
        )
        total = float(self.cfg.loss.prior) * losses["prior"]
        if stage == "stage1" or "motion" not in outputs:
            losses["total"] = total
            return losses

        if "pred_x0" in outputs:
            losses["diffusion"] = F.mse_loss(outputs["pred_x0"], batch["motion"])
        else:
            losses["diffusion"] = outputs["motion"].new_zeros(())
        task = motion_losses(outputs["motion"], batch["motion"], int(self.cfg.data.observed_frames))
        losses.update(task)
        losses["contact"] = contact_loss(outputs["contact_logits"], batch["contacts"])
        physical = physical_losses(outputs["motion"], outputs["contact_logits"], batch, self.cfg)
        losses.update(physical)

        total = total + float(self.cfg.loss.diffusion) * losses["diffusion"]
        total = total + float(self.cfg.loss.task) * losses["task"]
        total = total + float(self.cfg.loss.future) * losses["future"]
        total = total + float(self.cfg.loss.observed_consistency) * losses["observed_consistency"]
        if stage != "stage2":
            total = total + float(self.cfg.loss.contact) * losses["contact"]
            for name in (
                "collision", "foot_contact", "foot_height", "ground_penetration",
                "head_hand_alignment", "temporal_smoothness",
            ):
                total = total + float(self.cfg.loss[name]) * losses[name]
        losses["total"] = total
        return losses
