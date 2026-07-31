from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ContactAwareDecoder(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.joints = int(cfg.data.num_joints)
        self.features = int(cfg.data.motion_features)
        self.contact_count = len(cfg.data.contact_joints)
        d_model = int(cfg.model.d_model)
        flat_dim = self.joints * self.features
        self.motion_projection = nn.Linear(flat_dim, d_model)
        self.contact_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, self.contact_count)
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model * 2 + self.contact_count, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, flat_dim),
        )

    def forward(
        self, denoised: torch.Tensor, denoiser_hidden: torch.Tensor, scene_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, j, f = denoised.shape
        motion_token = self.motion_projection(denoised.reshape(b, t, j * f))
        contact_logits = self.contact_head(torch.cat([denoiser_hidden, scene_context], dim=-1))
        residual = self.output_head(torch.cat([motion_token, scene_context, torch.sigmoid(contact_logits)], dim=-1))
        final = denoised + residual.reshape(b, t, j, f)
        return final, contact_logits
