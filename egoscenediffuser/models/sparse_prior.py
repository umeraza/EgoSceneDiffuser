from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .encoders import SinusoidalPositionEncoding, _init_transformer


class SparseMotionPrior(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.total_frames = int(cfg.data.observed_frames) + int(cfg.data.future_frames)
        self.joints = int(cfg.data.num_joints)
        self.features = int(cfg.data.motion_features)
        d_model = int(cfg.model.d_model)
        self.query_tokens = nn.Parameter(torch.randn(self.total_frames, d_model) * 0.02)
        self.position = SinusoidalPositionEncoding(d_model, int(cfg.model.max_sequence_length))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=int(cfg.model.n_heads),
            dim_feedforward=int(cfg.model.dim_feedforward),
            dropout=float(cfg.model.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=max(1, int(cfg.model.temporal_layers) // 2))
        self.mean_head = nn.Linear(d_model, self.joints * self.features)
        self.sigma_head = nn.Linear(d_model, self.joints * self.features)
        self.epsilon = float(cfg.model.uncertainty_epsilon)
        _init_transformer(self.decoder)

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = memory.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        hidden = self.decoder(self.position(queries), memory)
        shape = (b, self.total_frames, self.joints, self.features)
        mean = self.mean_head(hidden).reshape(shape)
        sigma = torch.nn.functional.softplus(self.sigma_head(hidden)).reshape(shape) + self.epsilon
        return mean, sigma, hidden

    @staticmethod
    def reparameterize(mean: torch.Tensor, sigma: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return mean
        return mean + sigma * torch.randn_like(sigma)
