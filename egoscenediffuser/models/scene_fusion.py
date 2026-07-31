from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class SpatiallyBiasedCrossAttention(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.d_model = int(cfg.model.d_model)
        self.n_heads = int(cfg.model.n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.head_dim = self.d_model // self.n_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.bias_mlp = nn.Sequential(nn.Linear(4, self.d_model // 2), nn.GELU(), nn.Linear(self.d_model // 2, self.n_heads))
        self.dropout = nn.Dropout(float(cfg.model.dropout))
        self.fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        body_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        trajectory_tokens: torch.Tensor,
        head_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat([scene_tokens, visual_tokens, trajectory_tokens], dim=1)
        q = self._reshape_heads(self.q_proj(body_tokens))
        k = self._reshape_heads(self.k_proj(context))
        v = self._reshape_heads(self.v_proj(context))
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        total = body_tokens.shape[1]
        if head_positions.shape[1] < total:
            pad = head_positions[:, -1:].expand(-1, total - head_positions.shape[1], -1)
            head_positions = torch.cat([head_positions, pad], dim=1)
        head_positions = head_positions[:, :total]
        relative = scene_xyz[:, None, :, :] - head_positions[:, :, None, :]
        distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
        spatial_features = torch.cat([relative, distance], dim=-1)
        spatial_bias = self.bias_mlp(spatial_features).permute(0, 3, 1, 2)
        logits[..., : scene_tokens.shape[1]] += spatial_bias

        attention = self.dropout(torch.softmax(logits, dim=-1))
        attended = torch.matmul(attention, v).transpose(1, 2).contiguous().view(
            body_tokens.shape[0], body_tokens.shape[1], self.d_model
        )
        attended = self.out_proj(attended)
        fused = self.fuse(torch.cat([attended, body_tokens], dim=-1))
        return fused, attention
