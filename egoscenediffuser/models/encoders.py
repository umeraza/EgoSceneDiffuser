from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from egoscenediffuser.utils.geometry import sample_scene_tokens


def _init_transformer(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[0]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max positional length")
        return x + self.encoding[: x.shape[1]].to(x.dtype).unsqueeze(0)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseTemporalEncoder(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        d_model = int(cfg.model.d_model)
        dropout = float(cfg.model.dropout)
        history_dim = int(cfg.data.num_joints) * int(cfg.data.motion_features)
        self.sparse_encoder = MLPEncoder(int(cfg.data.sparse_dim), d_model, dropout)
        self.history_encoder = MLPEncoder(history_dim, d_model, dropout)
        self.trajectory_encoder = MLPEncoder(int(cfg.data.trajectory_dim), d_model, dropout)
        self.modality_embeddings = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.mask_embeddings = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.position = SinusoidalPositionEncoding(d_model, int(cfg.model.max_sequence_length) * 4)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.model.n_heads),
            dim_feedforward=int(cfg.model.dim_feedforward),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=int(cfg.model.temporal_layers))
        _init_transformer(self.temporal_encoder)

    def forward(
        self,
        sparse: torch.Tensor,
        history: torch.Tensor,
        trajectory: torch.Tensor,
        modality_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, t, _, _ = history.shape
        sparse_tokens = self.sparse_encoder(sparse) + self.modality_embeddings[0]
        history_tokens = self.history_encoder(history.reshape(b, t, -1)) + self.modality_embeddings[1]
        trajectory_tokens = self.trajectory_encoder(trajectory) + self.modality_embeddings[2]
        if modality_masks is not None:
            names = ("sparse", "history", "trajectory")
            tokens = [sparse_tokens, history_tokens, trajectory_tokens]
            for index, (name, token) in enumerate(zip(names, tokens)):
                mask = modality_masks.get(name)
                if mask is not None:
                    replacement = self.mask_embeddings[index].view(1, 1, -1).expand_as(token)
                    tokens[index] = torch.where(mask[:, None, None], replacement, token)
            sparse_tokens, history_tokens, trajectory_tokens = tokens
        streams = [sparse_tokens, history_tokens, trajectory_tokens]
        memory = torch.cat([self.position(stream) for stream in streams], dim=1)
        memory = self.temporal_encoder(memory)
        return memory, {
            "sparse": sparse_tokens,
            "history": history_tokens,
            "trajectory": trajectory_tokens,
        }


class TinyVisualEncoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(96, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = images.shape
        features = self.backbone(images.reshape(b * t, c, h, w)).flatten(1)
        return self.projection(features).reshape(b, t, -1)


class VisualEncoder(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.use_timm = bool(cfg.model.use_timm)
        self.visual_stride = max(1, int(cfg.model.get("visual_stride", 1)))
        d_model = int(cfg.model.d_model)
        if not self.use_timm:
            self.backbone = TinyVisualEncoder(d_model)
            self.is_tiny = True
            return
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install the vision extra with `pip install -e .[vision]`") from exc
        backbone = timm.create_model(
            str(cfg.model.visual_backbone),
            pretrained=bool(cfg.model.visual_pretrained),
            num_classes=0,
            global_pool="avg",
        )
        if bool(cfg.model.freeze_visual_backbone):
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        output_dim = int(getattr(backbone, "num_features"))
        self.backbone = backbone
        self.projection = nn.Linear(output_dim, d_model)
        self.is_tiny = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images[:, :: self.visual_stride]
        if self.is_tiny:
            return self.backbone(images)
        b, t, c, h, w = images.shape
        features = self.backbone(images.reshape(b * t, c, h, w))
        return self.projection(features).reshape(b, t, -1)


class PointNetSceneEncoder(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.token_count = int(cfg.data.scene_tokens)
        d_model = int(cfg.model.d_model)
        self.point_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.global_mlp = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sampled = sample_scene_tokens(points, self.token_count)
        local = self.point_mlp(sampled)
        global_feature = local.max(dim=1, keepdim=True).values.expand_as(local)
        return self.global_mlp(torch.cat([local, global_feature], dim=-1)), sampled


class BodyContextEncoder(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        input_dim = int(cfg.data.num_joints) * int(cfg.data.motion_features) + int(cfg.data.sparse_dim) + 3
        self.encoder = MLPEncoder(input_dim, int(cfg.model.d_model), float(cfg.model.dropout))

    def forward(
        self, prior_sample: torch.Tensor, sparse: torch.Tensor, head_positions: torch.Tensor
    ) -> torch.Tensor:
        b, total, joints, features = prior_sample.shape
        obs = sparse.shape[1]
        sparse_full = sparse.new_zeros(b, total, sparse.shape[-1])
        sparse_full[:, :obs] = sparse
        head_full = head_positions.new_zeros(b, total, 3)
        head_full[:, : head_positions.shape[1]] = head_positions
        if obs < total:
            sparse_full[:, obs:] = sparse[:, -1:].expand(-1, total - obs, -1)
            head_full[:, obs:] = head_positions[:, -1:].expand(-1, total - obs, -1)
        flattened = prior_sample.reshape(b, total, joints * features)
        return self.encoder(torch.cat([flattened, sparse_full, head_full], dim=-1))
