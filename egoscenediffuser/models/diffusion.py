from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .encoders import SinusoidalPositionEncoding, _init_transformer


def _cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-6, 0.999).float()


def _linear_beta_schedule(steps: int) -> torch.Tensor:
    scale = 1000.0 / steps
    return torch.linspace(scale * 1e-4, min(scale * 2e-2, 0.999), steps).clamp(max=0.999)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class MotionDenoiser(nn.Module):
    """Transformer denoiser with uncertainty FiLM conditioning."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.joints = int(cfg.data.num_joints)
        self.features = int(cfg.data.motion_features)
        flat_dim = self.joints * self.features
        d_model = int(cfg.model.d_model)
        dropout = float(cfg.model.dropout)
        self.noisy_projection = nn.Linear(flat_dim, d_model)
        self.prior_projection = nn.Linear(flat_dim, d_model)
        self.uncertainty_projection = nn.Linear(flat_dim, d_model * 2)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        self.position = SinusoidalPositionEncoding(d_model, int(cfg.model.max_sequence_length))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.model.n_heads),
            dim_feedforward=int(cfg.model.dim_feedforward),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(cfg.model.denoiser_layers))
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, flat_dim)
        _init_transformer(self.transformer)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        timesteps: torch.Tensor,
        prior_sample: torch.Tensor,
        uncertainty: torch.Tensor,
        context: torch.Tensor,
        use_uncertainty: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, j, f = noisy_motion.shape
        noisy = self.noisy_projection(noisy_motion.reshape(b, t, j * f))
        prior = self.prior_projection(prior_sample.reshape(b, t, j * f))
        time_token = self.timestep_mlp(timestep_embedding(timesteps, noisy.shape[-1]))[:, None, :]
        hidden = noisy + prior + context + time_token
        if use_uncertainty:
            scale, shift = self.uncertainty_projection(uncertainty.reshape(b, t, j * f)).chunk(2, dim=-1)
            hidden = hidden * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        hidden = self.transformer(self.position(hidden))
        hidden = self.output_norm(hidden)
        prediction = self.output(hidden).reshape(b, t, j, f)
        return prediction, hidden


class UncertaintyConditionedDiffusion(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.train_steps = int(cfg.model.diffusion.train_steps)
        self.inference_steps = int(cfg.model.diffusion.inference_steps)
        self.ddim_eta = float(cfg.model.diffusion.ddim_eta)
        self.repaint_observed = bool(cfg.model.diffusion.repaint_observed)
        self.repaint_mode = str(cfg.model.diffusion.get("repaint_mode", "full" if self.repaint_observed else "none"))
        lower = set(map(int, cfg.data.lower_body_joints))
        self.upper_repaint_joint_ids = [i for i in range(int(cfg.data.num_joints)) if i not in lower]
        self.noise_strength = float(cfg.model.uncertainty_noise_strength)
        schedule = str(cfg.model.diffusion.beta_schedule).lower()
        betas = _cosine_beta_schedule(self.train_steps) if schedule == "cosine" else _linear_beta_schedule(self.train_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.denoiser = MotionDenoiser(cfg)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.reshape(out.shape[0], *([1] * (ndim - 1)))

    @staticmethod
    def normalize_uncertainty(sigma: torch.Tensor) -> torch.Tensor:
        flat = sigma.flatten(1)
        lo = flat.amin(dim=1, keepdim=True)
        hi = flat.amax(dim=1, keepdim=True)
        norm = (flat - lo) / (hi - lo + 1e-6)
        return norm.reshape_as(sigma)

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        uncertainty: torch.Tensor,
        noise: torch.Tensor | None = None,
        use_uncertainty: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self._extract(self.alpha_bar, timesteps, clean.ndim)
        if use_uncertainty:
            modulation = torch.sqrt(1.0 + self.noise_strength * self.normalize_uncertainty(uncertainty))
        else:
            modulation = torch.ones_like(uncertainty)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * modulation * noise
        return noisy, noise

    def training_step(
        self,
        clean: torch.Tensor,
        prior_sample: torch.Tensor,
        uncertainty: torch.Tensor,
        context: torch.Tensor,
        use_uncertainty: bool = True,
    ) -> dict[str, torch.Tensor]:
        timesteps = torch.randint(0, self.train_steps, (clean.shape[0],), device=clean.device)
        noisy, noise = self.q_sample(clean, timesteps, uncertainty, use_uncertainty=use_uncertainty)
        pred_x0, hidden = self.denoiser(
            noisy, timesteps, prior_sample, uncertainty, context, use_uncertainty=use_uncertainty
        )
        return {"pred_x0": pred_x0, "hidden": hidden, "noisy": noisy, "noise": noise, "timesteps": timesteps}

    @torch.no_grad()
    def sample(
        self,
        prior_sample: torch.Tensor,
        uncertainty: torch.Tensor,
        context: torch.Tensor,
        observed_motion: torch.Tensor | None = None,
        use_uncertainty: bool = True,
        inference_steps: int | None = None,
        repaint_observed: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = int(inference_steps or self.inference_steps)
        repaint = self.repaint_observed if repaint_observed is None else repaint_observed
        mode = self.repaint_mode if repaint else "none"
        times = torch.linspace(self.train_steps - 1, 0, steps, device=prior_sample.device).long().unique_consecutive()
        initial_t = times[0].expand(prior_sample.shape[0])
        x, _ = self.q_sample(prior_sample, initial_t, uncertainty, use_uncertainty=use_uncertainty)
        last_hidden = context
        for index, time in enumerate(times):
            t = time.expand(prior_sample.shape[0])
            pred_x0, last_hidden = self.denoiser(x, t, prior_sample, uncertainty, context, use_uncertainty)
            if index == len(times) - 1:
                x = pred_x0
            else:
                next_time = times[index + 1]
                alpha_t = self.alpha_bar[time]
                alpha_next = self.alpha_bar[next_time]
                epsilon = (x - alpha_t.sqrt() * pred_x0) / (1.0 - alpha_t).sqrt().clamp_min(1e-6)
                sigma = self.ddim_eta * torch.sqrt(
                    ((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next)).clamp_min(0.0)
                )
                direction = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0.0)) * epsilon
                x = alpha_next.sqrt() * pred_x0 + direction
                if self.ddim_eta > 0:
                    x = x + sigma * torch.randn_like(x)
            if mode != "none" and observed_motion is not None:
                should_repaint = mode in {"full", "root_upper"} or (mode == "final" and index == len(times) - 1)
                if should_repaint:
                    obs = min(observed_motion.shape[1], x.shape[1])
                    if mode == "root_upper":
                        ids = self.upper_repaint_joint_ids
                        x[:, :obs, ids] = observed_motion[:, :obs, ids]
                    else:
                        x[:, :obs] = observed_motion[:, :obs]
        return x, last_hidden
