from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .contact_decoder import ContactAwareDecoder
from .diffusion import UncertaintyConditionedDiffusion
from .encoders import BodyContextEncoder, PointNetSceneEncoder, SparseTemporalEncoder, VisualEncoder
from .scene_fusion import SpatiallyBiasedCrossAttention
from .sparse_prior import SparseMotionPrior


def _get_ablation(cfg: Any, key: str, default: Any) -> Any:
    ablation = cfg.get("ablation", {}) if isinstance(cfg, dict) else getattr(cfg, "ablation", {})
    return ablation.get(key, default) if isinstance(ablation, dict) else getattr(ablation, key, default)


class EgoSceneDiffuser(nn.Module):
    """Runnable reference implementation reconstructed from the manuscript."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs = int(cfg.data.observed_frames)
        self.total = self.obs + int(cfg.data.future_frames)
        d_model = int(cfg.model.d_model)
        self.sparse_temporal = SparseTemporalEncoder(cfg)
        self.prior = SparseMotionPrior(cfg)
        self.visual = VisualEncoder(cfg)
        self.scene = PointNetSceneEncoder(cfg)
        self.body_context = BodyContextEncoder(cfg)
        self.fusion = SpatiallyBiasedCrossAttention(cfg)
        self.diffusion = UncertaintyConditionedDiffusion(cfg)
        self.contact_decoder = ContactAwareDecoder(cfg)
        self.visual_mask = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scene_mask = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.drop_modalities = set(map(str, _get_ablation(cfg, "drop_modalities", [])))
        self.deterministic_prior = bool(_get_ablation(cfg, "deterministic_prior", False))
        self.disable_reparameterization = bool(_get_ablation(cfg, "disable_reparameterization", False))
        self.disable_uncertainty_conditioning = bool(_get_ablation(cfg, "disable_uncertainty_conditioning", False))
        self.deterministic_refiner = bool(_get_ablation(cfg, "deterministic_refiner", False))
        self.disable_contact_decoder = bool(_get_ablation(cfg, "disable_contact_decoder", False))

    def _sample_mask(self, batch: int, probability: float, device: torch.device, forced: bool) -> torch.Tensor:
        if forced:
            return torch.ones(batch, dtype=torch.bool, device=device)
        if not self.training or probability <= 0:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        return torch.rand(batch, device=device) < probability

    @staticmethod
    def _replace_tokens(tokens: torch.Tensor, mask: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[:, None, None], replacement.expand(tokens.shape[0], tokens.shape[1], -1), tokens)

    def forward_prior(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sparse = batch["sparse"]
        history = batch["history"]
        trajectory = batch["trajectory"]
        b = sparse.shape[0]
        probs = self.cfg.model.mask_probabilities
        masks = {
            name: self._sample_mask(b, float(probs[name]), sparse.device, name in self.drop_modalities)
            for name in ("sparse", "history", "trajectory")
        }
        memory, _ = self.sparse_temporal(sparse, history, trajectory, modality_masks=masks)
        mean, sigma, prior_hidden = self.prior(memory)
        if self.deterministic_prior:
            sigma = torch.zeros_like(sigma) + float(self.cfg.model.uncertainty_epsilon)
        prior_sample = self.prior.reparameterize(
            mean, sigma, enabled=not self.disable_reparameterization and not self.deterministic_prior
        )
        return {
            "prior_mean": mean, "prior_sigma": sigma, "prior_sample": prior_sample, "prior_hidden": prior_hidden
        }

    def forward(self, batch: dict[str, torch.Tensor], sample: bool | None = None) -> dict[str, torch.Tensor]:
        sparse = batch["sparse"]
        history = batch["history"]
        trajectory = batch["trajectory"]
        images = batch["images"]
        scene_points = batch["scene_points"]
        b = sparse.shape[0]
        probs = self.cfg.model.mask_probabilities
        masks = {
            name: self._sample_mask(b, float(probs[name]), sparse.device, name in self.drop_modalities)
            for name in ("sparse", "history", "trajectory", "visual", "scene")
        }

        memory, modality_tokens = self.sparse_temporal(
            sparse, history, trajectory, modality_masks={k: masks[k] for k in ("sparse", "history", "trajectory")}
        )
        mean, sigma, prior_hidden = self.prior(memory)
        if self.deterministic_prior:
            sigma = torch.zeros_like(sigma) + float(self.cfg.model.uncertainty_epsilon)
        reparameterize = not self.disable_reparameterization and not self.deterministic_prior
        prior_sample = self.prior.reparameterize(mean, sigma, enabled=reparameterize)

        visual_tokens = self.visual(images)
        visual_tokens = self._replace_tokens(visual_tokens, masks["visual"], self.visual_mask)
        scene_tokens, scene_xyz = self.scene(scene_points)
        scene_tokens = self._replace_tokens(scene_tokens, masks["scene"], self.scene_mask)
        if "scene" in self.drop_modalities:
            scene_xyz = torch.zeros_like(scene_xyz)

        head_positions = batch.get("sparse_positions", None)
        head_positions = head_positions[:, :, 0] if head_positions is not None else trajectory[:, :, :3]
        body_tokens = self.body_context(prior_sample, sparse, head_positions)
        trajectory_tokens = modality_tokens["trajectory"]
        fused, attention = self.fusion(
            body_tokens, scene_tokens, scene_xyz, visual_tokens, trajectory_tokens, head_positions
        )
        use_uncertainty = not self.disable_uncertainty_conditioning

        if sample is None:
            sample = not self.training
        if self.deterministic_refiner:
            denoised, denoiser_hidden = prior_sample, fused
            diffusion_data = {}
        elif sample:
            denoised, denoiser_hidden = self.diffusion.sample(
                prior_sample,
                sigma,
                fused,
                observed_motion=batch.get("observed_motion"),
                use_uncertainty=use_uncertainty,
            )
            diffusion_data = {}
        else:
            diffusion_data = self.diffusion.training_step(
                batch["motion"], prior_sample, sigma, fused, use_uncertainty=use_uncertainty
            )
            denoised = diffusion_data["pred_x0"]
            denoiser_hidden = diffusion_data["hidden"]

        if self.disable_contact_decoder:
            final_motion = denoised
            contact_logits = denoised.new_zeros(denoised.shape[0], denoised.shape[1], len(self.cfg.data.contact_joints))
        else:
            final_motion, contact_logits = self.contact_decoder(denoised, denoiser_hidden, fused)
        if sample and self.diffusion.repaint_observed and batch.get("observed_motion") is not None:
            observed = batch["observed_motion"]
            obs = min(observed.shape[1], final_motion.shape[1])
            if self.diffusion.repaint_mode == "root_upper":
                ids = self.diffusion.upper_repaint_joint_ids
                final_motion[:, :obs, ids] = observed[:, :obs, ids]
            elif self.diffusion.repaint_mode in {"full", "final"}:
                final_motion[:, :obs] = observed[:, :obs]
        return {
            "motion": final_motion,
            "denoised": denoised,
            "contact_logits": contact_logits,
            "prior_mean": mean,
            "prior_sigma": sigma,
            "prior_sample": prior_sample,
            "prior_hidden": prior_hidden,
            "scene_context": fused,
            "attention": attention,
            "modality_masks": torch.stack(list(masks.values()), dim=-1),
            **diffusion_data,
        }
