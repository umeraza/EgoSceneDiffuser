import torch

from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataloader
from egoscenediffuser.losses import CompositeLoss
from egoscenediffuser.models import EgoSceneDiffuser


def _batch():
    cfg = load_config("configs/smoke.yaml")
    return cfg, next(iter(build_dataloader(cfg, "train", batch_size=2)))


def test_training_forward_and_backward():
    cfg, batch = _batch()
    model = EgoSceneDiffuser(cfg).train()
    outputs = model(batch, sample=False)
    assert outputs["motion"].shape == batch["motion"].shape
    assert outputs["prior_sigma"].min() > 0
    loss = CompositeLoss(cfg)(outputs, batch, "stage3")["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_sampling_repaints_observed_prefix():
    cfg, batch = _batch()
    model = EgoSceneDiffuser(cfg).eval()
    with torch.no_grad():
        outputs = model(batch, sample=True)
    assert torch.allclose(outputs["denoised"][:, : cfg.data.observed_frames], batch["observed_motion"], atol=1e-6)


def test_ablation_switches_load():
    cfg = load_config("configs/ablations/deterministic_refiner.yaml", ["data.name=synthetic", "data.synthetic_samples=2",
        "data.observed_frames=8", "data.future_frames=4", "data.num_joints=12", "data.contact_joints=[7,8,10,11]",
        "data.head_joint=6", "data.hand_joints=[9,10]", "data.lower_body_joints=[1,2,3,4,7,8,10,11]",
        "data.scene_points=64", "data.scene_tokens=16", "data.image_size=32", "data.num_workers=0",
        "model.d_model=48", "model.n_heads=4", "model.temporal_layers=1", "model.denoiser_layers=2",
        "model.dim_feedforward=96", "model.max_sequence_length=32", "model.use_timm=false",
        "model.diffusion.train_steps=20", "model.diffusion.inference_steps=4"])
    model = EgoSceneDiffuser(cfg)
    assert model.deterministic_refiner
