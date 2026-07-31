from egoscenediffuser.config import load_config


def test_base_inheritance_and_override():
    cfg = load_config("configs/smoke.yaml", ["model.diffusion.inference_steps=3"])
    assert cfg.data.name == "synthetic"
    assert cfg.model.diffusion.inference_steps == 3
    assert cfg.training.learning_rate == 1.0e-3
