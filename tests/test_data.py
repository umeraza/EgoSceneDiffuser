from egoscenediffuser.config import load_config
from egoscenediffuser.data import build_dataset


def test_synthetic_shapes():
    cfg = load_config("configs/smoke.yaml")
    sample = build_dataset(cfg, "train")[0]
    assert sample["sparse"].shape == (8, 36)
    assert sample["history"].shape == (8, 12, 3)
    assert sample["motion"].shape == (12, 12, 3)
    assert sample["scene_points"].shape == (64, 3)
    assert sample["contacts"].shape == (12, 4)
