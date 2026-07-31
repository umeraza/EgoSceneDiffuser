# Explicit implementation assumptions

These defaults are engineering choices, not facts stated by the manuscript.

- Motion is represented as direct 3D joint positions by default: `[time, joint, xyz]`.
- Default skeleton has 22 joints; all indices are configurable.
- Sparse input dimension is 36: three devices x (3D position + 6D rotation + 3D velocity).
- The temporal width is 256, with 8 heads, 4 encoder layers, and 6 denoiser layers.
- Scene points are encoded with a PointNet-style MLP and reduced to 128 scene tokens.
- ViT-B/16 is loaded through `timm` when enabled; the smoke test uses a tiny CNN. The default freezes the backbone and samples every fourth RGB frame because the manuscript does not explain how its large sequence/batch combination fits in memory.
- Missing modality masking probabilities are 0.10/0.10/0.20/0.10/0.20 for sparse/history/visual/trajectory/scene.
- Diffusion uses a cosine beta schedule, 1000 training steps, 20 deterministic DDIM steps, and x0 prediction.
- Uncertainty scales the forward noise variance with strength 0.5 after per-sample normalization.
- Forecasting repainting uses the available observed full-body prefix. It is disabled in the EgoBody reconstruction config to avoid target-prefix leakage.
- Contact joints default to four foot/ankle indices and must be changed for the selected skeleton.
- Ground height defaults to y=0.
- The default collision term is a nearest-point clearance surrogate because the manuscript never defines the claimed COAP/SDF formulation. It is not equivalent to signed penetration depth and must be replaced for exact reproduction.
- The optimizer is AdamW with the manuscript's LR, weight decay, and clipping values. Betas use PyTorch defaults.
- No author-specific checkpoints, baseline code, or unpublished preprocessing are assumed.
