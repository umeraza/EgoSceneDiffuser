# Reproducibility audit of the manuscript

## Verdict

The proposed pipeline is implementable, but the paper is not currently an exact experimental specification. A repository can reproduce the *idea*; it cannot honestly reproduce the claimed tables without author-only information.

## Framework issues

1. **Motion representation is undefined in implementable terms.** The paper writes a tuple of canonical transforms but never states the final tensor dimensions, rotation representation, joint order, root variables, SMPL-X parameter subset, or conversion to evaluation joints.
2. **Sparse input layout is incomplete.** Positions, rotations, and velocities are named, but rotation encoding, device order, velocity calculation, missing-hand handling, and synchronization are unspecified.
3. **Two prior-loss definitions disagree.** One divides squared error by `sigma^2`; the later compact form divides the residual by `sigma` before squaring and replaces the summed log term with an L1 norm. These are related but not identical normalizations.
4. **Diffusion notation is internally inconsistent.** The forward process uses step `k`, while the training equation calls the noisy sample `Y^t` and passes `t`. The covariance expression does not state whether uncertainty is a diagonal covariance, variance multiplier, or standard-deviation multiplier.
5. **Spatial bias is underspecified.** `Delta(head trajectory, point cloud)` is not defined, and no head-to-scene token correspondence or bias shape is given.
6. **Feature-wise affine uncertainty conditioning is mentioned but not parameterized.** The injection layers, normalization, per-joint mapping, and whether uncertainty is detached are absent.
7. **Contact and collision mechanisms are not mathematically defined.** The ablation names a COAP collision loss that does not appear in the method equations. Binary contact labels, contact joints, thresholds, and signed-distance computation are missing.
8. **Forecasting repainting lacks a precise algorithm.** The paper does not state which clean observed signal is reinserted, whether it is re-noised at each step, or how repainting behaves when historical full-body motion is unavailable.

## Dataset and protocol issues

1. **EgoBody split conflict.** The official release provides sequence splits, but the paper claims a subject-disjoint 70/15/15 split without listing subjects or publishing a split file.
2. **Target identity is ambiguous on EgoBody.** EgoBody contains camera-wearer and interactee annotations. The manuscript needs to state which body is predicted and how HoloLens head/hand streams map to that target.
3. **GIMO has no native three-device VR stream.** The paper says proxy head/wrist inputs are generated but gives no joint indices, coordinate transforms, noise model, controller orientation construction, or train/test policy.
4. **Window generation is absent.** Stride, padding, scene boundary handling, missing-frame policy, and the relationship between 60 observed and 90 forecast frames are not stated.
5. **RGB sampling is absent.** The manuscript says 224x224 ViT-B/16 but not frame rate, crop, normalization, augmentation, backbone checkpoint, or temporal pooling.
6. **Scene processing is incomplete.** A 2.5 m crop and 2048 points are stated, but crop center, mesh sampling, normals/colors, coordinate frame, and sampling method are not.
7. **Pseudo-contact thresholds are missing.** This makes contact precision/recall/F1 and skating metrics non-reproducible.

## Training issues

Missing values include model width, attention heads, encoder/decoder layers, dropout, positional encoding, initialization, LR schedule, warmup, optimizer betas, EMA, mixed precision, random seeds, number of GPUs, gradient accumulation, stage-freezing policy, masking probabilities, every loss weight, uncertainty-noise strength, beta schedule, and DDIM eta.

## Results and ablation issues

1. Main-table MPJPE is **43.6 mm**, while the 20-step diffusion ablation uses **42.9 mm** as the full-model reference.
2. The text says the 50-step model gives the lowest errors, but it also states MPJPE rises from 42.9 to 43.2 mm.
3. The uncertainty-prior ablation has a variant with lower lower-body MPJPE (54.7) than the claimed full model (55.2), so the full model is not best on every listed metric.
4. Ablations report a single aggregate run with no variance, despite stochastic sampling and training.
5. The uncertainty-prior table caption is duplicated and corrupted.
6. The qualitative figure is captioned as an architecture diagram.
7. Multiple ablation figures reuse or mismatch filenames/captions (`A5_T7`, multimodal caption for repainting).
8. Baseline modality support and numerical values require public evaluation scripts/checkpoints. The manuscript does not establish that every method was retrained under a common protocol or measured on identical hardware.
9. Runtime fairness is not established because methods use different modalities and potentially different implementations.

## Repository policy

This code resolves unspecified choices through configuration defaults, not hidden guesses. All such choices are listed in `ASSUMPTIONS.md`. Reported manuscript numbers are preserved only as audit references; they are not hard-coded as generated results.
