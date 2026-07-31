# Critical paper-to-repository analysis

## Executive verdict

EgoSceneDiffuser is conceptually coherent: sparse device signals create an uncertain body prior;
visual, trajectory, historical-motion, and scene evidence refine it; diffusion handles one-to-many
ambiguity; contact and geometry losses target physical plausibility; repainting protects a known
prefix during forecasting. That is enough to build a reference implementation.

It is not enough to reproduce the reported tables. The manuscript mixes a method proposal with
unpublished engineering choices and several contradictory results. Exact reproduction requires the
authors' processed windows, skeleton/feature definitions, split files, checkpoints, baseline
execution code, and full hyperparameter record.

## Proposed framework

### Stated pipeline

The paper's pipeline decomposes into:

1. sparse HMD/controller, historical-body, and trajectory encoders;
2. a temporal representation;
3. mean and positive joint-wise uncertainty heads;
4. reparameterized sparse-motion sampling;
5. visual and point-cloud encoders;
6. body-context construction;
7. spatially biased cross-attention over scene, visual, and trajectory tokens;
8. uncertainty-modulated forward diffusion and a conditioned denoiser;
9. masked multimodal training;
10. contact prediction and final motion decoding;
11. physical losses and forecasting repainting.

The repository maps those blocks to `models/encoders.py`, `sparse_prior.py`, `scene_fusion.py`,
`diffusion.py`, `contact_decoder.py`, and `model.py`.

### Strong aspects

- The architecture addresses the real under-constraint of sparse VR tracking rather than pretending
  a deterministic inverse exists.
- Separating prior uncertainty from scene-conditioned correction is defensible and testable.
- The multimodal ablations are aligned with the claimed contributions.
- Forecasting repainting is a reasonable mechanism for protecting a known prefix.
- Contact and temporal objectives target failure modes that MPJPE alone misses.

### Unresolved or inconsistent aspects

- The head-centric state is written as transformations, while experiments report joint-position and
  joint-rotation metrics. The conversion and final tensor layout are absent.
- The manuscript never fixes the joint set, root representation, rotation parameterization, device
  order, or velocity convention.
- Two versions of the heteroscedastic prior loss use different notation and reduction.
- The uncertainty-modulated covariance does not clarify whether uncertainty scales variance or
  standard deviation.
- `Delta(head, scene)` and the spatial attention-bias shape are undefined.
- Feature-wise affine uncertainty injection is named but not parameterized.
- The claimed COAP collision loss is introduced only in an ablation, not in the method.
- Repainting does not specify whether the prefix is clean, re-noised, all-joint, partial-joint, or
  applied before or after final decoding.

## Dataset analysis

### EgoBody

The manuscript uses EgoBody for current-motion reconstruction and scene/contact evaluation. That is
plausible because the release contains egocentric streams, tracking, calibrated scenes, and SMPL-X
annotations. The protocol remains ambiguous:

- EgoBody includes two interacting people; the predicted target identity is not stated.
- The official release already provides sequence splits, but the paper claims a separate 70/15/15
  subject-disjoint split without publishing subject IDs.
- The mapping from HoloLens head/hand data to three sparse VR devices is not specified.
- Calibration, frame synchronization, missing-track handling, and scene coordinate conversion are
  not specified.
- A 2.5 m scene crop and 2048 samples are stated, but crop origin, mesh sampling, normals/colors,
  and signed-distance construction are missing.

### GIMO

GIMO is reasonable for forecasting because it includes body motion, egocentric video, gaze, and
scene scans. It is not a native HMD-plus-two-controller dataset. The manuscript creates proxy sparse
inputs from head and upper-limb/hand trajectories but omits:

- exact joints;
- synthetic controller orientation;
- velocity calculation;
- coordinate transforms;
- whether gaze is used or discarded;
- sensor noise/dropout assumptions;
- window stride and split policy.

Any raw converter that silently chooses these would be manufacturing a protocol. The repository
therefore requires standardized author-preprocessed windows.

### Window and split problems

The stated 30 Hz, 60 observed frames, and 90 future frames imply a 2-second observation and 3-second
forecast. The paper does not state stride, overlap, short-sequence policy, temporal augmentation,
scene-boundary handling, or whether windows from one subject/sequence can cross splits. These details
can strongly change sample count and leakage risk.

## Experimental setup

The experimental objective covers reconstruction, forecasting, and physical plausibility. Matching
modalities only to what each baseline supports sounds fair but is not enough. A valid comparison also
needs:

- exact baseline commits and checkpoints;
- retrained versus published-checkpoint status;
- identical target skeleton and coordinate transforms;
- identical dataset windows and splits;
- modality preprocessing and missing-input rules;
- hardware, precision, batch size, and timing protocol;
- repeated seeds and uncertainty intervals.

The manuscript provides none of that. The statement that methods are evaluated under a matched
protocol is therefore not independently auditable.

## Evaluation metrics

### Correctly motivated metrics

The metric families are appropriate:

- reconstruction: MPJPE and body-region MPJPE;
- rotations: MPJRE;
- forecasting: ADE, FDE, and horizon errors;
- dynamics: velocity and acceleration errors;
- physical behavior: skating, ground/body-scene penetration, and contact scores;
- uncertainty: error correlation and Gaussian NLL;
- efficiency: latency, FPS, sampling steps, parameter count, and peak memory.

### Missing metric definitions

- MPJPE alignment is not fully specified: root-relative, head-centric, world, scale-adjusted, or
  rigidly aligned.
- Lower-body and contact-sensitive joint lists are absent.
- Penetration requires a mesh/SDF convention, but the paper reports a single depth value without one.
- Skating requires foot joints, a contact rule, and a velocity unit/threshold.
- Pseudo-contact thresholds are explicitly left unstated.
- NLL assumes an isotropic 3D Gaussian per joint, while the model describes component-wise
  uncertainty; the reduction and calibration target are not reconciled.
- MPJRE requires a predicted rotation representation that the implementation section never gives.
- FPS is not comparable without timing boundaries, preprocessing inclusion, precision, warm-up, and
  baseline implementations.

The repository implements all metrics it can define explicitly and labels penetration/collision
quantities as proxies where a true signed geometry representation is unavailable.

## Training strategy

### Stated schedule

The manuscript gives three stages:

1. prior pretraining for 80 epochs, batch 64;
2. diffusion/multimodal training for 100 epochs, batch 32;
3. contact/physics fine-tuning for 40 epochs, batch 16.

It also gives AdamW, learning rate `2e-4`, weight decay `1e-4`, clipping `1.0`, 1000 diffusion
training steps, 20 DDIM inference steps, A100 hardware, 30 Hz data, 224×224 RGB, 2.5 m scene crops,
and 2048 scene points.

### Missing training controls

The paper omits optimizer betas/epsilon, LR schedule, warm-up, stage freezing, visual-backbone
freezing, gradient accumulation, mixed precision, number of GPUs, distributed strategy, seed,
initialization, EMA, validation cadence, early stopping, checkpoint selection, all loss weights,
masking probabilities, diffusion beta schedule, DDIM eta, uncertainty-noise strength, and data
augmentation.

The repository exposes these choices in YAML. Defaults are engineering assumptions, not recovered
facts.

## Implementation details

The stated ViT-B/16 and Transformer denoiser are insufficient to instantiate the model. Processing every one of 60 RGB frames with ViT-B/16 at the stated stage-II batch size of 32 is also implausibly memory-heavy unless features are frozen/precomputed, frames are subsampled, gradients are accumulated, or multiple GPUs are used; none is stated. Width, heads,
layers, feed-forward size, dropout, token counts, projection design, scene encoder, positional
encoding, and uncertainty FiLM are missing. The reference implementation chooses transparent defaults
and manually reinitializes cloned PyTorch Transformer layers.

The default model predicts direct joint positions. A position-plus-6D-rotation layout is supported by
configuration and the MPJRE utility, but exact SMPL-X reconstruction is not claimed.

## Ablation audit

### Good coverage

The study tests the main causal claims: uncertainty, reparameterization, scene conditioning,
modalities, masking, physical constraints, sampling steps, and repainting.

### Statistical and numerical weaknesses

- Ablations are single aggregate runs with no standard deviation or confidence interval despite
  stochastic training and diffusion.
- Main MPJPE is 43.6 mm, but the diffusion ablation describes the 20-step full model as 42.9 mm.
- The text says 50 steps gives the lowest errors while also saying MPJPE worsens from 42.9 to 43.2.
- Removing reparameterization produces a better lower-body MPJPE (54.7) than the full model (55.2).
- The prior-table caption is duplicated and corrupted.
- The qualitative figure repeats the architecture caption.
- Repainting and multimodal figures reuse or mismatch filenames/captions.
- No repeated runs are reported.

All reported numbers are transcribed under `paper/` and explicitly marked manuscript-reported.

## What this repository can and cannot establish

It can establish that the proposed computational graph is implementable, trainable, testable, and
configurable. It can support a genuine reproduction once the missing data and settings are supplied.
It cannot establish that the manuscript's claimed SOTA numbers, improvements, or FPS values are real.
Those claims require released evidence, not a plausible code reconstruction.
