# EgoSceneDiffuser

A runnable PyTorch **reference implementation** reconstructed from the manuscript
*EgoSceneDiffuser: Uncertainty-Guided Scene-Conditioned Diffusion for Sparse VR Full-Body Motion
Estimation and Forecasting*.

The repository implements:

1. head-centric sparse HMD/controller, history, and trajectory encoding;
2. mean/uncertainty motion-prior prediction with optional reparameterization;
3. RGB and local point-cloud encoders;
4. spatially biased body-to-scene cross-attention;
5. uncertainty-conditioned diffusion training and DDIM inference;
6. full, final-step, root/upper-body, or disabled prefix repainting;
7. contact-aware output decoding and configurable physical losses;
8. three-stage training, evaluation metrics, runtime benchmarking, and manuscript ablations.

## Reproducibility verdict

The paper is not an exact executable specification. It omits the motion tensor layout, joint order,
rotation representation, architecture dimensions, loss weights, mask probabilities, collision/SDF
formulation, pseudo-contact thresholds, raw-data conversion, split identities, and baseline execution
protocol. It also contains conflicting full-model values. This code therefore reproduces the stated
architecture **concept**, not the paper's reported tables.

Read these before training:

- [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md)
- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md)
- [`docs/DATA.md`](docs/DATA.md)
- [`paper/reported_inconsistencies.csv`](paper/reported_inconsistencies.csv)

Do not claim exact reproduction until the authors release their processed split files, mappings,
checkpoints, full hyperparameters, and baseline protocol.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional ViT, mesh, and SMPL-X dependencies:

```bash
pip install -e .[all]
```

## Verified smoke workflow

```bash
python scripts/train.py --config configs/smoke.yaml --stage stage1
python scripts/train.py --config configs/smoke.yaml --stage stage2
python scripts/train.py --config configs/smoke.yaml --stage stage3
python scripts/evaluate.py --config configs/smoke.yaml \
  --checkpoint outputs/smoke/stage3_best.pt
pytest -q
```

The smoke setup uses synthetic data, a tiny CNN, and a short diffusion schedule. It verifies code and
tensor flow; it is not evidence for the manuscript's accuracy.

## Data preparation

EgoBody, GIMO, and SMPL-X assets are not redistributed. Convert licensed raw data into the explicit
schema in `docs/DATA.md`, then index it:

```bash
python scripts/prepare_egobody.py \
  --processed-root data/processed/egobody \
  --manifest data/processed/egobody/manifest.csv \
  --official-split-csv /path/to/data_splits.csv

python scripts/prepare_gimo.py \
  --processed-root data/processed/gimo \
  --manifest data/processed/gimo/manifest.csv \
  --split-csv /path/to/gimo_splits.csv
```

These commands deliberately do not invent the manuscript's missing raw conversion. For a new,
non-official 70/15/15 subject split:

```bash
python scripts/make_subject_split.py \
  --manifest data/processed/egobody/manifest.csv \
  --output data/processed/egobody/manifest_subject_split.csv
```

## Training

```bash
python scripts/train.py --config configs/egobody.yaml --stage stage1
python scripts/train.py --config configs/egobody.yaml --stage stage2 \
  --resume outputs/egobody/stage1_best.pt
python scripts/train.py --config configs/egobody.yaml --stage stage3 \
  --resume outputs/egobody/stage2_best.pt

python scripts/train.py --config configs/gimo.yaml --stage all
```

CLI overrides use dotted keys:

```bash
python scripts/train.py --config configs/gimo.yaml \
  --override model.d_model=384 \
  --override loss.collision=0
```

## Evaluation, inference, and runtime

```bash
python scripts/evaluate.py --config configs/gimo.yaml \
  --checkpoint outputs/gimo/stage3_best.pt --split test

python scripts/infer.py --config configs/gimo.yaml \
  --checkpoint outputs/gimo/stage3_best.pt \
  --input sample_window.npz --output prediction.npz

python scripts/benchmark.py --config configs/gimo.yaml \
  --checkpoint outputs/gimo/stage3_best.pt
```

Implemented metrics include MPJPE and body subsets, optional MPJRE for position+6D-rotation layouts,
ADE/FDE and horizon MPJPE, root/head-hand/velocity/acceleration errors, contact scores, ground
penetration and skating proxies, uncertainty NLL/correlations, latency, parameter count, and peak CUDA
memory.

## Ablations

```bash
python scripts/run_ablation.py --configs \
  configs/ablations/no_scene.yaml \
  configs/ablations/no_repainting.yaml \
  configs/ablations/ddim_50.yaml
```

See [`docs/ABLATIONS.md`](docs/ABLATIONS.md) for the full mapping. External baseline predictions can be evaluated through `scripts/evaluate_external_predictions.py`; the required fairness record is in [`docs/BASELINE_PROTOCOL.md`](docs/BASELINE_PROTOCOL.md).

## Repository layout

```text
configs/                   base, dataset, smoke, and ablation YAML files
docs/                      audit, assumptions, schema, model card, ablation map
egoscenediffuser/data/     standardized loaders, splits, synthetic data
egoscenediffuser/models/   encoders, prior, fusion, diffusion, contact decoder
egoscenediffuser/losses/   prior, task, contact, and physical objectives
egoscenediffuser/metrics/  reconstruction, forecasting, physics, calibration
egoscenediffuser/training/ staged trainer and checkpointing
scripts/                   train, evaluate, infer, benchmark, validate, prepare
paper/                     transcribed manuscript values and contradiction log
tests/                     configuration, data, model, diffusion, metric tests
```

## Licensing

Repository code is MIT licensed. EgoBody, GIMO, SMPL-X assets, pretrained weights, and third-party
baseline implementations retain their own licenses. This repository's license does not override them.
