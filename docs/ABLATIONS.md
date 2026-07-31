# Ablation mapping

Every manuscript ablation that can be represented without unpublished data is exposed as a YAML file.
These switches reproduce the *structural intervention*, not the manuscript's claimed number.

| Paper ablation | Configuration |
|---|---|
| Deterministic sparse prior | `configs/ablations/deterministic_prior.yaml` |
| Uncertainty not propagated to diffusion | `no_uncertainty_conditioning.yaml` |
| No heteroscedastic loss | `no_heteroscedastic_loss.yaml` |
| No reparameterized sample | `no_reparameterization.yaml` |
| Deterministic regression/refiner | `deterministic_refiner.yaml` |
| No scene conditioning/tokens | `no_scene.yaml` |
| Sparse HMD/hand only | `sparse_only.yaml` |
| No history, visual, or trajectory | `no_history.yaml`, `no_visual.yaml`, `no_trajectory.yaml` |
| No visual and trajectory cues | `no_visual_trajectory.yaml` |
| No masked multimodal training | `no_masked_training.yaml` |
| 10 or 50 DDIM steps | `ddim_10.yaml`, `ddim_50.yaml` |
| No contact decoder | `no_contact_decoder.yaml` |
| No contact and physical losses | `no_contact_or_physics.yaml` |
| No individual physical loss | `no_collision_loss.yaml`, `no_foot_contact_loss.yaml`, `no_foot_height_ground_loss.yaml`, `no_head_hand_alignment_loss.yaml`, `no_temporal_smoothness_loss.yaml` |
| No repainting / observed loss only | `no_repainting.yaml`, `observed_loss_only.yaml` |
| Final-step repainting | `repaint_final_step.yaml` |
| Root/upper-body repainting | `repaint_root_upper.yaml` |

Run configurations with:

```bash
python scripts/run_ablation.py --configs configs/ablations/no_scene.yaml --stage all
```

The manuscript does not define the exact root/upper-body joint set. This repository uses the
complement of `data.lower_body_joints`, which is an explicit configurable assumption.
