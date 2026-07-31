# Data contract

## Why a standardized format is required

The manuscript does not define a complete raw-data conversion. EgoBody and GIMO use different annotations and coordinate systems. Training directly from their raw releases without an explicit exporter would hide critical assumptions. The repository therefore uses one auditable per-window NPZ schema.

## Per-window NPZ schema

Required arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `sparse` | `[T_obs, D_sparse]` | head and two hand/controller features |
| `history` | `[T_obs, J, F]` | historical body state |
| `trajectory` | `[T_obs, D_traj]` | head/device trajectory features |
| `scene_points` | `[N, 3]` | local scene points in world or canonical coordinates |
| `motion` | `[T_obs + T_future, J, F]` | target motion |

Optional arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `images` | `[T_obs, 3, H, W]` | uint8 RGB or float RGB in `[0,1]`; loader applies ImageNet normalization |
| `contacts` | `[T_total, C]` | binary contact labels |
| `observed_motion` | `[T_obs, J, F]` | repainting prefix |
| `sparse_positions` | `[T_obs, 3, 3]` | head, left hand, right hand positions |
| `scene_signed_distance` | `[T_total, J]` | signed distance at predicted/target joints, negative inside |
| `joint_rotations` | `[T_total, J, 3, 3]` | rotation-metric target |
| `valid_mask` | `[T_total]` | valid frames |

## Manifest

`manifest.csv` must contain:

```text
path,split,dataset,sequence_id,subject_id,scene_id,start_frame
```

`path` can be absolute or relative to the configured dataset root.

## EgoBody

The official release includes `data_info_release.csv`, `data_splits.csv`, egocentric RGB, HoloLens head/hand/eye CSV files, scene meshes, calibration files, and SMPL-X annotations. The included preparation script validates and indexes author-preprocessed windows. It does not claim to perform the missing raw SMPL-X-to-window conversion, which requires licensed parameters and a declared target identity/calibration policy.

## GIMO

GIMO provides body motion, egocentric video, gaze, and scene scans. The manuscript's proxy VR construction is not published. The preparation script therefore indexes author-preprocessed windows; producing those windows requires an explicit joint map and proxy-controller policy that the manuscript does not provide.

## Split policy

- `official_split: true` uses release-provided splits when available.
- `subject_disjoint_split: true` hashes sorted subject IDs with the configured seed and writes the generated split file. Do not call this the official split.
