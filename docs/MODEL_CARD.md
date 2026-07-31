# Model card

## Intended use

Research on sparse-VR full-body motion estimation and egocentric motion forecasting using licensed EgoBody/GIMO data.

## Not intended for

Safety-critical tracking, clinical assessment, biometric identification, surveillance, or claims of exact manuscript reproduction without the authors' missing settings and checkpoints.

## Inputs

Sparse head/hand features, historical body states, RGB frames, ego trajectory, and local scene points.

## Outputs

Full-body joint trajectories, joint-wise uncertainty, and contact probabilities.

## Known limitations

The default direct-joint representation is not a full SMPL-X body model. Scene collision requires signed distances. Fine contact, occlusion, dynamic objects, and human-human interaction remain difficult. Dataset biases and licensing restrictions apply.
