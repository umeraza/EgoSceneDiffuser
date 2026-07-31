# Baseline integrations

Third-party baseline repositories are not vendored because their licenses, dependencies, input
modalities, checkpoints, and preprocessing differ. Export each baseline prediction into one NPZ per
`sequence_id` containing:

- `motion`: `[T, J, F]` in the same coordinates and joint order as the target manifest;
- optional `contact_probability`: `[T, C]`;
- optional `uncertainty`: `[T, J, F]`.

Then run `scripts/evaluate_external_predictions.py`. A comparison is invalid unless every prediction
uses the same target windows, transforms, joints, units, and split.
