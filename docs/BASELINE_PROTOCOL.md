# Baseline evaluation protocol

The manuscript's baseline table cannot be reproduced from method names alone. For each baseline,
record:

1. repository URL, exact commit, license, and checkpoint;
2. whether the model is retrained or evaluated from a published checkpoint;
3. input modalities actually supplied;
4. raw-to-standardized preprocessing and coordinate transforms;
5. target skeleton/joint mapping and units;
6. split manifest hash;
7. inference precision, batch size, warm-up, and hardware;
8. whether image/scene preprocessing is included in latency;
9. seed(s), repeated runs, mean, and standard deviation.

Export predictions using `baselines/README.md`. Do not copy manuscript numbers into generated result
files. The CSVs in `paper/` are transcriptions, not repository outputs.
