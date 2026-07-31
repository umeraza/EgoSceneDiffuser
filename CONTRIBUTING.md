# Contributing

Use small, auditable changes. Any change to preprocessing, coordinate frames, skeleton indices,
losses, diffusion, or metrics must update the relevant configuration and documentation. Do not
commit dataset files, SMPL-X parameters, checkpoints, or manuscript numbers presented as reproduced
results.

Before opening a pull request:

```bash
python -m compileall -q egoscenediffuser scripts
pytest -q
```

New dataset adapters must emit the schema in `docs/DATA.md` and include a synthetic or tiny fixture
test. New ablations must have a YAML config and a mapping in `docs/ABLATIONS.md`.
