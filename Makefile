.PHONY: install test smoke evaluate benchmark clean

install:
	python -m pip install -e .[dev]

test:
	python -m compileall -q egoscenediffuser scripts
	pytest -q

smoke:
	python scripts/train.py --config configs/smoke.yaml --stage all

evaluate:
	python scripts/evaluate.py --config configs/smoke.yaml

benchmark:
	python scripts/benchmark.py --config configs/smoke.yaml

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ outputs
