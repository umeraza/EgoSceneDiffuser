FROM python:3.11-slim
WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY egoscenediffuser ./egoscenediffuser
RUN pip install --no-cache-dir -e .
COPY configs ./configs
COPY scripts ./scripts
ENTRYPOINT ["python", "scripts/train.py"]
CMD ["--config", "configs/smoke.yaml", "--stage", "stage1"]
