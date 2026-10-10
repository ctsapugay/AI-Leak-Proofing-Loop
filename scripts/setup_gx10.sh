#!/usr/bin/env bash
# One-time GPU env on the GX10 (ARM64, CUDA 13). Run from the repo root.
# Each teammate keeps their own clone (we all log in as `asus`), e.g. ~/clara/AI-Hackathon.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv transformers accelerate numpy scikit-learn faker openai pyyaml pytest
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
