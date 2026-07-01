#!/usr/bin/env bash
# Step 1 local orchestration env (GPU-free). Deferred: run this on the box that will drive
# inference (locally for dry-runs with the stub, or on vast to hit a vLLM/API endpoint).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
echo "OK: .venv ready. Activate with 'source .venv/bin/activate'."
echo "Static checks:  ruff check harness_rl && mypy harness_rl && pytest -q"
