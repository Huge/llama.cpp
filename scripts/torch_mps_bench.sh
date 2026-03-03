#!/usr/bin/env bash

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.2.2}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.51.3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKENIZERS_PARALLELISM=false uv run \
  --python "${PYTHON_VERSION}" \
  --with "torch==${TORCH_VERSION}" \
  --with "numpy<2" \
  --with "transformers==${TRANSFORMERS_VERSION}" \
  --with safetensors \
  --with sentencepiece \
  python "${SCRIPT_DIR}/torch_mps_bench.py" "$@"
