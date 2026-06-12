#!/bin/bash
set -e  # Exit on any error

VENV_PATH="/nfs/team361/sb75/.venvs/nicheformer"
REPO_URL="https://github.com/theislab/nicheformer.git"
REPO_DIR="nicheformer"

echo "==> Creating virtual environment at $VENV_PATH"
uv venv "$VENV_PATH" --python=3.10

echo "==> Activating virtual environment"
source "$VENV_PATH/bin/activate"

echo "==> Installing PyTorch with CUDA 12.4"
uv pip install "torch==2.5.1" torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> Cloning nicheformer repository"
if [ -d "$REPO_DIR" ]; then
    echo "    Directory '$REPO_DIR' already exists, skipping clone"
else
    git clone "$REPO_URL"
fi

echo "==> Installing nicheformer in editable mode"
cd "$REPO_DIR"
uv pip install -e .

echo "==> Installing scib_metrics"
uv pip install scib_metrics

echo ""
echo "==> Done! Activate your environment with:"
echo "    source $VENV_PATH/bin/activate"