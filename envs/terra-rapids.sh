#!/usr/bin/env bash
# Creates the terra-rapids virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/terra-reproducibility/envs/terra-rapids.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/terra-rapids"
# Optional: editable install of the local terra/terra package. Leave empty
# to skip, or: TERRA_DIR=/nfs/team361/sb75/terra ./terra-rapids.sh
TERRA_DIR="${TERRA_DIR:-}"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

echo "Installing setuptools (<81, provides pkg_resources) + wheel ..."
uv pip install "setuptools<81" wheel

# torch and RAPIDS share nvidia-*-cu12 runtime wheels, so they are resolved in
# a SINGLE pass (NVIDIA + PyTorch indexes both visible) to keep one consistent
# CUDA-lib set. cu12 wheels run on any CUDA 12.x driver (incl. 12.9).
# --only-binary rapids-singlecell: force its pre-built (pure-python) wheel.
# With unsafe-best-match + the NVIDIA index, uv otherwise picks an sdist and
# tries to BUILD rapids-singlecell from source -> "Failed to find nvcc" ->
# the whole script aborts (under set -e) before scib_metrics etc. install.
echo "Installing torch + RAPIDS 25.10 + rapids-singlecell (GPU Leiden) ..."
# cuvs-cu12 (RAPIDS Vector Search) is needed by rapids-singlecell's neighbors
# module ('import cuvs') but is NOT pulled by cudf/cuml/cugraph alone.
uv pip install \
    --index-strategy unsafe-best-match \
    --only-binary rapids-singlecell \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch rapids-singlecell \
    "cudf-cu12==25.10.*" "cuml-cu12==25.10.*" "cugraph-cu12==25.10.*" \
    "cuvs-cu12==25.10.*"

# Pin scikit-learn into cuml's supported range. cuml 25.10's cuml.accel wraps
# the sklearn internal BaseEstimator._get_default_requests, which only exists
# in scikit-learn <1.6; a newer sklearn (pulled by scanpy/scib-metrics/etc.)
# makes 'import cuml' fail with AttributeError. Pin BEFORE the science stack
# so those packages resolve against the compatible version.
echo "Pinning scikit-learn (<1.6) for cuml compatibility ..."
uv pip install "scikit-learn<1.6"

echo "Installing scanpy + squidpy + spatialdata ..."
uv pip install scanpy anndata squidpy spatialdata

echo "Installing scib_metrics ..."
uv pip install scib_metrics

echo "Installing celltypist + cellphonedb + leidenalg ..."
uv pip install celltypist cellphonedb leidenalg

echo "Installing transformers + datasets + peft + accelerate + wandb ..."
uv pip install transformers datasets peft accelerate wandb

echo "Installing pyensembl + scikit-misc + session-info + gdown + wget ..."
uv pip install pyensembl scikit-misc session-info gdown wget

echo "Installing dask + distributed ..."
uv pip install "dask[dataframe]" distributed

echo "Installing JupyterLab ..."
uv pip install jupyterlab ipykernel

# Modern dask (>=2025, pulled by RAPIDS) removed the legacy dataframe, but
# spatialdata sets dataframe.query-planning=False on import -> squidpy fails to
# import. Pre-initialise dask.dataframe with query-planning=True at interpreter
# startup (a .pth + module) so spatialdata's later set(False) is a no-op.
echo "Installing the squidpy/dask query-planning startup fix ..."
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cat > "${SITE_PACKAGES}/_squidpy_dask_fix.py" <<'PY'
"""Make squidpy/spatialdata importable under modern dask (>=2025).

spatialdata sets dataframe.query-planning=False on import, which modern dask
rejects (legacy implementation removed). Pre-initialising dask.dataframe with
query-planning=True here -- at startup, before spatialdata is imported --
makes spatialdata's later set(False) a no-op. Auto-loaded via the companion
_squidpy_dask_fix.pth file.
"""
try:
    import dask

    dask.config.set({"dataframe.query-planning": True})
    import dask.dataframe  # noqa: F401  (initialise before spatialdata)
except Exception:
    pass
PY
echo "import _squidpy_dask_fix" > "${SITE_PACKAGES}/zzz_squidpy_dask_fix.pth"

if [ -n "${TERRA_DIR}" ]; then
    # --no-deps: all dependencies are already installed above with carefully
    # pinned versions (scikit-learn<1.6 for cuml, the cu12 RAPIDS wheels, the
    # CUDA torch build). Re-resolving them here would risk upgrading
    # scikit-learn past 1.6 (via scib-metrics) and re-breaking cuml. Editable
    # src-layout install exposes both 'terra' and 'app'.
    echo "Installing terra (editable, --no-deps) from ${TERRA_DIR} ..."
    uv pip install -e "${TERRA_DIR}" --no-deps
fi

echo "Registering Jupyter kernel 'terra-rapids' ..."
python -m ipykernel install --user --name terra-rapids \
    --display-name "Python (terra-rapids)"

echo "Done. Environment terra-rapids is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
