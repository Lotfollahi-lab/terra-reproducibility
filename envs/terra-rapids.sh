#!/usr/bin/env bash
# Creates the terra-rapids virtual environment and installs all dependencies.
# Script location: /nfs/team361/sb75/terra-reproducibility/envs/terra-rapids.sh

set -euo pipefail

VENV_PATH="/nfs/team361/sb75/.venvs/terra-rapids"
# TERRA is installed from PyPI (terra-st) by default. Set TERRA_DIR to a local
# clone to editable-install that instead (for development), e.g.:
#   TERRA_DIR=/nfs/team361/sb75/terra ./terra-rapids.sh
TERRA_DIR="${TERRA_DIR:-}"

# CUDA toolkit for cupy's runtime JIT. rapids-singlecell's neighbors JIT-compiles
# CUDA kernels via cupy, which needs a real toolkit (headers + nvrtc) at runtime;
# the pip cu12 wheels ship none, so cupy fails with "Failed to find CUDA headers"
# / "failed to open libnvrtc-builtins.so". We bake CUDA_PATH + LD_LIBRARY_PATH
# (this path + /lib64) into the Jupyter kernelspec below so the kernel finds them
# without a manual `module load`. Defaults to the cuda-12.1.1 spack module (its
# nvrtc compiles fine for a 12.4 driver; the path is on the shared /software
# filesystem, stable across nodes). Override CUDA_TOOLKIT if that path changes,
# or point it at any other CUDA 12.x toolkit.
CUDA_TOOLKIT="${CUDA_TOOLKIT:-/software/spack_environments/default/00/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-11.4.0/cuda-12.1.1-5xymjb2qc4qsjuazbljh4gcox2bvafhi}"

echo "Creating virtual environment at ${VENV_PATH} ..."
uv venv "${VENV_PATH}" --python 3.12

echo "Activating virtual environment ..."
source "${VENV_PATH}/bin/activate"

# cuml 25.10's cuml.accel wraps a scikit-learn internal (BaseEstimator.
# _get_default_requests) that scikit-learn removed in 1.8, so `import cuml` /
# rapids-singlecell break on a newer scikit-learn. Constrain scikit-learn <1.8
# for the WHOLE build via UV_CONSTRAINT, so every install below (RAPIDS, scanpy,
# scib-metrics, ...) resolves against a compatible version. A one-off
# `uv pip install "scikit-learn<1.x"` does NOT hold -- a later `uv pip install
# scanpy` re-resolves and pulls scikit-learn forward again. (RAPIDS 25.10 works
# with scikit-learn 1.6-1.7; <1.8 also keeps scanpy modern, avoiding a needless
# scanpy downgrade that <1.6 would force.)
echo "Constraining scikit-learn<1.8 (cuml compatibility) for the whole build ..."
echo "scikit-learn<1.8" > "${VENV_PATH}/constraints.txt"
export UV_CONSTRAINT="${VENV_PATH}/constraints.txt"

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
# (cupy's runtime JIT needs a CUDA toolkit for headers/nvrtc -- that is handled
# by the kernelspec CUDA_PATH/LD_LIBRARY_PATH below, not a pip extra: the
# cupy-cuda12x[ctk] header wheels do not satisfy nvrtc here.)
uv pip install \
    --index-strategy unsafe-best-match \
    --only-binary rapids-singlecell \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch rapids-singlecell \
    "cudf-cu12==25.10.*" "cuml-cu12==25.10.*" "cugraph-cu12==25.10.*" \
    "cuvs-cu12==25.10.*"

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

# Install TERRA. --no-deps in both cases: TERRA's dependencies are already
# installed above, and the CUDA torch / cu12 RAPIDS wheels must not be
# re-resolved (a plain resolve could swap them for default wheels).
# scikit-learn stays <1.8 via UV_CONSTRAINT regardless.
if [ -n "${TERRA_DIR}" ]; then
    echo "Installing terra (editable, --no-deps) from ${TERRA_DIR} ..."
    uv pip install -e "${TERRA_DIR}" --no-deps
else
    echo "Installing terra-st from PyPI (--no-deps) ..."
    uv pip install terra-st --no-deps
fi

echo "Registering Jupyter kernel 'terra-rapids' ..."
python -m ipykernel install --user --name terra-rapids \
    --display-name "Python (terra-rapids)"

# Bake the CUDA toolkit into the kernel's environment so cupy's runtime JIT
# (rapids-singlecell neighbors -> compute_umap) can compile without a manual
# `module load`. The kernelspec env is applied when the kernel process is
# launched, so LD_LIBRARY_PATH is set before the dynamic linker runs (unlike
# setting it from inside Python, which is too late for nvrtc's builtins .so).
echo "Baking CUDA_PATH/LD_LIBRARY_PATH into the terra-rapids kernelspec ..."
CUDA_TOOLKIT="${CUDA_TOOLKIT}" python - <<'PY'
import json, os, pathlib
from jupyter_client.kernelspec import KernelSpecManager

toolkit = os.environ["CUDA_TOOLKIT"]
spec_path = pathlib.Path(
    KernelSpecManager().get_kernel_spec("terra-rapids").resource_dir
) / "kernel.json"
spec = json.loads(spec_path.read_text())
spec.setdefault("env", {}).update({
    "CUDA_PATH": toolkit,
    # Prepend so nvrtc finds libnvrtc-builtins.so.* in the toolkit's lib64.
    # The ${LD_LIBRARY_PATH} tail keeps any inherited entries (harmless if the
    # launcher does not expand it -- a bogus path component is simply ignored).
    "LD_LIBRARY_PATH": f"{toolkit}/lib64:${{LD_LIBRARY_PATH}}",
})
spec_path.write_text(json.dumps(spec, indent=1))
print(f"Patched {spec_path} with CUDA_PATH={toolkit}")
PY

echo "Done. Environment terra-rapids is ready."
echo "To activate it in your shell, run:"
echo "  source ${VENV_PATH}/bin/activate"
