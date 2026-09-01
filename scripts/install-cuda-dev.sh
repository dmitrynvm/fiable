#!/usr/bin/env bash
# Install cuBLAS headers/symlinks so CMake can find CUDA::cublas.
# Runtime libcublas is often present without the -dev package.
set -euo pipefail

if [[ "${FIABLE_NO_CUDA:-}" =~ ^(1|true|yes)$ ]]; then
  echo "FIABLE_NO_CUDA set; skipping cuBLAS dev install"
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  echo "nvcc not found; skipping cuBLAS dev install"
  exit 0
fi

NVCC="$(command -v nvcc 2>/dev/null || true)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

if [[ -f /usr/local/cuda/include/cublas.h && -e /usr/local/cuda/lib64/libcublas.so ]]; then
  echo "cuBLAS headers already present"
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; install libcublas-dev for your CUDA toolkit" >&2
  exit 1
fi

release="$("$NVCC" --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1-\2/p' | head -n1)"
if [[ -z "$release" ]]; then
  echo "Could not parse CUDA version from nvcc" >&2
  exit 1
fi

pkg="libcublas-dev-${release}"
echo "Installing ${pkg} (needed for CUDA::cublas)"
export DEBIAN_FRONTEND=noninteractive
apt_cmd=(apt-get)
if [[ "$(id -u)" -ne 0 ]]; then
  apt_cmd=(sudo apt-get)
fi
"${apt_cmd[@]}" update -qq
"${apt_cmd[@]}" install -y "$pkg"
