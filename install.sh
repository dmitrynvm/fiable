#!/usr/bin/env bash
# Bootstrap system deps and install the fiable CLI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DEBIAN_FRONTEND=noninteractive

if command -v apt-get >/dev/null 2>&1; then
  apt_cmd=(apt-get)
  if [[ "$(id -u)" -ne 0 ]]; then
    apt_cmd=(sudo apt-get)
  fi
  "${apt_cmd[@]}" update -qq
  "${apt_cmd[@]}" install -y build-essential cmake git pkg-config
  bash "${ROOT}/scripts/install-cuda-dev.sh"
fi

PYTHON="${PYTHON:-python3}"
if [[ -x /venv/main/bin/python ]]; then
  PYTHON=/venv/main/bin/python
fi

"$PYTHON" -m pip install -e "$ROOT"
echo "Installed. Next: fiable download && fiable quantize"
