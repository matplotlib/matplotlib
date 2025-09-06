#!/bin/bash
set -euo pipefail  

MM_VERSION="${MM_VERSION:-latest}"

if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    MM_PLATFORM="linux-64"
else
    echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2
    exit 1
fi

INSTALL_DIR="/usr/local/bin"
if ! command -v micromamba &> /dev/null; then
    curl -Ls "https://micro.mamba.pm/api/micromamba/${MM_PLATFORM}/${MM_VERSION}" \
        | tar -xvj -C "${INSTALL_DIR}" bin/micromamba --strip-components=1
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/codespace/micromamba}"
eval "$(micromamba shell hook --shell=bash)"

micromamba env create -f environment.yml --yes || \
    micromamba env update -f environment.yml --yes --prune

if [ -d "/opt/conda" ]; then
    echo "envs_dirs:
  - ${MAMBA_ROOT_PREFIX}/envs" > /opt/conda/.condarc
fi

ENV_NAME=$(sed -n 's/^name:\s*\(["\x27]\?\)\([^"\x27]*\)\1\s*$/\2/p' environment.yml || echo "mpl-dev")

echo "Installation complete. To activate, run: micromamba activate ${ENV_NAME}"
