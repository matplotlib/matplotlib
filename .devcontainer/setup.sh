#!/usr/bin/env bash
set -euo pipefail

MM_VERSION="${MM_VERSION:-latest}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)   MM_PLATFORM="linux-64"  ;;
  Linux-aarch64)  MM_PLATFORM="linux-aarch64" ;;
  Darwin-x86_64)  MM_PLATFORM="osx-64" ;;
  Darwin-arm64)   MM_PLATFORM="osx-arm64" ;;
  *) echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

INSTALL_DIR="/usr/local/bin"

if ! command -v micromamba >/dev/null; then
  tmp_tar=$(mktemp)
  curl -Ls "https://micro.mamba.pm/api/micromamba/${MM_PLATFORM}/${MM_VERSION}" -o "${tmp_tar}"
  tar --no-same-owner --strip-components=1 -xvjf "${tmp_tar}" -C "${INSTALL_DIR}" "bin/micromamba"
  rm -f "${tmp_tar}"
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
eval "$(micromamba shell hook --shell bash)"

micromamba env create --file environment.yml --platform "${MM_PLATFORM}" --yes \
|| micromamba env update --file environment.yml --platform "${MM_PLATFORM}" --yes --prune

if [[ -d /opt/conda ]]; then
  { echo 'envs_dirs:'; echo "  - ${MAMBA_ROOT_PREFIX}/envs"; } | sudo tee /opt/conda/.condarc >/dev/null || true
fi

env_name=$(grep -E '^name:' environment.yml | sed -E 's/^name:[[:space:]]*//')
env_name=${env_name:-mpl-dev}

echo "micromamba ready. Activate with: micromamba activate ${env_name}"
