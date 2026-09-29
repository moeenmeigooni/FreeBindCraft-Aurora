#!/usr/bin/env bash
# Project-local installer for the requested shared Aurora Conda prefix.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${FREEBINDCRAFT_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
project_root="${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}"
env_prefix="${FREEBINDCRAFT_AURORA_ENV:-${project_root}/conda_envs/FreeBindCraft}"
params_dir="${FREEBINDCRAFT_AF2_PARAMS:-${repo_dir}/af2-params}"

if [[ ! -d "${repo_dir}" || ! -f "${repo_dir}/aurora/bootstrap_aurora.sh" ]]; then
  echo "FreeBindCraft checkout or bootstrap script is missing: ${repo_dir}" >&2
  exit 2
fi

export FREEBINDCRAFT_AURORA_ENV="${env_prefix}"
export FREEBINDCRAFT_CACHE_ROOT="${FREEBINDCRAFT_CACHE_ROOT:-${project_root}/conda_envs/.FreeBindCraft-cache}"

exec bash "${repo_dir}/aurora/bootstrap_aurora.sh" \
  --env "${env_prefix}" \
  --params-dir "${params_dir}" \
  "$@"
