#!/usr/bin/env bash
# Source this after resolving the Conda prefix.  Keep every runtime cache out
# of the user's real home directory, which is typically quota-limited on HPC.

if [[ $# -ne 1 ]]; then
  echo "Usage: source aurora/runtime_env.sh CONDA_PREFIX" >&2
  return 2 2>/dev/null || exit 2
fi

freebindcraft_env="$1"
freebindcraft_env_parent="$(dirname -- "${freebindcraft_env}")"
freebindcraft_env_name="$(basename -- "${freebindcraft_env}")"
freebindcraft_cache_root="${FREEBINDCRAFT_CACHE_ROOT:-${freebindcraft_env_parent}/.${freebindcraft_env_name}-cache}"
freebindcraft_runtime_home="${FREEBINDCRAFT_RUNTIME_HOME:-${freebindcraft_env_parent}/.${freebindcraft_env_name}-runtime-home}"
freebindcraft_conda_pkgs="${FREEBINDCRAFT_CONDA_PKGS_DIRS:-${freebindcraft_cache_root}/conda-pkgs}"

mkdir -p \
  "${freebindcraft_conda_pkgs}" \
  "${freebindcraft_cache_root}/pip" \
  "${freebindcraft_cache_root}/xdg" \
  "${freebindcraft_cache_root}/huggingface" \
  "${freebindcraft_cache_root}/torch" \
  "${freebindcraft_cache_root}/jax" \
  "${freebindcraft_cache_root}/matplotlib" \
  "${freebindcraft_cache_root}/python-userbase" \
  "${freebindcraft_runtime_home}"

export FREEBINDCRAFT_CACHE_ROOT="${freebindcraft_cache_root}"
export FREEBINDCRAFT_RUNTIME_HOME="${freebindcraft_runtime_home}"
export HOME="${freebindcraft_runtime_home}"
export CONDA_PKGS_DIRS="${freebindcraft_conda_pkgs}"
export PIP_CACHE_DIR="${freebindcraft_cache_root}/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUSERBASE="${freebindcraft_cache_root}/python-userbase"
export XDG_CACHE_HOME="${freebindcraft_cache_root}/xdg"
export HF_HOME="${freebindcraft_cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${freebindcraft_cache_root}/torch"
export JAX_COMPILATION_CACHE_DIR="${freebindcraft_cache_root}/jax"
export MPLCONFIGDIR="${freebindcraft_cache_root}/matplotlib"
