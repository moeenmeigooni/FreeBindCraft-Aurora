#!/usr/bin/env bash
# Backward-compatible name; use aurora/install_aurora.sh for new installs.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/install_aurora.sh" "$@"
