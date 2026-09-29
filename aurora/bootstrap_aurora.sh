#!/usr/bin/env bash
# Create FreeBindCraft's Aurora environment on an Aurora UAN (not a compute
# node).  The compute nodes cannot reach PyPI/GitHub without a configured
# proxy, so this intentionally installs before a PBS job is submitted.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
env_prefix="${FREEBINDCRAFT_AURORA_ENV:-${repo_dir}/.venv-aurora}"
params_dir="${FREEBINDCRAFT_AF2_PARAMS:-${repo_dir}/params}"
skip_weights=false
no_weights=false
faspr_commit="0d55732fd6307f373018c6bddd842291c355c5f7"

usage() {
  cat <<'EOF'
Usage: aurora/bootstrap_aurora.sh [--env PREFIX] [--params-dir DIRECTORY] [--skip-weights | --no-weights]

Run this on an Aurora UAN.  The environment needs Intel's OpenXLA PJRT plugin,
which is JAX's Intel-GPU backend.  --skip-weights is useful when the AlphaFold
parameters have already been staged in --params-dir.  --no-weights creates (or
reuses) only the software environment; it is appropriate for the Aurora smoke
test but cannot run an AF2 design.
EOF
}

while (($#)); do
  case "$1" in
    --env)
      env_prefix="$2"
      shift 2
      ;;
    --params-dir)
      params_dir="$2"
      shift 2
      ;;
    --skip-weights)
      skip_weights=true
      shift
      ;;
    --no-weights)
      no_weights=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v module >/dev/null 2>&1; then
  echo "This installer must run inside Aurora's module environment." >&2
  exit 1
fi

# Aurora's framework module supplies the compatible Level Zero/OpenCL runtime
# and the oneAPI library paths used by the PJRT wheel.  It no longer supplies
# JAX itself, so the pinned packages in environment_aurora.yml are installed
# independently below.
# Lmod's Aurora initialization references shell-specific variables that may be
# unset in a non-interactive bash launched from zsh.  Do not combine nounset
# with this module call; restore it immediately afterwards for the installer.
set +u
module load frameworks
set -u

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not made available by 'module load frameworks'." >&2
  exit 1
fi

if [[ "${skip_weights}" == true && "${no_weights}" == true ]]; then
  echo "--skip-weights and --no-weights cannot be used together." >&2
  exit 2
fi

# Redirect Conda, Python, and ML caches out of the user's true home directory.
# The helper stores them next to the requested prefix by default, including
# during a first install when the prefix itself does not yet exist.
source "${script_dir}/runtime_env.sh" "${env_prefix}"

if [[ -e "${env_prefix}" ]]; then
  if [[ ! -x "${env_prefix}/bin/python" ]]; then
    echo "Environment path exists but is not a usable conda environment: ${env_prefix}" >&2
    exit 1
  fi
  echo "Reusing existing environment: ${env_prefix}"
else
  # Aurora's site Conda configuration can point at a shared project cache that
  # is full or quota-limited. runtime_env.sh gives this installation its own
  # project-filesystem cache without touching $HOME.
  conda env create --prefix "${env_prefix}" --file "${script_dir}/environment_aurora.yml"
fi

# Existing port environments predate the Rust Kabsch dependency.  Install the
# exact wheel after creating or reusing the environment so all SYCL jobs use
# the same tested CPU-side realignment implementation.  runtime_env.sh above
# routes Pip's download cache to the project filesystem.
"${env_prefix}/bin/python" -m pip install --no-deps "rust-simulation-tools==0.2.2"

# ColabDesign performs a Kabsch SVD in its binder loss even when its `realign`
# option is false.  Intel Extension for OpenXLA 0.6.0 does not lower the eigh
# primitive used by that SVD.  Apply this narrowly-scoped patch to the exact
# ColabDesign revision pinned in environment_aurora.yml: it skips the unused
# alignment calculation only when the caller explicitly selected realign=False.
site_packages="$("${env_prefix}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
colabdesign_loss="${site_packages}/colabdesign/af/loss.py"
colabdesign_patch="${script_dir}/patches/colabdesign-sycl-realign.patch"
if [[ ! -f "${colabdesign_loss}" ]]; then
  echo "Pinned ColabDesign loss module is missing: ${colabdesign_loss}" >&2
  exit 1
fi
if grep -Fq '# the Kabsch SVD altogether when it is disabled' "${colabdesign_loss}"; then
  echo "ColabDesign SYCL realignment patch already applied."
elif grep -Fq 'align_fn = get_rmsd_loss(inputs, outputs, L=tL)["align"]' "${colabdesign_loss}"; then
  patch --batch --forward -d "${site_packages}" -p1 < "${colabdesign_patch}"
else
  echo "Pinned ColabDesign loss module does not match the expected revision; refusing to patch it." >&2
  exit 1
fi

# The upstream package expects data_dir/params.  A symlink lets a project use
# one shared, read-only AlphaFold parameter cache without changing JSON files.
if [[ "${no_weights}" == false && "${skip_weights}" == false ]]; then
  mkdir -p "${params_dir}"
  weights_tar="${params_dir}/alphafold_params_2022-12-06.tar"
  if [[ ! -f "${params_dir}/params_model_5_ptm.npz" ]]; then
    curl --fail --location \
      --output "${weights_tar}" \
      "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    tar -xf "${weights_tar}" -C "${params_dir}" --no-same-owner
    rm -f "${weights_tar}"
  fi
fi

if [[ "${no_weights}" == false && ! -f "${params_dir}/params_model_5_ptm.npz" ]]; then
  echo "Missing AlphaFold parameters in ${params_dir}; rerun without --skip-weights or stage them there." >&2
  exit 1
fi

if [[ "${no_weights}" == true ]]; then
  echo "Skipping AlphaFold parameter setup (--no-weights)."
elif [[ ! -e "${repo_dir}/params" ]]; then
  ln -s "${params_dir}" "${repo_dir}/params"
elif [[ "$(cd -- "${repo_dir}/params" && pwd -P)" != "$(cd -- "${params_dir}" && pwd -P)" ]]; then
  echo "${repo_dir}/params already exists but is not ${params_dir}; update your advanced JSON af_params_dir instead." >&2
  exit 1
fi

chmod +x "${repo_dir}/functions/dssp" "${repo_dir}/functions/sc" "${repo_dir}/functions/FASPR"

# The FASPR executable shipped by upstream was built against glibc 2.34, while
# Aurora currently provides glibc 2.31.  Build the upstream FASPR source on
# Aurora instead of replacing the tracked binary.  The job templates set
# FASPR_BIN to this output and provide the required rotamer library beside it.
faspr_source_dir="${repo_dir}/aurora/deps/FASPR"
faspr_bin_dir="${repo_dir}/aurora/bin"
faspr_bin="${faspr_bin_dir}/FASPR"
if [[ ! -d "${faspr_source_dir}/.git" ]]; then
  mkdir -p "$(dirname -- "${faspr_source_dir}")"
  git clone https://github.com/tommyhuangthu/FASPR.git "${faspr_source_dir}"
fi
if [[ "$(git -C "${faspr_source_dir}" rev-parse HEAD)" != "${faspr_commit}" ]]; then
  git -C "${faspr_source_dir}" fetch --depth 1 origin "${faspr_commit}"
  git -C "${faspr_source_dir}" checkout --detach "${faspr_commit}"
fi
if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ is required to build FASPR for Aurora but was not found." >&2
  exit 1
fi
mkdir -p "${faspr_bin_dir}"
g++ -O3 --fast-math -o "${faspr_bin}" "${faspr_source_dir}"/src/*.cpp
ln -sfn ../../functions/dun2010bbdep.bin "${faspr_bin_dir}/dun2010bbdep.bin"

echo "Environment created at: ${env_prefix}"
echo "Runtime home: ${FREEBINDCRAFT_RUNTIME_HOME}"
echo "Cache root: ${FREEBINDCRAFT_CACHE_ROOT}"
echo "Aurora-built FASPR: ${faspr_bin}"
echo "Run: qsub aurora/smoke_test.pbs"
