# FreeBindCraft on ALCF Aurora

## Clone this Aurora port

```bash
git clone https://github.com/moeenmeigooni/FreeBindCraft-Aurora.git
cd FreeBindCraft-Aurora
```

## Port status

This is a **JAX/SYCL proof-of-port**, not a claim of validated protein-design
results yet.  FreeBindCraft's hot path is ColabDesign (AlphaFold2 and
ProteinMPNN), which imports JAX directly.  It does not use PyTorch, so Intel
Extension for PyTorch cannot accelerate it.  The intended accelerator backend
is Intel Extension for OpenXLA's PJRT plug-in.  The plug-in exposes each Intel
GPU as a JAX device with `platform == "sycl"`.

## PE 26.1810 runtime compatibility

The Intel OpenXLA 0.6.0 wheel uses the oneAPI 2025 runtime ABI. On Aurora's
current PE 26.181.0 image, load the rebuilt PE 26.26.0 runtime with
`module load oneapi/release/2025.3.1` before running the JAX workflows.
`aurora/runtime_env.sh` loads it by default; set `AURORA_JAX_ONEAPI_MODULE` to
override the module name.

## Live Aurora smoke result

On 2026-08-31, one Aurora debug-node job validated the software stack on one
Intel GPU tile.  It reported `devices: [sycl(id=0)]`, compiled and executed a
JAX matrix kernel (result `16384.0`), enumerated OpenMM's `OpenCL` platform,
and completed a PDL1 OpenMM/FASPR relaxation.  The relaxation took 29.06 s,
reported a best energy of -20210.09 kJ/mol, and wrote a 150,135-byte relaxed
PDB.  The bundled FASPR binary failed because it requires glibc 2.34; the port
now compiles FASPR upstream revision `0d55732fd6307f373018c6bddd842291c355c5f7`
on Aurora instead.  This validates the XPU runtime and no-PyRosetta relaxation
route, not the full AF2/ProteinMPNN design loop.

The source-side compatibility changes span `functions/generic_utils.py`,
`functions/colabdesign_utils.py`, and `bindcraft.py`: FreeBindCraft recognizes
the `sycl` platform alongside JAX's standard `gpu` platform, counts all
trajectory outcomes against `max_trajectories`, and selects ColabDesign's
existing `realign=False` option on SYCL only. Counting all outcomes prevents
an acceptance-gate run from looping when its only trajectory is low-confidence
or clashing. The pinned patch in `aurora/patches/` makes `realign=False`
suppress the otherwise unused Kabsch coordinate realignment; without it,
ColabDesign eagerly calls the SVD whose `eigh` lowering is absent from this
Intel plug-in. CUDA and other JAX backends retain ColabDesign's default
realignment. The remaining files in this directory are an Aurora-specific
environment and launch overlay.

On SYCL, FreeBindCraft now restores the PDB coordinate frame immediately after
each AF2 write with `rust-simulation-tools==0.2.2` rather than asking JAX to
lower Kabsch/SVD.  It uses the output target C-alpha atom indices to fit the
input target coordinates and applies the resulting rigid transform to the
complete complex, including the binder.  The target chain mapping is strict:
missing chains or unequal C-alpha counts stop the run before relaxation, MPNN,
or filtering can consume a coordinate-frame mismatch.  The bootstrap installs
the pinned wheel even when it reuses an already-created environment.

On 2026-09-01, the bounded PDL1 AF2 gate completed all three design stages on
one Intel GPU tile after applying the pinned realignment patch. With one
iteration per stage it took about two minutes, reached losses 12.18, 9.32, and
9.80, and wrote a 119,095-byte trajectory PDB. Its final binder pLDDT was
0.27, so FreeBindCraft correctly categorized it as low-confidence and did not
invoke MPNN. This demonstrates real AF2/XLA design execution, but not a
scientifically useful binder or the MPNN path.

The follow-up XPU sampler test did exercise ProteinMPNN against that saved
scaffold: it completed without a backend error and returned a non-empty sample
with score 1.883175. Because the input scaffold was deliberately
low-confidence, this is MPNN execution evidence only, not a candidate sequence
or a quality validation.

## What is supported and what is deliberately disabled

| Component | Aurora route | Status/risk |
| --- | --- | --- |
| AF2 hallucination and ProteinMPNN | JAX 0.4.38 + `intel-extension-for-openxla` 0.6.0 | AF2 three-stage gate, MPNN sampler, and both two- and five-validation-model full-path fixture gates passed. The normal-gated iterative reference below remains required before a scale pilot. |
| ColabDesign target-coordinate realignment | JAX-side Kabsch is disabled only on SYCL; `rust-simulation-tools==0.2.2` restores the frame on CPU after every trajectory and re-predicted complex PDB write | The transform is fitted to selected input-target C-alpha atoms and applied to the full complex before relaxation, MPNN templating, RMSD, and filtering. A chain/atom mismatch is an error, never a silent truncation. |
| Structural relaxation | OpenMM's OpenCL platform using Aurora's Intel OpenCL driver | Supported by OpenMM for Intel GPUs; validate numerically on a representative structure. |
| PyRosetta | `--no-pyrosetta` | Required. PyRosetta is a CPU package and FreeBindCraft's OpenMM/FASPR alternatives were made for this mode. |
| FASPR side-chain packing | Built from upstream FASPR source on Aurora | Required: the bundled binary targets glibc 2.34, newer than Aurora's glibc 2.31. |
| DSSP and sc-rs helper executables | Existing x86-64 Linux binaries | Tested on Aurora by the smoke job. |
| Dockerfile | Do not use | It is built from `nvidia/cuda` and registers NVIDIA's OpenCL ICD. Use a native conda environment first; convert it to Apptainer only after validation. |

## Why the original installer cannot work

`install_bindcraft.sh` requests CUDA `jaxlib`, `cuda-nvcc`, and `cudnn`; the
Dockerfile starts from NVIDIA CUDA and registers `libnvidia-opencl.so.1`.  None
exist on Intel Data Center GPU Max cards.  It also pins JAX 0.6.0, whereas the
currently released Intel PJRT plugin requires JAX/JAXLIB 0.4.38.  This version
gap is the main compatibility risk, so do not substitute a CPU JAX wheel or
silently upgrade packages.

## Installation on Aurora

Clone this port to a directory visible from compute nodes, then run the
bootstrap only on an Aurora UAN:

```bash
module load frameworks
cd /lus/flare/projects/PROJECT/FreeBindCraft
bash aurora/bootstrap_aurora.sh --env /lus/flare/projects/PROJECT/freebindcraft-aurora
```

The script creates the pinned environment, builds the pinned FASPR source using
Aurora's compiler, downloads the AF2 parameter archive only if needed, and
creates `params -> <shared params directory>`.  To reuse an existing parameter
cache:

```bash
bash aurora/bootstrap_aurora.sh \
  --env /lus/flare/projects/PROJECT/freebindcraft-aurora \
  --params-dir /lus/flare/projects/PROJECT/af2-params \
  --skip-weights
```

For the initial software-only smoke test, omit the weight download entirely:

```bash
bash aurora/bootstrap_aurora.sh --no-weights
```

Rerun the bootstrap without `--no-weights` before the first AF2 design; it
reuses the already-created environment and stages the parameters.

## Cache policy and fixed project install

Every installer and PBS template sources `aurora/runtime_env.sh` after it
resolves the Conda prefix. It redirects `HOME`, Conda/Pip/XDG/HuggingFace/Torch
and JAX caches, Python user packages, and Matplotlib configuration to a
project-filesystem directory beside the environment. Consequently, model
weights and runtime caches do not go under the user's `/home` directory. Set
`FREEBINDCRAFT_CACHE_ROOT`, `FREEBINDCRAFT_RUNTIME_HOME`, or
`FREEBINDCRAFT_CONDA_PKGS_DIRS` to select other project-visible locations.

The included installer derives the checkout path from its own location. Set the
shared project root and run it from the checkout:

```bash
export AURORA_PROJECT_ROOT="/lus/flare/projects/FRAME-IDP/${USER}"
export FREEBINDCRAFT_REPO="$PWD"
bash aurora/install_FreeBindCraft.sh
```

By default, it installs the environment to
`${AURORA_PROJECT_ROOT}/conda_envs/FreeBindCraft` and places AF2 parameters in
the checkout's `af2-params/` directory. Override
`FREEBINDCRAFT_AURORA_ENV`, `FREEBINDCRAFT_AF2_PARAMS`, or
`FREEBINDCRAFT_CACHE_ROOT` to use other shared locations. To reuse an existing
parameter cache, set its path and add `--skip-weights`:

```bash
export FREEBINDCRAFT_AF2_PARAMS=/path/to/shared/af2-params
bash aurora/install_FreeBindCraft.sh --skip-weights
```

Set `FREEBINDCRAFT_AURORA_ENV` to the same path before submitting a job.  Pass
the account and queue to `qsub`, then edit the three input paths at the bottom
of `aurora/run_bindcraft.pbs`.  For example:

```bash
qsub -A FRAME-IDP -q debug aurora/smoke_test.pbs
```

## Bounded AF2/ProteinMPNN acceptance job

After the software-only smoke test passes and parameters are staged, submit
exactly one deliberately tiny end-to-end gate:

```bash
qsub -A FRAME-IDP -q debug aurora/af2_mpnn_smoke.pbs
```

This uses `settings_target/PDL1_af2_mpnn_smoke.json` and
`settings_advanced/af2_mpnn_smoke.json`: PDL1, one 65-residue binder, one
iteration in each 3-stage AF2 phase, one MPNN sample, and one maximum
trajectory.  It is an execution-compatibility test, not a useful design run.
Review its scheduler stdout/stderr and inspect the configured result directory
(`aurora_af2_mpnn_smoke_sycl_realign_off`) before using the production
template.

Because that deliberately short AF2 run is expected to fail the normal 0.70
trajectory pLDDT threshold, it may not invoke MPNN.  After it has produced a
trajectory PDB, use the independent sampler smoke test to validate the MPNN
XPU path without weakening that threshold:

```bash
qsub -A FRAME-IDP -q debug aurora/mpnn_xpu_smoke.pbs
```

It samples exactly one sequence from the saved scaffold and prints it only as
an execution artifact, never as a design candidate.

## Rust-alignment iterative design smoke

Use the following bounded gate after running the installer updated with the
Rust alignment dependency:

```bash
export FREEBINDCRAFT_AURORA_ENV=${AURORA_PROJECT_ROOT}/conda_envs/FreeBindCraft
qsub -A FRAME-IDP -q debug \
  -v FREEBINDCRAFT_AURORA_ENV,FREEBINDCRAFT_AF2_SMOKE_SETTINGS=aurora/settings_target/PDL1_rust_alignment_iterative_smoke.json,FREEBINDCRAFT_AF2_SMOKE_ADVANCED=aurora/settings_advanced/af2_rust_alignment_iterative_smoke.json \
  aurora/af2_mpnn_smoke.pbs
```

This keeps the AF2 stages deliberately minimal (one logit, softmax, and
one-hot iteration) but performs **three independent trajectories**, each with
the normal acceptance gates intact.  It is a workflow smoke, not a design
campaign.  A passing job must show three `Restored target coordinate frame
with Rust Kabsch` messages, each with a finite post-align target RMSD, and
exactly three unique PDB basenames across `Trajectory`, `LowConfidence`,
`Clashing`, and `Relaxed` in `aurora_pdl1_rust_alignment_iterative_smoke/`.
Do not loosen the 0.70 pLDDT or clash gates merely to force MPNN in this smoke;
the existing independent MPNN sampler remains the correct execution test for
an intentionally tiny AF2 trajectory.

### Completed validation

On 2026-09-02, Aurora job `8797236` completed this three-trajectory gate with
exit status 0 in 7m41s on one Intel GPU tile.  It wrote exactly three PDBs:
one clashing trajectory and two low-confidence trajectories, so the normal
quality gates correctly did not invoke MPNN re-prediction.  Each PDB reported
the Rust Kabsch restoration over 115 target C-alpha atoms.  The direct
target-coordinate RMSDs of the saved PDBs were 1.177, 0.678, and 0.494 A;
the corresponding raw pre-fit coordinate-frame offsets were 67.802, 66.707,
and 65.202 A.  These are execution and coordinate-frame validation results,
not evidence of a viable PDL1 binder.

## Full no-PyRosetta integration gate

`aurora/full_path_xpu_smoke.py` is a deterministic engineering test for the
post-trajectory branch that short AF2 runs often do not reach.  It starts with
the saved `PDL1_XPU_sycl_realign_off_l65_s223675.pdb` fixture and calls the
real ProteinMPNN, AF2 complex re-prediction, Rust target-frame restoration,
OpenMM/FASPR relaxation, no-PyRosetta scoring, binder-monomer prediction,
filtering, and accepted-design ranking functions.  It first checks an
intentionally impossible early filter, then uses the provided all-null filter
file only to prove the remaining plumbing.  Its results are explicitly marked
as a fixture and are never candidates.

```bash
export FREEBINDCRAFT_AURORA_ENV=${AURORA_PROJECT_ROOT}/conda_envs/FreeBindCraft
qsub -A FRAME-IDP -q debug \
  -v FREEBINDCRAFT_AURORA_ENV \
  aurora/full_path_xpu_smoke.pbs
```

Success requires `Full no-PyRosetta integration smoke passed.`, the expected
normal validation-model list, a non-empty `MPNN/Relaxed` ensemble, and exactly
one PDB in `aurora_full_path_xpu_smoke/Accepted/Ranked/`.  Inspect
`integration_summary.json`; do not use the fixture's metrics for binder
selection.

The shipped `standard` mode covers FreeBindCraft's usual two-model validation
profile.  To exercise the separate upstream branch that validates with all
five multimer models, submit a separate output directory:

```bash
qsub -A FRAME-IDP -q debug \
  -v FREEBINDCRAFT_AURORA_ENV,FREEBINDCRAFT_FULL_PATH_MODE=five-model,FREEBINDCRAFT_FULL_PATH_OUTPUT=aurora_full_path_xpu_five_model_smoke \
  aurora/full_path_xpu_smoke.pbs
```

Run this only after the standard fixture gate has passed; it is an additional
branch-coverage test, not another design campaign.

### Completed full-path fixture gates

Aurora job `8805548` completed the standard (two-validation-model) gate on
2026-09-04. It reported a SYCL Intel GPU, project-scoped runtime/cache roots,
a 65-residue ProteinMPNN sample, the deliberately impossible early-filter
rejection, normal complex validation and OpenMM/FASPR relaxation, binder-only
prediction, final filtering, and a ranked fixture PDB. Its outputs are marked
non-candidates.

Aurora job `8816454` completed the alternate five-validation-model gate on
2026-09-10 with exit status 0 in 8m49s. It produced five non-empty complex
PDBs, five relaxed-complex PDBs, five binder-only PDBs, and one ranked fixture
PDB. Model-specific no-PyRosetta shape-complementarity values were 0.42--0.59;
the run did not use a placeholder SC score.

During this validation, a pre-existing no-PyRosetta fallback was corrected:
if `sc-rs` cannot calculate shape complementarity, FreeBindCraft now records
the conservative value `0.00`, rather than a favorable `0.70` placeholder.
The regression test covers absent binaries, empty output, command failure, and
valid JSON. This prevents a failed geometry calculation from passing a quality
filter as a seemingly good interface.

## Normal-gated iterative reference

After the fixture gate passes, run the bounded normal-gated PDL1 reference:

```bash
export FREEBINDCRAFT_AURORA_ENV=${AURORA_PROJECT_ROOT}/conda_envs/FreeBindCraft
qsub -A FRAME-IDP -q debug \
  -v FREEBINDCRAFT_AURORA_ENV,FREEBINDCRAFT_AF2_SMOKE_SETTINGS=aurora/settings_target/PDL1_normal_gated_reference.json,FREEBINDCRAFT_AF2_SMOKE_ADVANCED=aurora/settings_advanced/af2_normal_gated_reference.json \
  aurora/af2_mpnn_smoke.pbs
```

This keeps the original four-stage optimization settings and default quality
filters, caps the run at three independent trajectories, and permits at most
two ProteinMPNN samples per viable trajectory.  It does not weaken confidence,
clash, or final quality thresholds to force an accepted result.  A job that
has no viable trajectory is a valid negative execution result; it is not an
end-to-end scientific validation.  Only an accepted design from this normal
gate provides a real candidate for independent structural and CUDA-comparison
validation.

For a replayable validation or future shard, set
`FREEBINDCRAFT_RANDOM_SEED` to one integer when submitting.  It seeds both
NumPy and Python's random generator before a trajectory ID or length is drawn;
when unset, upstream random behavior is unchanged.

## Required validation gates

1. Submit `qsub aurora/smoke_test.pbs`.  It must report a JAX `sycl` device,
   execute a compiled JAX matmul, enumerate an OpenMM `OpenCL` platform, launch
   the bundled helper programs, and produce an OpenMM-relaxed PDB.
2. Run one deliberately small design with the production PBS template and
   `--no-pyrosetta`.  Check that it writes non-empty trajectory and MPNN CSVs,
   PDBs, and no OpenMM fallback-copy warning.
3. Compare that mini-run with a known CUDA run using the same target, seed, and
   settings.  Compare AF2 confidence metrics, interface contacts, MPNN output,
   and relaxed-PDB geometry.  Exact trajectories need not match across XLA
   backends, but large systematic differences require investigation.
4. Only then launch a trajectory ensemble.  FreeBindCraft has no distributed
   JAX implementation; run independent process/output-directory/seed shards,
   one per tile, and merge results afterwards.  Never point concurrent jobs at
   the same `design_path`, CSV, or `Trajectory` directory.

## Aurora resource model

Aurora nodes have six GPU cards, each with two tiles.  The PBS templates bind a
single process to tile `0.0` using `ZE_AFFINITY_MASK` and ask OpenMM for
`OpenCL,CPU` (not CUDA).  That gives a clean correctness baseline.  Production
throughput comes from independent shards, not from making one ColabDesign
process use all 12 tiles.  Each shard must have a unique output directory and
seed range.

## Known risks and fallback

Intel's OpenXLA PJRT plugin is experimental and pins an older JAX release.
ColabDesign's current revision is therefore pinned in the environment file.
If the end-to-end AF2/MPNN mini-run exposes a missing XLA lowering or a numerical
regression, stop the XPU rollout rather than accepting CPU fallback.  The
practical alternatives are: build the Intel OpenXLA plugin from source against
Aurora's installed oneAPI stack, or use a different AF2/MPNN implementation
with a maintained PyTorch XPU backend.  The latter is a pipeline migration, not
a small FreeBindCraft port.

The included ColabDesign patch covers the non-redesign binder route used by
the supplied templates.  On SYCL, leave `predict_initial_guess` and
`predict_bigbang` disabled: their binder-redesign prediction route makes RMSD
a real loss term and still requires SVD/eigh support.
