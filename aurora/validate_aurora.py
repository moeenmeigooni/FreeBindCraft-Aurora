#!/usr/bin/env python3
"""Fail-fast validation for the FreeBindCraft Aurora software stack.

This does not download AF2 weights or run a design.  It proves that JAX sees an
Intel PJRT/SYCL device, OpenMM can initialize Aurora's OpenCL platform, and the
optional no-PyRosetta relaxation path can import.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    os.chdir(repo_dir)
    # Executing this file by path makes Python place ``aurora/`` (rather than
    # the repository root) at the front of sys.path.  FreeBindCraft's support
    # modules live in the root-level ``functions`` package.
    sys.path.insert(0, str(repo_dir))

    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    print(f"JAX {jax.__version__}; devices: {devices}")
    sycl_devices = [device for device in devices if device.platform == "sycl"]
    if not sycl_devices:
        fail("Intel's JAX backend was not detected. Expected a JAX device with platform 'sycl'.")

    # Compile and execute a small XLA graph; merely enumerating devices can
    # succeed even when the PJRT plug-in cannot submit a kernel.
    @jax.jit
    def kernel(x: jax.Array) -> jax.Array:
        return jnp.tanh(x @ x.T).sum()

    value = kernel(jnp.ones((128, 128), dtype=jnp.float32)).block_until_ready()
    print(f"JAX SYCL kernel result: {float(value):.6f}")

    import openmm
    from openmm import Platform

    platform_names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    print(f"OpenMM platforms: {platform_names}")
    if "OpenCL" not in platform_names:
        fail("OpenMM was installed without its OpenCL platform.")

    from functions import pr_alternative_utils

    importlib.import_module("pdbfixer")
    openmm_version = getattr(openmm, "__version__", None)
    if openmm_version is None:
        openmm_version = getattr(getattr(openmm, "version", None), "version", "unknown")
    print(f"OpenMM version: {openmm_version}")
    print(f"FreeBindCraft no-PyRosetta relax module: {pr_alternative_utils.__file__}")

    helper_binaries = (
        ("functions/dssp", repo_dir / "functions/dssp"),
        ("functions/sc", repo_dir / "functions/sc"),
        ("FASPR", Path(os.environ.get("FASPR_BIN", repo_dir / "functions/FASPR"))),
    )
    for binary, binary_path in helper_binaries:
        if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
            fail(f"Required helper binary is unavailable or non-executable: {binary_path}")
        try:
            result = subprocess.run([str(binary_path), "--help"], capture_output=True, text=True, timeout=15)
        except OSError as error:
            fail(f"Helper binary cannot execute on Aurora: {binary_path}\n{error}")
        # The bundled tools use different help conventions.  A normal usage
        # error is acceptable; an exec-format/shared-library failure is not.
        combined = (result.stdout or "") + (result.stderr or "")
        lowered = combined.lower()
        if (
            "not found" in lowered
            or "exec format" in lowered
            or "error while loading shared libraries" in lowered
        ):
            fail(f"Helper binary cannot execute on Aurora: {binary_path}\n{combined}")
        print(f"Helper binary launched: {binary} (exit code {result.returncode})")

    print("Aurora software-stack validation passed.")


if __name__ == "__main__":
    main()
