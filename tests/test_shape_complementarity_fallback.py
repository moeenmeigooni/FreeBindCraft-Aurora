"""Guard against favorable placeholder SC scores in no-PyRosetta mode."""

import importlib
import subprocess
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    # Avoid functions/__init__.py: it eagerly loads the complete JAX/
    # ColabDesign pipeline, while this unit test only needs the structural
    # scoring module and its direct relative imports.
    package_name = "_freebindcraft_sc_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "functions")]
    sys.modules[package_name] = package

    # pr_alternative_utils imports these helpers for its larger relaxation and
    # scoring API.  The shape-complementarity helper under test does not call
    # them, so tiny stubs prevent this unit test from initializing JAX on an
    # Aurora login node via generic_utils.
    generic_utils = types.ModuleType(f"{package_name}.generic_utils")
    generic_utils.clean_pdb = lambda *_args, **_kwargs: None
    sys.modules[generic_utils.__name__] = generic_utils

    logging_utils = types.ModuleType(f"{package_name}.logging_utils")
    logging_utils.vprint = lambda *_args, **_kwargs: None
    sys.modules[logging_utils.__name__] = logging_utils

    biopython_utils = types.ModuleType(f"{package_name}.biopython_utils")
    for name in (
        "hotspot_residues",
        "biopython_align_all_ca",
        "compute_target_segment_lengths",
        "compute_target_chain_lengths",
        "split_chain_into_subchains",
        "merge_chains_into_single",
    ):
        setattr(biopython_utils, name, lambda *_args, **_kwargs: None)
    sys.modules[biopython_utils.__name__] = biopython_utils

    alternative_utils = importlib.import_module(f"{package_name}.pr_alternative_utils")
except ModuleNotFoundError:
    alternative_utils = None


@unittest.skipIf(alternative_utils is None, "FreeBindCraft structural dependencies are unavailable")
class TestShapeComplementarityFallback(unittest.TestCase):
    def test_sc_rs_failure_returns_non_passing_zero(self):
        error = subprocess.CalledProcessError(
            1,
            ["sc"],
            stderr="Error: Failed to read radii: No molecular dots generated",
        )
        with mock.patch.object(alternative_utils.subprocess, "run", side_effect=error):
            self.assertEqual(alternative_utils._calculate_shape_complementarity("fixture.pdb"), 0.0)

    def test_empty_sc_rs_output_returns_non_passing_zero(self):
        with mock.patch.object(
            alternative_utils.subprocess,
            "run",
            return_value=SimpleNamespace(stdout=""),
        ):
            self.assertEqual(alternative_utils._calculate_shape_complementarity("fixture.pdb"), 0.0)

    def test_missing_sc_binary_returns_non_passing_zero(self):
        with mock.patch.object(alternative_utils.os.path, "isfile", return_value=False):
            self.assertEqual(alternative_utils._calculate_shape_complementarity("fixture.pdb"), 0.0)

    def test_valid_sc_rs_json_is_preserved(self):
        with mock.patch.object(
            alternative_utils.subprocess,
            "run",
            return_value=SimpleNamespace(stdout='{"sc": 0.42}'),
        ):
            self.assertEqual(alternative_utils._calculate_shape_complementarity("fixture.pdb"), 0.42)


if __name__ == "__main__":
    unittest.main()
