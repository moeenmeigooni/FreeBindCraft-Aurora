"""Regression tests for Aurora's Rust replacement for ColabDesign realignment."""

import importlib
import sys
import tempfile
import types
import unittest
from importlib.metadata import version as package_version
from pathlib import Path

try:
    import numpy as np
    from Bio.PDB import PDBParser

    # Import this one module under a lightweight synthetic package rather than
    # importing functions/__init__.py, which eagerly imports the full JAX and
    # ColabDesign pipeline.  The realignment helper itself only needs NumPy,
    # BioPython, SciPy, and rust-simulation-tools.
    package_name = "_freebindcraft_realign_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "functions")]
    sys.modules[package_name] = package
    realign_module = importlib.import_module(f"{package_name}.biopython_utils")
    realign_complex_to_input_target = realign_module.realign_complex_to_input_target
except ModuleNotFoundError:
    np = None
    PDBParser = None
    realign_complex_to_input_target = None


def atom_line(serial, residue_name, chain, residue_id, xyz):
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d}  CA  {residue_name:>3s} {chain}{residue_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
    )


def write_pdb(path, chains):
    lines = []
    serial = 1
    for chain, residues in chains:
        for residue_id, xyz in enumerate(residues, start=1):
            lines.append(atom_line(serial, "ALA", chain, residue_id, xyz))
            serial += 1
        lines.append("TER\n")
    path.write_text("".join(lines) + "END\n")


@unittest.skipIf(realign_complex_to_input_target is None, "FreeBindCraft structural dependencies are unavailable")
class TestSyclCpuRealign(unittest.TestCase):
    def test_uses_the_pinned_rust_alignment_release(self):
        self.assertEqual(package_version("rust-simulation-tools"), "0.2.2")

    def test_realigns_entire_complex_using_target_ca_atoms(self):
        target = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        binder = [(0.0, 0.0, 2.0)]

        # 90-degree z rotation followed by a translation.  The binder must
        # receive the same inverse transform as the target.
        def transformed(point):
            x, y, z = point
            return (-y + 4.0, x - 3.0, z + 2.0)

        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            reference_pdb = temporary_dir / "reference.pdb"
            complex_pdb = temporary_dir / "complex.pdb"
            write_pdb(reference_pdb, [("A", target)])
            write_pdb(complex_pdb, [("A", [transformed(point) for point in target]), ("B", [transformed(point) for point in binder])])

            result = realign_complex_to_input_target(str(complex_pdb), str(reference_pdb), "A")

            self.assertEqual(result["target_ca_count"], 3)
            self.assertGreater(result["target_ca_rmsd_before"], 1.0)
            self.assertLess(result["target_ca_rmsd_after"], 1e-10)
            structure = PDBParser(QUIET=True).get_structure("aligned", complex_pdb)[0]
            aligned_target = [residue["CA"].coord for residue in structure["A"]]
            aligned_binder = [residue["CA"].coord for residue in structure["B"]]
            np.testing.assert_allclose(aligned_target, target, atol=1e-3)
            np.testing.assert_allclose(aligned_binder, binder, atol=1e-3)

    def test_rejects_a_target_atom_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            reference_pdb = temporary_dir / "reference.pdb"
            complex_pdb = temporary_dir / "complex.pdb"
            write_pdb(reference_pdb, [("A", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])])
            write_pdb(complex_pdb, [("A", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])])

            with self.assertRaisesRegex(ValueError, "count mismatch"):
                realign_complex_to_input_target(str(complex_pdb), str(reference_pdb), "A")


if __name__ == "__main__":
    unittest.main()
