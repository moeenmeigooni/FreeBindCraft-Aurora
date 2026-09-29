#!/usr/bin/env python3
"""Exercise FreeBindCraft's real ProteinMPNN sampler on an Aurora XPU.

This is intentionally an execution smoke test.  Its input can be an AF2
trajectory that failed the normal confidence gate, so the emitted sequence is
not a design candidate and must never be evaluated as one.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, type=Path)
    parser.add_argument(
        "--advanced",
        default=Path("aurora/settings_advanced/af2_mpnn_smoke.json"),
        type=Path,
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from functions.colabdesign_utils import mpnn_gen_sequence
    from functions.generic_utils import check_jax_gpu

    if not args.pdb.is_file():
        raise FileNotFoundError(f"ProteinMPNN smoke input is missing: {args.pdb}")

    with args.advanced.open() as handle:
        advanced_settings = json.load(handle)
    advanced_settings.update({"mpnn_fix_interface": False, "num_seqs": 1})

    check_jax_gpu()
    sequences = mpnn_gen_sequence(
        str(args.pdb), binder_chain="B", trajectory_interface_residues="",
        advanced_settings=advanced_settings,
    )
    sequence = str(sequences["seq"][0])
    if not sequence:
        raise RuntimeError("ProteinMPNN returned an empty sequence")

    print("ProteinMPNN XPU smoke passed.")
    print(f"Sampled sequence: {sequence}")
    print(f"MPNN score: {float(sequences['score'][0]):.6f}")


if __name__ == "__main__":
    main()
