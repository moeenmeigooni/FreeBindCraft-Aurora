#!/usr/bin/env python3
"""Exercise FreeBindCraft's post-trajectory pipeline on an Aurora XPU.

This is an engineering integration gate, not a binder-design run.  It starts
from a saved PDL1/binder trajectory fixture, generates one ProteinMPNN
sequence, exercises the early-AF2 rejection path, then runs the same sequence
through all normal complex-validation models, Rust frame restoration, OpenMM/
FASPR relaxation, no-PyRosetta scoring, binder-monomer prediction, filtering,
and accepted-design ranking.  The filters used for the positive path are
deliberately permissive, so every output is explicitly marked non-candidate.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def _required_pdb(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} is missing or empty: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("aurora/artifacts/PDL1_XPU_sycl_realign_off_l65_s223675.pdb"),
        help="Existing target+binder PDB used only as an integration fixture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aurora_full_path_xpu_smoke"),
        help="New output directory; the test refuses to overwrite it.",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("standard", "five-model"),
        default="standard",
        help=(
            "Use FreeBindCraft's standard two-model validation profile, or "
            "exercise its alternate five-model validation branch."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    fixture = (repo_root / args.fixture).resolve() if not args.fixture.is_absolute() else args.fixture
    output_root = (repo_root / args.output).resolve() if not args.output.is_absolute() else args.output
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing integration output: {output_root}")
    _required_pdb(fixture, "Integration fixture")

    from colabdesign import mk_afdesign_model
    from functions.biopython_utils import (
        calc_ss_percentage,
        calculate_clash_score,
        target_pdb_rmsd,
        validate_design_sequence,
    )
    from functions.colabdesign_utils import (
        mpnn_gen_sequence,
        predict_binder_alone,
        predict_binder_complex,
    )
    from functions.generic_utils import (
        af2_backend_compat_kwargs,
        calculate_averages,
        check_accepted_designs,
        check_filters,
        check_jax_gpu,
        create_dataframe,
        generate_dataframe_labels,
        generate_directories,
        generate_filter_pass_csv,
        insert_data,
        load_af2_models,
        perform_advanced_settings_check,
    )
    from functions.pyrosetta_utils import score_interface, unaligned_rmsd

    check_jax_gpu()

    with (repo_root / "aurora/settings_advanced/af2_mpnn_smoke.json").open() as handle:
        advanced_settings = json.load(handle)
    advanced_settings.update(
        {
            "mpnn_fix_interface": False,
            "num_seqs": 1,
            "max_mpnn_sequences": 1,
            "num_recycles_validation": 1,
            "remove_unrelaxed_complex": False,
            "remove_binder_monomer": False,
            "save_design_animations": False,
            "save_design_trajectory_plots": False,
            "zip_animations": False,
            "zip_plots": False,
        }
    )
    if args.validation_mode == "five-model":
        # This is FreeBindCraft's native alternate profile: its design stage
        # uses the non-multimer route and validation uses all five multimer
        # models.  It is kept separate from the normal shipped profile.
        advanced_settings["use_multimer_design"] = False
    advanced_settings = perform_advanced_settings_check(advanced_settings, str(repo_root))

    target_settings = {
        "design_path": str(output_root),
        "binder_name": "PDL1_full_path_xpu_smoke",
        "starting_pdb": str(repo_root / "example/PDL1.pdb"),
        "chains": "A",
        "target_hotspot_residues": "56",
        "lengths": [65, 65],
        "number_of_final_designs": 1,
    }
    _required_pdb(Path(target_settings["starting_pdb"]), "PDL1 target")

    with (repo_root / "settings_filters/no_filters.json").open() as handle:
        permissive_filters = json.load(handle)
    early_reject_filters = {"1_pLDDT": {"threshold": 1.1, "higher": True}}

    design_paths = generate_directories(str(output_root))
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()
    trajectory_csv = output_root / "trajectory_stats.csv"
    mpnn_csv = output_root / "mpnn_design_stats.csv"
    final_csv = output_root / "final_design_stats.csv"
    failure_csv = output_root / "failure_csv.csv"
    for csv_path, labels in (
        (trajectory_csv, trajectory_labels),
        (mpnn_csv, design_labels),
        (final_csv, final_labels),
    ):
        create_dataframe(str(csv_path), labels)
    generate_filter_pass_csv(str(failure_csv), str(repo_root / "settings_filters/no_filters.json"))

    # ProteinMPNN's target chain stays fixed.  The saved fixture itself is not
    # a design candidate, so no interface positions are frozen in this test.
    mpnn_output = mpnn_gen_sequence(
        str(fixture), binder_chain="B", trajectory_interface_residues="", advanced_settings=advanced_settings
    )
    length = 65
    binder_sequence = str(mpnn_output["seq"][0])[-length:]
    if len(binder_sequence) != length:
        raise RuntimeError(f"ProteinMPNN returned {len(binder_sequence)} binder residues; expected {length}")
    mpnn_name = "PDL1_full_path_xpu_smoke_mpnn1"

    _, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])
    complex_prediction_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        use_initial_guess=False,
        use_initial_atom_pos=False,
        **af2_backend_compat_kwargs(),
    )
    complex_prediction_model.prep_inputs(
        pdb_filename=target_settings["starting_pdb"],
        chain=target_settings["chains"],
        binder_len=length,
        rm_target_seq=advanced_settings["rm_template_seq_predict"],
        rm_target_sc=advanced_settings["rm_template_sc_predict"],
    )

    # Validate the normal early-rejection bookkeeping without making a quality
    # claim or retaining a PDB.  The threshold is intentionally impossible.
    _, early_passed, early_failures = predict_binder_complex(
        complex_prediction_model,
        binder_sequence,
        f"{mpnn_name}_early_reject",
        target_settings["starting_pdb"],
        target_settings["chains"],
        length,
        str(fixture),
        prediction_models,
        advanced_settings,
        early_reject_filters,
        design_paths,
        str(failure_csv),
        use_pyrosetta=False,
    )
    if early_passed or "1_pLDDT" not in early_failures:
        raise RuntimeError(f"The intentional early-filter rejection did not occur: {early_failures}")

    prediction_stats, full_path_passed, full_path_failures = predict_binder_complex(
        complex_prediction_model,
        binder_sequence,
        mpnn_name,
        target_settings["starting_pdb"],
        target_settings["chains"],
        length,
        str(fixture),
        prediction_models,
        advanced_settings,
        permissive_filters,
        design_paths,
        str(failure_csv),
        use_pyrosetta=False,
    )
    if not full_path_passed or full_path_failures:
        raise RuntimeError(f"Permissive full-path filters unexpectedly failed: {full_path_failures}")
    if sorted(prediction_stats) != [model + 1 for model in prediction_models]:
        raise RuntimeError(f"Incomplete AF2 validation ensemble: {sorted(prediction_stats)}")

    complex_statistics = {}
    for model_num in prediction_models:
        model_id = model_num + 1
        complex_pdb = Path(design_paths["MPNN"]) / f"{mpnn_name}_model{model_id}.pdb"
        relaxed_pdb = Path(design_paths["MPNN/Relaxed"]) / f"{mpnn_name}_model{model_id}.pdb"
        _required_pdb(complex_pdb, f"AF2 complex model {model_id}")
        _required_pdb(relaxed_pdb, f"OpenMM/FASPR-relaxed complex model {model_id}")

        scores, interface_aas, interface_residues = score_interface(str(relaxed_pdb), "B", use_pyrosetta=False)
        (
            alpha,
            beta,
            loops,
            alpha_interface,
            beta_interface,
            loops_interface,
            interface_plddt,
            ss_plddt,
        ) = calc_ss_percentage(str(complex_pdb), advanced_settings, "B")
        complex_statistics[model_id] = {
            **prediction_stats[model_id],
            "i_pLDDT": interface_plddt,
            "ss_pLDDT": ss_plddt,
            "Unrelaxed_Clashes": calculate_clash_score(str(complex_pdb)),
            "Relaxed_Clashes": calculate_clash_score(str(relaxed_pdb)),
            "Binder_Energy_Score": scores["binder_score"],
            "Surface_Hydrophobicity": scores["surface_hydrophobicity"],
            "ShapeComplementarity": scores["interface_sc"],
            "PackStat": scores["interface_packstat"],
            "dG": scores["interface_dG"],
            "dSASA": scores["interface_dSASA"],
            "dG/dSASA": scores["interface_dG_SASA_ratio"],
            "Interface_SASA_%": scores["interface_fraction"],
            "Interface_Hydrophobicity": scores["interface_hydrophobicity"],
            "n_InterfaceResidues": scores["interface_nres"],
            "n_InterfaceHbonds": scores["interface_interface_hbonds"],
            "InterfaceHbondsPercentage": scores["interface_hbond_percentage"],
            "n_InterfaceUnsatHbonds": scores["interface_delta_unsat_hbonds"],
            "InterfaceUnsatHbondsPercentage": scores["interface_delta_unsat_hbonds_percentage"],
            "Interface_Helix%": alpha_interface,
            "Interface_BetaSheet%": beta_interface,
            "Interface_Loop%": loops_interface,
            "Binder_Helix%": alpha,
            "Binder_BetaSheet%": beta,
            "Binder_Loop%": loops,
            "InterfaceAAs": interface_aas,
            "Hotspot_RMSD": unaligned_rmsd(str(fixture), str(complex_pdb), "B", "B", use_pyrosetta=False),
            "Target_RMSD": target_pdb_rmsd(str(complex_pdb), target_settings["starting_pdb"], target_settings["chains"]),
        }

    complex_averages = calculate_averages(complex_statistics, handle_aa=True)
    binder_prediction_model = mk_afdesign_model(
        protocol="hallucination",
        use_templates=False,
        initial_guess=False,
        use_initial_atom_pos=False,
        num_recycles=advanced_settings["num_recycles_validation"],
        data_dir=advanced_settings["af_params_dir"],
        use_multimer=multimer_validation,
        **af2_backend_compat_kwargs(),
    )
    binder_prediction_model.prep_inputs(length=length)
    binder_statistics = predict_binder_alone(
        binder_prediction_model,
        binder_sequence,
        mpnn_name,
        length,
        str(fixture),
        "B",
        prediction_models,
        advanced_settings,
        design_paths,
        use_pyrosetta=False,
    )
    for model_num in prediction_models:
        model_id = model_num + 1
        binder_pdb = Path(design_paths["MPNN/Binder"]) / f"{mpnn_name}_model{model_id}.pdb"
        _required_pdb(binder_pdb, f"Binder monomer model {model_id}")
        binder_statistics[model_id]["Binder_RMSD"] = unaligned_rmsd(
            str(fixture), str(binder_pdb), "B", "A", use_pyrosetta=False
        )
    binder_averages = calculate_averages(binder_statistics)

    statistics_labels = [
        "pLDDT", "pTM", "i_pTM", "pAE", "i_pAE", "ipSAE", "i_pLDDT", "ss_pLDDT",
        "Unrelaxed_Clashes", "Relaxed_Clashes", "Binder_Energy_Score", "Surface_Hydrophobicity",
        "ShapeComplementarity", "PackStat", "dG", "dSASA", "dG/dSASA", "Interface_SASA_%",
        "Interface_Hydrophobicity", "n_InterfaceResidues", "n_InterfaceHbonds",
        "InterfaceHbondsPercentage", "n_InterfaceUnsatHbonds", "InterfaceUnsatHbondsPercentage",
        "Interface_Helix%", "Interface_BetaSheet%", "Interface_Loop%", "Binder_Helix%",
        "Binder_BetaSheet%", "Binder_Loop%", "InterfaceAAs", "Hotspot_RMSD", "Target_RMSD",
    ]
    selected_model = max(prediction_models, key=lambda model_num: prediction_stats[model_num + 1]["pLDDT"]) + 1
    mpnn_data = [
        mpnn_name,
        "integration-smoke",
        length,
        0,
        0,
        target_settings["target_hotspot_residues"],
        binder_sequence,
        interface_residues,
        round(float(mpnn_output["score"][0]), 2),
        round(float(mpnn_output["seqid"][0]), 2),
    ]
    for label in statistics_labels:
        mpnn_data.append(complex_averages.get(label))
        for model_id in range(1, 6):
            mpnn_data.append(complex_statistics.get(model_id, {}).get(label))
    for label in ("pLDDT", "pTM", "pAE", "Binder_RMSD"):
        mpnn_data.append(binder_averages.get(label))
        for model_id in range(1, 6):
            mpnn_data.append(binder_statistics.get(model_id, {}).get(label))
    mpnn_data.extend(
        [
            "integration-smoke; not a candidate",
            validate_design_sequence(binder_sequence, complex_averages.get("Relaxed_Clashes", 0), advanced_settings),
            "fixture",
            "no_filters",
            "af2_mpnn_smoke",
        ]
    )
    if check_filters(mpnn_data, design_labels, permissive_filters) is not True:
        raise RuntimeError("Permissive final filters unexpectedly rejected the integration fixture")
    negative_result = check_filters(
        mpnn_data, design_labels, {"Average_pLDDT": {"threshold": 1.1, "higher": True}}
    )
    if negative_result != ["Average_pLDDT"]:
        raise RuntimeError(f"Negative final-filter test failed: {negative_result}")

    insert_data(str(mpnn_csv), mpnn_data)
    insert_data(str(final_csv), [""] + mpnn_data)
    selected_relaxed = Path(design_paths["MPNN/Relaxed"]) / f"{mpnn_name}_model{selected_model}.pdb"
    shutil.copyfile(selected_relaxed, Path(design_paths["Accepted"]) / selected_relaxed.name)
    if not check_accepted_designs(
        design_paths,
        str(mpnn_csv),
        final_labels,
        str(final_csv),
        advanced_settings,
        target_settings,
        design_labels,
        rank_by="Average_i_pTM",
    ):
        raise RuntimeError("Accepted-design ranking did not complete")
    ranked = list(Path(design_paths["Accepted/Ranked"]).glob("*.pdb"))
    if len(ranked) != 1:
        raise RuntimeError(f"Expected one ranked fixture PDB, found {ranked}")

    summary = {
        "candidate": False,
        "validation_mode": args.validation_mode,
        "fixture": str(fixture),
        "mpnn_sequence_length": len(binder_sequence),
        "prediction_models": [model + 1 for model in prediction_models],
        "early_rejection": early_failures,
        "full_path_filter_passed": full_path_passed,
        "negative_final_filter": negative_result,
        "selected_model": selected_model,
        "ranked_pdb": str(ranked[0]),
        "complex_averages": complex_averages,
        "binder_averages": binder_averages,
    }
    with (output_root / "integration_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("Full no-PyRosetta integration smoke passed.")
    print(f"AF2 validation models: {summary['prediction_models']}")
    print(f"Ranked fixture output: {summary['ranked_pdb']}")


if __name__ == "__main__":
    main()
