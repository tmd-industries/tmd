# Copyright 2025 Justin Gullingsrud
# Modifications Copyright 2025-2026, Forrest York
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import jax

# Enable 64 bit jax
jax.config.update("jax_enable_x64", True)


import numpy as np
from rdkit import Chem

# This is needed for pickled mols to preserve their properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from dataclasses import replace
from typing import Any

from rbfe_common import COMPLEX_LEG, SOLVENT_LEG, compute_total_ns

from tmd.constants import DEFAULT_FF, DEFAULT_TEMP, KCAL_TO_KJ
from tmd.fe.absolute.abfe import get_initial_state, optimize_abfe_initial_state, sample_for_restraints
from tmd.fe.absolute.free_energy import AbsoluteBindingFreeEnergy, RestraintParams
from tmd.fe.absolute.plots import generate_restraint_plot
from tmd.fe.free_energy import (
    AbsoluteFreeEnergy,
    HREXParams,
    HREXSimulationResult,
    LocalMDParams,
    MDParams,
    SimulationResult,
    WaterSamplingParams,
    make_pair_bar_plots,
    run_sims_bisection,
)
from tmd.fe.plots import (
    plot_as_png_fxn,
    plot_forward_and_reverse_dg,
    plot_retrospective,
    plot_water_proposals_by_state,
)
from tmd.fe.rbfe import (
    DEFAULT_NUM_WINDOWS,
    HostConfig,
    estimate_relative_free_energy_bisection_hrex_impl,
    setup_optimized_host,
)
from tmd.fe.topology import BaseTopology
from tmd.fe.utils import get_mol_experimental_value, get_mol_name, read_sdf_mols_by_name, set_romol_conf
from tmd.ff import Forcefield
from tmd.md.builders import build_membrane_system, build_protein_system, build_water_system
from tmd.md.exchange.utils import get_radius_of_mol_pair
from tmd.parallel.client import (
    AbstractFileClient,
    CUDAMPSPoolClient,
    FileClient,
    iterate_completed_futures,
)
from tmd.parallel.utils import get_gpu_count


def write_result_csv(
    file_client: AbstractFileClient,
    mols_by_name: dict[str, Chem.Mol],
    leg_results: dict[str, dict[str, Any]],
    experimental_field: str,
    experimental_units: str,
):
    # Write out the DG csv
    csv_header = ["mol"]
    legs = list(sorted(set(leg for legs in leg_results.values() for leg in legs.keys())))
    for leg in legs:
        csv_header.append(f"{leg}_pred_dg (kcal/mol)")
        csv_header.append(f"{leg}_pred_dg_err (kcal/mol)")
    compute_dg = SOLVENT_LEG in legs and COMPLEX_LEG in legs
    if compute_dg:
        csv_header.append("complex correction (kcal/mol)")
        csv_header.append("pred_dg (kcal/mol)")
        csv_header.append("pred_dg_err (kcal/mol)")
        csv_header.append("exp_dg (kcal/mol)")
    ddg_path = file_client.full_path("dg_results.csv")
    with open(ddg_path, "w", newline="") as ofs:
        writer = csv.writer(ofs)
        writer.writerow(csv_header)
        for (name), leg_summaries in leg_results.items():
            row = [name]
            for leg in legs:
                leg_res = leg_summaries.get(leg)
                if leg_res is None:
                    # Add empty values, since the leg didn't run
                    row.append("")
                    row.append("")
                else:
                    leg_pred = leg_res["pred_dg"] / KCAL_TO_KJ
                    leg_err = leg_res["pred_dg_err"] / KCAL_TO_KJ
                    row.append(str(leg_pred))
                    row.append(str(leg_err))
            if compute_dg:
                try:
                    correction = leg_summaries[COMPLEX_LEG]["correction"]
                    edge_dg = (
                        -leg_summaries[COMPLEX_LEG]["pred_dg"] + correction + leg_summaries[SOLVENT_LEG]["pred_dg"]
                    )
                    edge_dg_err = np.linalg.norm(
                        [leg_summaries[COMPLEX_LEG]["pred_dg_err"], leg_summaries[SOLVENT_LEG]["pred_dg_err"]]
                    )
                    row.append(str(correction / KCAL_TO_KJ))
                    row.append(str(edge_dg / KCAL_TO_KJ))
                    row.append(str(edge_dg_err / KCAL_TO_KJ))
                except KeyError:
                    row.extend(["", "", ""])

                exp = None
                try:
                    exp = get_mol_experimental_value(mols_by_name[name], experimental_field, experimental_units)
                except KeyError:
                    pass
                if exp is not None:
                    row.append(str(exp / KCAL_TO_KJ))
                else:
                    row.append("")
            writer.writerow(row)
    if compute_dg:
        x = []
        y = []
        y_err = []
        for row in csv.DictReader(open(ddg_path)):  # type: ignore
            assert isinstance(row, dict)
            if len(row["exp_dg (kcal/mol)"]) > 0 and len(row["pred_dg (kcal/mol)"]) > 0:
                x.append(float(row["exp_dg (kcal/mol)"]))
                y.append(float(row["pred_dg (kcal/mol)"]))
                y_err.append(float(row["pred_dg_err (kcal/mol)"]))
        if len(x) > 0:
            plot_retrospective(
                np.array(y),  # type: ignore
                np.array(x),  # type: ignore
                "ABFE",
                file_client.full_path("dg_plot.png"),
                pred_kcal_errs=np.array(y_err),  # type: ignore
            )
        else:
            print("No experimental labels, skipping plot")


def estimate_abfe_leg(
    mol,
    ff: Forcefield,
    leg: str,
    host_config: HostConfig,
    prefix,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    rst_params: RestraintParams,
):
    host_config = setup_optimized_host(host_config, [mol], ff)
    host_conf = host_config.conf
    bt = BaseTopology(mol, ff)
    temperature = DEFAULT_TEMP

    afe = AbsoluteFreeEnergy(mol, bt)
    if leg == COMPLEX_LEG:
        # Run short equilibration to obtain trajectory used to pick restraint atoms
        initial_state = get_initial_state(afe, ff, host_config, host_conf, temperature, md_params.seed, 0.0)
        minimized_state = optimize_abfe_initial_state(initial_state)
        # TBD: How many frames do you want from here?
        sample_md_params = replace(md_params, n_eq_steps=200_000)
        trj = sample_for_restraints(minimized_state, sample_md_params, replicas=1)

        afe = AbsoluteBindingFreeEnergy.create(bt, host_config, trj, rst_params)

        with open("restraints.svg", "w") as ofs:
            ofs.write(generate_restraint_plot(mol, host_config, afe.lig_atoms, afe.rec_atoms))

        # get equilibrated coordinates and box
        host_conf = afe.x0[: len(host_conf)]
        set_romol_conf(afe.mol, afe.x0[len(host_conf) :])
        host_config = replace(host_config, conf=host_conf, box=afe.box0)
    else:
        # Disable water sampling
        md_params = replace(md_params, water_sampling_params=None)

    def create_abfe_initial_state(lamb):
        return get_initial_state(afe, ff, host_config, host_conf, temperature, md_params.seed, lamb)

    if md_params.hrex_params is None:
        bisection_params = md_params

        def create_minimized_abfe_initial_state(lamb):
            return optimize_abfe_initial_state(create_abfe_initial_state(lamb))

        initial_lambdas = [0.0, 1.0]
        list_of_results, trjs = run_sims_bisection(
            initial_lambdas,
            create_minimized_abfe_initial_state,
            bisection_params,
            n_bisections=n_windows - len(initial_lambdas),
            temperature=temperature,
            min_overlap=min_overlap,
        )
        results = list_of_results[-1]

        plots = make_pair_bar_plots(results, temperature, prefix)
        sim_result = SimulationResult(results, plots, trjs, md_params, list_of_results)
    else:
        sim_result = estimate_relative_free_energy_bisection_hrex_impl(
            temperature,
            0.0,
            1.0,
            md_params,
            n_windows,
            create_abfe_initial_state,
            optimize_abfe_initial_state,
            combined_prefix=prefix,
            min_overlap=min_overlap,
        )
    if leg == COMPLEX_LEG:
        sim_result.correction = afe.get_restraint_correction(temperature)  # type: ignore
        sim_result.rec_atoms = afe.rec_atoms  # type: ignore
        sim_result.lig_atoms = afe.lig_atoms  # type: ignore
    return sim_result


def run_abfe(
    file_client: AbstractFileClient,
    mol_path: Path,
    mol: Chem.Mol,
    leg: str,
    ff: Forcefield,
    pdb_path: str,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    write_trajectories: bool,
    force_overwrite: bool,
    add_membrane: bool,
) -> dict[str, Any]:
    """Run an ABFE calculation.

    Will store results using the file_client to a new directory that name of the ligand being run.

    Stores the following files:

    * results.npz - Predictions, overlaps and the number of windows
    * lambda*_traj.npz - Store the endstate trajectories (if write_trajectory is set to True)
    * final_pairbar_result.pkl - Pickled copy of the final PairBarResult object
    * host_config.pkl - Pickled HostConfig, if the leg is not vacuum
    * dg_errors.png - PNG of the dg errors
    * overlap_summary.png - PNG of the pair bar overlap between windows
    * forward_and_reverse_dg.png - PNG of forward and reverse dG for evaluating convergence
    * hrex_transition_matrix.png - PNG of the transition matrix plot
    * hrex_replica_state_distribution_heatmap.png - PNG of the HREX replica state distribution heatmap
    * water_sampling_acceptances.png - PNG of water sampling acceptances by window

    Parameters
    ----------
    file_client : FileClient
        File client for storing results of the simulation
    mol_path: Path
        Path to write out molecule results to
    mol : Chem.Mol
        Molecule in the system.
    leg: str
        Either complex or solvent
    ff : Forcefield
        Forcefield
    pdb_path : str
        Path to a PDB file
    md_params : MDParams
        Parameters for the RBFE simulation.
    n_windows : int
        Maximum number of windows to generate during bisection.
    min_overlap : float
        Minimum overlap used during bisection.
    write_trajectories: bool
        Whether or not to write trajectories
    force_overwrite: bool
        If results already exist, overwrite the results
    add_membrane: bool
        Build the protein with a POPC membrane

    Returns
    -------
    Summary data
        Data contained in the results.npz. Will include pred_dg
    """
    assert leg in (COMPLEX_LEG, SOLVENT_LEG)
    # Ensure the output directories exists
    leg_path = mol_path / leg
    Path(file_client.full_path(leg_path)).mkdir(parents=True, exist_ok=True)
    results_path = Path(file_client.full_path(leg_path / "results.npz"))
    if not force_overwrite and results_path.is_file():
        print(f"Skipping abfe {leg} calculation: {get_mol_name(mol)}")
        return dict(np.load(results_path))

    with open(file_client.full_path(mol_path / "md_params.pkl"), "wb") as ofs:
        pickle.dump(md_params, ofs)
    with Chem.SDWriter(file_client.full_path(mol_path / "mol.sdf")) as writer:
        writer.write(mol)

    np.random.seed(md_params.seed)
    init_dir = os.getcwd()
    # Change the working directory, so any files written out go to the appropriate directory
    os.chdir(file_client.full_path(leg_path))

    start = time.perf_counter()
    if leg == COMPLEX_LEG:
        if not add_membrane:
            host_config = build_protein_system(pdb_path, ff.protein_ff, ff.water_ff, mols=[mol], box_margin=0.1)
        else:
            host_config = build_membrane_system(pdb_path, ff.protein_ff, ff.water_ff, mols=[mol], box_margin=0.1)
    else:
        host_config = build_water_system(4.0, ff.water_ff, mols=[mol], box_margin=0.1)
    # TBD: Expose restraint params?
    res = estimate_abfe_leg(
        mol, ff, leg, host_config, f"{get_mol_name(mol)}_{leg}", md_params, n_windows, min_overlap, RestraintParams()
    )
    took = time.perf_counter() - start
    os.chdir(init_dir)

    pred_dg = float(np.sum(res.final_result.dGs))
    pred_dg_err = float(np.linalg.norm(res.final_result.dG_errs))

    correction = 0.0 if leg == SOLVENT_LEG else res.correction

    print(
        f"{get_mol_name(mol)} {leg} (kJ/mol) {pred_dg:.2f} +- {pred_dg_err:.2f}, Correction {correction:.2f}, {took:.0f} Seconds"
    )

    summary_data = {
        "time": took,
        "total_ns": compute_total_ns(res, md_params),
        "pred_dg": pred_dg,
        "pred_dg_err": pred_dg_err,
        "n_windows": len(res.final_result.initial_states),
        "overlaps": res.final_result.overlaps,
    }
    if isinstance(res, HREXSimulationResult):
        summary_data["bisected_windows"] = len(res.intermediate_results[-1].initial_states)
        summary_data["normalized_kl_divergence"] = res.hrex_diagnostics.normalized_kl_divergence
    if leg == COMPLEX_LEG:
        summary_data["correction"] = res.correction
        summary_data["receptor_restraint_atoms"] = res.rec_atoms
        summary_data["ligand_restraint_atoms"] = res.lig_atoms

    np.savez_compressed(results_path, **summary_data)

    if write_trajectories:
        np.savez_compressed(
            file_client.full_path(leg_path / "lambda0_traj.npz"),
            coords=np.array(res.trajectories[0].frames),
            boxes=np.asarray(res.trajectories[0].boxes),
        )
    if host_config is not None:
        file_client.store(leg_path / "host_config.pkl", pickle.dumps(host_config))

    if isinstance(res, HREXSimulationResult):
        file_client.store(leg_path / "hrex_transition_matrix.png", res.hrex_plots.transition_matrix_png)
        file_client.store(
            leg_path / "hrex_replica_state_distribution_heatmap.png",
            res.hrex_plots.replica_state_distribution_heatmap_png,
        )
        if res.water_sampling_diagnostics is not None:
            file_client.store(
                leg_path / "water_sampling_acceptances.png",
                plot_as_png_fxn(
                    plot_water_proposals_by_state,
                    [state.lamb for state in res.final_result.initial_states],
                    res.water_sampling_diagnostics.cumulative_proposals_by_state,
                ),
            )
    file_client.store(leg_path / "dg_errors.png", res.plots.dG_errs_png)
    file_client.store(leg_path / "overlap_summary.png", res.plots.overlap_summary_png)
    u_kln_by_lambda = res.final_result.u_kln_by_component_by_lambda.sum(1)
    file_client.store(
        leg_path / "forward_and_reverse_dg.png",
        plot_forward_and_reverse_dg(u_kln_by_lambda, frames_per_step=min(100, u_kln_by_lambda.shape[-1])),
    )
    # Contains initial states and the complete u_kln
    file_client.store(leg_path / "final_pairbar_result.pkl", pickle.dumps(res.final_result))
    return summary_data


def main():
    parser = ArgumentParser(description="Run ABFE for a set of compounds")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
    parser.add_argument("--pdb_path", help="Path to pdb file containing structure")
    parser.add_argument("--mps_workers", type=int, default=1, help="Number of MPS processes per GPU")
    parser.add_argument("--n_eq_steps", default=200_000, type=int, help="Number of steps to perform equilibration")
    parser.add_argument("--n_frames", default=2000, type=int, help="Number of frames to generation")
    parser.add_argument("--steps_per_frame", default=400, type=int, help="Steps per frame")
    parser.add_argument(
        "--n_windows", default=DEFAULT_NUM_WINDOWS, type=int, help="Max number of windows from bisection"
    )
    parser.add_argument("--min_overlap", default=0.667, type=float, help="Overlap to target in bisection")
    parser.add_argument(
        "--target_overlap", default=0.667, type=float, help="Overlap to optimize final HREX schedule to"
    )
    parser.add_argument("--seed", default=2025, type=int, help="Seed")
    parser.add_argument("--forcefield", default=DEFAULT_FF)
    parser.add_argument(
        "--n_gpus", default=None, type=int, help="Number of GPUs to use, defaults to all GPUs if not provided"
    )
    parser.add_argument(
        "--water_sampling_padding",
        type=float,
        default=0.4,
        help="How much to expand the radius of the sphere used for water sampling (nm). Half of the largest intramolecular distance is used as the starting radius to which the padding is added: dist/2 + padding",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to output results, else generates a directory based on the time"
    )
    parser.add_argument("--legs", default=[COMPLEX_LEG, SOLVENT_LEG], nargs="+")
    parser.add_argument(
        "--bisection_frames", type=int, default=100, help="Number of frames to collect during bisection"
    )
    parser.add_argument("--local_md_k", default=10_000.0, type=float, help="Local MD k parameter")
    parser.add_argument("--local_md_radius", default=1.2, type=float, help="Local MD radius")
    parser.add_argument("--local_md_free_reference", action="store_true")
    parser.add_argument(
        "--local_md_steps",
        default=0,
        type=int,
        help="Number of steps to run with Local MD. Must be less than or equal to --steps_per_frame. If set to 0, no local MD is run",
    )
    parser.add_argument(
        "--local_md_iterations",
        default=1,
        type=int,
        help="Number of independent local MD iterations to make. local_md_steps // local_md_iterations steps per iteration",
    )
    parser.add_argument(
        "--store_trajectories",
        action="store_true",
        help="Store the trajectories of the simulations. Can take up a large amount of space",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing predictions, otherwise will skip the completed legs",
    )
    parser.add_argument(
        "--experimental_field", default="kcal/mol experimental dG", help="Field that contains the experimental label."
    )
    parser.add_argument(
        "--experimental_units",
        default="kcal/mol",
        choices=["kcal/mol", "kJ/mol", "uM", "nM"],
        help="Units of the experimental label.",
    )
    parser.add_argument(
        "--add_membrane",
        action="store_true",
        help="Add a POPC membrane to the protein. Refer to OpenMM for preparing proteins for adding Membranes",
    )
    args = parser.parse_args()
    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"abfe_graph_{date_str}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = ""
    if COMPLEX_LEG in args.legs:
        if args.pdb_path is None:
            raise ValueError("Must provide pdb path to run the complex leg")
        pdb_path = str(Path(args.pdb_path).resolve())

    with Chem.SDWriter(dest_dir / "mols.sdf") as writer:
        for mol in mols_by_name.values():
            writer.write(mol)

    file_client = FileClient(dest_dir)

    ff = Forcefield.load_from_file(args.forcefield)

    with open(dest_dir / "ff.py", "w") as ofs:
        ofs.write(ff.serialize())

    num_gpus = args.n_gpus
    if num_gpus is None:
        num_gpus = get_gpu_count()

    # Set max_tasks_per_child=1 to reduce potential for accumulating memory
    pool = CUDAMPSPoolClient(num_gpus, workers_per_gpu=args.mps_workers, max_tasks_per_child=1)
    pool.verify()
    futures = []
    future_id_to_leg = {}
    for name, mol in mols_by_name.items():
        # TBD: Fix this
        mol_radius = get_radius_of_mol_pair(mol, mol)
        md_params = MDParams(
            n_eq_steps=args.n_eq_steps,
            n_frames=args.n_frames,
            steps_per_frame=args.steps_per_frame,
            seed=args.seed,
            hrex_params=HREXParams(
                optimize_target_overlap=args.target_overlap,
                n_frames_bisection=args.bisection_frames,
            ),
            local_md_params=LocalMDParams(
                args.local_md_steps,
                k=args.local_md_k,
                min_radius=args.local_md_radius,
                max_radius=args.local_md_radius,
                freeze_reference=not args.local_md_free_reference,
                iterations=args.local_md_iterations,
            )
            if args.local_md_steps > 0
            else None,
            water_sampling_params=WaterSamplingParams(radius=mol_radius + args.water_sampling_padding),
        )
        for leg in args.legs:
            fut = pool.submit(
                run_abfe,
                file_client,
                Path(name),
                mol,
                leg,
                ff,
                pdb_path,
                md_params,
                args.n_windows,
                args.min_overlap,
                args.store_trajectories,
                args.force_overwrite,
                args.add_membrane,
            )
            future_id_to_leg[fut.id] = (name, leg)
            futures.append(fut)
    leg_results = defaultdict(dict)
    for fut in iterate_completed_futures(futures):
        mol, leg = future_id_to_leg[fut.id]
        try:
            data = fut.result()
        except Exception as e:
            print(f"Leg {leg} {mol} failed: {e}")
            continue
        leg_results[mol][leg] = data
    write_result_csv(file_client, mols_by_name, leg_results, args.experimental_field, args.experimental_units)


if __name__ == "__main__":
    main()
