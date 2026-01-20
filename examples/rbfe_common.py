import csv
import pickle
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS, KCAL_TO_KJ
from tmd.fe import atom_mapping
from tmd.fe.free_energy import MDParams, compute_total_ns
from tmd.fe.mle import infer_node_vals_and_errs_networkx
from tmd.fe.plots import plot_as_png_fxn, plot_forward_and_reverse_dg, plot_water_proposals_by_state
from tmd.fe.rbfe import (
    HREXSimulationResult,
    run_complex,
    run_solvent,
    run_vacuum,
)
from tmd.fe.utils import get_mol_experimental_value, get_mol_name, plot_atom_mapping_grid
from tmd.ff import Forcefield
from tmd.parallel.client import AbstractFileClient

VACUUM_LEG = "vacuum"
SOLVENT_LEG = "solvent"
COMPLEX_LEG = "complex"


def write_result_csvs(
    file_client: AbstractFileClient,
    mols_by_name: dict[str, Chem.Mol],
    leg_results: dict[tuple[str, str], dict[str, Any]],
    experimental_field: str,
    experimental_units: str,
):
    legs = list(sorted(set(leg for legs in leg_results.values() for leg in legs.keys())))
    g = nx.DiGraph()
    edge_records: dict[tuple[str, str], dict] = {}
    for (name_a, name_b), leg_summaries in leg_results.items():
        edge: dict[str, Any] = {
            "mol_a": name_a,
            "mol_b": name_b,
        }
        for leg in legs:
            leg_res = leg_summaries.get(leg)
            if leg_res is not None:
                edge[f"{leg}_pred_dg (kcal/mol)"] = leg_res["pred_dg"] / KCAL_TO_KJ
                edge[f"{leg}_pred_dg_err (kcal/mol)"] = leg_res["pred_dg_err"] / KCAL_TO_KJ

        exp_a = None
        exp_b = None
        try:
            exp_a = get_mol_experimental_value(mols_by_name[name_a], experimental_field, experimental_units)
            g.add_node(name_a, node_exp=exp_a)
        except KeyError:
            pass
        try:
            exp_b = get_mol_experimental_value(mols_by_name[name_b], experimental_field, experimental_units)
            g.add_node(name_b, node_exp=exp_b)
        except KeyError:
            pass
        if exp_a is not None and exp_b is not None:
            edge["exp_ddg (kcal/mol)"] = (exp_b - exp_a) / KCAL_TO_KJ

        if COMPLEX_LEG in leg_summaries and SOLVENT_LEG in leg_summaries:
            edge["pred_ddg (kcal/mol)"] = (
                leg_summaries[COMPLEX_LEG]["pred_dg"] - leg_summaries[SOLVENT_LEG]["pred_dg"]
            ) / KCAL_TO_KJ
            edge["pred_ddg_err (kcal/mol)"] = (
                np.linalg.norm([leg_summaries[COMPLEX_LEG]["pred_dg_err"], leg_summaries[SOLVENT_LEG]["pred_dg_err"]])
                / KCAL_TO_KJ
            )

        edge_records[(name_a, name_b)] = edge

    for edge in edge_records.values():
        if "pred_ddg (kcal/mol)" in edge:
            g.add_edge(
                edge["mol_a"],
                edge["mol_b"],
                edge_pred=edge["pred_ddg (kcal/mol)"],
                edge_pred_std=edge["pred_ddg_err (kcal/mol)"],
            )

    res = None
    if len(g.edges) > 0:
        res = infer_node_vals_and_errs_networkx(
            g,
            edge_diff_prop="edge_pred",
            edge_stddev_prop="edge_pred_std",
            ref_node_val_prop="node_exp",
            ref_node_stddev_prop="node_exp_std",
        )
        dg_csv_header = ["mol", "smiles", "pred_dg (kcal/mol)", "pred_dg_err (kcal/mol)", "exp_dg (kcal/mol)"]
        dg_path = file_client.full_path("dg_results.csv")
        with open(dg_path, "w", newline="") as ofs:
            writer = csv.writer(ofs)
            writer.writerow(dg_csv_header)
            for n, data in res.nodes(data=True):
                writer.writerow(
                    [
                        n,
                        Chem.MolToSmiles(Chem.RemoveHs(mols_by_name[n])),
                        data["inferred_dg"] / KCAL_TO_KJ if "inferred_dg" in data else "",
                        data["inferred_dg_stddev"] / KCAL_TO_KJ if "inferred_dg" in data else "",
                        data["node_exp"] / KCAL_TO_KJ if "node_exp" in data else "",
                    ]
                )
        for edge in edge_records.values():
            name_a = edge["mol_a"]
            name_b = edge["mol_b"]
            # If both nodes had predictions, add the implied MLE ddG
            if name_a in res.nodes and name_b in res.nodes:
                data_a = res.nodes[name_a]
                data_b = res.nodes[name_b]
                edge["mle_ddg (kcal/mol)"] = data_b["inferred_dg"] - data_a["inferred_dg"]
                edge["mle_ddg_err (kcal/mol)"] = np.linalg.norm(
                    [data_b["inferred_dg_stddev"], data_a["inferred_dg_stddev"]]
                )

    ddg_path = file_client.full_path("ddg_results.csv")
    # Get the edge with the most fields
    field_names = list(max(edge_records.values(), key=len).keys())
    ddg_writer = csv.DictWriter(open(ddg_path, "w"), fieldnames=field_names)
    ddg_writer.writeheader()
    for edge in edge_records.values():
        ddg_writer.writerow(edge)


def run_rbfe_leg(
    file_client: AbstractFileClient,
    edge_path: Path,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray | None,
    leg_name: str,
    ff: Forcefield,
    pdb_path: str | None,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    write_trajectories: bool,
    force_overwrite: bool,
) -> dict[str, Any]:
    """Run an RBFE leg (vacuum, solvent, or complex).

    Will store results using the file_client to a new directory that has the name of the leg being run.

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
    edge_path: Path
        Path to directory that contains the edge level data
    mol_a : Chem.Mol
        First molecule in the system.
    mol_b : Chem.Mol
        Second molecule in the system.
    core : NDArray
        The atom mapping between the two molecules
    leg_name : str
        Name of the leg to run. Can be "vacuum", "solvent", or "complex".
    ff : Forcefield
        Forcefield
    pdb_path : str, optional
        Path to a PDB file if running a "complex" leg. Optional.
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

    Returns
    -------
    Summary data
        Data contained in the results.npz. Will include pred_dg
    """
    # Ensure the output directories exists
    Path(file_client.full_path(edge_path)).mkdir(parents=True, exist_ok=True)
    leg_path = Path(edge_path) / leg_name
    Path(file_client.full_path(leg_path)).mkdir(parents=True, exist_ok=True)
    results_path = Path(file_client.full_path(leg_path / "results.npz"))
    if not force_overwrite and results_path.is_file():
        print(f"Skipping existing leg {leg_name}: {get_mol_name(mol_a)} -> {get_mol_name(mol_b)}")
        return dict(np.load(results_path))

    if core is None:
        core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]

    # Store top level data
    file_client.store(edge_path / "atom_mapping.svg", plot_atom_mapping_grid(mol_a, mol_b, core).encode("utf-8"))
    with open(file_client.full_path(edge_path / "md_params.pkl"), "wb") as ofs:
        pickle.dump(md_params, ofs)
    with open(file_client.full_path(edge_path / "core.pkl"), "wb") as ofs:
        pickle.dump(core, ofs)
    with open(file_client.full_path(edge_path / "ff.py"), "w") as ofs:
        ofs.write(ff.serialize())
    with Chem.SDWriter(file_client.full_path(edge_path / "mols.sdf")) as writer:
        writer.write(mol_a)
        writer.write(mol_b)

    np.random.seed(md_params.seed)
    start = time.perf_counter()
    host_config = None
    if leg_name == VACUUM_LEG:
        res = run_vacuum(
            mol_a,
            mol_b,
            core,
            ff,
            None,
            md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    elif leg_name == SOLVENT_LEG:
        res, host_config = run_solvent(
            mol_a,
            mol_b,
            core,
            ff,
            None,
            md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    elif leg_name == COMPLEX_LEG:
        assert pdb_path is not None, "No pdb data provided"
        res, host_config = run_complex(
            mol_a,
            mol_b,
            core,
            ff,
            str(Path(pdb_path).expanduser()),
            md_params,
            n_windows=n_windows,
            min_overlap=min_overlap,
        )
    else:
        assert 0, f"Invalid leg: {leg_name}"
    took = time.perf_counter() - start

    pred_dg = float(np.sum(res.final_result.dGs))
    pred_dg_err = float(np.linalg.norm(res.final_result.dG_errs))
    print(
        " | ".join(
            [
                f"{get_mol_name(mol_a)} -> {get_mol_name(mol_b)} (kJ/mol)",
                f"{leg_name} {pred_dg:.2f} +- {pred_dg_err:.2f}",
                f"{took:.0f} Seconds",
            ]
        ),
    )

    summary_data = {
        "time": took,
        "total_ns": compute_total_ns(res, md_params),
        "pred_dg": pred_dg,
        "pred_dg_err": pred_dg_err,
        "overlaps": res.final_result.overlaps,
        "n_windows": len(res.final_result.initial_states),
    }
    if isinstance(res, HREXSimulationResult):
        summary_data["bisected_windows"] = len(res.intermediate_results[-1].initial_states)
        summary_data["normalized_kl_divergence"] = res.hrex_diagnostics.normalized_kl_divergence

    np.savez_compressed(results_path, **summary_data)

    if write_trajectories:
        np.savez_compressed(
            file_client.full_path(leg_path / "lambda0_traj.npz"),
            coords=np.array(res.trajectories[0].frames),
            boxes=np.asarray(res.trajectories[0].boxes),
        )
        np.savez_compressed(
            file_client.full_path(leg_path / "lambda1_traj.npz"),
            coords=np.array(res.trajectories[-1].frames),
            boxes=np.asarray(res.trajectories[-1].boxes),
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
