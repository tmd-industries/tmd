import pickle
import time
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe import atom_mapping
from tmd.fe.free_energy import MDParams
from tmd.fe.plots import plot_as_png_fxn, plot_forward_and_reverse_dg, plot_water_proposals_by_state
from tmd.fe.rbfe import (
    HREXSimulationResult,
    SimulationResult,
    run_complex,
    run_solvent,
    run_vacuum,
)
from tmd.fe.utils import get_mol_name, plot_atom_mapping_grid
from tmd.ff import Forcefield
from tmd.parallel.client import AbstractFileClient


def compute_total_ns(res: SimulationResult | HREXSimulationResult, md_params: MDParams):
    total_steps = 0
    n_windows = len(res.final_result.initial_states)
    # Equilibration steps have to be handled differently for Bisection vs serial
    if len(res.intermediate_results) == 0:
        total_steps += md_params.n_eq_steps * n_windows
    else:
        n_bisection_windows = len(res.intermediate_results[-1].initial_states)
        total_steps += md_params.n_eq_steps * n_bisection_windows
    total_steps += md_params.steps_per_frame * md_params.n_frames * n_windows

    dt = res.final_result.initial_states[0].integrator.dt
    dt_in_fs = 1000 * dt
    steps_per_picosecond = 1000 / dt_in_fs
    ps = total_steps / steps_per_picosecond
    return ps / 1000


def run_rbfe_leg(
    file_client: AbstractFileClient,
    edge_path: Path,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    core: NDArray | None,
    leg_name: Literal["complex", "solvent", "vacuum"],
    ff: Forcefield,
    pdb_path: str | None,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    write_trajectories: bool,
):
    """Run an RBFE leg (vacuum, solvent, or complex).

    Will store results using the file_client to a new directory that has the name of the leg being run.

    Stores the following files:

    * results.npz - Predictions, overlaps and the number of windows
    * lambda*_traj.npz - Store the endstate trajectories
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
    """
    # Ensure the output directories exists
    Path(file_client.full_path(edge_path)).mkdir(parents=True, exist_ok=True)
    leg_path = Path(edge_path) / leg_name
    Path(file_client.full_path(leg_path)).mkdir(parents=True, exist_ok=True)

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
    if leg_name == "vacuum":
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
    elif leg_name == "solvent":
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
    elif leg_name == "complex":
        assert pdb_path is not None
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
        if res.hrex_diagnostics.mbar_estimates is not None:
            summary_data["mbar_estimates"] = res.hrex_diagnostics.mbar_estimates

    np.savez_compressed(file_client.full_path(leg_path / "results.npz"), **summary_data)

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
