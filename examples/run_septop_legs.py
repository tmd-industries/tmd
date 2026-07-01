# (C) 2026 Justin Gullingsrud

import os
import pickle
import time
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import jax

# Enable 64 bit jax
jax.config.update("jax_enable_x64", True)


import numpy as np
from rdkit import Chem

# This is needed for pickled mols to preserve their properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from tmd.constants import DEFAULT_FF, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, KCAL_TO_KJ
from tmd.fe.absolute.free_energy import RestraintParams
from tmd.fe.free_energy import HREXParams, MDParams, compute_total_ns
from tmd.fe.plots import plot_forward_and_reverse_dg
from tmd.fe.rbfe import BATCH_MODE_ENV_VAR, DEFAULT_NUM_WINDOWS, HREXSimulationResult
from tmd.fe.septop import SepTopResult, estimate_septop
from tmd.fe.utils import get_mol_name, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.md.builders import build_protein_system, build_water_system, compute_solvent_box_size, verify_pdb_structure
from tmd.parallel.client import AbstractFileClient, CUDAPoolClient, FileClient, SerialClient
from tmd.parallel.utils import get_gpu_count

SOLVENT_LEG = "solvent"
COMPLEX_LEG = "complex"

# Map the leg name used on the command line to the SepTop phase argument.
LEG_TO_PHASE = {SOLVENT_LEG: "aqueous", COMPLEX_LEG: "complex"}


def run_septop_leg(
    file_client: AbstractFileClient,
    edge_path: Path,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    leg_name: str,
    ff: Forcefield,
    pdb_path: str | None,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    rst_params: RestraintParams,
    decharge_lambda: float,
    eps_scale_lambda: float,
    w_lambda: float,
    enable_batching: bool,
    write_trajectories: bool,
    force_overwrite: bool,
    water_box_size: float = 4.0,
) -> dict[str, Any]:
    """Run a SepTop leg (solvent or complex).

    Stores results using the file_client under a directory named for the leg.

    Stores the following files:

    * results.npz - Predictions, corrections, overlaps and the number of windows
    * lambda*_traj.npz - Endstate trajectories (if write_trajectories is True)
    * final_pairbar_result.pkl - Pickled copy of the final PairBarResult object
    * host_config.pkl - Pickled HostConfig
    * dg_errors.png - PNG of the dg errors
    * overlap_summary.png - PNG of the pair bar overlap between windows
    * forward_and_reverse_dg.png - PNG of forward and reverse dG for convergence
    * hrex_transition_matrix.png - PNG of the transition matrix plot (HREX only)
    * hrex_replica_state_distribution_heatmap.png - HREX replica state heatmap (HREX only)

    Parameters
    ----------
    file_client : AbstractFileClient
        File client for storing results of the simulation.
    edge_path : Path
        Path to the directory that contains the edge-level data.
    mol_a, mol_b : Chem.Mol
        The two molecules. mol_a is decoupled from lambda=0 to lambda=1 while
        mol_b is coupled over the same interval.
    leg_name : str
        Name of the leg to run. Either "solvent" or "complex".
    ff : Forcefield
        Forcefield.
    pdb_path : str, optional
        Path to a PDB file, required when running the "complex" leg.
    md_params : MDParams
        Parameters for the SepTop simulation.
    n_windows : int
        Maximum number of windows to generate during bisection.
    min_overlap : float
        Minimum overlap used during bisection.
    rst_params : RestraintParams
        Boresch restraint force constants (complex leg) and the zero-length
        tether strength (solvent leg).
    decharge_lambda, eps_scale_lambda, w_lambda : float
        Alchemical schedule knobs. Decharging happens over the symmetric
        interval [decharge_lambda, 1 - decharge_lambda], LJ epsilon scaling over
        [0, eps_scale_lambda], and the W-coordinate (4D) shift over
        [w_lambda, 1.0].
    enable_batching : bool
        Batch the per-window MD during bisection. Enables batching for both the
        non-HREX path (via ``estimate_septop``) and the HREX path (via the
        ``TMD_BATCH_MODE`` environment variable).
    write_trajectories : bool
        Whether to write out the endstate trajectories.
    force_overwrite : bool
        If results already exist, overwrite them; otherwise skip the leg.
    water_box_size : float
        Size of the water box for the solvent leg. Should be large enough to
        avoid molecules interacting with copies of themselves across PBCs. Use
        ``tmd.md.builders.compute_solvent_box_size`` to pick an appropriate size.

    Returns
    -------
    Summary data
        Data contained in results.npz. Includes ``pred_dg`` (kJ/mol).
    """
    Path(file_client.full_path(edge_path)).mkdir(parents=True, exist_ok=True)
    leg_path = Path(edge_path) / leg_name
    Path(file_client.full_path(leg_path)).mkdir(parents=True, exist_ok=True)
    results_path = Path(file_client.full_path(leg_path / "results.npz"))
    if not force_overwrite and results_path.is_file():
        print(f"Skipping existing leg {leg_name}: {get_mol_name(mol_a)} / {get_mol_name(mol_b)}")
        return dict(np.load(results_path))

    # Store top level data
    with open(file_client.full_path(edge_path / "md_params.pkl"), "wb") as ofs:
        pickle.dump(md_params, ofs)
    with open(file_client.full_path(edge_path / "ff.py"), "w") as ofs:
        ofs.write(ff.serialize())
    with Chem.SDWriter(file_client.full_path(edge_path / "mols.sdf")) as writer:
        writer.write(mol_a)
        writer.write(mol_b)

    np.random.seed(md_params.seed)

    # Batching in the HREX bisection path is controlled by an environment
    # variable rather than a function argument, so set it here (inside the
    # worker process) to enable batching throughout when requested.
    if enable_batching:
        os.environ[BATCH_MODE_ENV_VAR] = "on"

    if leg_name == SOLVENT_LEG:
        host_config = build_water_system(water_box_size, ff.water_ff, mols=[mol_a, mol_b])
    elif leg_name == COMPLEX_LEG:
        assert pdb_path is not None, "No pdb data provided"
        host_config = build_protein_system(
            str(Path(pdb_path).expanduser()), DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b]
        )
    else:
        assert 0, f"Invalid leg: {leg_name}"

    prefix = f"{leg_name}_{get_mol_name(mol_a)}_{get_mol_name(mol_b)}"

    start = time.perf_counter()
    result: SepTopResult = estimate_septop(
        mol_a,
        mol_b,
        ff,
        host_config,
        prefix,
        md_params,
        n_windows,
        min_overlap,
        rst_params,
        decharge_lambda=decharge_lambda,
        eps_scale_lambda=eps_scale_lambda,
        w_lambda=w_lambda,
        enable_batching=enable_batching,
        phase=LEG_TO_PHASE[leg_name],
    )
    took = time.perf_counter() - start

    res = result.sim_result

    # Raw dG is the sum of the per-window dGs. The complex leg additionally
    # carries the analytical Boresch restraint correction; the corrected leg dG
    # is raw - (correction_a - correction_b). The solvent leg correction is 0.
    raw_dg = float(np.sum(res.final_result.dGs))
    correction = float(result.correction)
    pred_dg = raw_dg - correction
    pred_dg_err = float(np.linalg.norm(res.final_result.dG_errs))
    print(
        " | ".join(
            [
                f"{get_mol_name(mol_a)} / {get_mol_name(mol_b)} (kJ/mol)",
                f"{leg_name} {pred_dg:.2f} +- {pred_dg_err:.2f}",
                f"correction {correction:.2f}",
                f"{took:.0f} Seconds",
            ]
        ),
    )

    summary_data: dict[str, Any] = {
        "time": took,
        "total_ns": compute_total_ns(res, md_params),
        "pred_dg": pred_dg,
        "pred_dg_err": pred_dg_err,
        "raw_dg": raw_dg,
        "correction": correction,
        "correction_a": result.correction_a,
        "correction_b": result.correction_b,
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

    file_client.store(leg_path / "host_config.pkl", pickle.dumps(host_config))

    if isinstance(res, HREXSimulationResult):
        file_client.store(leg_path / "hrex_transition_matrix.png", res.hrex_plots.transition_matrix_png)
        file_client.store(
            leg_path / "hrex_replica_state_distribution_heatmap.png",
            res.hrex_plots.replica_state_distribution_heatmap_png,
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
    parser = ArgumentParser(description="Run the SepTop legs for a pair of molecules")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
    parser.add_argument("--mol_a", help="Name of mol a in sdf_path", required=True)
    parser.add_argument("--mol_b", help="Name of mol b in sdf_path", required=True)
    parser.add_argument("--pdb_path", help="Path to pdb file containing structure")
    parser.add_argument("--n_eq_steps", default=200_000, type=int, help="Number of steps to perform equilibration")
    parser.add_argument("--n_frames", default=2000, type=int, help="Number of frames to simulate")
    parser.add_argument("--steps_per_frame", default=400, type=int, help="Steps per frame")
    parser.add_argument(
        "--n_windows", default=DEFAULT_NUM_WINDOWS, type=int, help="Max number of windows from bisection"
    )
    parser.add_argument("--min_overlap", default=0.667, type=float, help="Overlap to target in bisection")
    parser.add_argument(
        "--target_overlap", default=0.667, type=float, help="Overlap to optimize final HREX schedule to"
    )
    parser.add_argument("--seed", default=2025, type=int, help="Seed")
    parser.add_argument("--legs", default=[SOLVENT_LEG, COMPLEX_LEG], nargs="+", choices=[SOLVENT_LEG, COMPLEX_LEG])
    parser.add_argument("--forcefield", default=DEFAULT_FF)
    parser.add_argument(
        "--n_gpus", default=None, type=int, help="Number of GPUs to use, defaults to all GPUs if not provided"
    )
    parser.add_argument("--kb", default=500.0, type=float, help="Bond restraint force constant (kcal/mol/nm^2)")
    parser.add_argument("--ka", default=200.0, type=float, help="Angle restraint force constant (kcal/mol/rad^2)")
    parser.add_argument("--kd", default=10.0, type=float, help="Dihedral restraint force constant (kcal/mol)")
    parser.add_argument(
        "--decharge_lambda",
        default=0.25,
        type=float,
        help="Ligand decharging happens over the symmetric interval [decharge_lambda, 1 - decharge_lambda]",
    )
    parser.add_argument(
        "--eps_scale_lambda",
        default=0.25,
        type=float,
        help="LJ epsilon scaling happens over the interval [0, eps_scale_lambda]",
    )
    parser.add_argument(
        "--w_lambda",
        default=0.5,
        type=float,
        help="W-coordinate (4D) decoupling happens over the interval [w_lambda, 1.0]",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to output results, else generates a directory based on the time"
    )
    parser.add_argument(
        "--solvent_padding", default=1.0, type=float, help="Padding to add to solvent boxes, defaults to 1.0 nanometer."
    )
    parser.add_argument(
        "--serial", action="store_true", help="Run without spawning subprocesses, useful when wanting to profile."
    )
    parser.add_argument(
        "--enable_batching",
        action="store_true",
        help="Batch the per-window MD during bisection (both HREX and non-HREX paths).",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite existing predictions, otherwise will skip the completed legs",
    )
    args = parser.parse_args()

    if COMPLEX_LEG in args.legs:
        assert args.pdb_path is not None, "Must provide PDB to run complex leg"

    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    mol_a = mols_by_name[args.mol_a]
    mol_b = mols_by_name[args.mol_b]

    water_box_size = 4.0
    if SOLVENT_LEG in args.legs:
        water_box_size = compute_solvent_box_size([mol_a, mol_b], padding=args.solvent_padding)

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"septop_{date_str}_{args.mol_a}_{args.mol_b}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_client = FileClient(dest_dir)

    ff = Forcefield.load_from_file(args.forcefield)

    if args.pdb_path is not None:
        verify_pdb_structure(args.pdb_path, ff)

    md_params = MDParams(
        n_eq_steps=args.n_eq_steps,
        n_frames=args.n_frames,
        steps_per_frame=args.steps_per_frame,
        seed=args.seed,
        hrex_params=HREXParams(optimize_target_overlap=args.target_overlap),
    )

    rst_params = RestraintParams(kb=args.kb, ka=args.ka, kd=args.kd)

    num_gpus = args.n_gpus
    if num_gpus is None:
        num_gpus = get_gpu_count()

    if args.serial:
        pool = SerialClient()
    else:
        pool = CUDAPoolClient(num_gpus)
    pool.verify()

    futures = []
    for leg_name in args.legs:
        fut = pool.submit(
            run_septop_leg,
            file_client,
            Path(""),  # Empty path, as the file client is prefixed on the top level directory
            mol_a,
            mol_b,
            leg_name,
            ff,
            args.pdb_path,
            md_params,
            args.n_windows,
            args.min_overlap,
            rst_params,
            args.decharge_lambda,
            args.eps_scale_lambda,
            args.w_lambda,
            args.enable_batching,
            True,  # Always write out the trajectories
            args.force_overwrite,
            water_box_size=water_box_size,
        )
        futures.append(fut)

    leg_results: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for leg, fut in zip(args.legs, futures):
        res = fut.result()
        leg_results[(get_mol_name(mol_a), get_mol_name(mol_b))][leg] = res

    summaries = leg_results[(get_mol_name(mol_a), get_mol_name(mol_b))]
    if COMPLEX_LEG in summaries and SOLVENT_LEG in summaries:
        pred_ddg = summaries[COMPLEX_LEG]["pred_dg"] - summaries[SOLVENT_LEG]["pred_dg"]
        pred_ddg_err = float(
            np.linalg.norm([summaries[COMPLEX_LEG]["pred_dg_err"], summaries[SOLVENT_LEG]["pred_dg_err"]])
        )
        print(
            f"{get_mol_name(mol_a)} / {get_mol_name(mol_b)} "
            f"pred_ddg {pred_ddg / KCAL_TO_KJ:.2f} +- {pred_ddg_err / KCAL_TO_KJ:.2f} kcal/mol"
        )


if __name__ == "__main__":
    main()
