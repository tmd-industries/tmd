import pickle
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import cached_property, partial
from pathlib import Path

import jax

# Enable 64 bit jax
jax.config.update("jax_enable_x64", True)


import numpy as np
from rdkit import Chem

# This is needed for pickled mols to preserve their properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from tmd.constants import DEFAULT_FF, DEFAULT_TEMP
from tmd.fe.free_energy import (
    HREXParams,
    HREXSimulationResult,
    LocalMDParams,
    MDParams,
    RESTParams,
)
from tmd.fe.lambda_schedule import bisection_lambda_schedule
from tmd.fe.rbfe import (
    DEFAULT_NUM_WINDOWS,
    estimate_relative_free_energy_bisection_hrex_impl,
    optimize_initial_state_from_pre_optimized,
    setup_initial_state,
    setup_initial_states,
)
from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.fe.utils import get_mol_name, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.md.builders import HostConfig, build_water_system
from tmd.parallel.client import AbstractFileClient, CUDAMPSPoolClient, FileClient, iterate_completed_futures
from tmd.parallel.utils import get_gpu_count


class IdentitySTRest(SingleTopologyREST):
    # Overriden to make all atoms part of the REST region (probably too aggressive)
    @cached_property
    def base_rest_region_atom_idxs(self) -> set[int]:
        """Returns the set of indices of atoms in the combined ligand that are in the REST region.

        Here the REST region is defined to include combined ligand atoms involved in bond, angle, or improper torsion
        interactions that differ in the end states. Note that proper torsions are omitted from this heuristic as this
        tends to result in larger REST regions than seem desirable.
        """

        aligned_potentials = [
            self.aligned_bond,
            self.aligned_angle,
            self.aligned_improper,
        ]

        idxs = {
            int(idx)
            for aligned in aligned_potentials
            for idxs in aligned.idxs
            for idx in idxs  # type: ignore[attr-defined]
        }

        # Ensure all dummy atoms are included in the REST region
        idxs |= self.get_dummy_atoms_a()
        idxs |= self.get_dummy_atoms_b()

        return idxs


def simulate_solvent_rest(
    st: IdentitySTRest, md_params: MDParams, host_config: HostConfig, n_windows: int, min_overlap: float
) -> HREXSimulationResult:
    # You could also do 0.0 -> 1.0 to get twice the endstates
    lambda_interval = (0.0, 0.5)
    lambda_min, lambda_max = lambda_interval[0], lambda_interval[1]
    temperature = DEFAULT_TEMP

    lambda_grid = bisection_lambda_schedule(n_windows, lambda_interval=lambda_interval)
    initial_states = setup_initial_states(
        st, host_config, temperature, lambda_grid, md_params.seed, False, min_cutoff=0.7
    )

    make_initial_state_fn = partial(
        setup_initial_state,
        st,
        host=host_config,
        temperature=temperature,
        seed=md_params.seed,
        verify_constraints=False,  # Speeds up construction of initial state
    )

    make_optimized_initial_state_fn = partial(
        optimize_initial_state_from_pre_optimized,
        optimized_initial_states=initial_states,
    )

    prefix = get_mol_name(st.mol_a)

    return estimate_relative_free_energy_bisection_hrex_impl(
        temperature,
        lambda_min,
        lambda_max,
        md_params,
        n_windows,
        make_initial_state_fn,
        make_optimized_initial_state_fn,
        prefix,
        min_overlap,
    )


def simulate_rest(
    file_client: AbstractFileClient,
    output_path: Path,
    mol: Chem.Mol,
    ff: Forcefield,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
):
    Path(file_client.full_path(output_path)).mkdir(parents=True, exist_ok=True)
    np.random.seed(md_params.seed)
    n_ligand_atoms = mol.GetNumAtoms()
    core = np.tile(np.arange(n_ligand_atoms)[:, None], (1, 2))
    assert md_params.hrex_params is not None and md_params.hrex_params.rest_params is not None
    st = IdentitySTRest(
        mol,
        mol,
        core,
        ff,
        md_params.hrex_params.rest_params.max_temperature_scale,
        temperature_scale_interpolation=md_params.hrex_params.rest_params.temperature_scale_interpolation,
    )

    host_config = build_water_system(4.0, ff.water_ff, mols=[mol], box_margin=0.1)
    start = time.perf_counter()
    res = simulate_solvent_rest(st, md_params, host_config, n_windows, min_overlap)
    took = time.perf_counter() - start

    pred_dg = float(np.sum(res.final_result.dGs))
    pred_dg_err = float(np.linalg.norm(res.final_result.dG_errs))
    print(
        " | ".join(
            [
                f"{get_mol_name(mol)} (kJ/mol)",
                f"{pred_dg:.2f} +- {pred_dg_err:.2f}",
                f"{took:.0f} Seconds",
            ]
        ),
    )

    summary_data = {
        "time": took,
        "pred_dg": pred_dg,
        "pred_dg_err": pred_dg_err,
        "overlaps": res.final_result.overlaps,
        "n_windows": len(res.final_result.initial_states),
    }
    if isinstance(res, HREXSimulationResult):
        summary_data["bisected_windows"] = len(res.intermediate_results[-1].initial_states)
        summary_data["normalized_kl_divergence"] = res.hrex_diagnostics.normalized_kl_divergence

    np.savez_compressed(Path(file_client.full_path(output_path / "summary.npz")), **summary_data)  # type:ignore

    np.savez_compressed(
        file_client.full_path(output_path / "endstate_traj.npz"),
        coords=np.array(res.trajectories[0].frames),
        boxes=np.asarray(res.trajectories[0].boxes),
    )
    file_client.store(output_path / "host_config.pkl", pickle.dumps(host_config))

    file_client.store(output_path / "hrex_transition_matrix.png", res.hrex_plots.transition_matrix_png)
    file_client.store(
        output_path / "hrex_replica_state_distribution_heatmap.png",
        res.hrex_plots.replica_state_distribution_heatmap_png,
    )


def main():
    parser = ArgumentParser(description="Run REST on compounds in solvent")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
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
        "--rest_max_temperature_scale",
        default=3.0,
        type=float,
        help="Maximum scale factor for the effective temperature of REST-softened interactions. Setting to 1.0 effectively disables REST.",
    )
    parser.add_argument(
        "--rest_temperature_scale_interpolation",
        default="exponential",
        type=str,
        help="Functional form to use for temperature scale interpolation in REST",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Directory to output results, else generates a directory based on the time"
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
    args = parser.parse_args()

    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"rest_sampling_{date_str}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

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
    pool = pool = CUDAMPSPoolClient(num_gpus, workers_per_gpu=args.mps_workers)
    pool.verify()
    futures = []
    for name, mol in mols_by_name.items():
        md_params = MDParams(
            n_eq_steps=args.n_eq_steps,
            n_frames=args.n_frames,
            steps_per_frame=args.steps_per_frame,
            seed=args.seed,
            hrex_params=HREXParams(
                optimize_target_overlap=args.target_overlap,
                rest_params=RESTParams(args.rest_max_temperature_scale, args.rest_temperature_scale_interpolation),
            ),
            local_md_params=LocalMDParams(
                args.local_md_steps,
                k=args.local_md_k,
                min_radius=args.local_md_radius,
                max_radius=args.local_md_radius,
                freeze_reference=not args.local_md_free_reference,
            )
            if args.local_md_steps > 0
            else None,
        )

        fut = pool.submit(
            simulate_rest,
            file_client,
            Path(name),
            mol,
            ff,
            md_params,
            args.n_windows,
            args.min_overlap,
        )
        futures.append(fut)
    for fut in iterate_completed_futures(futures):
        try:
            fut.result()
        except Exception as e:
            print(f"Failure: {e}")
            continue


if __name__ == "__main__":
    main()
