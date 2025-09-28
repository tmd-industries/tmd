import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import jax

# Enable 64 bit jax
jax.config.update("jax_enable_x64", True)


import numpy as np
from rdkit import Chem

# This is needed for pickled mols to preserve their properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

from rbfe_common import run_rbfe_leg

from tmd.constants import DEFAULT_FF
from tmd.fe.free_energy import (
    EarlyTerminationParams,
    HREXParams,
    LocalMDParams,
    MDParams,
    RESTParams,
    WaterSamplingParams,
)
from tmd.fe.rbfe import (
    DEFAULT_NUM_WINDOWS,
)
from tmd.fe.utils import get_mol_name, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.md.exchange.utils import get_radius_of_mol_pair
from tmd.parallel.client import CUDAMPSPoolClient, FileClient, iterate_completed_futures
from tmd.parallel.utils import get_gpu_count


def main():
    parser = ArgumentParser(description="Run RBFE for a set of compounds given a JSON graph topology")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
    parser.add_argument("--graph_json", help="Path to JSON file containing mols", required=True)
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
    parser.add_argument("--legs", default=["vacuum", "solvent", "complex"], nargs="+")
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
    parser.add_argument(
        "--early_term_interval",
        default=0,
        type=int,
        help="Interval to collect samples for early termination, if zero disables early termination",
    )
    parser.add_argument(
        "--early_term_threshold",
        default=0.25,
        type=float,
        help="Max difference in estimates before allowing termination",
    )
    parser.add_argument(
        "--store_trajectories",
        action="store_true",
        help="Store the trajectories of the edges. Can take up a large amount of space",
    )
    args = parser.parse_args()

    if "complex" in args.legs:
        assert args.pdb_path is not None, "Must provide PDB to run complex leg"

    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    with open(Path(args.graph_json).expanduser()) as ifs:
        edges_data = json.load(ifs)
    assert all(isinstance(x, dict) for x in edges_data)

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"rbfe_graph_{date_str}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with Chem.SDWriter(dest_dir / "mols.sdf") as writer:
        for mol in mols_by_name.values():
            writer.write(mol)
    with open(dest_dir / "edges.json", "w") as ofs:
        json.dump(edges_data, ofs)

    file_client = FileClient(dest_dir)

    ff = Forcefield.load_from_file(args.forcefield)

    num_gpus = args.n_gpus
    if num_gpus is None:
        num_gpus = get_gpu_count()

    pool = CUDAMPSPoolClient(num_gpus, workers_per_gpu=args.mps_workers)
    pool.verify()
    futures = []

    for edge in edges_data:
        mol_a = mols_by_name[edge["mol_a"]]
        mol_b = mols_by_name[edge["mol_b"]]
        core = None
        if "core" in edge:
            core = np.array(edge["core"])

        mol_radius = get_radius_of_mol_pair(mol_a, mol_b)

        md_params = MDParams(
            n_eq_steps=args.n_eq_steps,
            n_frames=args.n_frames,
            steps_per_frame=args.steps_per_frame,
            seed=args.seed,
            hrex_params=HREXParams(
                optimize_target_overlap=args.target_overlap,
                rest_params=RESTParams(args.rest_max_temperature_scale, args.rest_temperature_scale_interpolation),
                early_termination_params=EarlyTerminationParams(
                    args.early_term_threshold, interval=args.early_term_interval
                )
                if args.early_term_interval > 0
                else None,
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
            water_sampling_params=WaterSamplingParams(radius=mol_radius + args.water_sampling_padding),
        )

        for leg_name in args.legs:
            fut = pool.submit(
                run_rbfe_leg,
                file_client,
                Path(f"{get_mol_name(mol_a)}_{get_mol_name(mol_b)}"),
                mol_a,
                mol_b,
                core,
                leg_name,
                ff,
                args.pdb_path,
                md_params,
                args.n_windows,
                args.min_overlap,
                args.store_trajectories,
            )
            futures.append(fut)
    for fut in iterate_completed_futures(futures):
        fut.result()


if __name__ == "__main__":
    main()
