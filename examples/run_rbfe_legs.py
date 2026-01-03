# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025 Forrest York
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

from rbfe_common import COMPLEX_LEG, SOLVENT_LEG, VACUUM_LEG, run_rbfe_leg, write_result_csvs

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_FF
from tmd.fe import atom_mapping
from tmd.fe.free_energy import HREXParams, LocalMDParams, MDParams, RESTParams, WaterSamplingParams
from tmd.fe.rbfe import DEFAULT_NUM_WINDOWS
from tmd.fe.utils import get_mol_name, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.md.exchange.utils import get_radius_of_mol_pair
from tmd.parallel.client import CUDAPoolClient, FileClient, SerialClient
from tmd.parallel.utils import get_gpu_count


def main():
    parser = ArgumentParser(description="Run the RBFE legs for a pair of molecules")
    parser.add_argument("--sdf_path", help="Path to sdf file containing mols", required=True)
    parser.add_argument("--mol_a", help="Name of mol a in sdf_path", required=True)
    parser.add_argument("--mol_b", help="Name of mol b in sdf_path", required=True)
    parser.add_argument("--pdb_path", help="Path to pdb file containing structure")
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
    parser.add_argument("--legs", default=[SOLVENT_LEG, COMPLEX_LEG, VACUUM_LEG], nargs="+")
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
    parser.add_argument("--local_md_radius", default=2.0, type=float, help="Local MD radius")
    parser.add_argument(
        "--local_md_steps",
        default=0,
        type=int,
        help="Number of steps to run with Local MD. Must be less than or equal to --steps_per_frame. If set to 0, no local MD is run",
    )
    parser.add_argument(
        "--serial", action="store_true", help="Run without spawning subprocesses, useful when wanting to profile."
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
    args = parser.parse_args()

    if "complex" in args.legs:
        assert args.pdb_path is not None, "Must provide PDB to run complex leg"

    mols_by_name = read_sdf_mols_by_name(args.sdf_path)
    np.random.seed(args.seed)

    mol_a = mols_by_name[args.mol_a]
    mol_b = mols_by_name[args.mol_b]

    output_dir = args.output_dir
    if output_dir is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_dir = f"rbfe_{date_str}_{args.mol_a}_{args.mol_b}"
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    file_client = FileClient(dest_dir)

    ff = Forcefield.load_from_file(args.forcefield)

    mol_radius = get_radius_of_mol_pair(mol_a, mol_b)

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
            args.local_md_steps, k=args.local_md_k, min_radius=args.local_md_radius, max_radius=args.local_md_radius
        )
        if args.local_md_steps > 0
        else None,
        water_sampling_params=WaterSamplingParams(radius=mol_radius + args.water_sampling_padding),
    )

    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]

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
            run_rbfe_leg,
            file_client,
            Path(""),  # Empty path, as the file client is prefixed on the top level directory
            mol_a,
            mol_b,
            core,
            leg_name,
            ff,
            args.pdb_path,
            md_params,
            args.n_windows,
            args.min_overlap,
            True,  # Always write out the trajectories
            args.force_overwrite,
        )
        futures.append(fut)
    leg_results = defaultdict(dict)
    for leg, fut in zip(args.legs, futures):
        res = fut.result()
        leg_results[(get_mol_name(mol_a), get_mol_name(mol_b))][leg] = res
    write_result_csvs(file_client, mols_by_name, leg_results, args.experimental_field, args.experimental_units)


if __name__ == "__main__":
    main()
