from argparse import ArgumentParser
from dataclasses import dataclass, replace

import jax
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Enable 64 bit jax, only to remove the noise of the Pymbar warning
jax.config.update("jax_enable_x64", True)

import time
from datetime import datetime

from tmd import constants
from tmd.fe.free_energy import (
    LocalMDParams,
    MDParams,
    sample_with_context_iter,
)
from tmd.fe.model_utils import apply_hmr
from tmd.fe.rbfe import setup_in_env
from tmd.fe.single_topology import SingleTopology
from tmd.fe.utils import get_romol_conf
from tmd.ff import Forcefield
from tmd.lib import Context, LangevinIntegrator, MonteCarloBarostat
from tmd.md import builders
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.parallel.client import CUDAMPSPoolClient
from tmd.potentials import HarmonicBond
from tmd.potentials.potential import get_bound_potential_by_type
from tmd.testsystems.dhfr import setup_dhfr
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass
class MDSystemData:
    x0: NDArray
    box0: NDArray
    masses: NDArray
    pots: list
    intg: LangevinIntegrator
    baro: MonteCarloBarostat
    params: MDParams
    ligand_idxs: NDArray


def run_steps(system_data):
    bps = []

    for potential in system_data.pots:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    movers = [system_data.baro.impl(bps)]

    ctxt = Context(
        system_data.x0,
        np.zeros_like(system_data.x0),
        system_data.box0,
        system_data.intg.impl(),
        bps,
        movers=movers,
    )
    start = time.perf_counter()
    for _ in sample_with_context_iter(ctxt, system_data.params, constants.DEFAULT_TEMP, system_data.ligand_idxs, 1):
        pass

    return time.perf_counter() - start


def main():
    parser = ArgumentParser()
    parser.add_argument("--processes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--local_md", action="store_true")
    parser.add_argument("--local_md_k", default=10_000.0, type=float)
    parser.add_argument("--local_md_rad", default=2.0, type=float)
    parser.add_argument("--local_md_free_reference", action="store_true", default=False)
    parser.add_argument("--steps", default=2000, type=int)
    parser.add_argument("--frames", default=10, type=int)
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument(
        "--active_thread_percentage",
        default=None,
        type=float,
        help="Specify CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for each MPS worker",
    )
    parser.add_argument("--system", default="dhfr", choices=["dhfr", "hif2a-rbfe", "hif2a"])
    args = parser.parse_args()

    output_suffix = args.output_suffix
    if output_suffix is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_suffix = f"_{date_str}"

    seed = 1234
    temperature = constants.DEFAULT_TEMP
    pressure = constants.DEFAULT_PRESSURE
    ff = Forcefield.load_default()

    if args.system == "dhfr":
        host_fns, host_masses, x0, box0 = setup_dhfr()
        # Treat the entire system as the ligand indices
        ligand_idxs = np.arange(len(x0))
        harmonic_bond_potential = get_bound_potential_by_type(host_fns, HarmonicBond).potential
        bond_list = get_bond_list(harmonic_bond_potential)
        hmr_masses = apply_hmr(host_masses, bond_list)

        group_idxs = get_group_indices(bond_list, len(hmr_masses))
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            25,  # Run Barostat every 25 steps if not running local MD
            seed,
        )
    elif args.system == "hif2a-rbfe":
        mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

        with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as protein_path:
            host_config = builders.load_pdb_system(str(protein_path), ff.protein_ff, ff.water_ff, box_margin=0.1)

        lamb = 0.0

        st = SingleTopology(mol_a, mol_b, core, ff)

        conf_a = get_romol_conf(mol_a)
        conf_b = get_romol_conf(mol_b)

        ligand_conf = st.combine_confs(conf_a, conf_b, lamb)

        x0, hmr_masses, host_fns, baro = setup_in_env(st, host_config, ligand_conf, lamb, temperature, seed)
        box0 = host_config.box
        ligand_idxs = np.arange(len(host_config.conf), len(x0))
    elif args.system == "hif2a":
        with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as protein_path:
            host_config = builders.load_pdb_system(str(protein_path), ff.protein_ff, ff.water_ff)

        host_fns, host_masses, x0, box0 = (
            host_config.host_system.get_U_fns(),
            host_config.masses,
            host_config.conf,
            host_config.box,
        )
        ligand_idxs = np.arange(len(x0))
        harmonic_bond_potential = get_bound_potential_by_type(host_fns, HarmonicBond).potential
        bond_list = get_bond_list(harmonic_bond_potential)
        hmr_masses = apply_hmr(host_masses, bond_list)

        group_idxs = get_group_indices(bond_list, len(hmr_masses))
        baro = MonteCarloBarostat(
            x0.shape[0],
            pressure,
            temperature,
            group_idxs,
            25,  # Run Barostat every 25 steps if not running local MD
            seed,
        )
    else:
        assert 0, f"Unknown system {args.system}"

    v0 = np.zeros_like(x0)

    dt = 2.5e-3

    intg = LangevinIntegrator(temperature, dt, 1.0, np.asarray(hmr_masses), seed)

    bps = []

    for potential in host_fns:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    movers = []

    movers.append(baro.impl(bps))

    ctxt = Context(
        x0,
        v0,
        box0,
        intg.impl(),
        bps,
        movers=movers,
    )

    num_samples = np.max(args.processes)

    # Get a range of samples that don't have the same neighborlist to add some variation
    # in the runtime (IE Neighborlist, etc)
    xs, boxes = ctxt.multiple_steps(400 * num_samples, store_x_interval=num_samples)

    md_params = MDParams(
        n_eq_steps=0,
        n_frames=args.frames,
        steps_per_frame=args.steps,
        seed=seed,
        local_md_params=LocalMDParams(
            local_steps=args.steps,
            k=args.local_md_k,
            min_radius=args.local_md_rad,
            max_radius=args.local_md_rad,
            freeze_reference=not args.local_md_free_reference,
        )
        if args.local_md
        else None,
    )

    state = MDSystemData(
        x0,
        box0,
        np.array(hmr_masses),
        host_fns,
        intg,
        baro,
        md_params,
        ligand_idxs,
    )

    ns_per_day_results = []
    for proc in args.processes:
        pool = CUDAMPSPoolClient(1, workers_per_gpu=proc, active_thread_usage_per_worker=args.active_thread_percentage)
        subset_xs = xs[:proc]
        subset_boxes = boxes[:proc]
        futures = [pool.submit(run_steps, replace(state, x0=x, box0=box)) for x, box in zip(subset_xs, subset_boxes)]
        results = [fut.result() for fut in futures]
        steps_per_second = (args.steps * args.frames) / np.array(results)
        total_steps_per_second = np.sum(steps_per_second)
        ns_per_day = total_steps_per_second * SECONDS_PER_DAY * dt * 1e-3
        print(f"{proc} Proccess: {ns_per_day} Ns per day")
        ns_per_day_results.append(ns_per_day)

    plt.plot(args.processes, ns_per_day_results)
    plt.ylabel("ns per day")
    plt.xlabel("Processes")
    plt.title(
        f"{args.system} MPS\nPeak {max(ns_per_day_results):.1f}ns/day"
        + (f"\nLocal MD Rad {args.local_md_rad}" if args.local_md else "")
    )
    plt.tight_layout()
    plt.savefig(f"{args.system}_ns_per_day{output_suffix}.png", dpi=150)
    plt.clf()

    plt.plot(args.processes, np.array(ns_per_day_results) / ns_per_day_results[0])
    plt.ylabel("Factor improvement")
    plt.xlabel("Processes")
    plt.title(f"{args.system} MPS" + (f"\nLocal MD Rad {args.local_md_rad}" if args.local_md else ""))
    plt.tight_layout()
    plt.savefig(f"{args.system}_factor_improvement{output_suffix}.png", dpi=150)
    plt.clf()


if __name__ == "__main__":
    main()
