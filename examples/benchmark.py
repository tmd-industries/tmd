from argparse import ArgumentParser
from dataclasses import dataclass, replace

import jax
import matplotlib.pyplot as plt
import numpy as np

# Enable 64 bit jax, only to remove the noise of the Pymbar warning
jax.config.update("jax_enable_x64", True)

import time
from datetime import datetime

from tmd import constants
from tmd.fe import atom_mapping
from tmd.fe.free_energy import (
    HREXParams,
    InitialState,
    LocalMDParams,
    MDParams,
    get_batched_context,
    get_context,
    initial_state_from_host_config,
    run_sims_hrex,
    sample_with_context_iter,
)
from tmd.fe.rbfe import setup_initial_state, setup_initial_states, setup_optimized_host
from tmd.fe.single_topology import SingleTopology
from tmd.fe.utils import read_sdf
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.md.thermostat.utils import sample_velocities
from tmd.parallel.client import CUDAMPSPoolClient
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass
class MDSystemData:
    initial_states: list[InitialState]
    params: MDParams
    hrex_states: int


def run_mps_steps(system_data):
    """Runs Simulations in parallel through Nvidia's MPS"""
    assert len(system_data.initial_states) == 1
    state = system_data.initial_states[0]
    ctxt = get_context(state, system_data.params)
    start = time.perf_counter()

    if system_data.params.hrex_params is None:
        ctxt = get_context(state, system_data.params)
        start = time.perf_counter()
        for _ in sample_with_context_iter(ctxt, system_data.params, state.integrator.temperature, state.ligand_idxs, 1):
            pass
    else:
        states = [
            replace(
                state,
                v0=sample_velocities(
                    state.integrator.masses, state.integrator.temperature, system_data.params.seed + i
                ),
            )
            for i in range(system_data.hrex_states)
        ]
        start = time.perf_counter()
        run_sims_hrex(states, system_data.params, print_diagnostics_interval=None, batch_simulations=False)
    return time.perf_counter() - start


def run_batched_steps(system_data):
    """Runs the simulations in the batch mode, which removes the need for MPS to improve throughput"""
    states = system_data.initial_states
    if system_data.params.hrex_params is None:
        if len(states) > 1:
            ctxt = get_batched_context(states, system_data.params)
        else:
            ctxt = get_context(states[0], system_data.params)
        start = time.perf_counter()
        for _ in sample_with_context_iter(
            ctxt,
            system_data.params,
            states[0].integrator.temperature,
            states[0].ligand_idxs,
            system_data.params.n_frames,
        ):
            pass
    else:
        start = time.perf_counter()
        run_sims_hrex(states, system_data.params, print_diagnostics_interval=None, batch_simulations=True)
    return time.perf_counter() - start


def generate_equilibrium_frames(initial_states, md_params, eq_steps, dt_fs):
    num_samples = len(initial_states)
    if all(state.lamb == initial_states[0].lamb for state in initial_states):
        equil_params = replace(
            md_params,
            n_eq_steps=eq_steps,
            n_frames=num_samples,
            local_md_params=None,
            steps_per_frame=int(1000 / dt_fs),  # Aim for 1 picosecond between frames
        )
        ctxt = get_context(initial_states[0], equil_params)
        xs, boxes, _ = next(
            sample_with_context_iter(
                ctxt,
                equil_params,
                initial_states[0].integrator.temperature,
                initial_states[0].ligand_idxs,
                equil_params.n_frames,
            )
        )
    else:
        equil_params = replace(
            md_params,
            n_eq_steps=eq_steps,
            n_frames=1,
            local_md_params=None,
            steps_per_frame=int(1000 / dt_fs),  # Aim for 1 picosecond between frames
        )
        xs = np.empty((num_samples, *initial_states[0].x0.shape))
        boxes = np.empty((num_samples, 3, 3))
        for i, state in enumerate(initial_states):
            ctxt = get_context(state, equil_params)
            x, b, _ = next(
                sample_with_context_iter(
                    ctxt,
                    equil_params,
                    initial_states[0].integrator.temperature,
                    initial_states[0].ligand_idxs,
                    equil_params.n_frames,
                )
            )
            xs[i] = x[-1]
            boxes[i] = b[-1]
    return xs, boxes


def main():
    parser = ArgumentParser()
    parser.add_argument("--processes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--local_md", action="store_true")
    parser.add_argument("--local_md_k", default=10_000.0, type=float)
    parser.add_argument("--local_md_rad", default=1.2, type=float)
    parser.add_argument("--local_md_steps", default=400, type=int)
    parser.add_argument("--local_md_free_reference", action="store_true", default=False)
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable Batching instead of MPS, will batch simulations together instead of across processes",
    )
    parser.add_argument("--steps", default=400, type=int)
    parser.add_argument("--frames", default=100, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument(
        "--active_thread_percentage",
        default=None,
        type=float,
        help="Specify CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for each MPS worker",
    )
    parser.add_argument("--hrex", action="store_true", help="Run HREX on N states")
    parser.add_argument(
        "--n_eq_steps",
        type=int,
        default=10_000,
        help="Number of global steps to run to equilibrate the system. Important to get reliable results from batch mode.",
    )
    parser.add_argument(
        "--hrex_states",
        type=int,
        default=8,
        help="Number of HREX states. Only applies to MPS, for batched the number of HREX states is the number of replicas",
    )
    parser.add_argument("--dt_fs", default=2.5, type=float, help="The timestep in femptoseconds")
    parser.add_argument("--system", default="dhfr", choices=["dhfr", "hif2a-rbfe", "hif2a", "pfkfb3-rbfe"])
    args = parser.parse_args()

    output_suffix = args.output_suffix
    if output_suffix is None:
        date = datetime.now()
        date_str = date.strftime("%Y_%b_%d_%H_%M")
        output_suffix = f"_{date_str}_{args.dt_fs:.1f}fs"

    seed = args.seed
    temperature = constants.DEFAULT_TEMP
    ff = Forcefield.load_default()

    assert args.dt_fs > 0.0
    dt = args.dt_fs * 1e-3

    num_samples = np.max(args.processes)

    if args.system in ("dhfr", "hif2a"):
        if args.system == "dhfr":
            with path_to_internal_file("tmd.testsystems.data", "5dfr_solv_equil.pdb") as pdb_path:
                host_config = builders.load_pdb_system(str(pdb_path), "amber99sbildn", "tip3p", cutoff=1.2)
        else:
            with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as protein_path:
                host_config = builders.load_pdb_system(str(protein_path), ff.protein_ff, ff.water_ff)
        initial_state = initial_state_from_host_config(host_config, dt=dt, temperature=temperature, seed=seed)
        initial_state.ligand_idxs = np.arange(len(host_config.conf), dtype=np.int32)
        if args.hrex:
            initial_states = [
                replace(initial_state, lamb=lamb)
                for lamb in np.linspace(0.0, 1.0, args.hrex_states if not args.batch_mode else num_samples)
            ]
        else:
            initial_states = [initial_state] * num_samples
    elif args.system.endswith("-rbfe"):
        if args.system == "hif2a-rbfe":
            mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()

            with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as protein_path:
                host_config = builders.load_pdb_system(str(protein_path), ff.protein_ff, ff.water_ff, box_margin=0.1)
        else:
            with path_to_internal_file("tmd.testsystems.fep_benchmark.pfkfb3", "ligands.sdf") as ligands_path:
                mols = read_sdf(ligands_path)
                # Select the first two mols, not important
                mol_a = mols[0]
                mol_b = mols[1]
            core = atom_mapping.get_cores(mol_a, mol_b, **constants.DEFAULT_ATOM_MAPPING_KWARGS)[0]
            with path_to_internal_file("tmd.testsystems.fep_benchmark.pfkfb3", "6hvi_prepared.pdb") as protein_path:
                host_config = builders.build_protein_system(
                    str(protein_path), ff.protein_ff, ff.water_ff, box_margin=0.1
                )
        st = SingleTopology(mol_a, mol_b, core, ff)
        host_config = setup_optimized_host(host_config, [mol_a, mol_b], ff)

        if args.hrex:
            initial_states = setup_initial_states(
                st,
                host_config,
                temperature,
                np.linspace(0.0, 1.0, args.hrex_states if not args.batch_mode else num_samples),
                seed,
                False,
                dt=dt,
            )
        else:
            initial_states = [setup_initial_state(st, 0.0, host_config, temperature, seed, False, dt=dt)] * num_samples
    else:
        assert 0, f"Unknown system {args.system}"

    # Resample the velocities to ensure they aren't all zero
    initial_states = [
        replace(state, v0=sample_velocities(state.integrator.masses, temperature, seed=seed + i))
        for i, state in enumerate(initial_states)
    ]

    md_params = MDParams(
        n_eq_steps=0,
        n_frames=args.frames,
        steps_per_frame=args.steps,
        seed=seed,
        local_md_params=LocalMDParams(
            local_steps=args.local_md_steps,
            k=args.local_md_k,
            min_radius=args.local_md_rad,
            max_radius=args.local_md_rad,
            freeze_reference=not args.local_md_free_reference,
        )
        if args.local_md
        else None,
        hrex_params=HREXParams() if args.hrex else None,
        dt=dt,
    )

    xs, boxes = generate_equilibrium_frames(initial_states, md_params, args.n_eq_steps, args.dt_fs)

    system_data = MDSystemData(initial_states, md_params, args.hrex_states)

    skip_single_proc = args.hrex and args.batch_mode and 1 in args.processes
    ns_per_day_results = []
    for proc in args.processes:
        if proc == 1 and skip_single_proc:
            print("Skipping single process for batch mode with HREX, must have at least 2 simulations")
            continue

        mps_workers = proc
        if args.batch_mode:
            mps_workers = 1
        pool = CUDAMPSPoolClient(
            1, workers_per_gpu=mps_workers, active_thread_usage_per_worker=args.active_thread_percentage
        )
        pool.verify()

        if not args.batch_mode:
            futures = [
                pool.submit(
                    run_mps_steps,
                    replace(
                        system_data,
                        initial_states=[replace(state, x0=xs[i], box0=boxes[i])],
                        params=replace(md_params, seed=md_params.seed + i + proc),
                    ),
                )
                for i, state in enumerate(initial_states[:proc])
            ]
        else:
            # Combine all steps into a single state
            futures = [
                pool.submit(
                    run_batched_steps,
                    replace(
                        system_data,
                        initial_states=[
                            replace(state, x0=xs[i], box0=boxes[i]) if not args.hrex else state
                            for i, state in enumerate(initial_states[:proc])
                        ],
                        params=replace(md_params, seed=md_params.seed + proc),
                    ),
                )
            ]
        results = [fut.result() for fut in futures]
        frames_run = md_params.steps_per_frame * md_params.n_frames
        if args.batch_mode:
            frames_run *= proc
            print(frames_run)
        elif args.hrex:
            frames_run *= args.hrex_states
        steps_per_second = frames_run / np.array(results)
        total_steps_per_second = np.sum(steps_per_second)
        ns_per_day = total_steps_per_second * SECONDS_PER_DAY * dt * 1e-3
        print(f"{proc} Process: {ns_per_day} Ns per day")
        ns_per_day_results.append(ns_per_day)
    if skip_single_proc:
        # Remove the 1 process example, else the results are skewed
        idx = args.processes.index(1)
        args.processes.pop(idx)

    plt.plot(args.processes, ns_per_day_results)
    plt.ylabel("ns per day")
    if args.batch_mode:
        plt.xlabel("Batched Replicas")
    else:
        plt.xlabel("MPS Replicas")
    plt.title(
        f"{args.system}\nPeak {max(ns_per_day_results):.1f}ns/day"
        + (f"\nLocal MD Rad {args.local_md_rad}" if args.local_md else "")
        + (f"\n{args.hrex_states} HREX Windows" if args.hrex and not args.batch_mode else "")
    )
    plt.tight_layout()
    plt.savefig(f"{args.system}_ns_per_day{output_suffix}.png", dpi=150)
    plt.clf()

    plt.plot(args.processes, np.array(ns_per_day_results) / ns_per_day_results[0])
    plt.ylabel("Factor improvement")
    plt.xlabel("Processes")
    plt.title(
        f"{args.system} {args.dt_fs:.1f}fs"
        + (f"\nLocal MD Rad {args.local_md_rad}" if args.local_md else "")
        + (f"\n{args.hrex_states} HREX Windows" if args.hrex and not args.batch_mode else "")
    )
    plt.tight_layout()
    plt.savefig(f"{args.system}_factor_improvement{output_suffix}.png", dpi=150)
    plt.clf()


if __name__ == "__main__":
    main()
