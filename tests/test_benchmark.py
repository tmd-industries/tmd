# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2026, Forrest York
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

"""Run vanilla "apo" MD on DHFR and HIF2A test systems,
and running an intermediate lambda window "rbfe" MD for a
relative binding free energy edge from the HIF2A test system"""

import hashlib
import time
from argparse import ArgumentParser
from dataclasses import dataclass, replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from tmd import constants
from tmd.fe.absolute import hydration as absolute_hydration
from tmd.fe.free_energy import (
    AbsoluteFreeEnergy,
    HostConfig,
    InitialState,
    LocalMDParams,
    MDParams,
    WaterSamplingParams,
    get_batched_context,
    get_context,
    sample_with_context_iter,
)
from tmd.fe.model_utils import apply_hmr
from tmd.fe.rbfe import setup_initial_state
from tmd.fe.single_topology import SingleTopology
from tmd.fe.topology import BaseTopology
from tmd.ff import Forcefield
from tmd.lib import ConstrainedLangevinIntegrator, LangevinIntegrator, MonteCarloBarostat, custom_ops
from tmd.md import builders
from tmd.md.barostat.utils import compute_box_volume, get_bond_list, get_group_indices
from tmd.md.thermostat.utils import sample_velocities
from tmd.potentials import (
    Nonbonded,
    NonbondedInteractionGroup,
    Potential,
    SummedPotential,
)
from tmd.testsystems.dhfr import setup_dhfr
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology
from tmd.utils import path_to_internal_file

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass
class BenchmarkConfig:
    num_batches: int
    steps_per_batch: int
    verbose: bool
    generate_plots: bool
    num_equil_batches: int = 1
    local_only: bool = False

    def __post_init__(self):
        assert self.num_batches >= 1
        assert self.steps_per_batch >= 1
        assert self.num_equil_batches >= 1

    def get_md_params(self):
        """Utility function to construct MD parameters implied by the config"""
        return MDParams(
            n_frames=1,
            n_eq_steps=0,
            steps_per_frame=self.steps_per_batch,
            seed=2026,
        )


def plot_batch_times(steps_per_batch: int, dt: float, batch_times: list[float], box_volumes: list[float], label: str):
    """
    Plot and save a figure of the batches of benchmarks run.

    Parameters
    ----------
        steps_per_batch: int
            Number of steps per each batch

        dt: float
            Timestep in femtoseconds

        batch_times: list of floats
            Times in seconds that each batch took

        label: str
            The label used as the file name as well as the title of the plot
    """
    ns_per_day = steps_per_batch / np.array(batch_times)
    ns_per_day = ns_per_day * SECONDS_PER_DAY * dt * 1e-3

    plt.title(label)
    fig, axes = plt.subplots(ncols=2)
    fig.suptitle(label)
    axes[0].plot(ns_per_day)
    axes[0].axhline(np.mean(ns_per_day), linestyle="--", c="gray", label="Mean")
    axes[0].legend()
    axes[0].set_xlabel("Batch")
    axes[0].set_ylabel("ns per day")

    axes[1].plot(box_volumes)
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Box Volume (nm^3)")
    fig.tight_layout()
    fig.savefig(f"{label}.png", dpi=150)
    plt.clf()
    plt.close()


@pytest.fixture(scope="module")
def hi2fa_test_frames():
    return generate_hif2a_frames(100, 10, seed=2022, barostat_interval=20)


def initial_state_from_host_config(
    host_config: HostConfig,
    dt: float,
    temperature: float = constants.DEFAULT_TEMP,
    seed: int = 2026,
    barostat_interval: int = 25,
) -> InitialState:
    system = host_config.host_system
    hmr_masses = apply_hmr(host_config.masses, system.bond.potential.idxs)

    group_idxs = get_group_indices(get_bond_list(system.bond.potential), len(hmr_masses))
    baro = None
    if barostat_interval > 0:
        baro = MonteCarloBarostat(
            len(hmr_masses), constants.DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, seed
        )

    x0 = host_config.conf
    # initialize integrator
    friction = 1.0
    intg: ConstrainedLangevinIntegrator | LangevinIntegrator
    if dt > 2.5e-3:
        assert host_config.constraints
        intg = ConstrainedLangevinIntegrator(temperature, dt, friction, hmr_masses, seed, host_config.constraints)
    else:
        intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)
    protein_idxs = np.arange(0, len(hmr_masses) - host_config.num_water_atoms, dtype=np.int32)
    ligand_idxs = np.array([], dtype=np.int32)

    v0 = sample_velocities(hmr_masses, temperature, seed)
    return InitialState(system.get_U_fns(), intg, baro, x0, v0, host_config.box, 0.0, ligand_idxs, protein_idxs)


def generate_hif2a_frames(n_frames: int, frame_interval: int, seed=None, barostat_interval: int = 5):
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as path_to_pdb:
        host_config = builders.load_pdb_system(str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff)
    initial_state = setup_initial_state(st, 0.1, host_config, constants.DEFAULT_TEMP, 2022, True)

    intg = initial_state.integrator.impl()

    bps = []

    for potential in initial_state.potentials:
        bps.append(potential.to_gpu(precision=np.float32).bound_impl)  # get the bound implementation

    movers = []

    if barostat_interval > 0:
        baro = initial_state.barostat
        assert baro is not None
        baro_impl = baro.impl(bps)
        baro_impl.set_interval(barostat_interval)
        movers.append(baro_impl)

    ctxt = custom_ops.Context_f32(
        initial_state.x0,
        initial_state.v0,
        initial_state.box0,
        intg,
        bps,
        movers=movers,
    )
    steps = n_frames * frame_interval
    coords, boxes = ctxt.multiple_steps(steps, frame_interval)
    assert coords.shape[0] == n_frames, f"Got {coords.shape[0]} frames, expected {n_frames}"
    return initial_state.potentials, coords, boxes, initial_state.ligand_idxs


def benchmark_potential(
    config: BenchmarkConfig,
    label: str,
    potential: Potential,
    precision,
    params: NDArray,
    coords: NDArray,
    boxes: NDArray,
    compute_du_dx: bool = True,
    compute_du_dp: bool = True,
    compute_u: bool = True,
):
    if precision == np.float32:
        label = label + "_f32"
    else:
        label = label + "_f64"
    unbound = potential.to_gpu(precision=precision).unbound_impl
    start = time.time()
    batch_times = []
    frames = coords.shape[0]
    param_batches = params.shape[0]
    runs_per_batch = frames * param_batches
    for _ in range(config.num_batches):
        batch_start = time.time()
        _, _, _ = unbound.execute_batch(
            coords,
            params,
            boxes,
            compute_du_dx,
            compute_du_dp,
            compute_u,
        )
        batch_end = time.time()
        delta = batch_end - batch_start

        batch_times.append(delta)
        runs_per_second = runs_per_batch / np.mean(batch_times)

        if config.verbose:
            print(f"executions per second: {runs_per_second:.3f}")
    print(
        f"{label}: N={coords.shape[1]} Frames={frames} Params={param_batches} speed: {runs_per_second:.2f} executions/seconds (ran {runs_per_batch * config.num_batches} potentials in {(time.time() - start):.2f}s)",
        f"du_dp={compute_du_dp}, du_dx={compute_du_dx}, u={compute_u}",
    )


def benchmark_initial_state(
    config: BenchmarkConfig, label: str, state: InitialState, md_params: MDParams, num_systems: int = 1
):
    if config.local_only and md_params.local_md_params is None:
        return
    if num_systems == 1:
        ctxt = get_context(state, md_params)
    else:
        ctxt = get_batched_context([state] * num_systems, md_params)

    expected_movers = 0
    if state.barostat is not None:
        expected_movers += 1
        label += f"-barostat-interval-{state.barostat.interval}"
    if md_params.water_sampling_params is not None:
        water_sampling_interval = md_params.water_sampling_params.interval
        expected_movers += 1
        label += f"-water-sampling-interval-{water_sampling_interval}"
        if config.steps_per_batch < water_sampling_interval:
            print("Warning::Not running water sampling every batch, interval is too large")
    assert len(ctxt.get_movers()) == expected_movers

    if md_params.local_md_params is not None:
        assert md_params.local_md_params.min_radius == md_params.local_md_params.max_radius

    batch_times = []
    box_volumes = []

    steps_per_batch = config.steps_per_batch
    num_batches = config.num_batches

    assert md_params.n_frames == 1
    assert md_params.steps_per_frame == steps_per_batch
    assert md_params.n_eq_steps == 0

    dt = state.integrator.dt

    temperature = state.integrator.temperature

    # run num_equil_batches before starting the time, can improve benchmark accuracy
    for _ in range(config.num_equil_batches):
        next(sample_with_context_iter(ctxt, md_params, temperature, state.ligand_idxs, 1))

    start = time.perf_counter()

    for batch in range(num_batches):
        # time the current batch
        batch_start = time.perf_counter()
        xs, boxes, _ = next(sample_with_context_iter(ctxt, md_params, temperature, state.ligand_idxs, 1))
        batch_end = time.perf_counter()
        if num_systems == 1:
            box_volumes.append(compute_box_volume(boxes[-1]))

        delta = batch_end - batch_start

        batch_times.append(delta)

        steps_per_second = steps_per_batch / np.mean(batch_times)
        steps_per_day = steps_per_second * SECONDS_PER_DAY

        ps_per_day = dt * steps_per_day
        ns_per_day = (ps_per_day * 1e-3) * num_systems

        if config.verbose:
            print(f"steps per second: {steps_per_second:.3f}")
            print(f"ns per day: {ns_per_day:.3f}")

    assert np.all(np.abs(ctxt.get_x_t()) < 1000)

    determinism_hash = hashlib.md5(xs[-1].tobytes()).hexdigest()[:8]

    if md_params.local_md_params is None:
        print(
            f"{label}: Systems={num_systems} N={state.x0.shape[0]} speed: {ns_per_day:.2f}ns/day dt: {dt * 1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.perf_counter() - start):.2f}s) | determinism hash: {determinism_hash}"
        )
    else:
        radius = md_params.local_md_params.min_radius
        k = md_params.local_md_params.k
        print(
            f"{label}: Systems={num_systems} N={state.x0.shape[0]} Radius={radius}, K={k} speed: {ns_per_day:.2f}ns/day dt: {dt * 1e3}fs (ran {steps_per_batch * num_batches} steps in {(time.time() - start):.2f}s) | determinism hash: {determinism_hash}"
        )

    if config.generate_plots and num_systems == 1:
        plot_batch_times(steps_per_batch, dt, batch_times, box_volumes, label)


def run_single_topology_benchmarks(
    config: BenchmarkConfig,
    stage: str,
    st: SingleTopology,
    host_config: Optional[HostConfig],
):
    for dt in [2.5e-3, 4.0e-3]:
        initial_state = setup_initial_state(st, 0.1, host_config, constants.DEFAULT_TEMP, 2022, True, dt=dt)

        for num_systems in [1, 2, 4]:
            if host_config is not None:
                for barostat_interval in [0, 25]:
                    apo_state = initial_state_from_host_config(host_config, dt, barostat_interval=barostat_interval)
                    benchmark_initial_state(
                        config,
                        f"{stage}-apo",
                        apo_state,
                        config.get_md_params(),
                        num_systems=num_systems,
                    )

            benchmark_initial_state(
                config,
                f"{stage}-rbfe",
                initial_state,
                config.get_md_params(),
                num_systems=num_systems,
            )

        if host_config is not None:
            # Local MD has some poor interplay with Constraints
            if isinstance(initial_state.integrator, LangevinIntegrator):
                benchmark_initial_state(
                    config,
                    f"{stage}-rbfe-local",
                    initial_state,
                    replace(
                        config.get_md_params(),
                        local_md_params=LocalMDParams(
                            config.steps_per_batch, min_radius=1.2, max_radius=1.2, k=10_000.0
                        ),
                    ),
                )

            # Only in the case where the ligand is in complex do we want to look at water sampling
            if host_config.num_water_atoms < host_config.conf.shape[0]:
                benchmark_initial_state(
                    config,
                    f"{stage}-rbfe",
                    initial_state,
                    replace(
                        config.get_md_params(),
                        water_sampling_params=WaterSamplingParams(interval=400, radius=1.0, batch_size=250),
                    ),
                )


def benchmark_dhfr(config: BenchmarkConfig):
    host_fns, host_masses, host_conf, box = setup_dhfr()

    with path_to_internal_file("tmd.testsystems.data", "5dfr_solv_equil.pdb") as pdb_path:
        host_config = builders.load_pdb_system(str(pdb_path), "amber99sbildn", "tip3p", cutoff=1.2)

    for dt in [2.5e-3, 4.0e-3]:
        for barostat_interval in [0, 25]:
            apo_state = initial_state_from_host_config(host_config, dt, barostat_interval=barostat_interval)
            for num_systems in [1, 2, 4]:
                benchmark_initial_state(
                    config,
                    "dhfr-apo",
                    apo_state,
                    config.get_md_params(),
                    num_systems=num_systems,
                )
        apo_state = initial_state_from_host_config(host_config, dt)
        benchmark_initial_state(
            config,
            "dhfr-local",
            replace(apo_state, ligand_idxs=np.arange(len(host_config.conf), dtype=np.int32)),
            replace(
                config.get_md_params(),
                local_md_params=LocalMDParams(config.steps_per_batch, min_radius=1.2, max_radius=1.2, k=10_000.0),
            ),
        )


def benchmark_hif2a(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    # build the protein system.
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_solv_equil.pdb") as path_to_pdb:
        host_config = builders.load_pdb_system(str(path_to_pdb), forcefield.protein_ff, forcefield.water_ff)

    run_single_topology_benchmarks(config, "hif2a", st, host_config)


def benchmark_solvent(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    host_config = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol_a, mol_b], box_margin=0.1)
    run_single_topology_benchmarks(config, "solvent", st, host_config)


def benchmark_vacuum(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)

    run_single_topology_benchmarks(config, "vacuum", st, None)


def benchmark_ahfe(config: BenchmarkConfig):
    # we use simple charge "sc" to be able to run on machines that don't have openeye licenses.
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2024
    mol, _, _ = get_hif2a_ligand_pair_single_topology()
    host_config = builders.build_water_system(4.0, forcefield.water_ff, mols=[mol], box_margin=0.1)
    bt = BaseTopology(mol, forcefield)
    afe = AbsoluteFreeEnergy(mol, bt)

    initial_state = absolute_hydration.setup_initial_states(
        afe, forcefield, host_config, constants.DEFAULT_TEMP, np.array([0.0]), seed
    )[0]

    for num_systems in [1, 2, 4]:
        benchmark_initial_state(
            config,
            "ahfe",
            initial_state,
            config.get_md_params(),
            num_systems=num_systems,
        )
        if host_config is not None:
            benchmark_initial_state(
                config,
                "ahfe-local",
                initial_state,
                replace(
                    config.get_md_params(),
                    local_md_params=LocalMDParams(config.steps_per_batch, min_radius=1.2, max_radius=1.2, k=10_000.0),
                ),
                num_systems=num_systems,
            )


def test_dhfr():
    benchmark_dhfr(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100, generate_plots=False))


def test_hif2a():
    benchmark_hif2a(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100, generate_plots=False))


def test_solvent():
    benchmark_solvent(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100, generate_plots=False))


def test_vacuum():
    benchmark_vacuum(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100, generate_plots=False))


def test_ahfe():
    benchmark_ahfe(BenchmarkConfig(verbose=True, num_batches=2, steps_per_batch=100, generate_plots=False))


def get_nonbonded_pot_params(bps):
    for bp in bps:
        if isinstance(bp.potential, Nonbonded):
            return bp.potential, bp.params
    else:
        raise AssertionError("Nonbonded potential not found")


def test_nonbonded_interaction_group_potential(hi2fa_test_frames):
    bps, frames, boxes, ligand_idxs = hi2fa_test_frames
    nonbonded_potential, nonbonded_params = get_nonbonded_pot_params(bps)

    config = BenchmarkConfig(num_batches=2, steps_per_batch=1, verbose=False, generate_plots=False)

    num_param_batches = 5
    cutoff = 1.2

    precisions = [np.float32, np.float64]
    nonbonded_params = np.stack([nonbonded_params] * num_param_batches)

    potential = NonbondedInteractionGroup(
        nonbonded_potential.num_atoms,
        ligand_idxs,
        cutoff,
    )
    class_name = potential.__class__.__name__
    for precision in precisions:
        benchmark_potential(
            config,
            class_name,
            potential,
            precision,
            nonbonded_params,
            frames,
            boxes,
        )


def test_hif2a_potentials(hi2fa_test_frames):
    bps, frames, boxes, _ = hi2fa_test_frames

    config = BenchmarkConfig(num_batches=2, steps_per_batch=1, verbose=False, generate_plots=False)

    num_param_batches = 5

    for bp in bps:
        potential = bp.potential
        class_name = potential.__class__.__name__
        if isinstance(potential, SummedPotential):
            class_name += "(" + ", ".join([pot.__class__.__name__ for pot in potential.potentials]) + ")"
        params = np.stack([bp.params] * num_param_batches)
        for precision in [np.float32, np.float64]:
            benchmark_potential(
                config,
                class_name,
                bp.potential,
                precision,
                params,
                frames,
                boxes,
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--equil_batches", default=10, type=int, help="Number of batches to run before taking timings")
    parser.add_argument("--num_batches", default=100, type=int)
    parser.add_argument("--steps_per_batch", default=1000, type=int)
    parser.add_argument("--skip_dhfr", action="store_true")
    parser.add_argument("--skip_hif2a", action="store_true")
    parser.add_argument("--skip_ahfe", action="store_true")
    parser.add_argument("--skip_solvent", action="store_true")
    parser.add_argument("--skip_vacuum", action="store_true")
    parser.add_argument("--skip_potentials", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = BenchmarkConfig(
        verbose=args.verbose,
        num_batches=args.num_batches,
        steps_per_batch=args.steps_per_batch,
        generate_plots=not args.skip_plots,
        num_equil_batches=args.equil_batches,
        local_only=args.local_only,
    )

    if not args.skip_dhfr:
        benchmark_dhfr(config)
    if not args.skip_hif2a:
        benchmark_hif2a(config)
    if not args.skip_solvent:
        benchmark_solvent(config)
    if not args.skip_ahfe:
        benchmark_ahfe(config)
    if not args.skip_vacuum and not args.local_only:
        benchmark_vacuum(config)

    if not args.skip_potentials:
        hif2a_frames = generate_hif2a_frames(1000, 20, seed=2022, barostat_interval=20)
        test_nonbonded_interaction_group_potential(hif2a_frames)
        test_hif2a_potentials(hif2a_frames)
