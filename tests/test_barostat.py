# Copyright 2019-2025, Relay Therapeutics
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

import itertools
from dataclasses import replace

import numpy as np
import pytest

from tmd.constants import AVOGADRO, BAR_TO_KJ_PER_NM3, BOLTZ, DEFAULT_PRESSURE, DEFAULT_TEMP
from tmd.fe import model_utils
from tmd.fe.free_energy import AbsoluteFreeEnergy
from tmd.fe.topology import BaseTopology
from tmd.ff import Forcefield
from tmd.lib import LangevinIntegrator, custom_ops
from tmd.md.barostat.moves import CentroidRescaler
from tmd.md.barostat.utils import compute_box_center, compute_box_volume, get_bond_list, get_group_indices
from tmd.md.builders import build_water_system
from tmd.md.enhanced import get_solvent_phase_system
from tmd.md.thermostat.utils import sample_velocities
from tmd.potentials import HarmonicBond, Nonbonded, NonbondedPairListPrecomputed
from tmd.potentials.potential import get_potential_by_type
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.memcheck
@pytest.mark.parametrize(
    "klass",
    [
        # Only test float32 to avoid additional conversion
        custom_ops.MonteCarloBarostat_f32,
        custom_ops.AnisotropicMonteCarloBarostat_f32,
    ],
)
def test_barostat_validation(klass):
    temperature = DEFAULT_TEMP  # kelvin
    pressure = DEFAULT_PRESSURE  # bar
    barostat_interval = 3  # step count
    seed = 2023

    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, lamb=0.0, minimize_energy=False
    )

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        u_impls.append(unbound_pot.bind(params).to_gpu(precision=np.float32).bound_impl)

    # Invalid interval
    with pytest.raises(RuntimeError, match="interval must be greater than 0"):
        klass(coords.shape[0], pressure, temperature, [[0, 1]], -1, u_impls, seed, True, 0.0)

    # Atom index over N
    with pytest.raises(RuntimeError, match="Grouped indices must be between 0 and N"):
        klass(
            coords.shape[0],
            pressure,
            temperature,
            [[0, coords.shape[0] + 1]],
            barostat_interval,
            u_impls,
            seed,
            True,
            0.0,
        )

    # Atom index < 0
    with pytest.raises(RuntimeError, match="Grouped indices must be between 0 and N"):
        klass(coords.shape[0], pressure, temperature, [[-1, 0]], barostat_interval, u_impls, seed, True, 0.0)

    # Atom index in two groups
    with pytest.raises(RuntimeError, match="All grouped indices must be unique"):
        klass(coords.shape[0], pressure, temperature, [[0, 1], [1, 2]], barostat_interval, u_impls, seed, True, 0.0)


@pytest.mark.memcheck
def test_barostat_with_clashes():
    temperature = DEFAULT_TEMP  # kelvin
    pressure = DEFAULT_PRESSURE  # bar
    timestep = 1.5e-3  # picosecond
    barostat_interval = 3  # step count
    collision_rate = 1.0  # 1 / picosecond
    seed = 2023

    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    # Construct water box without removing the waters around the ligand to ensure clashes
    host_config = build_water_system(3.0, ff.water_ff)
    # Shrink the box to ensure the energies are NaN
    host_config = replace(host_config, box=host_config.box - np.eye(3) * 0.1)
    bt = BaseTopology(mol_a, ff)
    afe = AbsoluteFreeEnergy(mol_a, bt)
    unbound_potentials, sys_params, masses = afe.prepare_host_edge(ff, host_config, 0.0)
    coords = afe.prepare_combined_coords(host_coords=host_config.conf)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    # Cut the number of groups in half
    group_indices = group_indices[len(group_indices) // 2 :]

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        u_impls.append(unbound_pot.bind(params).to_gpu(precision=np.float32).bound_impl)

    # The energy of the system should be non-finite
    nrg = np.sum([bp.execute(coords, host_config.box, compute_du_dx=False)[1] for bp in u_impls])
    assert not np.isfinite(nrg)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat_f32(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )
    assert baro.num_systems() == 1

    # The clashes will result in overflows, so the box should never change as no move is accepted
    ctxt = custom_ops.Context_f32(
        coords.astype(np.float32), v_0.astype(np.float32), host_config.box, integrator_impl, u_impls, movers=[baro]
    )
    # Will trigger the unstable check since the box is so small.
    with pytest.raises(
        RuntimeError,
        match="simulation unstable: dimensions of coordinates two orders of magnitude larger than max box dimension",
    ):
        ctxt.multiple_steps(barostat_interval * 100)
    assert np.all(host_config.box == ctxt.get_box())


@pytest.mark.memcheck
def test_barostat_zero_interval():
    pressure = DEFAULT_PRESSURE  # bar
    temperature = DEFAULT_TEMP  # kelvin
    seed = 2021
    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    unbound_potentials, sys_params, masses, coords, _ = get_solvent_phase_system(
        mol_a, ff, lamb=1.0, minimize_energy=False
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    with pytest.raises(RuntimeError):
        custom_ops.MonteCarloBarostat_f32(
            coords.shape[0], pressure, temperature, group_indices, 0, u_impls, seed, True, 0.0
        )
    # Setting it to 1 should be valid.
    baro = custom_ops.MonteCarloBarostat_f32(
        coords.shape[0], pressure, temperature, group_indices, 1, u_impls, seed, True, 0.0
    )
    assert baro.num_systems() == 1
    # Setting back to 0 should raise another error
    with pytest.raises(RuntimeError):
        baro.set_interval(0)


@pytest.mark.memcheck
def test_barostat_partial_group_idxs():
    """Verify that the barostat can handle a subset of the molecules
    rather than all of them. This test only verify that it runs, not the behavior"""
    lam = 1.0
    temperature = DEFAULT_TEMP  # kelvin
    timestep = 1.5e-3  # picosecond
    barostat_interval = 3  # step count
    collision_rate = 1.0  # 1 / picosecond

    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE  # bar
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(
        mol_a, ff, lam, minimize_energy=False
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    # Cut the number of groups in half
    group_indices = group_indices[len(group_indices) // 2 :]

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat_f32(
        coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )
    assert baro.num_systems() == 1

    ctxt = custom_ops.Context_f32(
        coords.astype(np.float32),
        v_0.astype(np.float32),
        complex_box.astype(np.float32),
        integrator_impl,
        u_impls,
        movers=[baro],
    )
    ctxt.multiple_steps(barostat_interval * 100)


@pytest.mark.parametrize(
    "box_width, iterations, num_systems",
    [
        pytest.param(3.0, 30, 1, marks=pytest.mark.memcheck),
        pytest.param(3.0, 30, 2, marks=pytest.mark.memcheck),
        (3.0, 30, 4),
        (3.0, 30, 12),
        # fyork: This test only fails 50-50 times. It tests a race condition in which the coordinates could be corrupted.
        # Needs to be a large system to trigger the failure.
        (10.0, 1000, 1),
    ],
)
@pytest.mark.parametrize("klass", [custom_ops.MonteCarloBarostat_f32, custom_ops.AnisotropicMonteCarloBarostat_f32])
def test_barostat_is_deterministic(box_width, iterations, num_systems, klass):
    """Verify that the barostat results in the same box size shift after a fixed number of steps
    This is important to debugging as well as providing the ability to replicate
    simulations
    """
    temperature = DEFAULT_TEMP
    timestep = 2.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, box_width=box_width, lamb=1.0, minimize_energy=False, margin=0.1
    )

    batch_coords = np.array([coords] * num_systems).astype(np.float32)
    batch_boxes = np.array([box] * num_systems).astype(np.float32)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    masses = model_utils.apply_hmr(masses, bond_list)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(params)
        for _ in range(num_systems - 1):
            bp = bp.combine(unbound_pot.bind(params))
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)
        assert bp_impl.num_systems() == num_systems

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        [masses] * num_systems,
        seed,
    )

    v_0 = np.array([sample_velocities(masses, temperature, seed + i) for i in range(num_systems)])

    baro = klass(coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0)

    ctxt = custom_ops.Context_f32(
        batch_coords.squeeze(),
        v_0.squeeze(),
        batch_boxes.squeeze(),
        integrator.impl(),
        u_impls,
        movers=[baro],
    )
    _, boxes = ctxt.multiple_steps(iterations * barostat_interval, 100)
    atm_box = ctxt.get_box()
    if num_systems == 1:
        # Verify that the volume of the box has changed
        assert compute_box_volume(atm_box) != compute_box_volume(box)
    else:
        for sys_box in atm_box:
            assert compute_box_volume(sys_box) != compute_box_volume(box)

    baro = klass(coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0)
    ctxt = custom_ops.Context_f32(
        batch_coords.squeeze(),
        v_0.squeeze(),
        batch_boxes.squeeze(),
        integrator.impl(),
        u_impls,
        movers=[baro],
    )
    ctxt.multiple_steps(iterations * barostat_interval)
    # Verify that we get back bitwise reproducible boxes
    np.testing.assert_array_equal(atm_box, ctxt.get_box())


def test_barostat_varying_pressure():
    temperature = DEFAULT_TEMP
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    # Start out with a very large pressure
    pressure = 1013.0
    host_config = build_water_system(3.0, ff.water_ff, box_margin=0.1)
    harmonic_bond_potential = host_config.host_system.bond
    bond_list = get_bond_list(harmonic_bond_potential.potential)
    group_indices = get_group_indices(bond_list, len(host_config.masses))

    u_impls = []
    for bp in host_config.host_system.get_U_fns():
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        host_config.masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(host_config.masses, temperature, seed)

    baro = custom_ops.MonteCarloBarostat_f32(
        host_config.conf.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
    )
    assert baro.num_systems() == 1

    ctxt = custom_ops.Context_f32(
        host_config.conf.astype(np.float32),
        v_0.astype(np.float32),
        host_config.box.astype(np.float32),
        integrator_impl,
        u_impls,
        movers=[baro],
    )
    ctxt.multiple_steps(1000)
    ten_atm_box = ctxt.get_box()
    ten_atm_box_vol = compute_box_volume(ten_atm_box)
    # Expect the box to shrink thanks to the barostat
    assert compute_box_volume(host_config.box) - ten_atm_box_vol > 0.4

    # Set the pressure to 1 atm
    baro.set_pressure(DEFAULT_PRESSURE)
    baro.set_step(0)

    ctxt.multiple_steps(1000)
    atm_box = ctxt.get_box()
    # Box will grow thanks to the lower pressure
    assert compute_box_volume(atm_box) > ten_atm_box_vol


# test that barostat only proposes properly re-centered coordinates
@pytest.mark.parametrize(
    "klass",
    [
        custom_ops.MonteCarloBarostat_f32,
        custom_ops.AnisotropicMonteCarloBarostat_f32,
    ],
)
def test_barostat_recentering_upon_acceptance(klass):
    lam = 1.0
    temperature = DEFAULT_TEMP
    pressure = DEFAULT_PRESSURE
    timestep = 1.5e-3
    barostat_interval = 10
    collision_rate = 1.0
    seed = 2023
    np.random.seed(seed)

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    unbound_potentials, sys_params, masses, coords, complex_box = get_solvent_phase_system(mol_a, ff, lam, margin=0.0)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )
    integrator_impl = integrator.impl()

    v_0 = sample_velocities(masses, temperature, seed)

    baro = klass(coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0)
    ctxt = custom_ops.Context_f32(
        coords.astype(np.float32),
        v_0.astype(np.float32),
        complex_box.astype(np.float32),
        integrator_impl,
        u_impls,
        movers=[baro],
    )
    # mini equilibrate the system to get barostat proposals to be reasonable
    ctxt.multiple_steps(1000)
    num_accepted = 0
    for _ in range(100):
        ctxt.multiple_steps(100)
        x_t = ctxt.get_x_t()
        box_t = ctxt.get_box()
        new_x_t, new_box_t = baro.move(x_t, box_t)
        if not np.all(box_t == new_box_t):
            for atom_idxs in group_indices:
                xyz = np.mean(new_x_t[atom_idxs], axis=0)
                ref_xyz = np.mean(model_utils.image_molecule(new_x_t[atom_idxs], new_box_t), axis=0)
                np.testing.assert_allclose(xyz, ref_xyz)
                x, y, z = xyz
                assert x > 0 and x < new_box_t[0][0]
                assert y > 0 and y < new_box_t[1][1]
                assert z > 0 and z < new_box_t[2][2]

            num_accepted += 1
        else:
            np.testing.assert_array_equal(new_x_t, x_t)
            np.testing.assert_array_equal(new_box_t, box_t)

    assert num_accepted > 0


@pytest.mark.parametrize("klass", [custom_ops.MonteCarloBarostat_f32, custom_ops.AnisotropicMonteCarloBarostat_f32])
def test_molecular_ideal_gas(klass):
    """


    References
    ----------
    OpenMM testIdealGas
    https://github.com/openmm/openmm/blob/d8ef57fed6554ec95684e53768188e1f666405c9/tests/TestMonteCarloBarostat.h#L86-L140
    """

    # simulation parameters
    timestep = 1.5e-3
    collision_rate = 1.0
    n_moves = 100_000
    barostat_interval = 5
    seed = 2021

    # thermodynamic parameters
    temperatures = np.array([300, 600, 1000])
    pressure = 100.0  # very high pressure, to keep the expected volume small

    # generate an alchemical system of a waterbox + alchemical ligand:
    # effectively discard ligands by running in AbsoluteFreeEnergy mode at lambda = 1.0
    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    _unbound_potentials, _sys_params, masses, coords, complex_box = get_solvent_phase_system(
        mol_a, ff, lamb=1.0, margin=0.0
    )

    unbound_potentials = list(_unbound_potentials)
    sys_params = list(_sys_params)

    # drop the nonbonded potentials
    for nb_type in (Nonbonded, NonbondedPairListPrecomputed):
        nb_pot_idx = next(i for i, pot in enumerate(unbound_potentials) if isinstance(pot, nb_type))
        unbound_potentials.pop(nb_pot_idx)
        sys_params.pop(nb_pot_idx)

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    volume_trajs = []

    relative_tolerance = 1e-2
    initial_relative_box_perturbation = 2 * relative_tolerance

    bound_potentials = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(np.asarray(params))
        bound_potentials.append(bp)

    u_impls = []
    for bp in bound_potentials:
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    # expected volume
    n_water_mols = len(group_indices) - 1  # 1 for the ligand
    expected_volume_in_md = (n_water_mols + 1) * BOLTZ * temperatures / (pressure * AVOGADRO * BAR_TO_KJ_PER_NM3)

    for i, temperature in enumerate(temperatures):
        # define a thermostat
        integrator = LangevinIntegrator(
            temperature,
            timestep,
            collision_rate,
            masses,
            seed,
        )
        integrator_impl = integrator.impl()

        v_0 = sample_velocities(masses, temperature, seed)

        # rescale the box to be approximately the desired box volume already
        rescaler = CentroidRescaler(group_indices)
        initial_volume = compute_box_volume(complex_box)
        initial_center = compute_box_center(complex_box)
        length_scale = ((1 + initial_relative_box_perturbation) * expected_volume_in_md[i] / initial_volume) ** (
            1.0 / 3
        )
        new_coords = rescaler.scale_centroids(coords, initial_center, length_scale)
        new_box = complex_box * length_scale

        baro = klass(
            new_coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0
        )

        ctxt = custom_ops.Context_f32(
            new_coords.astype(np.float32),
            v_0.astype(np.float32),
            new_box.astype(np.float32),
            integrator_impl,
            u_impls,
            movers=[baro],
        )
        _, boxes = ctxt.multiple_steps(n_moves, barostat_interval)
        vols = [compute_box_volume(box) for box in boxes]
        volume_trajs.append(vols)

    equil_time = len(volume_trajs[0]) // 2  # TODO: don't hard-code this?
    actual_volume_in_md = np.array([np.mean(volume_traj[equil_time:]) for volume_traj in volume_trajs])

    np.testing.assert_allclose(actual=actual_volume_in_md, desired=expected_volume_in_md, rtol=relative_tolerance)


def convert_to_fzset(grp_idxs):
    all_items = set()
    for grp in grp_idxs:
        items = set()
        for idx in grp:
            items.add(idx)
        items = frozenset(items)
        all_items.add(items)
    all_items = frozenset(all_items)
    return all_items


def assert_group_idxs_are_equal(set_a, set_b):
    assert convert_to_fzset(set_a) == convert_to_fzset(set_b)


@pytest.mark.nocuda
def test_get_group_indices():
    """
    Test that we generate correct group indices even when there are disconnected atoms (eg. ions) present

    Note that indices must be consecutive within each mol
    """

    bond_idxs = [[1, 0], [1, 2], [5, 6]]
    test_idxs = get_group_indices(bond_idxs, num_atoms=7)

    ref_idxs = [(0, 1, 2), (5, 6), (3,), (4,)]
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    test_idxs = get_group_indices([], num_atoms=4)
    ref_idxs = [(0,), (1,), (2,), (3,)]
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    test_idxs = get_group_indices([], num_atoms=0)
    ref_idxs = []
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    # slightly larger connected group
    test_idxs = get_group_indices([[0, 1], [1, 3], [3, 2]], num_atoms=5)
    ref_idxs = [(0, 1, 2, 3), (4,)]
    assert_group_idxs_are_equal(ref_idxs, test_idxs)

    with pytest.raises(ValueError, match="Group 0 is not constructed of consecutive atom indices"):
        # num_atoms <  an atom's index in bond_idxs
        get_group_indices([[0, 3]], num_atoms=3)


@pytest.mark.memcheck
@pytest.mark.parametrize("klass", [custom_ops.MonteCarloBarostat_f32, custom_ops.AnisotropicMonteCarloBarostat_f32])
def test_barostat_scaling_behavior(klass):
    """Verify that it is possible to retrieve and set the volume scaling factor. Also check that the adaptive behavior of the scaling can be disabled"""
    temperature = DEFAULT_TEMP
    timestep = 1.5e-3
    barostat_interval = 3
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, lamb=0.0, minimize_energy=False
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(params)
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )

    v_0 = sample_velocities(masses, temperature, seed)

    baro = klass(coords.shape[0], pressure, temperature, group_indices, barostat_interval, u_impls, seed, True, 0.0)
    # Initial volume scaling is 0
    assert np.all(np.array(baro.get_volume_scale_factor()) == 0.0)
    assert baro.get_adaptive_scaling()

    ctxt = custom_ops.Context_f32(
        coords.astype(np.float32),
        v_0.astype(np.float32),
        box.astype(np.float32),
        integrator.impl(),
        u_impls,
        movers=[baro],
    )
    ctxt.multiple_steps(15)

    # Verify that the volume scaling is non-zero
    scaling = baro.get_volume_scale_factor()
    assert np.all(np.array(scaling) > 0)

    # Set to an intentionally bad factor to ensure it adapts
    bad_scaling_factor = 0.5 * compute_box_volume(box)
    baro.set_volume_scale_factor(bad_scaling_factor)
    assert np.all(np.array(baro.get_volume_scale_factor()) == bad_scaling_factor)
    ctxt.multiple_steps(100)
    # The scaling should adapt between moves
    assert bad_scaling_factor > baro.get_volume_scale_factor()

    # Reset the scaling to the previous value
    baro.set_volume_scale_factor(scaling[0])
    assert np.all(np.array(baro.get_volume_scale_factor()) == scaling)

    # Set back to the initial volume scaling, effectively disabling the barostat
    baro.set_volume_scale_factor(0.0)
    baro.set_adaptive_scaling(False)
    assert not baro.get_adaptive_scaling()
    ctxt.multiple_steps(100)
    assert np.all(np.array(baro.get_volume_scale_factor()) == 0.0)

    # Turning adaptive scaling back on should change the scaling after some MD
    baro.set_adaptive_scaling(True)
    assert baro.get_adaptive_scaling()
    ctxt.multiple_steps(100)
    assert np.all(np.array(baro.get_volume_scale_factor()) != 0.0)

    # Check that the adaptive_scaling_enabled, initial_volume_scale_factor constructor arguments works as expected
    baro = klass(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        u_impls,
        seed,
        False,
        initial_volume_scale_factor=1.23,
    )
    assert not baro.get_adaptive_scaling()
    assert np.all(np.array(baro.get_volume_scale_factor()) == np.float32(1.23))


@pytest.mark.parametrize("scale_x, scale_y, scale_z", itertools.product([False, True], repeat=3))
def test_anisotropic_barostat(scale_x, scale_y, scale_z):
    """Verify anisotropic barostat only scales a subset of the dimensions"""
    iterations = 50
    temperature = DEFAULT_TEMP
    timestep = 1.5e-3
    barostat_interval = 2
    collision_rate = 1.0
    seed = 2021
    np.random.seed(seed)

    pressure = DEFAULT_PRESSURE

    mol_a, _, _ = get_hif2a_ligand_pair_single_topology()
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    unbound_potentials, sys_params, masses, coords, box = get_solvent_phase_system(
        mol_a, ff, lamb=0.0, minimize_energy=False
    )

    # get list of molecules for barostat by looking at bond table
    harmonic_bond_potential = get_potential_by_type(unbound_potentials, HarmonicBond)
    bond_list = get_bond_list(harmonic_bond_potential)
    group_indices = get_group_indices(bond_list, len(masses))

    u_impls = []
    for params, unbound_pot in zip(sys_params, unbound_potentials):
        bp = unbound_pot.bind(params)
        bp_impl = bp.to_gpu(precision=np.float32).bound_impl
        u_impls.append(bp_impl)

    if not (scale_x or scale_y or scale_z):
        with pytest.raises(RuntimeError, match="must scale at least one dimension"):
            custom_ops.AnisotropicMonteCarloBarostat_f32(
                coords.shape[0],
                pressure,
                temperature,
                group_indices,
                barostat_interval,
                u_impls,
                seed,
                True,
                0.0,
                scale_x,
                scale_y,
                scale_z,
            )
        return

    integrator = LangevinIntegrator(
        temperature,
        timestep,
        collision_rate,
        masses,
        seed,
    )

    v_0 = sample_velocities(masses, temperature, seed)

    # Reduce the default scaling factor
    baro = custom_ops.AnisotropicMonteCarloBarostat_f32(
        coords.shape[0],
        pressure,
        temperature,
        group_indices,
        barostat_interval,
        u_impls,
        seed,
        True,
        0.0011,
        scale_x,
        scale_y,
        scale_z,
    )
    assert baro.get_adaptive_scaling()

    ctxt = custom_ops.Context_f32(
        coords.astype(np.float32),
        v_0.astype(np.float32),
        box.astype(np.float32),
        integrator.impl(),
        u_impls,
        movers=[baro],
    )
    xs, boxes = ctxt.multiple_steps(barostat_interval * iterations, barostat_interval)

    adjusted_dims = np.array([scale_x, scale_y, scale_z])

    # Verify that all of the boxes are only changing in one dimension at a time
    assert all([(np.diag(box) != np.diag(new_box)).sum() <= 1 for box, new_box in zip(boxes, boxes[1:])])

    assert np.all(np.diag(box)[adjusted_dims] != np.diag(boxes[-1])[adjusted_dims])
    # Verify that the volume scaling is non-zero
    scaling = baro.get_volume_scale_factor()
    assert np.all(np.array(scaling) > 0)
