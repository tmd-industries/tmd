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


import jax
import numpy as np
import pytest
from common import ligand_from_smiles
from jax import grad, jit
from jax import numpy as jnp
from openmm import app, unit

from tmd.constants import BOLTZ, DEFAULT_ATOM_MAPPING_KWARGS, DEFAULT_TEMP, DEFAULT_WATER_FF
from tmd.fe import utils
from tmd.fe.atom_mapping import get_cores
from tmd.fe.model_utils import apply_hmr
from tmd.fe.single_topology import SingleTopology
from tmd.fe.system import HostSystem
from tmd.fe.topology import BaseTopology
from tmd.fe.utils import get_mol_masses, get_romol_conf
from tmd.ff import Forcefield, get_water_ff_model
from tmd.ff.handlers.openmm_deserializer import deserialize_constraints, deserialize_system
from tmd.integrator import ConstrainedLangevinIntegrator, ConstraintSolver, LangevinIntegrator
from tmd.md.builders import strip_units
from tmd.md.thermostat.utils import sample_velocities
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology


def _make_water():
    """Create a water molecule from SMILES with forcefield and topology."""
    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol = ligand_from_smiles("O", seed=2024)
    bt = BaseTopology(mol, forcefield)
    return mol, bt


def _force_from_topology(topo):
    u_fn = topo.setup_end_state().get_U_fn()
    du_dx = jax.grad(u_fn, argnums=0)

    def force(x):
        # Return numpy array to avoid implicit jax array conversions downstream
        return -np.array(du_dx(x))

    return force


@pytest.mark.nocuda
def test_constrained_langevin_single_water():
    """Simulate a single water molecule with constraints at 4fs."""
    mol, bt = _make_water()
    x0 = np.array(get_romol_conf(mol), dtype=np.float64)
    v0 = np.zeros_like(x0)
    masses = np.array(get_mol_masses(mol), dtype=np.float64)

    constraint_groups, constraint_distances = bt.get_constraint_groups()

    force = _force_from_topology(bt)

    integrator = ConstrainedLangevinIntegrator(
        force,
        masses,
        temperature=DEFAULT_TEMP,
        dt=4.0e-3,
        friction=1.0,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
    )

    rng = np.random.default_rng(2026)
    xs, vs = integrator.multiple_steps(x0, v0, n_steps=1000, rng=rng)

    for frame, velo in zip(xs, vs):
        verify_constraints(frame, velo, constraint_groups, constraint_distances, integrator._solver._tol)

    # Determinism check
    rng_comp = np.random.default_rng(2026)
    xs_comp, vs_comp = integrator.multiple_steps(x0, v0, n_steps=1000, rng=rng_comp)
    np.testing.assert_array_equal(xs, xs_comp)
    np.testing.assert_array_equal(vs, vs_comp)


@pytest.mark.nocuda
@pytest.mark.parametrize(
    "dt, n_steps",
    [
        (1.5e-3, 1000),
        (2.5e-3, 1000),
        (4.0e-3, 500),
    ],
)
def test_constrained_langevin_water_hmr(dt, n_steps):
    """Simulate water with HMR at various timesteps enabled by mass repartitioning."""
    seed = 2026
    mol, bt = _make_water()
    x0 = np.array(get_romol_conf(mol), dtype=np.float64)
    masses = np.array(get_mol_masses(mol), dtype=np.float64)
    v0 = sample_velocities(masses, DEFAULT_TEMP, seed)

    constraint_groups, constraint_distances = bt.get_constraint_groups()

    # Build bond list from constraint groups for HMR
    bond_list = []
    for group in constraint_groups:
        anchor = group[0]
        for atom in group[1:]:
            bond_list.append([anchor, atom])
    bond_list = np.array(bond_list)

    hmr_masses = apply_hmr(masses, bond_list, multiplier=2)

    force = _force_from_topology(bt)

    integrator = ConstrainedLangevinIntegrator(
        force,
        hmr_masses,
        temperature=DEFAULT_TEMP,
        dt=dt,
        friction=1.0,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
    )
    rng = np.random.default_rng(seed)

    xs, vs = integrator.multiple_steps(x0, v0, n_steps=n_steps, rng=rng)
    assert np.all(xs[0] == x0)
    assert np.all(vs[0] == v0)

    # Remove the first frame, since it is the input frames
    for frame, velo in zip(xs[1:], vs[1:]):
        verify_constraints(frame, velo, constraint_groups, constraint_distances, integrator._solver._tol)

    # Determinism check
    rng_comp = np.random.default_rng(seed)
    xs_comp, vs_comp = integrator.multiple_steps(x0, v0, n_steps=n_steps, rng=rng_comp)
    np.testing.assert_array_equal(xs, xs_comp)
    np.testing.assert_array_equal(vs, vs_comp)


@pytest.mark.nocuda
@pytest.mark.parametrize("seed", list(range(10)))
def test_constrained_langevin_inf_mass_atoms(seed):
    """Validate that constraints are still valid when atoms in constraint groups have np.inf mass."""
    mol, bt = _make_water()
    x0 = np.array(get_romol_conf(mol), dtype=np.float64)
    v0 = np.zeros_like(x0)
    rng = np.random.default_rng(seed)

    frozen_count = rng.integers(1) + 1
    frozen_idxs = rng.choice(mol.GetNumAtoms(), size=frozen_count)
    masses = np.array(get_mol_masses(mol), dtype=np.float64)
    masses[frozen_idxs] = np.inf

    constraint_groups, constraint_distances = bt.get_constraint_groups()

    force = _force_from_topology(bt)

    integrator = ConstrainedLangevinIntegrator(
        force,
        masses,
        temperature=DEFAULT_TEMP,
        dt=1.5e-3,
        friction=1.0,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
    )

    xs, vs = integrator.multiple_steps(x0, v0, n_steps=100, rng=rng)

    for frame, velo in zip(xs, vs):
        verify_constraints(frame, velo, constraint_groups, constraint_distances, integrator._solver._tol)


@pytest.mark.nocuda
def test_constrained_langevin_multiple_water_molecules():
    """Simulate multiple water molecules with constraints and HMR."""

    n_waters = 4

    water_ff = app.ForceField(f"{DEFAULT_WATER_FF}.xml")

    top = app.Topology()
    pos = unit.Quantity((), unit.angstroms)
    modeller = app.Modeller(top, pos)
    modeller.addSolvent(water_ff, numAdded=n_waters, neutralize=False, model=get_water_ff_model(DEFAULT_WATER_FF))

    # System with constraints
    omm_system = water_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )

    x0 = strip_units(modeller.positions).astype(np.float64)
    v0 = np.zeros_like(x0)

    # Verify the constraint groups from the system are reasonable
    constraint_groups, constraint_distances = deserialize_constraints(modeller.topology, x0, omm_system)
    atoms_by_idx = list(modeller.topology.atoms())
    for group in constraint_groups:
        assert len(group) == len(set(group))
        anchor_atom = group[0]
        assert atoms_by_idx[anchor_atom].element.atomic_number > 1
        for atom in group[1:]:
            assert atoms_by_idx[atom].element.atomic_number == 1

    (bond, angle, proper, improper, nonbonded), masses = deserialize_system(omm_system, cutoff=1.2)

    bond_list = bond.potential.idxs

    hmr_masses = apply_hmr(masses, bond_list, multiplier=2)

    host_system = HostSystem(
        bond=bond,
        angle=angle,
        proper=proper,
        improper=improper,
        nonbonded_all_pairs=nonbonded,
    )
    u_fn = host_system.get_U_fn()
    du_dx = jax.grad(u_fn, argnums=0)

    def force(x):
        return -np.array(du_dx(x))

    integrator = ConstrainedLangevinIntegrator(
        force,
        hmr_masses,
        temperature=DEFAULT_TEMP,
        dt=4.0e-3,
        friction=1.0,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
    )

    # Pre-constrain initial positions to satisfy constraints
    x0 = integrator._solver.apply_shake(x0)
    verify_constraints(x0, v0, constraint_groups, constraint_distances, integrator._solver._tol)

    rng = np.random.default_rng(2026)
    xs, vs = integrator.multiple_steps(x0, v0, n_steps=500, rng=rng)

    for frame, velo in zip(xs, vs):
        verify_constraints(frame, velo, constraint_groups, constraint_distances, integrator._solver._tol)

    # Determinism check
    rng_comp = np.random.default_rng(2026)
    xs_comp, vs_comp = integrator.multiple_steps(x0, v0, n_steps=500, rng=rng_comp)
    np.testing.assert_array_equal(xs, xs_comp)
    np.testing.assert_array_equal(vs, vs_comp)


@pytest.mark.nocuda
def test_constrained_langevin_constraint_group_max_7_atoms():
    """Verify that constraint groups with more than 7 atoms raises an error."""
    masses = np.concatenate([np.ones(8)]).astype(np.float64)

    bad_group = list(range(8))
    constraint_distances = [0.1] * 7

    def force(x):
        return np.zeros_like(x)

    with pytest.raises(ValueError, match="limited to 7 atoms per group"):
        ConstrainedLangevinIntegrator(
            force,
            masses,
            300.0,
            1.5e-3,
            1.0,
            constraint_groups=[bad_group],
            constraint_distances=[constraint_distances],
        )


@pytest.mark.nocuda
@pytest.mark.parametrize("dt", [1.5e-3, 4.0e-3])
def test_constrained_langevin_deterministic(dt):
    """Verify that the constrained integrator produces deterministic trajectories with same seed."""
    mol, bt = _make_water()
    x0 = np.array(get_romol_conf(mol), dtype=np.float64)
    v0 = np.zeros_like(x0)
    masses = np.array(get_mol_masses(mol), dtype=np.float64)

    constraint_groups, constraint_distances = bt.get_constraint_groups()

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    force = _force_from_topology(bt)

    integrator = ConstrainedLangevinIntegrator(
        force,
        masses,
        temperature=DEFAULT_TEMP,
        dt=dt,
        friction=1.0,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
    )

    xs1, vs1 = integrator.multiple_steps(x0, v0, n_steps=100, rng=rng1)
    xs2, vs2 = integrator.multiple_steps(x0, v0, n_steps=100, rng=rng2)

    np.testing.assert_array_equal(xs1, xs2)
    np.testing.assert_array_equal(vs1, vs2)


@pytest.mark.nocuda
@pytest.mark.parametrize("temperature", [200, DEFAULT_TEMP])
@pytest.mark.parametrize("dt", [0.1, 0.15])
@pytest.mark.parametrize("mass", [1.0, 2.0])
@pytest.mark.parametrize("friction", [0.1, +np.inf])
def test_reference_langevin_integrator(temperature, dt, friction, mass, threshold=1e-4):
    """Assert approximately canonical sampling of e^{-x^4 / kBT},
    for various settings of temperature, friction, timestep, and mass"""

    np.random.seed(2021)

    potential_fxn = lambda x: x**4
    force_fxn = lambda x: -4 * x**3

    # generate n_production_steps * n_copies samples
    n_copies = 2500
    langevin = LangevinIntegrator(force_fxn, mass, temperature, dt, friction)

    x0, v0 = 0.1 * np.ones((2, n_copies))
    xs, vs = langevin.multiple_steps(x0, v0, n_steps=2500)
    samples = xs[10:].flatten()

    # summarize using histogram
    y_empirical, edges = np.histogram(samples, bins=100, range=(-2, +2), density=True)
    x_grid = (edges[1:] + edges[:-1]) / 2

    # compare with e^{-U(x) / kB T} / Z
    y = np.exp(-potential_fxn(x_grid) / (BOLTZ * temperature))
    y_ref = y / np.trapezoid(y, x_grid)

    histogram_mse = np.mean((y_ref - y_empirical) ** 2)
    print("(temperature, friction, dt, mass) -> histogram_mse")
    print(f"{(temperature, friction, dt, mass)}".ljust(33), "->", histogram_mse)

    assert histogram_mse < threshold


@pytest.mark.nocuda
def test_reference_langevin_integrator_deterministic():
    """Asserts that trajectories are deterministic given a seed value"""
    force_fxn = lambda x: -4 * x**3
    langevin = LangevinIntegrator(force_fxn, masses=1.0, temperature=DEFAULT_TEMP, dt=0.1, friction=1.0)
    x0, v0 = 0.1 * jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 5))

    def assert_deterministic(f):
        xs1, vs1 = f(1)

        # same seed should yield same result
        xs2, vs2 = f(1)
        np.testing.assert_array_equal(xs1, xs2)
        np.testing.assert_array_equal(vs1, vs2)

        # different seed should give different result
        xs3, vs3 = f(2)
        assert not np.allclose(xs2, xs3)
        assert not np.allclose(vs2, vs3)

    assert_deterministic(lambda seed: langevin.multiple_steps(x0, v0, rng=np.random.default_rng(seed)))
    assert_deterministic(lambda seed: langevin.multiple_steps_lax(jax.random.PRNGKey(seed), x0, v0))


@pytest.mark.nocuda
def test_reference_langevin_integrator_consistent():
    """
    Asserts that the result of the implementation based on jax.lax
    primitives is consistent with a simple for-loop implementation
    """
    force_fxn = lambda x: -4 * x**3
    langevin = LangevinIntegrator(force_fxn, masses=1.0, temperature=DEFAULT_TEMP, dt=0.1, friction=1.0)
    x0, v0 = 0.1 * jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 5))
    key = jax.random.PRNGKey(1)

    def multiple_steps_reference(key, x, v, n_steps=1000):
        keys = jax.random.split(key, n_steps)
        xs, vs = [x], [v]

        for key in keys:
            new_x, new_v = langevin.step_lax(key, xs[-1], vs[-1])

            xs.append(new_x)
            vs.append(new_v)

        return np.array(xs), np.array(vs)

    xs1, vs1 = multiple_steps_reference(key, x0, v0)
    xs2, vs2 = langevin.multiple_steps_lax(key, x0, v0)

    # NOTE: result of the jax.lax implementation is NOT bitwise
    # equivalent to the pure Python implementation. This might be due
    # to loop-unrolling and reassociation optimizations performed by
    # XLA
    np.testing.assert_allclose(xs1, xs2)
    np.testing.assert_allclose(vs1, vs2)


def test_reference_langevin_integrator_with_custom_ops():
    """Run reference LangevinIntegrator on an alchemical ligand in vacuum under a few settings:
    * assert minimizer-like behavior when run at 0 temperature,
    * assert stability when run at room temperature"""

    seed = 2021
    np.random.seed(seed)
    temperature = 300
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    vac_sys = st.setup_intermediate_state(0.5)
    x_a = utils.get_romol_conf(st.mol_a)
    x_b = utils.get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    potentials = vac_sys.get_U_fns()
    masses = np.array(st.combine_masses())

    impls = [bp.to_gpu(np.float32).bound_impl for bp in potentials]
    box = 100 * np.eye(3)
    box = box.astype(np.float32)

    def custom_op_force_component(coords):
        coords = coords.astype(np.float32)

        du_dxs = np.array([bp.execute(coords, box)[0] for bp in impls])
        return -np.sum(du_dxs, 0)

    def jax_restraint(coords):
        center = jnp.mean(coords, 0)
        return jnp.sum(center**4)

    @jit
    def jax_force_component(coords):
        return -grad(jax_restraint)(coords)

    def force(coords):
        return custom_op_force_component(coords) + jax_force_component(coords)

    def F_norm(coords):
        return np.linalg.norm(force(coords))

    # define a few integrators
    dt, temperature, friction = 1.5e-3, 300.0, 10.0

    # zero temperature, infinite friction
    # (gradient descent, with no momentum)
    descender = LangevinIntegrator(force, masses, 0.0, dt, np.inf)

    # zero temperature, finite friction
    # (gradient descent, with momentum)
    dissipator = LangevinIntegrator(force, masses, 0.0, dt, friction)

    # finite temperature, finite friction
    # (Langevin, with momentum)
    sampler = LangevinIntegrator(force, masses, temperature, dt, friction)

    # apply them
    x_0 = np.array(coords)
    v_0 = np.zeros_like(x_0)

    # assert gradient descent doesn't go far, but makes force norm much smaller

    xs, vs = descender.multiple_steps(x_0, v_0, n_steps=1000)
    force_reduction_factor = F_norm(xs[0]) / F_norm(xs[-1])
    assert force_reduction_factor > 50
    assert np.abs(xs[-1] - xs[0]).max() < 0.1

    # assert *inertial* gradient descent doesn't go far, but makes force norm much smaller
    xs, vs = dissipator.multiple_steps(x_0, v_0, n_steps=1000)
    force_reduction_factor = F_norm(xs[0]) / F_norm(xs[-1])
    assert force_reduction_factor > 50
    assert np.abs(xs[-1] - xs[0]).max() < 1

    x_min = xs[-1]

    # assert that finite temperature simulation initialized from x_min
    # (1) doesn't blow up
    # (2) goes uphill
    # (3) doesn't go very far
    xs, vs = sampler.multiple_steps(x_min, v_0, n_steps=1000)
    assert F_norm(xs[-1]) / len(coords) < 1e3
    assert F_norm(xs[-1]) > F_norm(xs[0])
    assert np.abs(xs[-1] - xs[0]).max() < 1


@pytest.mark.nocuda
@pytest.mark.parametrize(
    "dt",
    [0.5e-3, 1e-3, 1.5e-3, 2.0e-3, 2.5e-3, 3e-3, 4.0e-3, 4.5e-3],
)
def test_reference_constrained_langevin_integrator_with_custom_ops(dt):
    """Run ConstrainedLangevinIntegrator on an alchemical ligand in vacuum with HMR + constraints,
    under different conditions:
    * assert minimizer-like behavior when run at 0 temperature,
    * assert stability when run at the default temperature.
    """
    seed = 2026
    np.random.seed(seed)
    temperature = DEFAULT_TEMP
    mol_a, mol_b, _ = get_hif2a_ligand_pair_single_topology()
    forcefield = Forcefield.load_default()

    # Have to use heavy matches heavy only when testing constraints
    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    kwargs["heavy_matches_heavy_only"] = True
    core = get_cores(mol_a, mol_b, **kwargs)[0]
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    vac_sys = st.setup_intermediate_state(0.5)
    x_a = utils.get_romol_conf(st.mol_a)
    x_b = utils.get_romol_conf(st.mol_b)
    coords = st.combine_confs(x_a, x_b)
    potentials = vac_sys.get_U_fns()
    masses = np.array(st.combine_masses(apply_hmr))

    impls = [bp.to_gpu(np.float32).bound_impl for bp in potentials]
    box = 100 * np.eye(3)
    box = box.astype(np.float32)

    def custom_op_force_component(coords):
        coords = coords.astype(np.float32)

        du_dxs = np.array([bp.execute(coords, box)[0] for bp in impls])
        return -np.sum(du_dxs, 0)

    def jax_restraint(coords):
        center = jnp.mean(coords, 0)
        return jnp.sum(center**4)

    @jit
    def jax_force_component(coords):
        return -grad(jax_restraint)(coords)

    def force(coords):
        return np.asarray(custom_op_force_component(coords) + jax_force_component(coords))

    def F_norm(coords):
        return np.linalg.norm(force(coords))

    # Build constraint groups for hydrogen bonds
    constraint_groups, constraint_distances = st.get_constraint_groups()

    friction = 10.0

    # zero temperature, infinite friction (gradient descent, no momentum)
    descender = ConstrainedLangevinIntegrator(
        force, masses, 0.0, dt, np.inf, constraint_groups=constraint_groups, constraint_distances=constraint_distances
    )

    # zero temperature, finite friction (gradient descent with momentum)
    dissipator = ConstrainedLangevinIntegrator(
        force, masses, 0.0, dt, friction, constraint_groups=constraint_groups, constraint_distances=constraint_distances
    )

    # finite temperature, finite friction (Langevin sampler with momentum)
    sampler = ConstrainedLangevinIntegrator(
        force,
        masses,
        temperature,
        dt,
        friction,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
    )

    # apply them
    x_0 = np.array(coords)
    v_0 = np.zeros_like(x_0)

    # Position change thresholds scale with total simulation time (n_steps * dt)
    # Reference: dt=1.5fs gives 1.5ps total; threshold scales linearly with total time
    ref_dt = 1.5e-3  # ps, reference timestep from original test
    pos_threshold_descender = 0.1 * (dt / ref_dt)
    pos_threshold_dissipator = 1.0 * (dt / ref_dt)
    pos_threshold_sampler = 1.0 * (dt / ref_dt)

    # assert gradient descent doesn't go far, but makes force norm much smaller
    xs, vs = descender.multiple_steps(x_0, v_0, n_steps=1000)
    force_reduction_factor = F_norm(xs[0]) / F_norm(xs[-1])
    assert force_reduction_factor > 2, f"Force reduction {force_reduction_factor:.2f} <= 5 at dt={dt}"
    assert np.abs(xs[-1] - xs[0]).max() < pos_threshold_descender, (
        f"Position change {np.abs(xs[-1] - xs[0]).max():.4f} > {pos_threshold_descender:.4f} at dt={dt}"
    )

    # assert *inertial* gradient descent doesn't go far, but makes force norm much smaller
    xs, vs = dissipator.multiple_steps(x_0, v_0, n_steps=1000)
    force_reduction_factor = F_norm(xs[0]) / F_norm(xs[-1])
    assert force_reduction_factor > 2, f"Force reduction {force_reduction_factor:.2f} <= 5 at dt={dt}"
    assert np.abs(xs[-1] - xs[0]).max() < pos_threshold_dissipator, (
        f"Position change {np.abs(xs[-1] - xs[0]).max():.4f} > {pos_threshold_dissipator:.4f} at dt={dt}"
    )

    x_min = xs[-1]

    # assert that finite temperature simulation initialized from x_min
    # (1) doesn't blow up
    # (2) goes uphill
    # (3) doesn't go very far
    xs, vs = sampler.multiple_steps(x_min, v_0, n_steps=1000)
    assert F_norm(xs[-1]) / len(coords) < 1e3
    assert F_norm(xs[-1]) > F_norm(xs[0])
    assert np.abs(xs[-1] - xs[0]).max() < pos_threshold_sampler, (
        f"Position change {np.abs(xs[-1] - xs[0]).max():.4f} > {pos_threshold_sampler:.4f} at dt={dt}"
    )

    # Verify constraints are satisfied throughout the trajectory
    for frame, velo in zip(xs, vs):
        verify_constraints(frame, velo, constraint_groups, constraint_distances, sampler._solver._tol)


@pytest.mark.nocuda
def test_solver_does_not_mutate_inputs():
    """solve() must not mutate input arrays.

    Currently apply_shake does x.copy() internally (x not mutated),
    while apply_rattle mutates v in-place. This inconsistency means
    callers can't rely on inputs being preserved.
    """
    mol, bt = _make_water()
    masses = np.array(get_mol_masses(mol), dtype=np.float64)

    # Get coords and constraint info from the molecule
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    x = np.array(get_romol_conf(mol), dtype=np.float64)

    # Non-zero velocities that violate velocity constraints
    v = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    x_orig = x.copy()
    v_orig = v.copy()

    solver = ConstraintSolver(masses, constraint_groups, constraint_distances)
    updated_x, updated_v = solver.solve(x, v)

    np.testing.assert_array_equal(x, x_orig)
    np.testing.assert_array_equal(v, v_orig)

    assert np.any(x != updated_x)
    assert np.any(v != updated_v)


def verify_constraints(x, v, groups: list, distances: list, tolerance: float, velo_tolerance: float = 0.3):
    for group, dists in zip(groups, distances):
        anchor = group[0]
        for atom, target_dist in zip(group[1:], dists):
            x_delta = x[anchor] - x[atom]
            v_delta = v[anchor] - v[atom]

            dist = np.linalg.norm(x_delta)
            assert np.all(np.abs(dist - target_dist) <= tolerance)

            # Weaken the velocity tolerance, it is more of an upper bound since there is no
            # final velocity constraints for performance purposes.
            vel_violation = np.abs(np.dot(x_delta, v_delta))
            try:
                assert vel_violation < velo_tolerance, (
                    f"RATTLE velocity constraint violated: |r·v_rel| = {vel_violation:.2e} for atoms ({anchor}, {atom})"
                )
            except AssertionError:
                print(velo_tolerance, anchor, atom, v[anchor], v[atom])
                raise
