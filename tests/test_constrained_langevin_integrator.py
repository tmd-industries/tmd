import numpy as np
import pytest
from common import ligand_from_smiles

from tmd.constants import DEFAULT_TEMP
from tmd.fe.single_topology import SingleTopology
from tmd.fe.topology import BaseTopology, DualTopology, MultiTopology
from tmd.fe.utils import get_mol_masses, get_romol_conf, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.lib import (
    ConstrainedLangevinIntegrator,
    ConstraintGroups,
    Context,
    LangevinIntegrator,
    custom_ops,
)
from tmd.md.builders import build_water_system
from tmd.testsystems.relative import get_hif2a_ligand_pair
from tmd.utils import path_to_internal_file


@pytest.fixture(scope="module")
def mols_by_name():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "ligands.sdf") as path_to_ligand:
        return read_sdf_mols_by_name(path_to_ligand)


def verify_constraints(constraints: ConstraintGroups, frames):
    constraint_groups = constraints.groups
    constraint_distances = constraints.distances
    tol = constraints.tolerance

    # Verify constraints are satisfied throughout the trajectory
    for frame in frames:
        for group, dists in zip(constraint_groups, constraint_distances):
            anchor = group[0]
            for atom, target_dist in zip(group[1:], dists):
                dist = np.linalg.norm(frame[anchor] - frame[atom])
                assert np.abs(dist - target_dist) <= tol, (
                    f"Constraint violated: |dist - target| = {np.abs(dist - target_dist):.2e} "
                    f"for atoms ({anchor}, {atom})"
                )


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrained_langevin_integrator_construction(precision, mols_by_name):
    """Test that the ConstrainedLangevinIntegrator class can be constructed."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)
    masses = get_mol_masses(mol).astype(np.float32)
    constraints = bt.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses.squeeze().astype(np.float64),
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(precision)
    assert intg_impl.num_systems() == 1


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrained_langevin_integrator_dataclass(precision, mols_by_name):
    """Test the Python dataclass wrapper for the constrained langevin integrator"""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)
    masses = get_mol_masses(mol)
    constraints = bt.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
        constraints=constraints,
    )

    intg_impl = intg.impl(precision)
    if precision == np.float32:
        assert isinstance(intg_impl, custom_ops.ConstrainedLangevinIntegrator_f32)
    else:
        assert isinstance(intg_impl, custom_ops.ConstrainedLangevinIntegrator_f64)
    assert intg_impl.num_systems() == 1


def test_constrained_langevin_integrator_dual_topology(mols_by_name):
    """Test creation with DualTopology (two ligands)."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol_a = mols_by_name["30"]
    mol_b = mols_by_name["67"]
    dt = DualTopology(mol_a, mol_b, ff)
    masses = np.concatenate([get_mol_masses(mol_a), get_mol_masses(mol_b)]).astype(np.float32)
    constraints = dt.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=4e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(np.float32)
    assert intg_impl.num_systems() == 1


@pytest.mark.memcheck
def test_constrained_langevin_integrator_single_topology():
    """Test creation with SingleTopology"""
    ff = Forcefield.load_default()
    seed = 2026
    precision = np.float32
    mol_a, mol_b, core = get_hif2a_ligand_pair(1, 4)
    st = SingleTopology(mol_a, mol_b, core, ff)
    masses = np.array(st.combine_masses(use_hmr=True))

    x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b), lamb=0.0)
    v0 = np.zeros_like(x0)
    box0 = np.eye(3) * 100.0

    guest_system = st.setup_intermediate_state(0.0)

    bps = []
    for pot in guest_system.get_U_fns():
        bp = pot.to_gpu(precision).bound_impl
        bps.append(bp)

    constraints = st.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=4e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(np.float32)
    assert intg_impl.num_systems() == 1

    ctxt = Context(x0, v0, box0, intg_impl, bps, precision=precision)

    steps = 1000
    interval = steps // 10
    xs, boxes = ctxt.multiple_steps(steps, store_x_interval=interval)

    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(boxes))

    verify_constraints(constraints, xs)

    # Determinism check
    ctxt2 = Context(x0, v0, box0, intg.impl(precision), bps, precision=precision)
    xs2, boxes2 = ctxt2.multiple_steps(steps, store_x_interval=interval)
    np.testing.assert_array_equal(xs, xs2)
    np.testing.assert_array_equal(boxes, boxes2)


def test_constrained_langevin_integrator_multi_topology(mols_by_name):
    """Test creation with MultiTopology (multiple ligands)."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mols = list(mols_by_name.values())
    mt = MultiTopology(mols, ff)
    masses = np.concatenate([get_mol_masses(m) for m in mols]).astype(np.float32)
    constraints = mt.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=4e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(np.float32)
    assert intg_impl.num_systems() == 1


def test_constrained_langevin_integrator_empty_constraints():
    """Test creation with no constraint groups (degenerates to regular Langevin)."""
    seed = 2026
    mol = ligand_from_smiles("C(F)(F)(F)F")
    masses = get_mol_masses(mol).astype(np.float64)

    constraints = ConstraintGroups(
        groups=[],
        distances=[],
        water_group_indices=np.array([], dtype=np.int_),
        tolerance=1e-6,
        max_iter=20,
    )
    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses,
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(np.float32)
    assert intg_impl.num_systems() == 1


def test_constrained_langevin_integrator_2d_masses(mols_by_name):
    """Test creation with 2D masses array (multiple systems)."""
    n_systems = 2
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)
    masses = np.tile(get_mol_masses(mol), (n_systems, 1)).astype(np.float64)
    constraints = bt.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses,
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(np.float32)
    assert intg_impl.num_systems() == n_systems


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_custom_ops_constrained_langevin_single_water(precision):
    """Simulate a single water molecule with constraints using the CUDA integrator."""
    seed = 2026
    mol = ligand_from_smiles("O", seed=seed)
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    bt = BaseTopology(mol, ff)

    x0 = get_romol_conf(mol).astype(precision)
    v0 = np.zeros_like(x0)
    masses = get_mol_masses(mol).astype(precision)

    constraints = bt.get_constraint_groups()

    guest_system = bt.setup_end_state()
    box0 = np.eye(3, dtype=precision) * 10.0

    bps = []
    for pot in guest_system.get_U_fns():
        bp = pot.to_gpu(precision).bound_impl
        bps.append(bp)

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=4e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
        constraints=constraints,
    )
    intg_impl = intg.impl(precision)

    ctxt = Context(x0, v0, box0, intg_impl, bps, precision=precision)

    steps = 1000
    interval = steps // 10
    xs, boxes = ctxt.multiple_steps(steps, store_x_interval=interval)

    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(boxes))

    verify_constraints(constraints, xs)

    # Determinism check
    ctxt2 = Context(x0, v0, box0, intg.impl(precision), bps, precision=precision)
    xs2, boxes2 = ctxt2.multiple_steps(steps, store_x_interval=interval)
    np.testing.assert_array_equal(xs, xs2)
    np.testing.assert_array_equal(boxes, boxes2)


@pytest.mark.parametrize("precision", [pytest.param(np.float32, marks=pytest.mark.memcheck), np.float64])
def test_custom_ops_constrained_langevin_bulk_solvent(precision):
    """Simulate a bulk solvent using the Constrained Langevin integrator"""
    seed = 2026
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    host_config = build_water_system(3.0, ff.water_ff)

    x0 = host_config.conf
    v0 = np.zeros_like(x0)
    masses = host_config.masses

    box0 = host_config.box

    bps = []
    for pot in host_config.host_system.get_U_fns():
        bp = pot.to_gpu(precision).bound_impl
        bps.append(bp)

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=4e-3,
        friction=1.0,
        masses=masses.astype(precision),
        seed=seed,
        constraints=host_config.constraints,
    )
    intg_impl = intg.impl(precision)

    ctxt = Context(x0, v0, box0, intg_impl, bps, precision=precision)

    steps = 1000
    interval = steps // 10
    xs, boxes = ctxt.multiple_steps(steps, store_x_interval=interval)

    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(boxes))

    verify_constraints(host_config.constraints, xs)

    # Determinism check
    ctxt2 = Context(x0, v0, box0, intg.impl(precision), bps, precision=precision)
    xs2, boxes2 = ctxt2.multiple_steps(steps, store_x_interval=interval)
    np.testing.assert_array_equal(xs, xs2)
    np.testing.assert_array_equal(boxes, boxes2)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrained_langevin_equivalence_without_constraints(mols_by_name, precision):
    """Verify that ConstrainedLangevinIntegrator produces identical results to
    LangevinIntegrator when no constraints are applied."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)

    x0 = get_romol_conf(mol).astype(precision)
    v0 = np.zeros_like(x0)
    masses = get_mol_masses(mol).astype(precision)
    box0 = np.eye(3, dtype=precision) * 10.0

    guest_system = bt.setup_end_state()

    bps = []
    for pot in guest_system.get_U_fns():
        bp = pot.to_gpu(precision).bound_impl
        bps.append(bp)

    # Empty constraints
    constraints = ConstraintGroups(
        groups=[],
        distances=[],
        water_group_indices=np.array([], dtype=np.int_),
        tolerance=1e-6,
        max_iter=20,
    )

    # Run with regular LangevinIntegrator
    intg_regular = LangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
    )
    intg_regular_impl = intg_regular.impl(precision)

    ctxt1 = Context(x0, v0, box0, intg_regular_impl, bps, precision=precision)

    steps = 100
    interval = 1
    xs1, boxes1 = ctxt1.multiple_steps(steps, store_x_interval=interval)

    # Run with ConstrainedLangevinIntegrator with empty constraints
    intg_constrained = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses.astype(np.float64),
        seed=seed,
        constraints=constraints,
    )
    intg_constrained_impl = intg_constrained.impl(precision)

    ctxt2 = Context(x0, v0, box0, intg_constrained_impl, bps, precision=precision)

    xs2, boxes2 = ctxt2.multiple_steps(steps, store_x_interval=interval)

    np.testing.assert_array_equal(xs1, xs2)
    np.testing.assert_array_equal(boxes1, boxes2)
