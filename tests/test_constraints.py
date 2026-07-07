import numpy as np
import pytest
from common import ligand_from_smiles

from tmd.fe.topology import BaseTopology
from tmd.fe.utils import get_mol_masses, get_romol_conf
from tmd.ff import Forcefield
from tmd.lib import ConstraintGroups, custom_ops


@pytest.fixture(scope="module")
def simple_mol():
    return ligand_from_smiles("C(F)(F)(F)F")


@pytest.fixture(scope="module")
def water_mol():
    return ligand_from_smiles("O")


@pytest.fixture(scope="module")
def ff():
    return Forcefield.load_from_file("smirnoff_2_0_0_sc.py")


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_empty_constraints(precision, simple_mol):
    masses = get_mol_masses(simple_mol).astype(precision)
    constraints = (
        custom_ops.ConstraintGroups_f32(masses, [], [], 15, 1e-8)
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(masses, [], [], 15, 1e-8)
    )
    assert constraints.num_systems() == 1
    assert constraints.num_atoms() == len(masses)
    assert constraints.n_groups() == 0


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_single_group_constraints(precision, water_mol, ff):
    bt = BaseTopology(water_mol, ff)
    masses = get_mol_masses(water_mol).astype(precision)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
    )
    assert constraints.num_systems() == 1
    assert constraints.n_groups() > 0


def test_invalid_iterations(simple_mol):
    masses = get_mol_masses(simple_mol).astype(np.float32)
    with pytest.raises(RuntimeError, match="iterations must be at least one"):
        custom_ops.ConstraintGroups_f32(masses, [], [], 0, 1e-8)


def test_invalid_tolerance(simple_mol):
    masses = get_mol_masses(simple_mol).astype(np.float32)
    with pytest.raises(RuntimeError, match="tolerance must be greater than 0.0"):
        custom_ops.ConstraintGroups_f32(masses, [], [], 15, -1e-8)


def test_mismatched_groups_distances(simple_mol):
    masses = get_mol_masses(simple_mol).astype(np.float32)
    with pytest.raises(RuntimeError, match="number of groups must match number of distance arrays"):
        custom_ops.ConstraintGroups_f32(masses, [[0, 1]], [], 15, 1e-8)


def test_group_too_small(simple_mol):
    masses = get_mol_masses(simple_mol).astype(np.float32)
    with pytest.raises(RuntimeError, match="must provide groups with at least 2 atoms"):
        custom_ops.ConstraintGroups_f32(masses, [[0]], [[1.0]], 15, 1e-8)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrain_positions_empty(precision, simple_mol, ff):
    bt = BaseTopology(simple_mol, ff)
    masses = get_mol_masses(simple_mol).astype(precision)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    # Create constraints with empty groups
    constraints = (
        custom_ops.ConstraintGroups_f32(masses, [], [], 15, 1e-8)
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(masses, [], [], 15, 1e-8)
    )

    x0 = get_romol_conf(simple_mol).astype(precision)
    original_coords = x0.copy()

    constrained = constraints.constrain_positions(x0)

    np.testing.assert_allclose(constrained, original_coords, rtol=1e-5)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrain_positions_water(precision, water_mol, ff):
    bt = BaseTopology(water_mol, ff)
    masses = get_mol_masses(water_mol).astype(precision)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
    )

    x0 = get_romol_conf(water_mol).astype(precision)

    # First verify that the original coords don't satisfy constraints
    # (they might by chance, but let's intentionally perturb them)
    perturbed = x0.copy()
    perturbed[1] += np.array([0.5, 0.0, 0.0])  # Move H atom significantly

    constrained = constraints.constrain_positions(perturbed)

    # Verify constraints are satisfied
    tol = 1e-5
    if precision == np.float64:
        tol = 1e-8
    for group, dists in zip(constraint_groups, constraint_distances):
        anchor = group[0]
        for atom, target_dist in zip(group[1:], dists):
            dist = np.linalg.norm(constrained[anchor] - constrained[atom])
            assert np.abs(dist - target_dist) <= tol, (
                f"Constraint violated: |dist - target| = {np.abs(dist - target_dist):.2e} for atoms ({anchor}, {atom})"
            )


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrain_positions_already_satisfied(precision, water_mol, ff):
    bt = BaseTopology(water_mol, ff)
    masses = get_mol_masses(water_mol).astype(precision)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
    )

    x0 = get_romol_conf(water_mol).astype(precision)
    constrained = constraints.constrain_positions(x0)

    # If the coords already satisfy constraints, they should be nearly unchanged
    np.testing.assert_allclose(constrained, x0, rtol=1e-4)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrain_velocities_empty(precision, simple_mol):
    masses = get_mol_masses(simple_mol).astype(precision)

    constraints = (
        custom_ops.ConstraintGroups_f32(masses, [], [], 15, 1e-8)
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(masses, [], [], 15, 1e-8)
    )

    x0 = get_romol_conf(simple_mol).astype(precision)
    v0 = np.random.randn(*x0.shape).astype(precision)

    constrained_v = constraints.constrain_velocities(x0, v0)

    np.testing.assert_allclose(constrained_v, v0, rtol=1e-5)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrain_velocities_water(precision, water_mol, ff):
    bt = BaseTopology(water_mol, ff)
    masses = get_mol_masses(water_mol).astype(precision)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraint_groups, [[float(x) for x in d] for d in constraint_distances], 15, 1e-8
        )
    )

    x0 = get_romol_conf(water_mol).astype(precision)
    v0 = np.random.randn(*x0.shape).astype(precision)

    constrained_v = constraints.constrain_velocities(x0, v0)

    # Verify velocity constraints: (v_anchor - v_atom) . delta_r = 0
    tol = 1e-4 if precision == np.float32 else 1e-7
    for group in constraint_groups:
        anchor = group[0]
        for atom in group[1:]:
            delta_r = x0[anchor] - x0[atom]
            delta_v = constrained_v[anchor] - constrained_v[atom]
            dot_product = np.dot(delta_r, delta_v)
            assert np.abs(dot_product) <= tol, (
                f"Velocity constraint violated: |dot| = {np.abs(dot_product):.2e} for atoms ({anchor}, {atom})"
            )

    @pytest.mark.parametrize("precision", [np.float32, np.float64])
    def test_dataclass_impl(precision, water_mol, ff):
        bt = BaseTopology(water_mol, ff)
        masses = get_mol_masses(water_mol)
        constraint_groups, constraint_distances = bt.get_constraint_groups()

        constraints = ConstraintGroups(
            constraint_groups=constraint_groups,
            constraint_distances=constraint_distances,
            water_group_indices=np.array([]),
            tolerance=1e-8,
            max_iter=15,
        )

        impl = constraints.impl(masses, precision)
        if precision == np.float32:
            assert isinstance(impl, custom_ops.ConstraintGroups_f32)
        else:
            assert isinstance(impl, custom_ops.ConstraintGroups_f64)
        assert impl.num_systems() == 1

    @pytest.mark.parametrize("precision", [np.float32, np.float64])
    def test_dataclass_sort(precision, water_mol, ff):
        bt = BaseTopology(water_mol, ff)
        constraint_groups, constraint_distances = bt.get_constraint_groups()

        # Create constraints with water group index
        constraints = ConstraintGroups(
            constraint_groups=constraint_groups,
            constraint_distances=constraint_distances,
            water_group_indices=np.array([0]),
        )

        sorted_constraints = constraints.sort()
        assert sorted_constraints.water_group_indices[0] == 0
        assert sorted_constraints.n_groups() == constraints.n_groups()
