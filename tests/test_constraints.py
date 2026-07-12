import numpy as np
import pytest
from common import ligand_from_smiles
from rdkit import Chem

from tmd.fe.topology import BaseTopology
from tmd.fe.utils import get_mol_masses, get_romol_conf
from tmd.ff import Forcefield
from tmd.lib import ConstraintGroups, custom_ops
from tmd.md.constraints.utils import get_hydrogen_bond_constraint_groups


@pytest.fixture(scope="module")
def simple_mol():
    return ligand_from_smiles("C(F)(F)(F)F")


@pytest.fixture(scope="module")
def water_mol():
    return ligand_from_smiles("O")


@pytest.fixture(scope="module")
def ff():
    return Forcefield.load_from_file("smirnoff_2_0_0_sc.py")


@pytest.mark.nogpu
def test_constraint_groups_from_mol(simple_mol, water_mol):
    constraints = get_hydrogen_bond_constraint_groups(simple_mol)
    assert len(constraints.groups) == 0

    constraints = get_hydrogen_bond_constraint_groups(water_mol)
    assert len(constraints.groups) == 1
    assert constraints.groups[0] == [0, 1, 2]
    assert len(constraints.distances[0]) == 2

    rng = np.random.default_rng(2026)
    # Verify that molecules with bonds that don't have hydrogens at the end function correctly
    mol = Chem.MolFromMolBlock(
        """7
     RDKit          3D

 40 41  0  0  1  0  0  0  0  0999 V2000
    1.7170    8.4485   31.3627 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6267    7.7483   30.9663 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.0102    8.1187   29.8231 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4951    9.1722   29.0211 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5948    9.9235   29.4650 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.2509    9.5513   30.6577 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.3867   10.1680   31.1426 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.9158   11.2978   30.4450 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.3491   11.5734   30.8875 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.6713   12.6161   31.7877 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.9888   12.7648   32.2651 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.0031   11.8972   31.8217 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.7063   10.9065   30.8723 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.3927   10.7566   30.3985 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0434    9.4184   28.0734 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9249   10.7488   28.8542 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.2629   12.1550   30.6040 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.9394   11.1387   29.3677 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.4951   13.7761   32.2650 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    7.2266   13.5581   32.9605 H   0  0  0  0  0  0  0  0  0  0  0  0
    9.0139   12.0044   32.1832 H   0  0  0  0  0  0  0  0  0  0  0  0
    8.4896   10.2611   30.5016 H   0  0  0  0  0  0  0  0  0  0  0  0
    6.0950    9.5937   29.1711 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    2.1858    8.1127   32.2760 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1308    7.3813   29.4659 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2070    6.5310   30.0143 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4486    8.0373   29.4835 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7504    8.5820   30.9069 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7190    7.7762   31.6405 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0064    9.3161   31.2192 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.0222    9.1937   30.9889 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.9249   10.1254   30.8267 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5398    9.1644   28.4412 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2181    8.8144   27.4650 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5550    9.5521   28.3399 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9052    9.9936   28.7262 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5054    6.9821   29.0980 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2673    6.5056   28.1465 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5949    6.2008   29.8498 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.4959    7.4269   28.9991 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  1  6  1  0
  1 24  1  0
  2  3  1  0
  3  4  2  0
  3 25  1  0
  4  5  1  0
  4 15  1  0
  5  6  2  0
  5 16  1  0
  6  7  1  0
  7  8  1  0
  8  9  1  0
  8 17  1  0
  8 18  1  0
  9 10  2  0
  9 14  1  0
 10 11  1  0
 10 19  1  0
 11 12  2  0
 11 20  1  0
 12 13  1  0
 12 21  1  0
 13 14  2  0
 13 22  1  0
 14 23  1  0
 25 26  1  0
 25 27  1  0
 27 28  1  0
 27 33  1  0
 27 37  1  0
 28 29  1  0
 28 30  1  0
 28 31  1  0
 31 32  1  0
 33 34  1  0
 33 35  1  0
 33 36  1  0
 37 38  1  0
 37 39  1  0
 37 40  1  0
M  END""",
        removeHs=False,
    )
    constraints = get_hydrogen_bond_constraint_groups(mol)
    assert len(constraints.groups) == 12

    # Shuffling the atom order shouldn't change the results
    for _ in range(10):
        atom_ordering = np.arange(mol.GetNumAtoms())
        rng.shuffle(atom_ordering)
        mol = Chem.RenumberAtoms(mol, atom_ordering.tolist())
        constraints = get_hydrogen_bond_constraint_groups(mol)
        assert len(constraints.groups) == 12


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
    constraints_obj = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
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
    masses = get_mol_masses(simple_mol).astype(precision)

    # Create constraints with empty groups
    klass = custom_ops.ConstraintGroups_f32
    if precision == np.float64:
        klass = custom_ops.ConstraintGroups_f64
    constraints = klass(masses, [], [], 15, 1e-8)

    x0 = get_romol_conf(simple_mol).astype(precision)
    original_coords = x0.copy()

    constrained = constraints.constrain_positions(x0)

    # There should be no change
    np.testing.assert_equal(constrained, original_coords)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrain_positions_water(precision, water_mol, ff):
    bt = BaseTopology(water_mol, ff)
    masses = get_mol_masses(water_mol).astype(precision)
    constraints_obj = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
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
    for group, dists in zip(constraints_obj.groups, constraints_obj.distances):
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
    constraints_obj = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
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
    constraints_obj = bt.get_constraint_groups()

    constraints = (
        custom_ops.ConstraintGroups_f32(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
        )
        if precision == np.float32
        else custom_ops.ConstraintGroups_f64(
            masses, constraints_obj.groups, [[float(x) for x in d] for d in constraints_obj.distances], 15, 1e-8
        )
    )

    x0 = get_romol_conf(water_mol).astype(precision)
    v0 = np.random.randn(*x0.shape).astype(precision)

    constrained_v = constraints.constrain_velocities(x0, v0)

    # Verify velocity constraints: (v_anchor - v_atom) . delta_r = 0
    tol = 1e-4 if precision == np.float32 else 1e-7
    for group in constraints_obj.groups:
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
    constraints_obj = bt.get_constraint_groups()

    constraints = ConstraintGroups(
        groups=constraints_obj.groups,
        distances=constraints_obj.distances,
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
    constraints_obj = bt.get_constraint_groups()

    # Create constraints with water group index
    constraints = ConstraintGroups(
        groups=constraints_obj.groups,
        distances=constraints_obj.distances,
        water_group_indices=np.array([0]),
    )

    sorted_constraints = constraints.sort()
    assert sorted_constraints.water_group_indices[0] == 0
    assert len(sorted_constraints.groups) == len(constraints.groups)
