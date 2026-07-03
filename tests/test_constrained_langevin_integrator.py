# Copyright 2026, Forrest York

import numpy as np
import pytest
from common import ligand_from_smiles

from tmd.constants import DEFAULT_TEMP
from tmd.fe.topology import BaseTopology, DualTopology, MultiTopology
from tmd.fe.utils import get_mol_masses, get_romol_conf, read_sdf_mols_by_name
from tmd.ff import Forcefield
from tmd.lib import ConstrainedLangevinIntegrator, Context, custom_ops
from tmd.utils import path_to_internal_file


@pytest.fixture(scope="module")
def mols_by_name():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "ligands.sdf") as path_to_ligand:
        return read_sdf_mols_by_name(path_to_ligand)


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrained_langevin_integrator_construction(precision, mols_by_name):
    """Test that the ConstrainedLangevinIntegrator class can be constructed."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)
    masses = get_mol_masses(mol).astype(np.float32)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    klass = custom_ops.ConstrainedLangevinIntegrator_f32
    if precision == np.float64:
        klass = custom_ops.ConstrainedLangevinIntegrator_f64

    intg = klass(
        masses.squeeze(),
        DEFAULT_TEMP,
        2e-3,
        1.0,
        seed,
        constraint_groups,
        [[float(x) for x in dist] for dist in constraint_distances],
        1e-6,
        20,
    )
    assert intg.num_systems() == 1


@pytest.mark.parametrize("precision", [np.float32, np.float64])
def test_constrained_langevin_integrator_dataclass(precision, mols_by_name):
    """Test the Python dataclass wrapper for the constrained langevin integrator"""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)
    masses = get_mol_masses(mol)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=2e-3,
        friction=1.0,
        masses=masses,
        seed=seed,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
        tolerance=1e-6,
        max_iter=20,
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
    constraint_groups, constraint_distances = dt.get_constraint_groups()

    intg = custom_ops.ConstrainedLangevinIntegrator_f32(
        masses,
        DEFAULT_TEMP,
        4e-3,
        1.0,
        seed,
        constraint_groups,
        constraint_distances,
        1e-8,
        15,
    )
    assert intg.num_systems() == 1


def test_constrained_langevin_integrator_multi_topology(mols_by_name):
    """Test creation with MultiTopology (multiple ligands)."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mols = list(mols_by_name.values())
    mt = MultiTopology(mols, ff)
    masses = np.concatenate([get_mol_masses(m) for m in mols]).astype(np.float32)
    constraint_groups, constraint_distances = mt.get_constraint_groups()

    intg = custom_ops.ConstrainedLangevinIntegrator_f32(
        masses,
        DEFAULT_TEMP,
        4e-3,
        1.0,
        seed,
        constraint_groups,
        constraint_distances,
        1e-8,
        15,
    )
    assert intg.num_systems() == 1


def test_constrained_langevin_integrator_empty_constraints():
    """Test creation with no constraint groups (degenerates to regular Langevin)."""
    seed = 2026
    mol = ligand_from_smiles("C(F)(F)(F)F")
    masses = get_mol_masses(mol).astype(np.float32)

    intg = custom_ops.ConstrainedLangevinIntegrator_f32(
        masses,
        DEFAULT_TEMP,
        2e-3,
        1.0,
        seed,
        [],  # no constraint groups
        [],  # no constraint distances
        1e-6,
        20,
    )
    assert intg.num_systems() == 1


def test_constrained_langevin_integrator_2d_masses(mols_by_name):
    """Test creation with 2D masses array (multiple systems)."""
    n_systems = 2
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    seed = 2026
    mol = mols_by_name["67"]
    bt = BaseTopology(mol, ff)
    masses = np.tile(get_mol_masses(mol), (n_systems, 1)).astype(np.float32)
    constraint_groups, constraint_distances = bt.get_constraint_groups()

    intg = custom_ops.ConstrainedLangevinIntegrator_f32(
        masses,
        DEFAULT_TEMP,
        2e-3,
        1.0,
        seed,
        constraint_groups,
        constraint_distances,
        1e-6,
        20,
    )
    assert intg.num_systems() == n_systems


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

    constraint_groups, constraint_distances = bt.get_constraint_groups()

    guest_system = bt.setup_end_state()
    box0 = np.eye(3, dtype=precision) * 10.0

    bps = []
    for pot in guest_system.get_U_fns():
        bp = pot.to_gpu(precision).bound_impl
        bps.append(bp)

    tol = 1e-5
    intg = ConstrainedLangevinIntegrator(
        temperature=DEFAULT_TEMP,
        dt=4e-3,
        friction=1.0,
        masses=masses,
        seed=seed,
        constraint_groups=constraint_groups,
        constraint_distances=constraint_distances,
        tolerance=tol,
    )
    intg_impl = intg.impl(precision)

    ctxt = Context(x0, v0, box0, intg_impl, bps, precision=precision)

    steps = 1000
    interval = steps // 10
    xs, boxes = ctxt.multiple_steps(steps, store_x_interval=interval)

    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(boxes))

    # Verify constraints are satisfied throughout the trajectory
    for frame in xs.reshape(-1, len(x0), 3):
        for group, dists in zip(constraint_groups, constraint_distances):
            anchor = group[0]
            for atom, target_dist in zip(group[1:], dists):
                dist = np.linalg.norm(frame[anchor] - frame[atom])
                assert np.abs(dist - target_dist) <= tol, (
                    f"Constraint violated: |dist - target| = {np.abs(dist - target_dist):.2e} "
                    f"for atoms ({anchor}, {atom})"
                )

    # Determinism check
    ctxt2 = Context(x0, v0, box0, intg.impl(precision), bps, precision=precision)
    xs2, boxes2 = ctxt2.multiple_steps(steps, store_x_interval=interval)
    np.testing.assert_array_equal(xs, xs2)
    np.testing.assert_array_equal(boxes, boxes2)
