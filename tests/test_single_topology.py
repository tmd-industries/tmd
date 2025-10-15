# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025, Forrest York
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

import functools
import time

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pytest
from common import ligand_from_smiles
from hypothesis import event, given, seed
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd import potentials
from tmd.constants import (
    DEFAULT_ATOM_MAPPING_KWARGS,
    DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    DEFAULT_CHIRAL_BOND_RESTRAINT_K,
)
from tmd.fe import atom_mapping, single_topology
from tmd.fe.dummy import MultipleAnchorWarning, canonicalize_bond
from tmd.fe.single_topology import (
    AtomMapMixin,
    ChargePertubationError,
    CoreBondChangeWarning,
    SingleTopology,
    assert_default_system_constraints,
    canonicalize_bonds,
    canonicalize_chiral_atom_idxs,
    canonicalize_improper_idxs,
    cyclic_difference,
    interpolate_w_coord,
    setup_dummy_interactions_from_ff,
)
from tmd.fe.system import minimize_scipy, simulate_system
from tmd.fe.utils import (
    get_mol_name,
    get_romol_conf,
    read_sdf,
    read_sdf_mols_by_name,
    set_romol_conf,
)
from tmd.ff import Forcefield
from tmd.md.builders import build_water_system
from tmd.potentials.jax_utils import pairwise_distances
from tmd.utils import path_to_internal_file

setup_chiral_dummy_interactions_from_ff = functools.partial(
    setup_dummy_interactions_from_ff,
    chiral_atom_k=DEFAULT_CHIRAL_ATOM_RESTRAINT_K,
    chiral_bond_k=DEFAULT_CHIRAL_BOND_RESTRAINT_K,
)


def _get_hif2a_mol_pairs(shuffle: bool = False, seed: int = 2029) -> list[Chem.Mol]:
    with path_to_internal_file("tmd.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    pairs = [(mol_a, mol_b) for mol_a in mols for mol_b in mols]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(pairs)
    return pairs


@pytest.mark.nocuda
def test_setup_chiral_dummy_atoms():
    """
    Test that we setup the correct geometries for each of the 8 types specified in single topology
    """
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol = Chem.MolFromMolBlock(
        """
  Mrv2311 02232401393D

  5  4  0  0  0  0            999 V2000
    1.8515    0.0946    2.1705 F   0  0  0  0  0  0  0  0  0  0  0  0
    1.0043    0.5689    1.2025 C   0  0  2  0  0  0  0  0  0  0  0  0
   -0.6276    0.0025    1.5238 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    1.5780   -0.0702   -0.5225 Br  0  0  0  0  0  0  0  0  0  0  0  0
    1.0321    2.6887    1.2159 I   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$"""
    )

    # First 3 tests, center is core
    dg_0 = [0, 2, 3]
    core_0 = [1]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_0, root_anchor_atom=1, nbr_anchor_atom=None, core_atoms=core_0
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_1 = [0, 3]
    core_1 = [1, 2]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_1, root_anchor_atom=1, nbr_anchor_atom=None, core_atoms=core_1
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_2 = [0]
    core_2 = [1, 2, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_2, root_anchor_atom=1, nbr_anchor_atom=None, core_atoms=core_2
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    # Next 3 tests, center is not core
    dg_3 = [1, 2, 3]
    core_3 = [0]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_3, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_3
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_4 = [1, 2]
    core_4 = [0, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_4, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_4
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K])

    dg_5 = [0, 1, 2, 3]
    core_5 = [4]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_5, root_anchor_atom=4, nbr_anchor_atom=None, core_atoms=core_5
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    np.testing.assert_array_equal(chiral_atom_idxs, [[1, 2, 0, 3], [1, 0, 2, 4], [1, 3, 0, 4], [1, 2, 3, 4]])
    np.testing.assert_array_equal(chiral_atom_params, [DEFAULT_CHIRAL_ATOM_RESTRAINT_K] * 4)

    # The next two should return empty
    dg_6 = [1]
    core_6 = [0, 2, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_6, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_6
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    assert len(chiral_atom_idxs) == 0
    assert len(chiral_atom_params) == 0

    dg_7 = []
    core_7 = [0, 1, 2, 3]
    idxs, params = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dg_7, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_7
    )
    chiral_atom_idxs = idxs[-1]
    chiral_atom_params = params[-1]
    assert len(chiral_atom_idxs) == 0
    assert len(chiral_atom_params) == 0


def assert_bond_sets_equal(bonds_a, bonds_b):
    def f(bonds):
        return {tuple(idxs) for idxs in bonds}

    return f(bonds_a) == f(bonds_b)


@pytest.mark.nocuda
def test_phenol():
    """
    Test that dummy interactions are setup correctly for a phenol. We want to check that bonds and angles
    are present when either a single root anchor is provided, or when a root anchor and a neighbor anchor is provided.
    """
    mol = ligand_from_smiles("c1ccccc1O", seed=2022)

    all_atoms_set = set([a for a in range(mol.GetNumAtoms())])

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    dg_0 = [6, 12]
    core_0 = list(all_atoms_set.difference(dg_0))

    # set [O,H] as the dummy group
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_0, root_anchor_atom=5, nbr_anchor_atom=None, core_atoms=core_0
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(5, 6), (6, 12)])
    assert_bond_sets_equal(angle_idxs, [(5, 6, 12)])
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    # set [O,H] as the dummy group but allow an extra angle
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_0, root_anchor_atom=5, nbr_anchor_atom=0, core_atoms=core_0
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(5, 6), (6, 12)])
    assert_bond_sets_equal(angle_idxs, [(5, 6, 12), (0, 5, 6)])
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    dg_1 = [12]
    core_1 = list(all_atoms_set.difference(dg_1))

    # set [H] as the dummy group, without neighbor anchor atom
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_1, root_anchor_atom=6, nbr_anchor_atom=None, core_atoms=core_1
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(6, 12)])
    assert len(angle_idxs) == 0
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    # set [H] as the dummy group, with neighbor anchor atom
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg_1, root_anchor_atom=6, nbr_anchor_atom=5, core_atoms=core_1
    )
    bond_idxs, angle_idxs, improper_idxs, chiral_atom_idxs = all_idxs

    assert_bond_sets_equal(bond_idxs, [(6, 12)])
    assert_bond_sets_equal(angle_idxs, [(5, 6, 12)])
    assert len(improper_idxs) == 0
    assert len(chiral_atom_idxs) == 0

    with pytest.raises(single_topology.MissingAngleError):
        all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
            ff, mol, dummy_group=dg_1, root_anchor_atom=6, nbr_anchor_atom=4, core_atoms=core_1
        )


@pytest.mark.nocuda
def test_methyl_chiral_atom_idxs():
    """
    Check that we're leaving the chiral restraints on correctly for a methyl, when only a single hydrogen is a core atom.
    """
    mol = ligand_from_smiles("C", seed=2022)

    dg = [1, 2, 3, 4]
    all_atoms_set = set([a for a in range(mol.GetNumAtoms())])
    core_atoms = list(all_atoms_set.difference(dg))

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    # set [O,H] as the dummy group
    all_idxs, _ = setup_chiral_dummy_interactions_from_ff(
        ff, mol, dummy_group=dg, root_anchor_atom=0, nbr_anchor_atom=None, core_atoms=core_atoms
    )
    _, _, _, chiral_atom_idxs = all_idxs

    expected_chiral_atom_idxs = [
        [
            (0, 1, 3, 4),
            (0, 3, 2, 4),
            (0, 2, 1, 4),
            (0, 1, 2, 3),
        ]
    ]

    assert_bond_sets_equal(chiral_atom_idxs, expected_chiral_atom_idxs)


@pytest.mark.nocuda
def test_find_dummy_groups_and_anchors():
    """
    Test that we can find the anchors and dummy groups when there's a single core anchor atom. When core bond
    is broken, we should disable one of the angle atoms.
    """
    mol_a = Chem.MolFromSmiles("OCCC")
    mol_b = Chem.MolFromSmiles("CCCF")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[1, 2], [2, 1], [3, 0]])

    dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == {2: (1, {3})}

    # angle should swap
    core_pairs = np.array([[1, 2], [2, 0], [3, 1]])

    with pytest.warns(CoreBondChangeWarning):
        dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == {2: (None, {3})}


@pytest.mark.nocuda
def test_find_dummy_groups_and_anchors_multiple_angles():
    """
    Test that when multiple angle groups are possible we can find one deterministically
    """
    mol_a = Chem.MolFromSmiles("CCC")
    mol_b = Chem.MolFromSmiles("CC(C)C")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[0, 2], [1, 1], [2, 3]])
    dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    assert dgs == {1: (2, {0})} or dgs == {1: (3, {0})}

    dgs_zero = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])

    # this code should be invariant to different random seeds and different ordering of core pairs
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs = single_topology.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero


@pytest.mark.nocuda
def test_find_dummy_groups_and_multiple_anchors():
    """
    Test that we can find anchors and dummy groups with multiple anchors, we expect to find only a single
    root anchor and neighbor core atom pair.
    """
    mol_a = Chem.MolFromSmiles("OCC")
    mol_b = Chem.MolFromSmiles("O1CC1")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_pairs = np.array([[1, 1], [2, 2]])

    with pytest.warns(MultipleAnchorWarning):
        dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
        assert dgs == {1: (2, {0})} or dgs == {2: (1, {0})}

    # test determinism, should be robust against seeds
    dgs_zero = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_pairs[:, 0], core_pairs[:, 1])
    for idx in range(100):
        np.random.seed(idx)
        core_pairs_shuffle = np.random.permutation(core_pairs)
        dgs = single_topology.find_dummy_groups_and_anchors(
            mol_a, mol_b, core_pairs_shuffle[:, 0], core_pairs_shuffle[:, 1]
        )
        assert dgs == dgs_zero

    mol_a = Chem.MolFromSmiles("C(C)(C)C")
    mol_b = Chem.MolFromSmiles("O1CCCC1")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core_a = [0, 1, 2, 3]
    core_b = [2, 1, 4, 3]

    with pytest.warns(MultipleAnchorWarning):
        dgs = single_topology.find_dummy_groups_and_anchors(mol_a, mol_b, core_a, core_b)
        assert dgs == {1: (2, {0})}


@pytest.mark.nocuda
def test_ethane_cyclobutadiene():
    """Test case where a naive heuristic for identifying dummy groups results in disconnected components"""

    mol_a = ligand_from_smiles("CC", seed=2022)
    mol_b = ligand_from_smiles("c1ccc1", seed=2022)

    core = np.array([[2, 0], [4, 2], [0, 3], [3, 7]])
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    g = nx.Graph()
    g.add_nodes_from(range(st.get_num_atoms()))
    g.add_edges_from(st.src_system.bond.potential.idxs)

    # bond graph should be connected (i.e. no floating bits)
    assert len(list(nx.connected_components(g))) == 1


@pytest.mark.nocuda
def test_charge_perturbation_is_invalid():
    mol_a = ligand_from_smiles("Cc1cc[nH]c1", seed=2022)
    mol_b = ligand_from_smiles("C[n+]1cc[nH]c1", seed=2022)

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    core = np.zeros((mol_a.GetNumAtoms(), 2), dtype=np.int32)
    core[:, 0] = np.arange(core.shape[0])
    core[:, 1] = core[:, 0]

    with pytest.raises(ChargePertubationError) as e:
        SingleTopology(mol_a, mol_b, core, ff)
    assert str(e.value) == "mol a and mol b don't have the same charge: a: 0 b: 1"


def bond_idxs_are_canonical(all_idxs):
    return np.all(all_idxs[:, 0] < all_idxs[:, -1])


def chiral_atom_idxs_are_canonical(all_idxs):
    return np.all((all_idxs[:, 1] < all_idxs[:, 2]) & (all_idxs[:, 1] < all_idxs[:, 3]))


def assert_improper_idxs_are_canonical(all_idxs):
    for idxs in all_idxs:
        np.testing.assert_array_equal(idxs, canonicalize_improper_idxs(idxs))


@pytest.mark.nogpu
@pytest.mark.nightly(reason="Takes awhile to run")
@pytest.mark.parametrize(
    "idx, mol_a, mol_b", [(i, a, b) for i, (a, b) in enumerate(_get_hif2a_mol_pairs(shuffle=True, seed=2024)[:25])]
)
def test_hif2a_end_state_stability(idx, mol_a, mol_b, num_pairs_to_simulate=5):
    """
    Pick some random pairs from the hif2a set and ensure that they're numerically stable at the
    end-states under a distance based atom-mapping protocol. For a subset of them, we will also run
    simulations.
    """

    seed = 2024

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    compute_distance_matrix = functools.partial(pairwise_distances, box=None)

    def get_max_distance(x0):
        dij = compute_distance_matrix(x0)
        return jnp.amax(dij)

    batch_distance_check = jax.vmap(get_max_distance)

    # this has been tested for up to 50 random pairs
    print("Checking", get_mol_name(mol_a), "->", get_mol_name(mol_b))
    core = _get_core_by_mcs(mol_a, mol_b)
    st = SingleTopology(mol_a, mol_b, core, ff)
    x0 = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    systems = [st.src_system, st.dst_system]

    for system in systems:
        # assert that the idxs are canonicalized.
        assert bond_idxs_are_canonical(system.bond.potential.idxs)
        assert bond_idxs_are_canonical(system.angle.potential.idxs)
        assert bond_idxs_are_canonical(system.proper.potential.idxs)
        assert_improper_idxs_are_canonical(system.improper.potential.idxs)
        assert bond_idxs_are_canonical(system.nonbonded_pair_list.potential.idxs)
        assert bond_idxs_are_canonical(system.chiral_bond.potential.idxs)
        assert chiral_atom_idxs_are_canonical(system.chiral_atom.potential.idxs)
        U_fn = jax.jit(system.get_U_fn())
        assert np.isfinite(U_fn(x0))
        x_min = minimize_scipy(U_fn, x0, seed=seed)
        assert np.all(np.isfinite(x_min))
        distance_cutoff = 2.5  # in nanometers
        assert get_max_distance(x_min) < distance_cutoff

        # test running simulations on the first 5 pairs
        if idx < num_pairs_to_simulate:
            batch_U_fn = jax.vmap(U_fn)
            frames = simulate_system(system.get_U_fn(), x0, num_samples=1000)
            nrgs = batch_U_fn(frames)
            assert np.all(np.isfinite(nrgs))
            assert np.all(np.isfinite(frames))
            assert np.all(batch_distance_check(frames) < distance_cutoff)


atom_idxs = st.integers(0, 100)


@st.composite
def bond_or_angle_idx_arrays(draw):
    n_idxs = draw(st.one_of(st.just(2), st.just(3)))
    idxs = st.lists(atom_idxs, min_size=n_idxs, max_size=n_idxs, unique=True).map(tuple)
    idx_arrays = st.lists(idxs, min_size=0, max_size=100, unique=True).map(
        lambda ixns: np.array(ixns).reshape(-1, n_idxs)
    )
    return draw(idx_arrays)


@given(bond_or_angle_idx_arrays())
@seed(2024)
def test_canonicalize_bonds(bonds):
    canonicalized_bonds = canonicalize_bonds(bonds)
    event("canonical" if bond_idxs_are_canonical(bonds) else "not canonical")
    assert all(set(canon_idxs) == set(idxs) for canon_idxs, idxs in zip(canonicalized_bonds, bonds))
    assert bond_idxs_are_canonical(canonicalized_bonds)


chiral_atom_idxs = st.lists(atom_idxs, min_size=4, max_size=4, unique=True).map(lambda x: tuple(x))
chiral_atom_idx_arrays = st.lists(chiral_atom_idxs, min_size=0, max_size=100, unique=True).map(
    lambda idxs: np.array(idxs).reshape(-1, 4)
)


@given(chiral_atom_idx_arrays)
@seed(2024)
def test_canonicalize_chiral_atom_idxs(chiral_atom_idxs):
    canonicalized_idxs = canonicalize_chiral_atom_idxs(chiral_atom_idxs)
    event("canonical" if chiral_atom_idxs_are_canonical(chiral_atom_idxs) else "not canonical")
    assert all(
        tuple(canon_idxs) in {tuple(idxs[p]) for p in [[0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]}
        for canon_idxs, idxs in zip(canonicalized_idxs, chiral_atom_idxs)
    )
    assert chiral_atom_idxs_are_canonical(canonicalized_idxs)


@pytest.mark.nocuda
def test_canonicalize_improper_idxs():
    # these are in the cw rotation set
    improper_idxs = [(0, 5, 1, 3), (1, 5, 3, 0), (3, 5, 0, 1)]

    for idxs in improper_idxs:
        # we should do nothing here.
        assert idxs == canonicalize_improper_idxs(idxs)

    # these are in the ccw rotation set
    assert canonicalize_improper_idxs((1, 5, 0, 3)) == (1, 5, 3, 0)
    assert canonicalize_improper_idxs((3, 5, 1, 0)) == (3, 5, 0, 1)
    assert canonicalize_improper_idxs((0, 5, 3, 1)) == (0, 5, 1, 3)


@pytest.mark.nocuda
def test_combine_masses():
    C_mass = Chem.MolFromSmiles("C").GetAtomWithIdx(0).GetMass()
    Br_mass = Chem.MolFromSmiles("Br").GetAtomWithIdx(0).GetMass()
    F_mass = Chem.MolFromSmiles("F").GetAtomWithIdx(0).GetMass()
    N_mass = Chem.MolFromSmiles("N").GetAtomWithIdx(0).GetMass()

    mol_a = Chem.MolFromSmiles("BrC1=CC=CC=C1")
    mol_b = Chem.MolFromSmiles("C1=CN=CC=C1F")

    AllChem.EmbedMolecule(mol_a, randomSeed=2022)
    AllChem.EmbedMolecule(mol_b, randomSeed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    st = SingleTopology(mol_a, mol_b, core, ff)

    test_masses = st.combine_masses()
    ref_masses = [Br_mass, C_mass, C_mass, max(C_mass, N_mass), C_mass, C_mass, C_mass, F_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)


@pytest.mark.nocuda
def test_combine_masses_hmr():
    C_mass = Chem.MolFromSmiles("C").GetAtomWithIdx(0).GetMass()
    Cl_mass = Chem.MolFromSmiles("Cl").GetAtomWithIdx(0).GetMass()
    Br_mass = Chem.MolFromSmiles("Br").GetAtomWithIdx(0).GetMass()
    F_mass = Chem.MolFromSmiles("F").GetAtomWithIdx(0).GetMass()

    mol_a = ligand_from_smiles("[H]C([H])([H])[H]")
    mol_b = ligand_from_smiles("[H]C(F)(Cl)Br")
    H_mass = mol_a.GetAtomWithIdx(1).GetMass()

    AllChem.EmbedMolecule(mol_a, randomSeed=2023)
    AllChem.EmbedMolecule(mol_b, randomSeed=2023)

    # only C mapped
    core = np.array([[0, 0]])

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # No HMR
    test_masses = st.combine_masses()
    ref_masses = [C_mass, H_mass, H_mass, H_mass, H_mass, F_mass, Cl_mass, Br_mass, H_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)

    # HMR
    test_masses = st.combine_masses(use_hmr=True)
    scale = 2 * H_mass
    ref_masses = [
        max(C_mass - 4 * scale, C_mass - scale),
        H_mass + scale,
        H_mass + scale,
        H_mass + scale,
        H_mass + scale,
        F_mass,
        Cl_mass,
        Br_mass,
        H_mass + scale,
    ]
    np.testing.assert_almost_equal(test_masses, ref_masses)

    # only C-H/C-F mapped
    core = np.array([[0, 0], [1, 1]])

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # No HMR
    test_masses = st.combine_masses()
    ref_masses = [C_mass, F_mass, H_mass, H_mass, H_mass, Cl_mass, Br_mass, H_mass]
    np.testing.assert_almost_equal(test_masses, ref_masses)

    # HMR
    test_masses = st.combine_masses(use_hmr=True)
    ref_masses = [
        max(C_mass - 4 * scale, C_mass - scale),
        F_mass,
        H_mass + scale,
        H_mass + scale,
        H_mass + scale,
        Cl_mass,
        Br_mass,
        H_mass + scale,
    ]
    np.testing.assert_almost_equal(test_masses, ref_masses)


@pytest.fixture()
def arbitrary_transformation():
    # NOTE: test system can probably be simplified; we just need
    # any SingleTopology and conformation
    with path_to_internal_file("tmd.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf_mols_by_name(path_to_ligand)

    mol_a = mols["206"]
    mol_b = mols["57"]

    core = _get_core_by_mcs(mol_a, mol_b)
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)
    conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    return st, conf


@pytest.mark.nocuda
def test_jax_transform_intermediate_potential(arbitrary_transformation):
    st, conf = arbitrary_transformation

    def U(x, lam):
        return st.setup_intermediate_state(lam).get_U_fn()(x)

    _ = jax.jit(U)(conf, 0.1)

    confs = jnp.array([conf for _ in range(10)])
    lambdas = jnp.linspace(0, 1, 10)
    _ = jax.vmap(U)(confs, lambdas)
    _ = jax.jit(jax.vmap(U))(confs, lambdas)


@pytest.mark.nocuda
def test_setup_intermediate_state_not_unreasonably_slow(arbitrary_transformation):
    st, _ = arbitrary_transformation
    n_states = 10
    start_time = time.perf_counter()
    for lam in np.linspace(0, 1, n_states):
        _ = st.setup_intermediate_state(lam)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # weak assertion to catch egregious perf issues while being unlikely to raise false positives
    assert elapsed_time / n_states <= 1.0


@pytest.mark.nocuda
def test_combine_achiral_ligand_with_host():
    """Verifies that combine_with_host correctly sets up all of the U functions"""
    mol_a = ligand_from_smiles("BrC1=CC=CC=C1", seed=2022)
    mol_b = ligand_from_smiles("C1=CN=CC=C1F", seed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")

    host_config = build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])
    st = SingleTopology(mol_a, mol_b, core, ff)
    combined_system = st.combine_with_host(
        host_config.host_system, 0.5, host_config.conf.shape[0], host_config.omm_topology
    )
    assert (
        set(type(bp.potential) for bp in combined_system.get_U_fns())
        == {
            potentials.HarmonicBond,
            potentials.HarmonicAngle,
            potentials.PeriodicTorsion,
            potentials.NonbondedPairListPrecomputed,
            potentials.Nonbonded,
            potentials.ChiralAtomRestraint,  # this is no longer missing since we now emit all potentials, even if they're length 0
            # potentials.ChiralBondRestraint,
            # NOTE: chiral bond restraints excluded
            # This should be updated when chiral restraints are re-enabled.
        }
    )


@pytest.mark.nocuda
def test_combine_chiral_ligand_with_host():
    """Verifies that combine_with_host correctly sets up all of the U functions"""
    mol_a = ligand_from_smiles("BrC1CCCCC1", seed=2022)
    mol_b = ligand_from_smiles("C1=CN=CC=C1F", seed=2022)

    core = np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]])
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    host_config = build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])
    st = SingleTopology(mol_a, mol_b, core, ff)
    combined_system = st.combine_with_host(
        host_config.host_system, 0.5, host_config.conf.shape[0], host_config.omm_topology
    )
    assert set(type(bp.potential) for bp in combined_system.get_U_fns()) == {
        potentials.HarmonicBond,
        potentials.HarmonicAngle,
        potentials.PeriodicTorsion,
        potentials.NonbondedPairListPrecomputed,
        potentials.Nonbonded,
        potentials.ChiralAtomRestraint,
        # potentials.ChiralBondRestraint,
        # NOTE: chiral bond restraints excluded
        # This should be updated when chiral restraints are re-enabled.
    }


def _get_core_by_mcs(mol_a, mol_b):
    all_cores = atom_mapping.get_cores(
        mol_a,
        mol_b,
        **DEFAULT_ATOM_MAPPING_KWARGS,
    )

    core = all_cores[0]
    return core


@pytest.mark.nocuda
def test_no_chiral_atom_restraints():
    mol_a = ligand_from_smiles("c1ccccc1")
    mol_b = ligand_from_smiles("c1(I)ccccc1")
    core = _get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_atom.potential.idxs) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


@pytest.mark.nocuda
def test_no_chiral_bond_restraints():
    mol_a = ligand_from_smiles("C")
    mol_b = ligand_from_smiles("CI")
    core = _get_core_by_mcs(mol_a, mol_b)

    forcefield = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, forcefield)
    init_conf = st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b))
    state = st.setup_intermediate_state(0.1)

    assert len(state.chiral_bond.potential.idxs) == 0
    U = state.get_U_fn()
    _ = U(init_conf)


finite_floats = functools.partial(st.floats, allow_nan=False, allow_infinity=False, allow_subnormal=False)

nonzero_force_constants = finite_floats(1e-9, 1e9)

lambdas = finite_floats(0.0, 1.0)


@pytest.mark.nocuda
def test_cyclic_difference():
    assert cyclic_difference(0, 0, 1) == 0
    assert cyclic_difference(0, 1, 2) == 1  # arbitrary, positive by convention
    assert cyclic_difference(0, 0, 3) == 0
    assert cyclic_difference(0, 1, 3) == 1
    assert cyclic_difference(0, 2, 3) == -1

    # antisymmetric
    assert cyclic_difference(0, 1, 3) == -cyclic_difference(1, 0, 3)
    assert cyclic_difference(0, 2, 3) == -cyclic_difference(2, 0, 3)

    # translation invariant
    assert cyclic_difference(0, 1, 3) == cyclic_difference(-1, 0, 3)
    assert cyclic_difference(0, 4, 8) == cyclic_difference(-2, 2, 8) == cyclic_difference(-4, 0, 8)

    # jittable
    _ = jax.jit(cyclic_difference)(0, 1, 1)


def assert_equal_cyclic(a, b, period):
    def f(x):
        x_mod = x % period
        return np.minimum(x_mod, period - x_mod)

    assert f(a) == f(b)


periods = st.integers(1, int(1e9))
bounded_ints = st.integers(-int(1e9), int(1e9))


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_inverse(a, b, period):
    x = cyclic_difference(a, b, period)
    assert np.abs(x) <= period / 2
    assert_equal_cyclic(a + x, b, period)


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_antisymmetric(a, b, period):
    assert cyclic_difference(a, b, period) + cyclic_difference(b, a, period) == 0


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_shift_by_n_periods(a, b, m, n, period):
    assert_equal_cyclic(
        cyclic_difference(a + m * period, b + n * period, period),
        cyclic_difference(a, b, period),
        period,
    )


@pytest.mark.nocuda
@given(bounded_ints, bounded_ints, bounded_ints, periods)
@seed(2022)
def test_cyclic_difference_translation_invariant(a, b, t, period):
    assert_equal_cyclic(
        cyclic_difference(a + t, b + t, period),
        cyclic_difference(a, b, period),
        period,
    )


def pairs(elem, unique=False):
    return st.lists(elem, min_size=2, max_size=2, unique=unique).map(tuple)


@pytest.mark.nocuda
@given(pairs(finite_floats()))
@seed(2022)
def test_interpolate_w_coord_valid_at_end_states(end_states):
    a, b = end_states
    f = functools.partial(interpolate_w_coord, a, b)
    assert f(0.0) == a
    assert f(1.0) == b


@pytest.mark.nocuda
def test_interpolate_w_coord_monotonic():
    lambdas = np.linspace(0.0, 1.0, 100)
    ws = interpolate_w_coord(0.0, 1.0, lambdas)
    assert np.all(np.diff(ws) >= 0.0)


@pytest.mark.nightly(reason="Test setting up hif2a pairs for single topology.")
@pytest.mark.nocuda
@pytest.mark.parametrize("mol_a, mol_b", _get_hif2a_mol_pairs())
def test_hif2a_pairs_setup_st(mol_a, mol_b):
    """
    Test that we can setup all-pairs single topology objects in hif2a.
    """
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    print(mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    SingleTopology(mol_a, mol_b, core, ff)  # Test that this doesn't not throw assertion


@pytest.mark.nocuda
def test_chiral_methyl_to_nitrile():
    # test that we do not turn off chiral atom restraints even if some of
    # the angle terms are planar
    #
    #     H        H
    #    .        /
    # N#C-H -> F-C-H
    #    .        \
    #     H        H

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02232412343D

  5  4  0  0  0  0            999 V2000
    0.4146   -0.0001    0.4976 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0830    0.0001    0.8564 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.5755   -0.0001   -1.0339 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0830    1.2574    1.0841 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0830   -1.2574    1.0841 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02232412343D

  3  2  0  0  0  0            999 V2000
    0.4146   -0.0001    0.4976 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0830    0.0001    0.8564 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.5755   -0.0001   -1.0339 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  2  3  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 2]])
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)
    # chiral force constants should be on for all chiral terms at lambda=0 and lambda=1
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4
    vs_1 = st.setup_intermediate_state(1.0)

    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4


@pytest.mark.nocuda
def test_chiral_methyl_to_nitrogen():
    # test that we maintain all 4 chiral idxs when morphing N#N into CH3
    #
    #     H        H
    #    /        /
    # N#N-H -> F-C-H
    #    \        \
    #     H        H
    #
    # (we need at least one restraint to be turned on to enable this)

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400273D

  5  4  0  0  0  0            999 V2000
    0.1976    0.0344    0.3479 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8624    0.0345    0.6018 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.3115    0.0344   -0.7361 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6707    0.9244    0.7630 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6707   -0.8555    0.7630 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
  1  2  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400253D

  2  1  0  0  0  0            999 V2000
    0.1976    0.0344    0.3479 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8624    0.0345    0.6018 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [4, 1]])

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4

    vs_1 = st.setup_intermediate_state(1.0)
    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_1) == 4
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4

    np.testing.assert_array_equal(chiral_idxs_0, chiral_idxs_1)


@pytest.mark.nocuda
def test_chiral_methyl_to_water():
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222411113D

  5  4  0  0  0  0            999 V2000
   -1.1951   -0.2262   -0.1811 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.1566   -0.1865    0.0446 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4366    0.8050    0.4004 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.6863   -0.4026   -0.8832 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.4215   -0.9304    0.7960 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  2  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222411123D

  3  2  0  0  0  0            999 V2000
   -1.1951   -0.2262   -0.1811 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.1566   -0.1865    0.0446 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.4215   -0.9304    0.7960 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2]])

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # chiral force constants should be on for all chiral terms at lambda=0 and lambda=1
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4
    vs_1 = st.setup_intermediate_state(1.0)

    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4


@pytest.mark.nocuda
def test_chiral_methyl_to_ammonia():
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02232411003D

  5  4  0  0  0  0            999 V2000
    0.0402    0.0126    0.1841 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2304   -0.7511    0.9383 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8502    0.0126   -0.5452 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0173    0.9900    0.6632 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9024   -0.2011   -0.3198 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02232411003D

  4  3  0  0  0  0            999 V2000
    0.0402    0.0126    0.1841 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.2304   -0.7511    0.9383 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0173    0.9900    0.6632 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9024   -0.2011   -0.3198 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )

    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)

    # chiral force constants should be on for all chiral terms at lambda=0 and lambda=1
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 4
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4
    vs_1 = st.setup_intermediate_state(1.0)

    # Note that NH3 is categorized as achiral
    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 3


@pytest.mark.nocuda
def test_ring_breaking_bond_position():
    """Test that when a ring breaking transformation is performed that the bond broken is
    further from the two atom anchors.

    This is done to reduce the phase space available to the broken ring in intermediate states
    """
    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2503 10152518042D

 10 11  0  0  0  0            999 V2000
  -13.5268    2.1085    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2412    1.6960    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2412    0.8709    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5268    0.4584    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8123    0.8709    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8123    1.6960    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0978    0.4584    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -11.3833    0.8708    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -11.3833    1.6959    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0978    2.1084    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  7  8  2  0  0  0  0
  8  9  1  0  0  0  0
  9 10  2  0  0  0  0
  5  7  1  0  0  0  0
 10  6  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )
    # Append hydrogens to the end
    mol_a = Chem.AddHs(mol_a)

    mol_b = Chem.MolFromMolBlock(
        """
  MJ250300

  9 10  0  0  0  0  0  0  0  0999 V2000
  -13.5268    2.3763    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2412    1.9638    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2412    1.1387    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -13.5268    0.7262    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8123    1.1387    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.8123    1.9638    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0277    0.8837    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -11.5427    1.5511    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0276    2.2186    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  7  8  1  0  0  0  0
  8  9  1  0  0  0  0
  5  7  1  0  0  0  0
  6  9  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )
    mol_b = Chem.AddHs(mol_b)
    # Give the hydrogens meaningful atoms
    AllChem.EmbedMolecule(mol_a, randomSeed=2025)
    AllChem.EmbedMolecule(mol_b, randomSeed=2025)

    # Map only the five member ring
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    AllChem.AlignMol(mol_a, mol_b, atomMap=core.tolist())

    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)
    vs_0 = st.setup_intermediate_state(1.0)

    frames = simulate_system(
        vs_0.get_U_fn(), st.combine_confs(get_romol_conf(mol_a), get_romol_conf(mol_b)), num_samples=1
    )

    with Chem.SDWriter("endstate.sdf") as writer:
        m = st.mol(1.0)
        set_romol_conf(m, frames[-1])
        writer.write(m)


@pytest.mark.nocuda
def test_chiral_core_ring_opening():
    # test that chiral restraints are maintained for dummy atoms when we open/close a ring,
    # at lambda=0, all 7 chiral restraints are turned on, but at lambda=1
    # only 4 chiral restraints are turned on.

    mol_a = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400433D

  6  6  0  0  0  0            999 V2000
   -0.2397    1.3763    0.4334 C   0  0  2  0  0  0  0  0  0  0  0  0
    0.2664   -0.0682    0.6077 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.8332    1.6232   -0.6421 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3412    0.1809   -0.4674 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0336    1.9673    1.3258 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2364    1.3849   -0.0078 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  3  4  1  0  0  0  0
  1  2  1  0  0  0  0
  4  2  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
M  END
$$$$
""",
        removeHs=False,
    )  # closed ring

    mol_b = Chem.MolFromMolBlock(
        """
  Mrv2311 02222400463D

  7  6  0  0  0  0            999 V2000
   -0.2397    1.3763    0.4334 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2664   -0.0682    0.6077 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8332    1.6232   -0.6421 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.3412    0.1809   -0.4674 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0336    1.9673    1.3258 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2364    1.3849   -0.0078 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.0787   -0.1553    0.4334 H   0  0  0  0  0  0  0  0  0  0  0  0
  3  4  1  0  0  0  0
  1  3  1  0  0  0  0
  1  5  1  0  0  0  0
  1  6  1  0  0  0  0
  1  7  1  0  0  0  0
  4  2  1  0  0  0  0
M  END
$$$$""",
        removeHs=False,
    )  # open ring

    # map everything except a single hydrogen at the end
    core = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    # chiral force constants should be on for all 7 chiral
    # terms at lambda=0
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    st = SingleTopology(mol_a, mol_b, core, ff)
    vs_0 = st.setup_intermediate_state(0.0)
    chiral_idxs_0 = vs_0.chiral_atom.potential.idxs
    chiral_params_0 = vs_0.chiral_atom.params
    assert len(chiral_idxs_0) == 7
    assert np.sum(chiral_params_0 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 7
    vs_1 = st.setup_intermediate_state(1.0)

    # chiral force constants should be on for all 4 of the 7
    # chiral terms at lambda=1
    chiral_idxs_1 = vs_1.chiral_atom.potential.idxs
    chiral_params_1 = vs_1.chiral_atom.params
    assert len(chiral_idxs_0) == len(chiral_idxs_1)

    assert np.sum(chiral_params_1 == 0) == 3
    assert np.sum(chiral_params_1 == DEFAULT_CHIRAL_ATOM_RESTRAINT_K) == 4


def permute_atom_indices(mol_a, mol_b, core, seed):
    """Randomly permute atom indices in mol_a, mol_b independently, and update core"""
    rng = np.random.default_rng(seed)

    perm_a = rng.permutation(mol_a.GetNumAtoms())
    perm_b = rng.permutation(mol_b.GetNumAtoms())

    # RenumberAtoms takes inverse permutations
    # e.g. [3, 2, 0, 1] means atom 3 in the original mol will be atom 0 in the new one
    inv_perm_a = np.argsort(perm_a)
    inv_perm_b = np.argsort(perm_b)
    mol_a = Chem.RenumberAtoms(mol_a, inv_perm_a.tolist())
    mol_b = Chem.RenumberAtoms(mol_b, inv_perm_b.tolist())

    core = np.array(core)
    core[:, 0] = perm_a[core[:, 0]]
    core[:, 1] = perm_b[core[:, 1]]

    return mol_a, mol_b, core


def get_vacuum_system_and_conf(mol_a, mol_b, core, lamb):
    ff = Forcefield.load_default()
    st = SingleTopology(mol_a, mol_b, core, ff)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    conf = st.combine_confs(conf_a, conf_b, lamb)
    return st.setup_intermediate_state(lamb), conf


def _assert_consistent_hamiltonian_term_impl(fwd_bonded_term, rev_bonded_term, rev_kv, canon_fn):
    canonical_map = dict()
    for fwd_idxs, fwd_params in zip(fwd_bonded_term.potential.idxs, fwd_bonded_term.params):
        fwd_key = canon_fn(fwd_idxs, fwd_params)
        canonical_map[fwd_key] = [fwd_params, None]

    for rev_idxs, rev_params in zip(rev_bonded_term.potential.idxs, rev_bonded_term.params):
        rev_key = canon_fn([rev_kv[x] for x in rev_idxs], rev_params)
        canonical_map[rev_key][1] = rev_params

    for fwd_params, rev_params in canonical_map.values():
        np.testing.assert_allclose(fwd_params, rev_params)


def _assert_u_and_grad_consistent(u_fwd, u_rev, x_fwd, fused_map, canon_fn):
    # test that the definition of the hamiltonian, the energies, and the forces are all consistent
    rev_kv = dict()
    fwd_kv = dict()
    for x, y in fused_map:
        fwd_kv[x] = y
        rev_kv[y] = x
    x_rev = np.zeros_like(x_fwd)
    for atom_idx, xyz in enumerate(x_fwd):
        x_rev[fwd_kv[atom_idx]] = xyz

    # check hamiltonian
    _assert_consistent_hamiltonian_term_impl(u_fwd, u_rev, rev_kv, canon_fn)

    # check energies and forces
    box = 100.0 * np.eye(3)
    np.testing.assert_allclose(u_fwd(x_fwd, box), u_rev(x_rev, box), rtol=1e-6)
    fwd_bond_grad_fn = jax.grad(u_fwd)
    rev_bond_grad_fn = jax.grad(u_rev)
    np.testing.assert_allclose(
        fwd_bond_grad_fn(x_fwd, box)[fused_map[:, 0]], rev_bond_grad_fn(x_rev, box)[fused_map[:, 1]], rtol=1e-5
    )


def _get_fused_map(mol_a, mol_b, core):
    amm_fwd = AtomMapMixin(mol_a, mol_b, core)
    amm_rev = AtomMapMixin(mol_b, mol_a, core[:, ::-1])
    fused_map = np.concatenate(
        [
            np.array([[x, y] for x, y in zip(amm_fwd.a_to_c, amm_rev.b_to_c)], dtype=np.int32).reshape(-1, 2),
            np.array(
                [[x, y] for x, y in zip(amm_fwd.b_to_c, amm_rev.a_to_c) if x not in core[:, 0] and y not in core[:, 1]],
                dtype=np.int32,
            ).reshape(-1, 2),
        ]
    )
    return fused_map


def assert_symmetric_interpolation(mol_a, mol_b, core):
    """
    Assert that the Single Topology interpolation code is symmetric, i.e. ST(mol_a, mol_b, lamb) == ST(mol_b, mol_a, 1-lamb)

    Where for each of the bond, angle, proper torsion, improper torsion, nonbonded, terms
        - the idxs, params are identical under atom-mapping + canonicalization
        - u_fwd, u_rev for an arbitrary conformation is identical under atom mapping
        - grad_fwd, grad_rev for an arbitrary conformation is identical under atom mapping

    """
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    # map atoms in the combined mol_ab to the atoms in the combined mol_ba
    fused_map = _get_fused_map(mol_a, mol_b, core)

    st_fwd = SingleTopology(mol_a, mol_b, core, ff)
    st_rev = SingleTopology(mol_b, mol_a, core[:, ::-1], ff)
    conf_a = get_romol_conf(mol_a)
    conf_b = get_romol_conf(mol_b)
    test_conf = st_fwd.combine_confs(conf_a, conf_b, 0)

    seed = 2024
    np.random.seed(seed)
    lambda_schedule = np.concatenate([np.linspace(0, 1, 12), np.random.rand(10)])

    for lamb in lambda_schedule:
        sys_fwd = st_fwd.setup_intermediate_state(lamb)
        sys_rev = st_rev.setup_intermediate_state(1 - lamb)

        assert_default_system_constraints(sys_fwd)
        assert_default_system_constraints(sys_rev)

        _assert_u_and_grad_consistent(
            sys_fwd.bond, sys_rev.bond, test_conf, fused_map, canon_fn=lambda idxs, _: tuple(canonicalize_bond(idxs))
        )

        _assert_u_and_grad_consistent(
            sys_fwd.angle, sys_rev.angle, test_conf, fused_map, canon_fn=lambda idxs, _: tuple(canonicalize_bond(idxs))
        )

        # for propers, we format the phase as a 5 decimal string to guard against loss of precision
        _assert_u_and_grad_consistent(
            sys_fwd.proper,
            sys_rev.proper,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, params: tuple([*canonicalize_bond(idxs), f"{params[1]:.5f}", int(round(params[2]))]),
        )

        _assert_u_and_grad_consistent(
            sys_fwd.improper,
            sys_rev.improper,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, _: tuple(canonicalize_improper_idxs(idxs)),
        )

        _assert_u_and_grad_consistent(
            sys_fwd.chiral_atom,
            sys_rev.chiral_atom,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, _: tuple(canonicalize_chiral_atom_idxs(np.array([idxs]))[0]),  # fn assumes ndim=2
        )

        _assert_u_and_grad_consistent(
            sys_fwd.nonbonded_pair_list,
            sys_rev.nonbonded_pair_list,
            test_conf,
            fused_map,
            canon_fn=lambda idxs, _: tuple(canonicalize_bond(idxs)),
        )


@pytest.mark.nocuda
def test_hif2a_end_state_symmetry_unit_test():
    with path_to_internal_file("tmd.testsystems.data", "ligands_40.sdf") as path_to_ligand:
        mols = read_sdf(path_to_ligand)

    mol_a = mols[0]
    mol_b = mols[1]
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    assert_symmetric_interpolation(mol_a, mol_b, core)


@pytest.mark.nocuda
@pytest.mark.nightly(reason="slow")
@pytest.mark.parametrize("mol_a, mol_b", _get_hif2a_mol_pairs(shuffle=True, seed=2029)[:25])
def test_hif2a_end_state_symmetry_nightly_test(mol_a, mol_b):
    """
    Test that end-states are symmetric for a large number of random pairs
    """
    print("testing", mol_a.GetProp("_Name"), "->", mol_b.GetProp("_Name"))
    core = atom_mapping.get_cores(mol_a, mol_b, **DEFAULT_ATOM_MAPPING_KWARGS)[0]
    assert_symmetric_interpolation(mol_a, mol_b, core)
