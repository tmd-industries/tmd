# Copyright 2026, Justin Gullingsrud
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

import mdtraj
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd.fe.absolute.restraints import (
    _RMSF_CUTOFF,
    _is_angle_linear,
    rdmol_to_mdtraj,
    select_ligand_atoms,
    select_receptor_atoms,
)
from tmd.fe.utils import read_sdf
from tmd.utils import path_to_internal_file

pytestmark = [pytest.mark.nogpu]


def _ligand_with_trajectory(seed: int = 2024, n_frames: int = 20):
    # naphthalene joined to a phenyl through a flexible propyl linker
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc2ccccc2c1CCCc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    ref = mol.GetConformer().GetPositions() / 10.0  # nm
    rng = np.random.default_rng(seed)
    xyz = ref[None] + rng.normal(scale=0.005, size=(n_frames, *ref.shape))
    # every atom looks rigid, as in a short equilibration
    rmsf = np.full(mol.GetNumAtoms(), 0.01)
    return mol, xyz, rmsf


def test_select_ligand_atoms_uses_largest_ring_system():
    mol, xyz, rmsf = _ligand_with_trajectory()
    naphthalene = set(mol.GetSubstructMatch(Chem.MolFromSmarts("c1ccc2ccccc2c1")))
    assert len(naphthalene) == 10

    l1, l2, l3 = select_ligand_atoms(mol, xyz, rmsf)

    assert len({l1, l2, l3}) == 3
    assert {l1, l2, l3} <= naphthalene
    assert not _is_angle_linear(xyz, (l1, l2, l3))


def test_select_ligand_atoms_prefers_stable_ring_system():
    mol, xyz, rmsf = _ligand_with_trajectory()
    naphthalene = list(mol.GetSubstructMatch(Chem.MolFromSmarts("c1ccc2ccccc2c1")))
    rmsf[naphthalene] = 2 * _RMSF_CUTOFF

    selected = select_ligand_atoms(mol, xyz, rmsf)

    phenyl = mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1"))
    assert any(set(selected) <= set(ring) - set(naphthalene) for ring in phenyl)


def test_select_ligand_atoms_skips_floppy_central_atom():
    mol, xyz, rmsf = _ligand_with_trajectory()
    naphthalene = list(mol.GetSubstructMatch(Chem.MolFromSmarts("c1ccc2ccccc2c1")))
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    ref = xyz.mean(axis=0)
    central = naphthalene[np.argmin(np.linalg.norm(ref[naphthalene] - ref[heavy].mean(axis=0), axis=-1))]
    xyz[:, central] += np.random.default_rng(0).normal(scale=0.05, size=(len(xyz), 3))

    selected = select_ligand_atoms(mol, xyz, rmsf)

    assert set(selected) <= set(naphthalene)
    assert central not in selected


def test_select_ligand_atoms_rejects_flexible_ligand():
    mol, xyz, rmsf = _ligand_with_trajectory()
    # independent atom motion leaves no triangle with fixed mutual distances
    xyz += np.random.default_rng(0).normal(scale=0.05, size=xyz.shape)

    with pytest.raises(ValueError, match="fixed mutual distances"):
        select_ligand_atoms(mol, xyz, rmsf)


@pytest.mark.parametrize("perm_seed", [0, 1, 2])
def test_select_ligand_atoms_invariant_to_atom_order(perm_seed):
    mol, xyz, rmsf = _ligand_with_trajectory()
    expected = select_ligand_atoms(mol, xyz, rmsf)

    perm = np.random.default_rng(perm_seed).permutation(mol.GetNumAtoms())
    # RenumberAtoms: new atom i is old atom perm[i]
    renumbered = Chem.RenumberAtoms(mol, perm.tolist())
    selected = select_ligand_atoms(renumbered, xyz[:, perm], rmsf[perm])

    assert [int(perm[i]) for i in selected] == expected


def test_select_receptor_atoms_invariant_to_ligand_image():
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "5tbm_prepared.pdb") as pdb_path:
        protein = mdtraj.load(str(pdb_path))
    with path_to_internal_file("tmd.testsystems.fep_benchmark.hif2a", "ligands.sdf") as ligand_path:
        mol = read_sdf(ligand_path)[0]
    trj = protein.stack(rdmol_to_mdtraj(mol))
    box_length = np.ptp(trj.xyz[0], axis=0).max() + 2.0
    trj.unitcell_vectors = box_length * np.eye(3)[None]
    rmsf = np.zeros(trj.n_atoms)
    lig = slice(protein.n_atoms, None)
    lig_ids = [protein.n_atoms + i for i in select_ligand_atoms(mol, trj.xyz[:, lig], rmsf[lig])]
    expected = select_receptor_atoms(trj, lig_ids, rmsf)

    # image the ligand into the neighboring cell, as mdtraj's image_molecules can
    trj.xyz[:, lig, 0] += box_length
    assert select_receptor_atoms(trj, lig_ids, rmsf) == expected
