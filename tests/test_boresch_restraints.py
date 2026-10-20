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

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmd.fe.absolute.restraints import _is_angle_linear, select_ligand_atoms

pytestmark = [pytest.mark.nocuda]


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


@pytest.mark.parametrize("perm_seed", [0, 1, 2])
def test_select_ligand_atoms_invariant_to_atom_order(perm_seed):
    mol, xyz, rmsf = _ligand_with_trajectory()
    expected = select_ligand_atoms(mol, xyz, rmsf)

    perm = np.random.default_rng(perm_seed).permutation(mol.GetNumAtoms())
    # RenumberAtoms: new atom i is old atom perm[i]
    renumbered = Chem.RenumberAtoms(mol, perm.tolist())
    selected = select_ligand_atoms(renumbered, xyz[:, perm], rmsf[perm])

    assert [int(perm[i]) for i in selected] == expected
