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

from rdkit import Chem

from .bond import CanonicalBond, mkbond


def get_aliphatic_ring_bonds(mol: Chem.rdchem.Mol) -> set[CanonicalBond]:
    return {
        mkbond(
            mol.GetBondWithIdx(bond_idx).GetBeginAtomIdx(),
            mol.GetBondWithIdx(bond_idx).GetEndAtomIdx(),
        )
        for ring_bond_idxs in mol.GetRingInfo().BondRings()
        for is_aromatic in [all(mol.GetBondWithIdx(bond_idx).GetIsAromatic() for bond_idx in ring_bond_idxs)]
        if not is_aromatic
        for bond_idx in ring_bond_idxs
    }


def get_rotatable_bonds(mol: Chem.rdchem.Mol) -> set[CanonicalBond]:
    """Identify rotatable bonds in a molecule.

    NOTE: This uses the same (non-strict) pattern for a rotatable bond as RDKit:

        https://github.com/rdkit/rdkit/blob/e640915d4eb2140fbca76a820b69a8e15216a908/rdkit/Chem/Lipinski.py#L41

    Parameters
    ----------
    mol: ROMol
        Input molecule

    Returns
    -------
    set of CanonicalBond
        Set of bonds identified as rotatable
    """

    pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(pattern, uniquify=1)
    return {mkbond(i, j) for i, j in matches}
