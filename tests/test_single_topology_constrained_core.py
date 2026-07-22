# Copyright 2026 Justin Gullingsrud
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
from common import ligand_from_smiles

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe.atom_mapping import get_cores
from tmd.fe.single_topology import (
    SingleTopology,
    filter_constraint_incompatible_hydrogens,
    verify_core_is_compatible_with_constraints,
)
from tmd.ff import Forcefield

pytestmark = [pytest.mark.nogpu]


def _h_neighbors(mol, idx):
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]


def test_verify_core_is_compatible_with_constraints():
    """Verify that verify_core_is_compatible_with_constraints combines all of the matches into a single error"""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol_a = ligand_from_smiles("C1CCCCC1")
    mol_b = ligand_from_smiles("C1C(Cl)CCCC1")

    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    kwargs["heavy_matches_heavy_only"] = False
    core = get_cores(mol_a, mol_b, **kwargs)[0]

    with pytest.raises(ValueError, match=r"Invalid Mappings: \(17, 2\)"):
        verify_core_is_compatible_with_constraints(mol_a, mol_b, core, ff)

    kwargs["heavy_matches_heavy_only"] = True
    core = get_cores(mol_a, mol_b, **kwargs)[0]

    verify_core_is_compatible_with_constraints(mol_a, mol_b, core, ff)


def test_single_topology_verify_constraints_check():
    """Verify that verification of the core is possible with the SingleTopology object to avoid waiting until systems are constructed"""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol_a = ligand_from_smiles("C1CCCCC1")
    mol_b = ligand_from_smiles("C1C(Cl)CCCC1")

    kwargs = DEFAULT_ATOM_MAPPING_KWARGS.copy()
    kwargs["heavy_matches_heavy_only"] = False
    core = get_cores(mol_a, mol_b, **kwargs)[0]

    SingleTopology(mol_a, mol_b, core, ff)

    with pytest.raises(ValueError, match=r"Invalid Mappings: \(17, 2\)"):
        SingleTopology(mol_a, mol_b, core, ff, verify_constraints=True)


def test_filter_drops_mismatched_hydrogen_lengths():
    """A hydrogen mapped between two positions with different bond lengths
    (here C-H vs O-H) is dropped, while a length-matching pair is kept."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol = ligand_from_smiles("CO")  # methanol

    c_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    o_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    ch_h = _h_neighbors(mol, c_idx)[0]
    oh_h = _h_neighbors(mol, o_idx)[0]

    # Pair methanol with itself; the cross C-H <-> O-H row must be dropped.
    core = np.array(
        [
            [c_idx, c_idx],
            [o_idx, o_idx],
            [ch_h, ch_h],  # C-H <-> C-H: identical length, keep
            [oh_h, ch_h],  # O-H <-> C-H: mismatched length, drop
        ],
        dtype=np.int32,
    )

    with pytest.raises(ValueError, match="Invalid Mappings:"):
        verify_core_is_compatible_with_constraints(mol, mol, core, ff)

    filtered, dropped = filter_constraint_incompatible_hydrogens(mol, mol, core, ff)

    verify_core_is_compatible_with_constraints(mol, mol, filtered, ff)

    assert [tuple(d) for d in dropped] == [(oh_h, ch_h)]
    filtered_rows = filtered.tolist()
    assert [oh_h, ch_h] not in filtered_rows
    assert [ch_h, ch_h] in filtered_rows
    assert [c_idx, c_idx] in filtered_rows
    assert [o_idx, o_idx] in filtered_rows


def test_filter_is_noop_when_all_lengths_match():
    """Mapping a molecule onto itself constrains nothing away."""
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol = ligand_from_smiles("CCO")  # ethanol
    n = mol.GetNumAtoms()
    core = np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.int32)

    filtered, dropped = filter_constraint_incompatible_hydrogens(mol, mol, core, ff)

    assert dropped == []
    np.testing.assert_array_equal(filtered, core)
