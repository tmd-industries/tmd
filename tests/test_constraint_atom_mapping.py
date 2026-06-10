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

"""Tests for making the atom map / single topology compatible with hydrogen
constraints.

Two stages cooperate:

1. The atom mapper preemptively refuses to map hydrogens onto a heavy atom that
   is transmuted between end states (different atomic number), without reference
   to a forcefield (``constrain_hydrogens`` on ``get_cores``).
2. Single topology construction additionally unmaps hydrogen pairs whose
   forcefield bond lengths disagree (``constrain_hydrogens`` on
   ``SingleTopology``), catching same-element retypings that stage 1 cannot see.
"""

import numpy as np
from common import ligand_from_smiles

from tmd.constants import DEFAULT_ATOM_MAPPING_KWARGS
from tmd.fe import atom_mapping
from tmd.fe.single_topology import (
    AtomMapFlags,
    SingleTopology,
    filter_constraint_incompatible_hydrogens,
)
from tmd.ff import Forcefield


def _h_neighbors(mol, idx):
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]


def _parent(mol, h_idx):
    (nbr,) = list(mol.GetAtomWithIdx(h_idx).GetNeighbors())
    return nbr


def test_filter_drops_mismatched_hydrogen_lengths():
    """A hydrogen mapped between two positions with different bond lengths
    (here C-H vs O-H) is dropped, while a length-matching pair is kept."""
    ff = Forcefield.load_default()
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

    filtered, dropped = filter_constraint_incompatible_hydrogens(mol, mol, core, ff)

    assert [tuple(d) for d in dropped] == [(oh_h, ch_h)]
    filtered_rows = filtered.tolist()
    assert [oh_h, ch_h] not in filtered_rows
    assert [ch_h, ch_h] in filtered_rows
    assert [c_idx, c_idx] in filtered_rows
    assert [o_idx, o_idx] in filtered_rows


def test_filter_is_noop_when_all_lengths_match():
    """Mapping a molecule onto itself constrains nothing away."""
    ff = Forcefield.load_default()
    mol = ligand_from_smiles("CCO")  # ethanol
    n = mol.GetNumAtoms()
    core = np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.int32)

    filtered, dropped = filter_constraint_incompatible_hydrogens(mol, mol, core, ff)

    assert dropped == []
    np.testing.assert_array_equal(filtered, core)


def test_mapper_skips_hydrogens_on_transmuted_parents():
    """With ``constrain_hydrogens``, no hydrogen is mapped across a parent atom
    whose atomic number changes (e.g. the O->N terminus of ethanol->ethylamine)."""
    mol_a = ligand_from_smiles("CCO")  # ethanol
    mol_b = ligand_from_smiles("CCN")  # ethylamine

    kwargs = dict(DEFAULT_ATOM_MAPPING_KWARGS)

    cores_default = atom_mapping.get_cores(mol_a, mol_b, **kwargs)
    cores_constrained = atom_mapping.get_cores(mol_a, mol_b, constrain_hydrogens=True, **kwargs)

    def mapped_hydrogen_pairs(core):
        return [
            (int(a), int(b))
            for a, b in core
            if mol_a.GetAtomWithIdx(int(a)).GetAtomicNum() == 1 and mol_b.GetAtomWithIdx(int(b)).GetAtomicNum() == 1
        ]

    # Invariant: every mapped hydrogen has parents of equal atomic number.
    for a, b in mapped_hydrogen_pairs(cores_constrained[0]):
        assert _parent(mol_a, a).GetAtomicNum() == _parent(mol_b, b).GetAtomicNum()

    # The flag can only remove hydrogen mappings, never add them.
    assert len(mapped_hydrogen_pairs(cores_constrained[0])) <= len(mapped_hydrogen_pairs(cores_default[0]))


def test_single_topology_constrain_hydrogens_prunes_incompatible_pairs():
    """SingleTopology(constrain_hydrogens=True) yields a core on which the
    forcefield-aware filter is a no-op, and which is no larger than the input."""
    # Simple-charge forcefield avoids the OpenEye AM1CCC dependency; bond lengths
    # (the only thing the filter consults) are independent of the charge model.
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol_a = ligand_from_smiles("CCO")  # ethanol
    mol_b = ligand_from_smiles("CCN")  # ethylamine

    # Use a core that is allowed to map hydrogens across the O->N transmutation,
    # so the single-topology stage has something to prune.
    kwargs = dict(DEFAULT_ATOM_MAPPING_KWARGS)
    core = atom_mapping.get_cores(mol_a, mol_b, **kwargs)[0]

    st = SingleTopology(mol_a, mol_b, core, ff, constrain_hydrogens=True)

    # Every mapped hydrogen remaining in the core is constraint-compatible.
    _, dropped = filter_constraint_incompatible_hydrogens(mol_a, mol_b, st.core, ff)
    assert dropped == []
    assert len(st.core) <= len(core)

    # The dropped hydrogens are demoted to per-state dummy atoms, not lost.
    constrained_pairs = {(int(a), int(b)) for a, b in st.core}
    for a, b in core:
        a, b = int(a), int(b)
        if (a, b) not in constrained_pairs:
            # a was a mol_a hydrogen that is now a dummy belonging to mol_a
            assert mol_a.GetAtomWithIdx(a).GetAtomicNum() == 1
            assert st.c_flags[a] == AtomMapFlags.MOL_A
