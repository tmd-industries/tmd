from tempfile import NamedTemporaryFile

import pytest
from rdkit import Chem

from tmd.fe.rest.single_topology import REST_REGION_ATOM_FLAG, SingleTopologyREST
from tmd.fe.rest.utils import assign_rest_atoms_from_smarts, match_smarts
from tmd.fe.utils import read_sdf
from tmd.ff import Forcefield
from tmd.testsystems.relative import get_hif2a_ligand_pair_single_topology


@pytest.mark.parametrize("smarts", ["[S](=O)(=O)", "[O]"])
def test_assign_rest_atoms_from_smarts(smarts):
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    mol_a, mol_b, core = get_hif2a_ligand_pair_single_topology()
    st = SingleTopologyREST(mol_a, mol_b, core, ff, 3.0)

    ref_rest_region = st.rest_region_atom_idxs
    # If the whole region is covered, this test doesn't make sense
    assert len(ref_rest_region) != st.get_num_atoms()

    # Molecule A should have no atoms with the rest region
    for atom in mol_a.GetAtoms():
        assert not atom.HasProp(REST_REGION_ATOM_FLAG)

    matches = match_smarts(mol_a, smarts)
    matched_atom_idxs = set([idx for match in matches for idx in match])

    # Remap to the alchemical molecule
    remapped_matched_atoms = set([st.a_to_c[idx] for idx in matched_atom_idxs])
    assert len(remapped_matched_atoms) > 0
    # At least one atom should not be in the REST region
    assert len(remapped_matched_atoms.difference(ref_rest_region)) > 0
    # Include the smarts match of mol_a, called twice to ensure idempotent behavior
    for _ in range(2):
        assign_rest_atoms_from_smarts(mol_a, smarts)
        for atom in mol_a.GetAtoms():
            if atom.GetIdx() in matched_atom_idxs:
                assert atom.GetBoolProp(REST_REGION_ATOM_FLAG)
            else:
                assert not atom.HasProp(REST_REGION_ATOM_FLAG)

    st = SingleTopologyREST(mol_a, mol_b, core, ff, 3.0)

    assert len(st.rest_region_atom_idxs.difference(ref_rest_region)) > 0
    assert st.rest_region_atom_idxs.intersection(matched_atom_idxs) == matched_atom_idxs

    # Verify that the region gets written out when using the SDWriter
    with NamedTemporaryFile(suffix="tmd.sdf") as temp:
        with Chem.SDWriter(temp.name) as writer:
            writer.write(mol_a)
            writer.write(mol_b)
        parsed_mols = read_sdf(temp.name)
        assert len(parsed_mols) == 2
        parsed_a = parsed_mols[0]
        for atom in parsed_a.GetAtoms():
            if atom.GetIdx() in matched_atom_idxs:
                assert atom.GetBoolProp(REST_REGION_ATOM_FLAG)
            else:
                assert not atom.HasProp(REST_REGION_ATOM_FLAG)

        parsed_b = parsed_mols[1]
        for atom in parsed_b.GetAtoms():
            assert not atom.HasProp(REST_REGION_ATOM_FLAG)


def test_assign_rest_atoms_from_smarts_no_match():
    mol, _, _ = get_hif2a_ligand_pair_single_topology()

    for atom in mol.GetAtoms():
        assert not atom.HasProp(REST_REGION_ATOM_FLAG)

    with pytest.warns(match="failed to find REST atoms with SMARTS"):
        assign_rest_atoms_from_smarts(mol, "[U]")

    for atom in mol.GetAtoms():
        assert not atom.HasProp(REST_REGION_ATOM_FLAG)
