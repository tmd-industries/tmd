import pytest

from tmd.fe.rest.single_topology import REST_REGION_ATOM_FLAG, SingleTopologyREST
from tmd.fe.rest.utils import assign_rest_atoms_from_smarts, match_smarts
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
    # At least one atom should not be in the REST region
    assert len(remapped_matched_atoms.difference(ref_rest_region)) > 0
    # Include the smarts match of mol_a
    assign_rest_atoms_from_smarts(mol_a, smarts)
    for atom in mol_a.GetAtoms():
        if atom.GetIdx() in matched_atom_idxs:
            assert atom.GetBoolProp(REST_REGION_ATOM_FLAG)
        else:
            assert not atom.HasProp(REST_REGION_ATOM_FLAG)

    st = SingleTopologyREST(mol_a, mol_b, core, ff, 3.0)

    assert len(st.rest_region_atom_idxs.difference(ref_rest_region)) > 0
    assert st.rest_region_atom_idxs.intersection(matched_atom_idxs) == matched_atom_idxs


def test_assign_rest_atoms_from_smarts_no_match():
    mol, _, _ = get_hif2a_ligand_pair_single_topology()

    for atom in mol.GetAtoms():
        assert not atom.HasProp(REST_REGION_ATOM_FLAG)

    with pytest.warns(match="failed to find REST atoms with SMARTS"):
        assign_rest_atoms_from_smarts(mol, "[U]")

    for atom in mol.GetAtoms():
        assert not atom.HasProp(REST_REGION_ATOM_FLAG)
