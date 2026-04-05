from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor

from tmd.fe.rest.single_topology import SingleTopologyREST
from tmd.fe.utils import get_mol_name


def plot_rest_region(single_top: SingleTopologyREST) -> Draw.MolsToGridImage:
    """Generate an SVG grid of a pair of molecules from a SingleToplogyREST object.

    Atoms and bonds that are in the REST region are highlighted in red, while torsions that are softened
    are in blue.
    """
    assert isinstance(single_top, SingleTopologyREST), "Must provide SingleTopologyREST object"

    mol_a_idxs, mol_b_idxs = single_top.split_combined_idxs(single_top.base_rest_region_atom_idxs)

    mol_a = Chem.Mol(single_top.mol_a)
    mol_b = Chem.Mol(single_top.mol_b)

    rdDepictor.SetPreferCoordGen(True)

    mol_a.RemoveAllConformers()
    mol_b.RemoveAllConformers()
    AllChem.Compute2DCoords(mol_a)
    AllChem.Compute2DCoords(mol_b)

    alchemical_rest_idxs = single_top.rest_region_atom_idxs

    mol_a_idxs = [a for a, c in enumerate(single_top.a_to_c) if c in alchemical_rest_idxs]
    mol_b_idxs = [b for b, c in enumerate(single_top.b_to_c) if c in alchemical_rest_idxs]

    weakened_torsion_idxs = set()
    # Include the proper torsions that are weakened
    for proper in single_top.target_propers.values():
        for idx in proper.idxs:
            if idx not in alchemical_rest_idxs:
                weakened_torsion_idxs.add(idx)

    # print(weakened_torsion_idxs)
    rest_region_color = (238 / 255, 144 / 255, 144 / 255)  # red
    weakened_torsion_color = (144 / 255, 144 / 255, 200 / 255)  # blue

    mol_a_weakened_torsions = [a for a, c in enumerate(single_top.a_to_c) if c in weakened_torsion_idxs]
    mol_b_weakened_torsions = [b for b, c in enumerate(single_top.b_to_c) if c in weakened_torsion_idxs]

    def generate_rest_bond_idxs_and_colors(mol, rest_atoms, weakened_torsions):
        bond_idxs = []
        bond_colors = dict()
        rest_atoms_set = set(rest_atoms)
        for bond in mol.GetBonds():
            src_a, dst_a = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if src_a in rest_atoms_set and dst_a in rest_atoms_set:
                bond_idxs.append(bond.GetIdx())
                if src_a in weakened_torsions and dst_a in weakened_torsions:
                    bond_colors[int(bond.GetIdx())] = weakened_torsion_color
                else:
                    bond_colors[int(bond.GetIdx())] = rest_region_color

        return bond_idxs, bond_colors

    atom_colors_a = {int(idx): rest_region_color for idx in mol_a_idxs}
    atom_colors_a.update({int(idx): weakened_torsion_color for idx in mol_a_weakened_torsions})
    atom_colors_b = {int(idx): rest_region_color for idx in mol_b_idxs}
    atom_colors_b.update({int(idx): weakened_torsion_color for idx in mol_b_weakened_torsions})
    mol_a_idxs.extend(list(mol_a_weakened_torsions))
    mol_b_idxs.extend(list(mol_b_weakened_torsions))

    # highlight bond idxs
    bond_idxs_a, bond_colors_a = generate_rest_bond_idxs_and_colors(mol_a, mol_a_idxs, mol_a_weakened_torsions)
    bond_idxs_b, bond_colors_b = generate_rest_bond_idxs_and_colors(mol_b, mol_b_idxs, mol_b_weakened_torsions)

    hals = [mol_a_idxs, mol_b_idxs]
    hacs = [atom_colors_a, atom_colors_b]
    hbls = [bond_idxs_a, bond_idxs_b]
    hbcs = [bond_colors_a, bond_colors_b]

    legends = [get_mol_name(mol_a), get_mol_name(mol_b)]

    return Draw.MolsToGridImage(
        [mol_a, mol_b],
        molsPerRow=2,
        highlightAtomLists=hals,
        highlightAtomColors=hacs,
        highlightBondLists=hbls,
        highlightBondColors=hbcs,
        subImgSize=(500, 300),
        legends=legends,
        useSVG=True,
    )
