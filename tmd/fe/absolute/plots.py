from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from tmd.fe.utils import get_mol_name
import numpy as np

def generate_restraint_plot(
    mol: Chem.Mol, host_config, lig_atoms, rec_atoms, size: tuple[int, int] = (400, 400)
) -> str:
    """Generate a plot showing the restrained atoms in the protein and ligand

    Based off of https://greglandrum.github.io/rdkit-blog/posts/2025-09-26-drawing-interactions-1.html
    """
    

    lig_with_interactions = Chem.RWMol(mol)

    lig_atoms = np.array(lig_atoms) - host_config.conf.shape[0]

    # add pseudo-atoms (and bonds to them) for the interacting residues:
    highlighted_atoms = []
    highlighted_colors = {}
    for atom in lig_atoms:
        highlighted_atoms.append(int(atom))
        highlighted_colors[int(atom)] = (1.0, 0.2, 1.0, 0.3)

    dummy_bond_to_add = None
    for i, receptor_atom in enumerate(rec_atoms):
        omm_atom = next(atom for atom in host_config.omm_topology.atoms() if atom.index == receptor_atom)
        new_atom = Chem.Atom(omm_atom.element.atomic_number)
        res = omm_atom.residue
        new_atom.SetProp("atomLabel", f"{res.name} {res.id}")
        new_id = lig_with_interactions.AddAtom(new_atom)
        # Bond between ligand atoms and the receptor atoms
        if i == 0:
            dummy_bond_to_add = (int(lig_atoms[0]), new_id)
        else:
            lig_with_interactions.AddBond(mol.GetNumAtoms() + i - 1, new_id, Chem.BondType.SINGLE)
        highlighted_atoms.append(new_id)
        highlighted_colors[new_id] = (0.5, 0.2, 0.5, 0.3)
    assert dummy_bond_to_add is not None

    lig_with_interactions.RemoveAllConformers()
    rdDepictor.Compute2DCoords(lig_with_interactions)
    lig_with_interactions.AddBond(*dummy_bond_to_add, Chem.BondType.ZERO)

    d2d = Draw.MolDraw2DSVG(size[0], size[1])

    # set the draw options so that we end up with circles under the pseudo-atoms:
    d2d.drawOptions().circleAtoms = True
    d2d.drawOptions().fillHighlights = True
    d2d.drawOptions().continuousHighlight = False
    d2d.drawOptions().highlightRadius = 0.5

    # now draw and return the result
    d2d.DrawMolecule(
        lig_with_interactions,
        legend=get_mol_name(mol),
        highlightAtoms=highlighted_atoms,
        highlightAtomColors=highlighted_colors,
    )
    d2d.FinishDrawing()
    return d2d.GetDrawingText()
