import networkx as nx
import numpy as np
from rdkit import Chem

from tmd.fe.utils import get_romol_conf
from tmd.lib import ConstraintGroups
from tmd.potentials.jax_utils import distance


def get_hydrogen_bond_constraint_groups(mol: Chem.Mol) -> ConstraintGroups:
    """Return hydrogen-bond constraint groups for a molecule.

    Builds connected components of heavy atoms and their bonded hydrogens.
    Each group has the heavy atom first, followed by hydrogen indices.

    Returns
    -------
    ConstraintGroups object
    """
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            graph.add_node(atom.GetIdx(), **{"hydrogen": True})
    starting_nodes = set(graph.nodes)
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if start_idx in starting_nodes or end_idx in starting_nodes:
            graph.add_edge(start_idx, end_idx)
    constraint_groups = []
    for component in nx.connected_components(graph):
        heavy_atom_first_component = list(
            sorted([atom for atom in component], key=lambda x: graph.nodes[x].get("hydrogen", False))
        )
        constraint_groups.append(heavy_atom_first_component)

    conf = get_romol_conf(mol)
    distances = []
    for group in constraint_groups:
        # Only support up to 6 bonds (+1 for the heavy atom)
        assert 1 < len(group) <= 7
        heavy = group[0]
        group_dist = [float(distance(conf[heavy], conf[idx], None)) for idx in group[1:]]
        distances.append(group_dist)
    return ConstraintGroups(constraint_groups, distances, np.array([], dtype=np.int_))
