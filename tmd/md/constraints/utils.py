from dataclasses import replace

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from tmd.fe.utils import get_romol_conf
from tmd.ff.handlers.utils import canonicalize_bond
from tmd.lib import ConstraintGroups
from tmd.potentials import BoundPotential, HarmonicBond
from tmd.potentials.jax_utils import distance
from tmd.potentials.types import Params


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


def prune_harmonic_bond_terms(
    pot: BoundPotential[HarmonicBond], constraints: ConstraintGroups
) -> BoundPotential[HarmonicBond]:
    """Remove harmonic bond terms that are constrained to avoid redundant computations.

    Parameters
    ----------
    pot : BoundPotential[HarmonicBond]
        Bound potential with harmonic bond terms to prune
    constraints : ConstraintGroups
        Constraint groups defining which atoms are constrained

    Returns
    -------
    BoundPotential[HarmonicBond]
        Filtered bound potential with constrained harmonic bond terms removed
    """
    assert isinstance(pot.potential, HarmonicBond)
    constrained_atoms = np.concatenate([group for group in constraints.groups])
    idxs = pot.potential.idxs
    # Handle the batched case
    num_batches = 1
    if isinstance(idxs, list):
        num_batches = len(idxs)
        idxs = np.asarray(idxs)

    assert idxs.shape[-1] == 2

    # Only look at bonds that are made up of entirely constrained atoms
    # Though still have to inspect further since the bond could be between two
    # constraint groups
    matches = np.isin(idxs, constrained_atoms).all(axis=-1)

    if np.count_nonzero(matches) > 0:
        pairs = set()
        for group in constraints.groups:
            for atom in group[1:]:
                pairs.add(canonicalize_bond((group[0], atom)))

        match_shape = matches.shape
        if len(matches.shape) > 1:
            match_idxs = np.arange(matches.size).reshape(matches.shape)[matches]
        else:
            match_idxs = np.arange(len(idxs))[matches]
        constrained_bond_matches = []
        for match_idx in match_idxs:
            # print(match_idx)
            pair = idxs[np.unravel_index(match_idx, match_shape)]
            if canonicalize_bond(tuple(pair)) in pairs:
                constrained_bond_matches.append(match_idx)

        keep_mask = np.ones_like(matches)
        keep_mask[np.unravel_index(constrained_bond_matches, matches.shape)] = False

        new_idxs: list | NDArray
        new_params: Params
        if num_batches > 1:
            assert len(keep_mask.shape) == 2 and keep_mask.shape[0] == num_batches
            new_idxs = [idxs[i][keep_mask[i]] for i in range(num_batches)]
            new_params = [pot.params[i][keep_mask[i]] for i in range(num_batches)]
        else:
            new_params = pot.params[keep_mask]
            new_idxs = idxs[keep_mask]

        bond_pot = replace(pot.potential, idxs=new_idxs)  # type: ignore

        pot = replace(pot, params=new_params, potential=bond_pot)
    return pot


def prune_constrained_valence_terms(
    bound_pots: list[BoundPotential], constraints: ConstraintGroups
) -> list[BoundPotential]:
    """Remove valence terms that are constrained to avoid redundant computations.

    Parameters
    ----------
    bound_pots : list[BoundPotential]
        List of bound potentials to process
    constraints : ConstraintGroups
        Constraint groups defining which atoms are constrained

    Returns
    -------
    list[BoundPotential]
        Filtered list of bound potentials with constrained valence terms removed

    Notes
    -----
    * Currently only prunes HarmonicBond terms that are constrained. Future versions may also prune harmonic angles.
    """

    output_pots = []
    for pot in bound_pots:
        if isinstance(pot.potential, HarmonicBond):
            pot = prune_harmonic_bond_terms(pot, constraints)
        output_pots.append(pot)

    return output_pots
