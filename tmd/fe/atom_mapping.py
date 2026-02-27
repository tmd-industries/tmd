# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025, Forrest York
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

from collections import defaultdict
from functools import partial
from itertools import combinations, permutations
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from tmd.fe import mcgregor
from tmd.fe.chiral_utils import (
    ChiralRestrIdxSet,
    has_chiral_atom_flips,
    setup_find_flipped_planar_torsions,
)
from tmd.fe.utils import get_romol_bonds, get_romol_conf

# (ytz): Just like how one should never re-write an MD engine, one should never rewrite an MCS library.
# Unless you have to. And now we have to. If you want to understand what this code is doing, it
# is strongly recommended that the reader reads:

# Backtrack search algorithms and the maximal common subgraph problem
# James J McGregor,  January 1982, https://doi.org/10.1002/spe.4380120103

# Theoretical Tricks
# ------------------
# Historically, MCS methods have relied on finding the largest core as soon as possible. However, this can pose difficulties
# since we may get stuck in a local region of poor quality (that end up having far smaller than the optimal). Our algorithm
# has several clever tricks up its sleeve in that we:

# - designed the method for free energy methods where the provided two molecules are aligned.
# - refine the row/cols of marcs (edge-edge mapping matrix) when a new atom-atom mapping is proposed
# - prune by looking at maximum number of row edges and column edges, i.e. arcs_left min(max_row_edges, max_col_edges)
# - only generate an atom-mapping between two mols, whereas RDKit generates a common substructure between N mols
# - operate on anonymous graphs whose atom-atom compatibility depends on a predicates matrix, such that a 1 is when
#   if atom i in mol_a is compatible with atom j in mol_b, and 0 otherwise. We do not implement a bond-bond compatibility matrix.
# - allow for the generation of disconnected atom-mappings, which is very useful for linker changes etc.
# - re-order the vertices in graph based on the degree, this penalizes None mapping by the degree of the vertex
# - when searching for atoms in mol_b to map, we prioritize based on distance
# - uses a best-first search ordering with an upper bound on the number of edges in correspondence (i.e. arcs_left) as
#   the heuristic. This guarantees that the optimal (in the sense of maximum number of edges) mappings are returned
#   first (see https://github.com/proteneer/timemachine/pull/1415#issue-2627969721 for details)
# - termination (without a timeout warning) guarantees optimality of the solution(s). If timeout occurs before an
#   exhaustive search can be performed, a warning is raised

# Engineering Tricks
# ------------------
# This is entirely written in python, which lends to its ease of use and modifiability. The following optimizations were
# implemented (without changing the number of nodes visited):
# - multiple representations of graph structures to improve efficiency
# - refinement of marcs matrix is done on uint8 arrays


def get_cores_and_diagnostics(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    heavy_matches_heavy_only,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
    initial_mapping,
) -> tuple[list[NDArray], mcgregor.MCSDiagnostics]:
    """Same as :py:func:`get_cores`, but additionally returns diagnostics collected during the MCS search."""
    assert max_cores > 0

    get_cores_ = partial(
        _get_cores_impl,
        ring_cutoff=ring_cutoff,
        chain_cutoff=chain_cutoff,
        max_visits=max_visits,
        max_connected_components=max_connected_components,
        min_connected_component_size=min_connected_component_size,
        max_cores=max_cores,
        enforce_core_core=enforce_core_core,
        ring_matches_ring_only=ring_matches_ring_only,
        heavy_matches_heavy_only=heavy_matches_heavy_only,
        enforce_chiral=enforce_chiral,
        disallow_planar_torsion_flips=disallow_planar_torsion_flips,
        min_threshold=min_threshold,
    )

    # we require that mol_a.GetNumAtoms() <= mol_b.GetNumAtoms()
    if mol_a.GetNumAtoms() > mol_b.GetNumAtoms():
        # reverse the columns of initial_mapping and the resulting cores
        initial_mapping_r = initial_mapping[:, ::-1] if initial_mapping is not None else None
        all_cores_r, mcs_diagnostics = get_cores_(mol_b, mol_a, initial_mapping=initial_mapping_r)
        all_cores = [core_r[:, ::-1] for core_r in all_cores_r]
    else:
        all_cores, mcs_diagnostics = get_cores_(mol_a, mol_b, initial_mapping=initial_mapping)
    return all_cores, mcs_diagnostics


def get_cores(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    heavy_matches_heavy_only,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
    initial_mapping,
) -> list[NDArray]:
    """
    Finds set of cores between two molecules that maximizes the number of common edges.

    If either atom i or atom j is in a ring then the dist(i,j) < ring_cutoff, otherwise dist(i,j) < chain_cutoff

    Additional notes
    ----------------
    1) The returned cores are jointly sorted in increasing order based on the number of core-dummy bonds broken,
       the sum of valence values changed, and the rmsd of the alignment.
    2) The number of cores atoms may vary slightly, but the number of mapped edges are the same.
    3) If a time-out has occurred due to max_visits, then an exception is thrown.

    Parameters
    ----------
    mol_a: Chem.Mol
        Input molecule a. Must have a conformation.

    mol_b: Chem.Mol
        Input molecule b. Must have a conformation.

    ring_cutoff: float
        The distance cutoff that ring atoms must satisfy.

    chain_cutoff: float
        The distance cutoff that non-ring atoms must satisfy.

    max_visits: int
        Maximum number of nodes we can visit to generate at least one core.

    max_connected_components: int or None
        Set to k to only keep mappings where the number of connected components is <= k.
        The definition of connected here is different from McGregor. Here it means there is a way to reach the mapped
        atom without traversing over a non-mapped atom.

    min_connected_component_size: int
        Set to n to only keep mappings where all connected components have size >= n.

    max_cores: int or float
        maximum number of maximal cores to store, this can be an +np.inf if you want
        every core - when set to 1 this enables a faster predicate that allows for more pruning.

    enforce_core_core: bool
        If we allow core-core bonds to be broken. This may be deprecated later on.

    ring_matches_ring_only: bool
        atom i in mol A can match atom j in mol B
        only if in_ring(i, A) == in_ring(j, B)

    heavy_matches_heavy_only: bool
        atom i in mol A can match atom j in mol B
        only if is_hydrogen(i, A) == is_hydrogen(i, B)

    enforce_chiral: bool
        Filter out cores that would flip atom chirality

    disallow_planar_torsion_flips: bool
        Filter out cores that would flip a mapped planar torsion (i.e. change the sign of the torsion volume)

    min_threshold: int
        Number of edges to require for a valid mapping

    Returns
    -------
    Returns a list of all_cores

    Raises
    ------
    tmd.fe.mcgregor.NoMappingError
        If no mapping is found
    """
    all_cores, _ = get_cores_and_diagnostics(
        mol_a,
        mol_b,
        ring_cutoff,
        chain_cutoff,
        max_visits,
        max_connected_components,
        min_connected_component_size,
        max_cores,
        enforce_core_core,
        ring_matches_ring_only,
        heavy_matches_heavy_only,
        enforce_chiral,
        disallow_planar_torsion_flips,
        min_threshold,
        initial_mapping,
    )

    return all_cores


def reorder_atoms_by_degree_and_initial_mapping(mol, initial_mapping):
    degrees = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
    for a in mol.GetAtoms():
        if a.GetIdx() in initial_mapping[:, 0]:
            degrees[a.GetIdx()] += np.inf
    perm = np.argsort(degrees, kind="stable")[::-1]

    old_to_new = {}
    for new, old in enumerate(perm):
        old_to_new[old] = new

    new_mol = Chem.RenumberAtoms(mol, perm.tolist())
    new_mapping = []
    for a, b in initial_mapping:
        new_mapping.append([old_to_new[a], b])
    new_mapping = np.array(new_mapping)

    return new_mol, perm, new_mapping


def _uniquify_core(core):
    core_list = []
    for a, b in core:
        core_list.append((a, b))
    return frozenset(core_list)


def _deduplicate_all_cores(all_cores):
    unique_cores = {}
    for core in all_cores:
        # Be careful with the unique core here, list -> set -> list is not consistent
        # across versions of python, use the frozen as the key, but return the untouched
        # cores
        unique_cores[_uniquify_core(core)] = core

    return list(unique_cores.values())


def core_bonds_broken_count(mol_a, mol_b, core):
    # count the number of core bonds broken in mol_a when applying the core atom map
    core_a_to_b = dict(core)
    count = 0
    for bond in mol_a.GetBonds():
        src_a, dst_a = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if src_a in core_a_to_b and dst_a in core_a_to_b:
            if mol_b.GetBondBetweenAtoms(int(core_a_to_b[src_a]), int(core_a_to_b[dst_a])):
                pass
            else:
                count += 1

    return count


def _build_h_removal_index_maps(mol_with_h, mol_without_h):
    """Build bidirectional index maps between a molecule and its H-stripped version.

    Chem.RemoveHs() may retain some Hs (e.g. those defining double-bond stereochemistry).
    This function identifies which atoms survived removal and builds the mapping.

    Parameters
    ----------
    mol_with_h : Chem.Mol
        Original molecule with all explicit Hs.
    mol_without_h : Chem.Mol
        Molecule after Chem.RemoveHs().

    Returns
    -------
    old_to_new : dict[int, int]
        Maps original atom index -> H-stripped atom index. Only contains atoms
        that survived removal.
    new_to_old : dict[int, int]
        Maps H-stripped atom index -> original atom index.
    removed_h_set : set[int]
        Set of original atom indices that were removed (all are hydrogens).
    """
    # RDKit's RemoveHs preserves the relative order of retained atoms,
    # so we can build the mapping by iterating and skipping removed Hs.
    n_old = mol_with_h.GetNumAtoms()
    n_new = mol_without_h.GetNumAtoms()

    old_to_new = {}
    new_to_old = {}
    removed_h_set = set()
    new_idx = 0

    for old_idx in range(n_old):
        atom = mol_with_h.GetAtomWithIdx(old_idx)
        if atom.GetAtomicNum() == 1:
            # Check if this H was retained (e.g. double-bond stereo H)
            if new_idx < n_new and mol_without_h.GetAtomWithIdx(new_idx).GetAtomicNum() == 1:
                # This H was retained
                old_to_new[old_idx] = new_idx
                new_to_old[new_idx] = old_idx
                new_idx += 1
            else:
                removed_h_set.add(old_idx)
        else:
            old_to_new[old_idx] = new_idx
            new_to_old[new_idx] = old_idx
            new_idx += 1

    assert new_idx == n_new
    return old_to_new, new_to_old, removed_h_set


def _get_removable_h_neighbors(mol, atom_idx, removed_h_set):
    """Return indices of hydrogen neighbors of atom_idx that were removed by RemoveHs.

    Parameters
    ----------
    mol : Chem.Mol
        Original molecule (with all Hs).
    atom_idx : int
        Atom index in the original molecule.
    removed_h_set : set[int]
        Set of atom indices that were removed by RemoveHs.

    Returns
    -------
    list[int]
        Indices of removed H neighbors, in original molecule numbering.
    """
    return [nb.GetIdx() for nb in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nb.GetIdx() in removed_h_set]


def _h_squared_distance(conf_a, conf_b, ha, hb):
    """Squared Euclidean distance between two hydrogen positions."""
    diff = conf_a[ha] - conf_b[hb]
    return np.dot(diff, diff)


def _assign_h_pairs_hungarian(h_neighbors_a, h_neighbors_b, conf_a, conf_b, sq_cutoff):
    """Use Hungarian algorithm to optimally match H neighbors, respecting a distance cutoff.

    Beyond-cutoff entries are replaced with a large sentinel so that
    ``linear_sum_assignment`` remains feasible.  Sentinel-assigned pairs
    are filtered out of the result.

    Returns
    -------
    list[list[int, int]]
        Matched ``[ha, hb]`` pairs within cutoff, or empty list.
    """
    n_a, n_b = len(h_neighbors_a), len(h_neighbors_b)
    cost = np.zeros((n_a, n_b))
    for ii, ha in enumerate(h_neighbors_a):
        for jj, hb in enumerate(h_neighbors_b):
            d2 = _h_squared_distance(conf_a, conf_b, ha, hb)
            cost[ii, jj] = np.inf if (sq_cutoff is not None and d2 >= sq_cutoff) else d2

    if not np.any(np.isfinite(cost)):
        return []

    # Replace inf with a large finite sentinel so scipy doesn't reject the matrix.
    max_finite = np.nanmax(cost[np.isfinite(cost)])
    sentinel = max_finite * 1e6 + 1e6
    solvable_cost = np.where(np.isfinite(cost), cost, sentinel)

    row_ind, col_ind = linear_sum_assignment(solvable_cost)
    return [[h_neighbors_a[r], h_neighbors_b[c]] for r, c in zip(row_ind, col_ind) if np.isfinite(cost[r, c])]


def _find_chiral_valid_h_assignment(
    h_neighbors_a,
    h_neighbors_b,
    k,
    conf_a,
    conf_b,
    sq_cutoff,
    mapping,
    current_pairs,
    center_has_conflict_fn,
    center_a,
):
    """Search all k-permutations of H neighbors for one that avoids a chiral conflict.

    Tries all k-subsets of A-side Hs crossed with all k-permutations of
    B-side Hs.  Returns the assignment that maximises surviving (within-cutoff)
    pairs with lowest cost, or ``None`` if no valid permutation exists.
    """
    best_pairs = None
    best_n = -1
    best_cost = float("inf")

    for combo_a in combinations(h_neighbors_a, k):
        for perm_b in permutations(h_neighbors_b, k):
            trial_pairs = [[ha, hb] for ha, hb in zip(combo_a, perm_b)]

            if sq_cutoff is not None:
                surviving = [p for p in trial_pairs if _h_squared_distance(conf_a, conf_b, p[0], p[1]) < sq_cutoff]
            else:
                surviving = trial_pairs

            # Build trial mapping: remove old pairs, add surviving ones
            trial_mapping = dict(mapping)
            for old_ha, _ in current_pairs:
                trial_mapping.pop(old_ha, None)
            for ha, hb in surviving:
                trial_mapping[ha] = hb

            if not center_has_conflict_fn(trial_mapping, center_a):
                n = len(surviving)
                cost = sum(_h_squared_distance(conf_a, conf_b, ha, hb) for ha, hb in surviving) if surviving else 0.0
                if n > best_n or (n == best_n and cost < best_cost):
                    best_n = n
                    best_cost = cost
                    best_pairs = surviving

    return best_pairs


def _augment_core_with_hydrogens(
    mol_a,
    mol_b,
    heavy_core,
    conf_a,
    conf_b,
    removed_h_a,
    removed_h_b,
    chiral_set_a=None,
    chiral_set_b=None,
    enforce_chiral=False,
    chain_cutoff=None,
):
    """Augment a heavy-atom core with optimal hydrogen mappings.

    For each mapped heavy-atom pair (a_i, b_j), finds hydrogen neighbors of a_i
    that were removed and hydrogen neighbors of b_j that were removed, then uses
    the Hungarian algorithm on squared distances to optimally match them.
    The ``chain_cutoff`` is enforced during the Hungarian assignment itself
    so that beyond-cutoff pairs are never selected.

    When ``enforce_chiral`` is True and full-molecule chiral sets are provided,
    the augmentation checks for chiral conflicts introduced by the H assignments.
    For each conflicting parent center, alternative permutations of the H
    assignment are tried before falling back to removing the H pairs entirely.

    Parameters
    ----------
    mol_a, mol_b : Chem.Mol
        Original molecules with all Hs.
    heavy_core : NDArray
        Shape (K, 2) array of mapped atom pairs in original molecule indices.
    conf_a, conf_b : NDArray
        Conformer coordinates, shape (N, 3), in nm.
    removed_h_a, removed_h_b : set[int]
        Sets of atom indices removed by RemoveHs.
    chiral_set_a, chiral_set_b : ChiralRestrIdxSet or None
        Chiral restraint index sets built from the full-H molecules.
    enforce_chiral : bool
        If True and chiral sets are provided, repair H assignments that
        would introduce chiral conflicts.
    chain_cutoff : float or None
        Maximum distance (in nm) for an H pair to be included.  Pairs
        whose Euclidean distance exceeds this value are never assigned.

    Returns
    -------
    NDArray
        Augmented core including hydrogen mappings, shape (K + M, 2).
    """
    sq_cutoff = chain_cutoff * chain_cutoff if chain_cutoff is not None else None

    # --- Phase 1: Hungarian assignment per parent pair (cutoff-aware) ---
    h_pairs_by_parent = {}  # (a_i, b_j) -> list of [ha, hb] pairs
    for a_i, b_j in heavy_core:
        a_i, b_j = int(a_i), int(b_j)
        h_a = _get_removable_h_neighbors(mol_a, a_i, removed_h_a)
        h_b = _get_removable_h_neighbors(mol_b, b_j, removed_h_b)
        if not h_a or not h_b:
            continue
        pairs = _assign_h_pairs_hungarian(h_a, h_b, conf_a, conf_b, sq_cutoff)
        if pairs:
            h_pairs_by_parent[(a_i, b_j)] = pairs

    # --- Phase 2: Repair chiral conflicts introduced by H assignments ---
    if enforce_chiral and chiral_set_a is not None and chiral_set_b is not None:
        _repair_chiral_conflicts(
            h_pairs_by_parent,
            heavy_core,
            mol_a,
            mol_b,
            conf_a,
            conf_b,
            removed_h_a,
            removed_h_b,
            chiral_set_a,
            chiral_set_b,
            sq_cutoff,
        )

    # --- Build and return augmented core ---
    all_h = [pair for pairs in h_pairs_by_parent.values() for pair in pairs]
    if all_h:
        return np.concatenate([heavy_core, np.array(all_h)], axis=0)
    return heavy_core


def _repair_chiral_conflicts(
    h_pairs_by_parent,
    heavy_core,
    mol_a,
    mol_b,
    conf_a,
    conf_b,
    removed_h_a,
    removed_h_b,
    chiral_set_a,
    chiral_set_b,
    sq_cutoff,
):
    """Detect and repair chiral conflicts caused by H assignments, in-place.

    For each parent center with a conflict, tries all permutations of H
    neighbors to find a valid assignment.  Falls back to removing the H
    pairs entirely if no permutation resolves the conflict.
    """

    def _center_has_chiral_conflict(mapping_a_to_b, center_a):
        for c_a, i_a, j_a, k_a in chiral_set_a.restr_idxs:
            if c_a != center_a:
                continue
            c_b = mapping_a_to_b.get(c_a)
            i_b = mapping_a_to_b.get(i_a)
            j_b = mapping_a_to_b.get(j_a)
            k_b = mapping_a_to_b.get(k_a)
            if c_b is None or i_b is None or j_b is None or k_b is None:
                continue
            if chiral_set_b.disallows((c_b, i_b, j_b, k_b)):
                return True
        return False

    # Build mapping from the current augmented core
    all_h = [pair for pairs in h_pairs_by_parent.values() for pair in pairs]
    if not all_h:
        return

    augmented = np.concatenate([heavy_core, np.array(all_h)], axis=0)
    mapping = {int(a): int(b) for a, b in augmented}

    # Identify and repair conflicting parents
    conflicting = [(a_i, b_j) for (a_i, b_j) in h_pairs_by_parent if _center_has_chiral_conflict(mapping, a_i)]

    for a_i, b_j in conflicting:
        h_a = _get_removable_h_neighbors(mol_a, a_i, removed_h_a)
        h_b = _get_removable_h_neighbors(mol_b, b_j, removed_h_b)
        k = len(h_pairs_by_parent[(a_i, b_j)])

        best = _find_chiral_valid_h_assignment(
            h_a,
            h_b,
            k,
            conf_a,
            conf_b,
            sq_cutoff,
            mapping,
            h_pairs_by_parent[(a_i, b_j)],
            _center_has_chiral_conflict,
            a_i,
        )

        if best is not None:
            h_pairs_by_parent[(a_i, b_j)] = best
            for ha, hb in best:
                mapping[ha] = hb
        else:
            for ha, _ in h_pairs_by_parent[(a_i, b_j)]:
                mapping.pop(ha, None)
            del h_pairs_by_parent[(a_i, b_j)]

    # Safety net: remove any remaining conflicts
    all_h = [pair for pairs in h_pairs_by_parent.values() for pair in pairs]
    if all_h:
        augmented = np.concatenate([heavy_core, np.array(all_h)], axis=0)
        mapping = {int(a): int(b) for a, b in augmented}
        for a_i, b_j in list(h_pairs_by_parent.keys()):
            if _center_has_chiral_conflict(mapping, a_i):
                del h_pairs_by_parent[(a_i, b_j)]


def _build_priority_idxs(
    mol_a, mol_b, conf_a, conf_b, initial_mapping, ring_cutoff, chain_cutoff, ring_matches_ring_only
):
    """Build distance-sorted, cutoff-filtered candidate lists for each atom in mol_a.

    For atoms covered by ``initial_mapping``, the candidate list is the single
    mapped partner.  For all others, candidates in mol_b are filtered by
    ring/chain cutoff and sorted by distance.

    Returns
    -------
    list[list[int]]
        One candidate list per atom in mol_a.
    """
    initial_mapping_kv = {int(src): int(dst) for src, dst in initial_mapping}
    priority_idxs = []

    for idx, a_xyz in enumerate(conf_a):
        if idx < len(initial_mapping):
            priority_idxs.append([initial_mapping_kv[idx]])
            continue

        atom_i = mol_a.GetAtomWithIdx(idx)
        dijs = []
        allowed_idxs = set()

        for jdx, b_xyz in enumerate(conf_b):
            atom_j = mol_b.GetAtomWithIdx(jdx)
            dij = np.linalg.norm(a_xyz - b_xyz)
            dijs.append(dij)

            if ring_matches_ring_only and (atom_i.IsInRing() != atom_j.IsInRing()):
                continue

            cutoff = ring_cutoff if (atom_i.IsInRing() or atom_j.IsInRing()) else chain_cutoff
            if dij < cutoff:
                allowed_idxs.add(jdx)

        priority_idxs.append([int(j) for j in np.argsort(dijs, kind="stable") if j in allowed_idxs])

    return priority_idxs


def _build_mcs_filter(mol_a, mol_b, conf_a, conf_b, enforce_chiral, disallow_planar_torsion_flips):
    """Build a composite filter function for MCS candidate cores.

    Returns
    -------
    callable
        A function ``(trial_core) -> bool`` that returns True if the core
        passes all enabled filters.
    """
    filter_fxns = []

    if enforce_chiral:
        chiral_set_a = ChiralRestrIdxSet.from_mol(mol_a, conf_a)
        chiral_set_b = ChiralRestrIdxSet.from_mol(mol_b, conf_b)

        def chiral_filter(trial_core):
            return not has_chiral_atom_flips(trial_core, chiral_set_a, chiral_set_b)

        filter_fxns.append(chiral_filter)

    if disallow_planar_torsion_flips:
        find_flipped = setup_find_flipped_planar_torsions(mol_a, mol_b)

        def planar_torsion_flip_filter(trial_core):
            return next(find_flipped(trial_core), None) is None

        filter_fxns.append(planar_torsion_flip_filter)

    def filter_fxn(trial_core):
        return all(f(trial_core) for f in filter_fxns)

    return filter_fxn


def _translate_core_to_original_indices(core, perm, new_to_old_a, new_to_old_b, swapped):
    """Undo atom reorder and H-stripped index translation for a single core.

    Parameters
    ----------
    core : NDArray
        Shape (K, 2) core from MCS on reordered/H-stripped molecules.
    perm : NDArray
        Permutation applied to mol_a atoms for degree reordering.
    new_to_old_a, new_to_old_b : dict[int, int]
        H-stripped → original atom index maps.
    swapped : bool
        Whether mol_a and mol_b were swapped for the n_a <= n_b requirement.

    Returns
    -------
    NDArray
        Core in original molecule indices, shape (K, 2).
    """
    # Undo degree reorder on column 0 (mol_a was reordered)
    core[:, 0] = perm[core[:, 0]]

    # Translate H-stripped → original indices
    orig_pairs = [[new_to_old_a[int(a)], new_to_old_b[int(b)]] for a, b in core]
    orig_core = np.array(orig_pairs)

    # Undo the swap if we swapped for the n_a <= n_b requirement
    if swapped:
        orig_core = orig_core[:, ::-1]

    return orig_core


def _score_and_sort_cores(cores, mol_a, mol_b, conf_a, conf_b):
    """Score cores and sort by (bonds broken, valence mismatch, mean squared distance).

    Parameters
    ----------
    cores : list[NDArray]
        Cores in original molecule indices.
    mol_a, mol_b : Chem.Mol
        Original molecules.
    conf_a, conf_b : NDArray
        Conformer coordinates.

    Returns
    -------
    list[NDArray]
        Sorted cores.
    """
    mean_sq_distances = []
    valence_mismatches = []
    cb_counts = []

    for core in cores:
        r2_ij = np.sum((conf_a[core[:, 0]] - conf_b[core[:, 1]]) ** 2)
        mean_sq_distances.append(r2_ij / len(core))

        v_count = sum(
            abs(mol_a.GetAtomWithIdx(int(i)).GetTotalValence() - mol_b.GetAtomWithIdx(int(j)).GetTotalValence())
            for i, j in core
        )
        valence_mismatches.append(v_count)

        cb_counts.append(
            core_bonds_broken_count(mol_a, mol_b, core) + core_bonds_broken_count(mol_b, mol_a, core[:, [1, 0]])
        )

    sort_vals = np.array(
        list(zip(cb_counts, valence_mismatches, mean_sq_distances)),
        dtype=[("cb", "i"), ("valence", "i"), ("msd", "f")],
    )
    return [cores[p] for p in np.argsort(sort_vals, order=["cb", "valence", "msd"])]


def _get_cores_impl(
    mol_a,
    mol_b,
    ring_cutoff,
    chain_cutoff,
    max_visits,
    max_connected_components: Optional[int],
    min_connected_component_size: int,
    max_cores,
    enforce_core_core,
    ring_matches_ring_only,
    heavy_matches_heavy_only,
    enforce_chiral,
    disallow_planar_torsion_flips,
    min_threshold,
    initial_mapping,
) -> tuple[list[NDArray], mcgregor.MCSDiagnostics]:
    if initial_mapping is None:
        initial_mapping = np.zeros((0, 2), dtype=np.intp)

    # Keep references to the original (full-H) molecules and their conformers
    mol_a_full = mol_a
    mol_b_full = mol_b
    conf_a_full = get_romol_conf(mol_a_full)
    conf_b_full = get_romol_conf(mol_b_full)

    # --- Prepare H-stripped (or identity) molecules and index maps ---
    if heavy_matches_heavy_only:
        mol_a_heavy = Chem.RemoveHs(mol_a)
        mol_b_heavy = Chem.RemoveHs(mol_b)
        old_to_new_a, new_to_old_a, removed_h_a = _build_h_removal_index_maps(mol_a_full, mol_a_heavy)
        old_to_new_b, new_to_old_b, removed_h_b = _build_h_removal_index_maps(mol_b_full, mol_b_heavy)

        heavy_initial_pairs = []
        for a_old, b_old in initial_mapping:
            a_old, b_old = int(a_old), int(b_old)
            if a_old in old_to_new_a and b_old in old_to_new_b:
                heavy_initial_pairs.append([old_to_new_a[a_old], old_to_new_b[b_old]])
        heavy_initial_mapping = np.array(heavy_initial_pairs, dtype=np.intp).reshape(-1, 2)
    else:
        mol_a_heavy = mol_a
        mol_b_heavy = mol_b
        n_a_full = mol_a.GetNumAtoms()
        n_b_full = mol_b.GetNumAtoms()
        old_to_new_a = {i: i for i in range(n_a_full)}
        old_to_new_b = {i: i for i in range(n_b_full)}
        new_to_old_a = {i: i for i in range(n_a_full)}
        new_to_old_b = {i: i for i in range(n_b_full)}
        removed_h_a = set()
        removed_h_b = set()
        heavy_initial_mapping = initial_mapping.copy() if len(initial_mapping) > 0 else initial_mapping

    # --- Ensure n_a <= n_b (may swap after H removal changes relative sizes) ---
    swapped_heavy = False
    if mol_a_heavy.GetNumAtoms() > mol_b_heavy.GetNumAtoms():
        mol_a_heavy, mol_b_heavy = mol_b_heavy, mol_a_heavy
        old_to_new_a, old_to_new_b = old_to_new_b, old_to_new_a
        new_to_old_a, new_to_old_b = new_to_old_b, new_to_old_a
        removed_h_a, removed_h_b = removed_h_b, removed_h_a
        mol_a_full, mol_b_full = mol_b_full, mol_a_full
        conf_a_full, conf_b_full = conf_b_full, conf_a_full
        heavy_initial_mapping = (
            heavy_initial_mapping[:, ::-1] if len(heavy_initial_mapping) > 0 else heavy_initial_mapping  # type: ignore[assignment]
        )
        swapped_heavy = True

    # --- Reorder atoms by degree for MCS efficiency ---
    mol_a_heavy, perm, heavy_initial_mapping = reorder_atoms_by_degree_and_initial_mapping(
        mol_a_heavy, heavy_initial_mapping
    )
    conf_a = get_romol_conf(mol_a_heavy)
    conf_b = get_romol_conf(mol_b_heavy)
    n_a = len(conf_a)
    n_b = len(conf_b)

    # --- Edge case: single heavy atom (e.g. methane) ---
    if n_a == 1:
        return _handle_single_atom_core(
            perm,
            conf_a,
            conf_b,
            new_to_old_a,
            new_to_old_b,
            swapped_heavy,
            heavy_matches_heavy_only,
            mol_a_full,
            mol_b_full,
            conf_a_full,
            conf_b_full,
            removed_h_a,
            removed_h_b,
            enforce_chiral,
            chain_cutoff,
        )

    # --- Build candidate lists and filters ---
    priority_idxs = _build_priority_idxs(
        mol_a_heavy,
        mol_b_heavy,
        conf_a,
        conf_b,
        heavy_initial_mapping,
        ring_cutoff,
        chain_cutoff,
        ring_matches_ring_only,
    )
    filter_fxn = _build_mcs_filter(
        mol_a_heavy,
        mol_b_heavy,
        conf_a,
        conf_b,
        enforce_chiral,
        disallow_planar_torsion_flips,
    )

    # --- Run MCS ---
    all_cores, _, mcs_diagnostics = mcgregor.mcs(
        n_a,
        n_b,
        priority_idxs,
        get_romol_bonds(mol_a_heavy),
        get_romol_bonds(mol_b_heavy),
        max_visits,
        max_cores,
        enforce_core_core,
        max_connected_components,
        min_connected_component_size,
        min_threshold,
        heavy_initial_mapping,
        filter_fxn,
    )
    all_cores = remove_cores_smaller_than_largest(all_cores)
    all_cores = _deduplicate_all_cores(all_cores)

    # --- Translate cores back to original indices ---
    augmented_cores = [
        _translate_core_to_original_indices(core, perm, new_to_old_a, new_to_old_b, swapped_heavy) for core in all_cores
    ]

    # Restore canonical mol_a_full / mol_b_full references after potential swap
    if swapped_heavy:
        mol_a_full, mol_b_full = mol_b_full, mol_a_full
        conf_a_full, conf_b_full = conf_b_full, conf_a_full
        removed_h_a, removed_h_b = removed_h_b, removed_h_a

    # --- Augment with hydrogen mappings ---
    if heavy_matches_heavy_only:
        chiral_set_a_full = None
        chiral_set_b_full = None
        if enforce_chiral:
            chiral_set_a_full = ChiralRestrIdxSet.from_mol(mol_a_full, conf_a_full)
            chiral_set_b_full = ChiralRestrIdxSet.from_mol(mol_b_full, conf_b_full)

        augmented_cores = [
            _augment_core_with_hydrogens(
                mol_a_full,
                mol_b_full,
                core,
                conf_a_full,
                conf_b_full,
                removed_h_a,
                removed_h_b,
                chiral_set_a=chiral_set_a_full,
                chiral_set_b=chiral_set_b_full,
                enforce_chiral=enforce_chiral,
                chain_cutoff=chain_cutoff,
            )
            for core in augmented_cores
        ]
        augmented_cores = remove_cores_smaller_than_largest(augmented_cores)
        augmented_cores = _deduplicate_all_cores(augmented_cores)

    # --- Score and sort ---
    return _score_and_sort_cores(augmented_cores, mol_a_full, mol_b_full, conf_a_full, conf_b_full), mcs_diagnostics


def _handle_single_atom_core(
    perm,
    conf_a,
    conf_b,
    new_to_old_a,
    new_to_old_b,
    swapped,
    heavy_matches_heavy_only,
    mol_a_full,
    mol_b_full,
    conf_a_full,
    conf_b_full,
    removed_h_a,
    removed_h_b,
    enforce_chiral,
    chain_cutoff,
):
    """Handle the edge case where the smaller molecule has only 1 heavy atom.

    Maps the single atom to the nearest atom in mol_b, optionally augments
    with hydrogens.
    """
    dists = np.linalg.norm(conf_b - conf_a[0], axis=1)
    best_b = int(np.argmin(dists))
    orig_a = int(perm[0])
    orig_core = np.array([[new_to_old_a[orig_a], new_to_old_b[best_b]]])
    if swapped:
        orig_core = orig_core[:, ::-1]
        mol_a_full, mol_b_full = mol_b_full, mol_a_full
        conf_a_full, conf_b_full = conf_b_full, conf_a_full
        removed_h_a, removed_h_b = removed_h_b, removed_h_a

    if heavy_matches_heavy_only:
        chiral_set_a_full = None
        chiral_set_b_full = None
        if enforce_chiral:
            chiral_set_a_full = ChiralRestrIdxSet.from_mol(mol_a_full, conf_a_full)
            chiral_set_b_full = ChiralRestrIdxSet.from_mol(mol_b_full, conf_b_full)

        augmented = _augment_core_with_hydrogens(
            mol_a_full,
            mol_b_full,
            orig_core,
            conf_a_full,
            conf_b_full,
            removed_h_a,
            removed_h_b,
            chiral_set_a=chiral_set_a_full,
            chiral_set_b=chiral_set_b_full,
            enforce_chiral=enforce_chiral,
            chain_cutoff=chain_cutoff,
        )
    else:
        augmented = orig_core

    return [augmented], mcgregor.MCSDiagnostics(
        total_nodes_visited=1, total_leaves_visited=1, core_size=len(augmented), num_cores=1
    )


def remove_cores_smaller_than_largest(cores):
    """measured by # mapped atoms"""
    cores_by_size = defaultdict(list)
    for core in cores:
        cores_by_size[len(core)].append(core)

    # Return the largest core(s)
    max_core_size = max(cores_by_size.keys())
    return cores_by_size[max_core_size]


def get_num_dummy_atoms(mol_a, mol_b, core):
    return mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - (2 * len(core))
