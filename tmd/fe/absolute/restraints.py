# Copyright 2025 Justin Gullingsrud
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

"""Absolute binding FE restraint selection.

Substantial portions of the code in this file are derived from the "femto"
project:

    https://github.com/Psivant/femto/blob/main/femto/fe/reference.py

which is freely available on github under the terms of the MIT License, reprinted
in its entirety below.

MIT License

Copyright (c) 2023 Simon Boothroyd.
Copyright (c) 2023 Psivant Therapeutics.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import itertools
import tempfile
from collections.abc import Sequence

import mdtraj
import networkx as nx
import numpy as np
import numpy.typing as npt

# TBD: replace these OpenMM derived objects with values from tmd.constants
import openmm
import scipy
from rdkit.Chem import Mol, PDBWriter

from tmd.potentials.bonded import kahan_angle, signed_torsion_angle

_COLLINEAR_THRESHOLD = 0.9  # roughly 25 degrees

# values taken from the SepTop reference implementation at commit 7af0b4d
_ANGLE_CHECK_FORCE_CONSTANT = 20.0 * openmm.unit.kilocalorie_per_mole
_ANGLE_CHECK_T = 298.15 * openmm.unit.kelvin
_ANGLE_CHECK_RT = openmm.unit.MOLAR_GAS_CONSTANT_R * _ANGLE_CHECK_T

_ANGLE_CHECK_FACTOR = 0.5 * _ANGLE_CHECK_FORCE_CONSTANT / _ANGLE_CHECK_RT
_ANGLE_CHECK_CUTOFF = 10.0  # units of kT

_ANGLE_CHECK_MAX_VAR = 100.0  # units of degrees^2

_DIHEDRAL_CHECK_CUTOFF = 150.0  # units of degrees
_DIHEDRAL_CHECK_MAX_VAR = 300.0  # units of degrees^2

_RMSF_CUTOFF = 0.1  # nm


def _is_angle_linear(coords: npt.NDArray, idxs: Sequence[int]) -> bool:
    """Check if angle is within 10 kT from 0 or 180.

    Follows the SepTop reference implementation.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the three atoms that form the angle.

    Returns:
        True if the angle is linear, False otherwise.
    """
    angles = []
    for frame in coords:
        # TBD: Handle PBCs correctly
        angle = kahan_angle(*frame[idxs, :], 0.0, np.eye(3) * 100.0)  # type: ignore
        angles.append(np.rad2deg(angle))

    angle_avg_rad = np.deg2rad(scipy.stats.circmean(angles, low=-180.0, high=180.0))
    angle_var_deg = scipy.stats.circvar(angles, low=-180.0, high=180.0)

    check_1 = _ANGLE_CHECK_FACTOR * angle_avg_rad**2
    check_2 = _ANGLE_CHECK_FACTOR * (angle_avg_rad - np.pi) ** 2

    return check_1 < _ANGLE_CHECK_CUTOFF or check_2 < _ANGLE_CHECK_CUTOFF or angle_var_deg > _ANGLE_CHECK_MAX_VAR


def _is_dihedral_trans(coords: npt.NDArray, idxs: Sequence[int]) -> bool:
    """Check if a dihedral angle is within -150 and 150 degrees.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the four atoms that form the dihedral.

    Returns:
        True if the dihedral is planar.
    """
    dihedrals = []
    for frame in coords:
        # TBD: Handle PBCs correctly
        dihedral = signed_torsion_angle(*frame[idxs, :], np.eye(3) * 100.0)  # type: ignore
        dihedrals.append(np.rad2deg(dihedral))

    dihedral_avg = scipy.stats.circmean(dihedrals, low=-180.0, high=180.0)
    dihedral_var = scipy.stats.circvar(dihedrals, low=-180.0, high=180.0)

    return np.abs(dihedral_avg) > _DIHEDRAL_CHECK_CUTOFF or dihedral_var > _DIHEDRAL_CHECK_MAX_VAR


def _are_collinear(coords: npt.NDArray, idxs: Sequence[int]) -> bool:
    """Check whether a sequence of coordinates are collinear.

    Args:
        coords: The full set of coordinates, either with ``shape=(n_coords, 3)`` or
            ``shape=(n_frames, n_coords, 3)``.
        idxs: The sequence of indices of those coordinates to check for collinearity.

    Returns:
        True if any sequential pair of vectors is collinear.
    """
    if coords.ndim == 2:
        coords = coords.reshape(1, *coords.shape)

    idxs = idxs if idxs is not None else list(range(coords.shape[1]))

    for i in range(len(idxs) - 2):
        v_1 = coords[:, idxs[i + 1], :] - coords[:, idxs[i], :]
        v_1 /= np.linalg.norm(v_1, axis=-1, keepdims=True)
        v_2 = coords[:, idxs[i + 2], :] - coords[:, idxs[i + 1], :]
        v_2 /= np.linalg.norm(v_2, axis=-1, keepdims=True)

        if (np.abs((v_1 * v_2).sum(axis=-1)) > _COLLINEAR_THRESHOLD).any():
            return True

    return False


def select_ligand_atoms_baumann(mol: Mol, rmsf: list[float]) -> list[int]:
    """Select ligand atoms for Boresch restraints.

    Args:
        ligand: input ligand
        rmsf: rmsf on ligand atoms

    Returns:
        atom indices

    The selection is probably not invariant to changes in input atom order.

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).
    """
    graph = nx.from_edgelist(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() > 1 and b.GetEndAtom().GetAtomicNum() > 1
    )

    all_paths = [path for _, node_path in nx.all_pairs_shortest_path(graph) for path in node_path.values()]
    path_lengths = {(path[0], path[-1]): len(path) for path in all_paths}
    longest_path = max(all_paths, key=len)

    center_idx = longest_path[len(longest_path) // 2]
    cycles = nx.cycle_basis(graph)
    if len(cycles) >= 1:
        cycles = [cycle for cycle in cycles if rmsf[cycle].max() < _RMSF_CUTOFF]

    if len(cycles) >= 1:
        open_list = [atom_idx for cycle in cycles for atom_idx in cycle]
    else:
        open_list = graph.nodes

    distances = [path_lengths[(center_idx, atom_idx)] for atom_idx in open_list]
    closest_idx = open_list[np.argmin(distances)]

    if len(cycles) >= 1:
        # restrict the list of reference atoms to select from to those that are in the
        # same cycle as the closest atom.
        cycle_idx = next(iter(i for i, cycle in enumerate(cycles) if closest_idx in cycle))
        open_list = cycles[cycle_idx]

        distances = [path_lengths[(closest_idx, atom_idx)] for atom_idx in open_list]

    open_list = [idx for _, idx in sorted(zip(distances, open_list, strict=True)) if idx != closest_idx]
    return [open_list[0], closest_idx, open_list[1]]


def _filter_receptor_atoms(
    trj: mdtraj.Trajectory,
    ligand_ref_idx: int,
    min_helix_size: int = 8,
    min_sheet_size: int = 8,
    skip_residues_start: int = 20,
    skip_residues_end: int = 10,
    minimum_distance_nm=1.0,
    maximum_distance_nm=3.0,
) -> list[int]:
    """Select possible protein atoms for Boresch-style restraints.

    Based on the criteria outlined by Baumann et al.

    Args:
        trj: The system trajectory.
        ligand_ref_idx: first restrained ligand atom
        min_helix_size: The minimum number of residues that have to be in an alpha-helix
            for it to be considered stable.
        min_sheet_size: The minimum number of residues that have to be in a beta-sheet
            for it to be considered stable.
        skip_residues_start: The number of residues to skip at the start of the protein
            as these tend to be more flexible.
        skip_residues_end: The number of residues to skip at the end of the protein
            as these tend to be more flexible
        minimum_distance_nm: Discard protein atoms that are closer than this distance
            to the ligand.
        maximum_distance_nm: Discard protein atoms that are further than this distance
            from the ligand.

    Returns:
        The indices of protein atoms that should be considered for use in Boresch-style
        restraints.

    Raises:
        ValueError: if no suitable receptor atoms could be found
    """
    assert min_helix_size >= 7, "helices must be at least 7 residues long"
    assert min_sheet_size >= 7, "sheets must be at least 7 residues long"

    backbone_idxs = trj.top.select("protein and (backbone or name CB)")
    backbone: mdtraj.Trajectory = trj.atom_slice(backbone_idxs)

    structure = mdtraj.compute_dssp(backbone, simplified=True).tolist()[0]

    # following the SepTop reference implementation we prefer to select from alpha
    # helices if they are dominant in the protein, but otherwise select from sheets
    # as well.
    n_helix_residues = structure.count("H")
    n_sheet_residues = structure.count("E")

    allowed_motifs = ["H"] if n_helix_residues >= n_sheet_residues else ["H", "E"]
    min_motif_size = {"H": min_helix_size, "E": min_sheet_size}

    residues_to_keep: list[str] = []

    structure = structure[skip_residues_start : -(skip_residues_end + 1)]

    for motif, _idxs in itertools.groupby(enumerate(structure), lambda x: x[1]):
        idxs = [(idx + skip_residues_start, motif) for idx, motif in _idxs]

        if motif not in allowed_motifs or len(idxs) < min_motif_size[motif]:
            continue

        # discard the first and last 3 residues of the helix / sheet
        start_idx, end_idx = idxs[0][0] + 3, idxs[-1][0] - 3

        residues_to_keep.extend(f"resid {idx}" for idx in range(start_idx, end_idx + 1))

    rigid_backbone_idxs = backbone.top.select(" ".join(residues_to_keep))

    if len(rigid_backbone_idxs) == 0:
        raise ValueError("no suitable receptor atoms could be found")

    if backbone.n_frames > 1:
        superposed = copy.deepcopy(backbone)
        superposed.superpose(superposed)

        rmsf = mdtraj.rmsf(superposed, superposed, 0)  # nm

        rigid_backbone_idxs = rigid_backbone_idxs[rmsf[rigid_backbone_idxs] < _RMSF_CUTOFF]

    distances = scipy.spatial.distance.cdist(backbone.xyz[0, rigid_backbone_idxs, :], trj.xyz[-1, [ligand_ref_idx], :])

    distance_mask = (distances > minimum_distance_nm).all(axis=1)
    distance_mask &= (distances <= maximum_distance_nm).any(axis=1)

    return backbone_idxs[rigid_backbone_idxs[distance_mask]].tolist()


def _is_valid_r1(trj: mdtraj.Trajectory, r1: int, l1: int, l2: int, l3: int) -> bool:
    """Check whether a given receptor atom would be a valid 'R1' atom.

    Uses the following criteria:

    * L2,L1,R1 angle not 'close' to 0 or 180 degrees
    * L3,L2,L1,R1 dihedral between -150 and 150 degrees
    """
    coords = trj.xyz

    if _are_collinear(coords, (r1, l1, l2, l3)):
        return False

    if _is_angle_linear(coords, (r1, l1, l2)):
        return False

    if _is_dihedral_trans(coords, (r1, l1, l2, l3)):
        return False

    return True


def _is_valid_r2(trj: mdtraj.Trajectory, r1: int, r2: int, l1: int, l2: int) -> bool:
    """Check whether a given receptor atom would be a valid 'R2' atom.

    Uses the following criteria:

    * R1,R2 are further apart than 5 Angstroms
    * R2,R1,L1,L2 are not collinear
    * R2,R1,L1 angle not 'close' to 0 or 180 degrees
    * R2,R1,L1,L2 dihedral between -150 and 150 degrees
    """
    coords = trj.xyz

    if r1 == r2:
        return False

    if np.linalg.norm(coords[:, r1, :] - coords[:, r2, :], axis=-1).mean() < 0.5:
        return False

    if _are_collinear(coords, (r2, r1, l1, l2)):
        return False

    if _is_angle_linear(coords, (r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r2, r1, l1, l2)):
        return False

    return True


def _is_valid_r3(trj: mdtraj.Trajectory, r1: int, r2: int, r3: int, l1: int) -> bool:
    """Check whether a given receptor atom would be a valid 'R3' atom.

    Uses the the following criteria:

    * R1,R2,R3,L1 are not collinear
    * R3,R2,R1,L1 dihedral between -150 and 150 degrees
    """
    coords = trj.xyz

    if len({r1, r2, r3}) != 3:
        return False

    if _are_collinear(coords[[0]], (r3, r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r3, r2, r1, l1)):
        return False

    return True


def rdmol_to_mdtraj(mol: Mol) -> mdtraj.Trajectory:
    """Convert oemol to mdtraj by round-tripping through pdb.

    Args:
        mol: input mol

    Returns:
        single-frame trj

    Bond orders might not be preserved but atom order is the important thing.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        writer = PDBWriter(tmp.name)
        writer.write(mol)
        writer.close()
        return mdtraj.load(tmp.name)


def select_receptor_atoms_baumann(
    trj: mdtraj.Trajectory,
    ligand_ref_idxs: list[int],
) -> list[int]:
    """Select possible protein atoms for Boresch-style restraints.

    Uses the method outlined by Baumann et al [1].

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).

    Args:
        trj: The trj containing the receptor and ligands.
        ligand_ref_idxs: The indices of the three ligands atoms that will be restrained.

    Returns:
        The indices of the three atoms to use for the restraint

    Raises:
        ValueError: if no suitable receptor atoms could be found
    """
    receptor_idxs = _filter_receptor_atoms(trj, ligand_ref_idxs[0])

    l1, l2, l3 = ligand_ref_idxs

    valid_r1_idxs = [r1 for r1 in receptor_idxs if _is_valid_r1(trj, r1, l1, l2, l3)]

    found_r1, found_r2 = next(
        ((r1, r2) for r1 in valid_r1_idxs for r2 in receptor_idxs if _is_valid_r2(trj, r1, r2, l1, l2)),
        (None, None),
    )

    if found_r1 is None or found_r2 is None:
        raise ValueError("could not find valid R1 / R2 atoms")

    valid_r3_idxs = [r3 for r3 in receptor_idxs if _is_valid_r3(trj, found_r1, found_r2, r3, l1)]

    if len(valid_r3_idxs) == 0:
        raise ValueError("could not find a valid R3 atom")

    r3_distances_per_frame = []

    for frame in trj.xyz:
        r3_r_distances = scipy.spatial.distance.cdist(frame[valid_r3_idxs, :], frame[[found_r1, found_r2], :])
        r3_l_distances = scipy.spatial.distance.cdist(frame[valid_r3_idxs, :], frame[[ligand_ref_idxs[0]], :])

        r3_distances_per_frame.append(np.hstack([r3_r_distances, r3_l_distances]))

    # chosen to match the SepTop reference implementation at commit 3705ba5
    max_distance = 0.8 * (trj.unitcell_lengths.mean(axis=0).min(axis=-1) / 2)

    r3_distances_avg = np.stack(r3_distances_per_frame).mean(axis=0)

    max_distance_mask = r3_distances_avg.max(axis=-1) < max_distance
    r3_distances_avg = r3_distances_avg[max_distance_mask]

    valid_r3_idxs = np.array(valid_r3_idxs)[max_distance_mask].tolist()

    r3_distances_prod = r3_distances_avg[:, 0] * r3_distances_avg[:, 1]
    found_r3 = valid_r3_idxs[r3_distances_prod.argmax()]

    return [found_r1, found_r2, found_r3]
