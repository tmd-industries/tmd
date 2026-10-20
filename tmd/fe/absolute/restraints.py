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

import itertools
import tempfile
from collections.abc import Sequence

import mdtraj
import numpy as np
import numpy.typing as npt

# TBD: replace these OpenMM derived objects with values from tmd.constants
import openmm
import scipy
from rdkit import Chem
from rdkit.Chem import Mol, PDBWriter

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
_LIGAND_DISTANCE_SD_CUTOFF = 0.03  # nm


def _is_angle_linear(coords: npt.NDArray, idxs: Sequence[int]) -> bool:
    """Check if angle is within 10 kT from 0 or 180.

    Follows the SepTop reference implementation.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the three atoms that form the angle.

    Returns:
        True if the angle is linear, False otherwise.
    """
    # TBD: Handle PBCs correctly
    a, b, c = (coords[:, i, :] for i in idxs)
    u, v = a - b, c - b
    angles = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u * v).sum(axis=-1)))

    angle_avg_rad = np.deg2rad(scipy.stats.circmean(angles, low=-180.0, high=180.0))
    # circvar is dimensionless in scipy >= 1.9; the limits are in degrees^2
    angle_var_deg = scipy.stats.circstd(angles, low=-180.0, high=180.0) ** 2

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
    # TBD: Handle PBCs correctly
    a, b, c, d = (coords[:, i, :] for i in idxs)
    b0, b1, b2 = b - a, c - b, d - c
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    y = (np.cross(n2, n1) * b1).sum(axis=-1) / np.linalg.norm(b1, axis=-1)
    dihedrals = np.rad2deg(np.arctan2(y, (n1 * n2).sum(axis=-1)))

    dihedral_avg = scipy.stats.circmean(dihedrals, low=-180.0, high=180.0)
    dihedral_var = scipy.stats.circstd(dihedrals, low=-180.0, high=180.0) ** 2

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


def _largest_ring_system(mol: Mol, coords: npt.NDArray, center: npt.NDArray) -> list[int]:
    """Heavy atoms of the largest set of rings joined by shared atoms.

    Ties go to the ring system whose centroid is nearest ``center``, so the choice
    does not depend on atom order.
    """
    systems: list[set[int]] = []
    for ring in Chem.GetSymmSSSR(mol):
        ring_atoms = {i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
        for system in [s for s in systems if s & ring_atoms]:
            ring_atoms |= system
            systems.remove(system)
        systems.append(ring_atoms)
    if not systems:
        return []

    def rank(system: set[int]) -> tuple[int, float]:
        return len(system), -float(np.linalg.norm(coords[sorted(system)].mean(axis=0) - center))

    largest: set[int] = max(systems, key=rank)
    return sorted(largest)


def select_ligand_atoms(mol: Mol, xyz: npt.NDArray, rmsf: npt.NDArray) -> list[int]:
    """Select ligand atoms for Boresch restraints.

    Candidates are rigid atoms of the largest ring system, with ties going to the
    one nearest the ligand's center. Substituents joined by
    rotatable bonds can look rigid in a short equilibration, but restraining them
    does little to hold the rest of the ligand.

    L1, the atom bonded to the receptor, is the candidate nearest the ligand's
    center, which minimizes how far restrained rotations move the rest of the
    ligand. L2 and L3 span the largest triangle with L1 among candidates whose
    mutual distances stay fixed, so the restrained frame tracks the whole ring
    system rather than a single ring.

    Args:
        mol: input ligand
        xyz: ligand coordinates with ``shape=(n_frames, n_atoms, 3)`` (nm), in a
            receptor-aligned frame
        rmsf: per-atom RMSF (nm) of the ligand in the same frame

    Returns:
        atom indices [L1, L2, L3]
    """
    heavy = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
    ref = xyz.mean(axis=0)
    center = ref[heavy].mean(axis=0)
    core = np.array(_largest_ring_system(mol, ref, center), dtype=int)
    if len(core) < 3:
        core = heavy
    rigid = core[rmsf[core] < _RMSF_CUTOFF]
    if len(rigid) < 3:
        rigid = core

    l1 = int(rigid[np.argmin(np.linalg.norm(ref[rigid] - center, axis=-1))])
    others = [int(i) for i in rigid if i != l1]

    dist_sd = np.linalg.norm(xyz[:, :, None, :] - xyz[:, None, :, :], axis=-1).std(axis=0)

    def area(l2: int, l3: int) -> float:
        return float(np.linalg.norm(np.cross(ref[l2] - ref[l1], ref[l3] - ref[l1])))

    def l2_angle_offset(l2: int, l3: int) -> float:
        u, v = ref[l1] - ref[l2], ref[l3] - ref[l2]
        return abs(float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v)) - np.pi / 2)

    pairs = sorted(itertools.combinations(others, 2), key=lambda p: area(*p), reverse=True)
    stiff = [p for p in pairs if max(dist_sd[l1, p[0]], dist_sd[l1, p[1]], dist_sd[p]) < _LIGAND_DISTANCE_SD_CUTOFF]
    for a, b in stiff or pairs:
        # order L2/L3 by geometry, not atom index, keeping the L1-L2-L3 angle near 90 degrees
        for l2, l3 in sorted([(a, b), (b, a)], key=lambda p: l2_angle_offset(*p)):
            if not _is_angle_linear(xyz, (l1, l2, l3)):
                return [l1, l2, l3]
    raise ValueError("no suitable ligand atoms could be found")


def _filter_receptor_atoms(
    trj: mdtraj.Trajectory,
    ligand_ref_idx: int,
    rmsf: npt.NDArray,
    minimum_distance_nm: float = 0.4,
    maximum_distance_nm: float = 3.0,
) -> list[int]:
    """Select possible protein atoms for Boresch-style restraints.

    Candidates are rigid (low RMSF) backbone and CB atoms, ordered by distance
    to the first restrained ligand atom. Preferring nearby atoms keeps the lever
    arm of the angle and dihedral restraints short, so the decoupled ligand is
    held close to its bound pose.

    Args:
        trj: The system trajectory.
        ligand_ref_idx: first restrained ligand atom
        rmsf: per-atom RMSF (nm) computed against the caller's chosen reference
            frame, indexed by full-system atom index.
        minimum_distance_nm: Discard protein atoms that are closer than this distance
            to the ligand.
        maximum_distance_nm: Discard protein atoms that are further than this distance
            from the ligand.

    Returns:
        Indices of candidate protein atoms, nearest first.

    Raises:
        ValueError: if no suitable receptor atoms could be found
    """
    candidates = trj.top.select("protein and (backbone or name CB)")
    candidates = candidates[rmsf[candidates] < _RMSF_CUTOFF]
    distances = np.linalg.norm(trj.xyz[-1, candidates] - trj.xyz[-1, ligand_ref_idx], axis=-1)
    mask = (distances > minimum_distance_nm) & (distances <= maximum_distance_nm)
    if not mask.any():
        raise ValueError("no suitable receptor atoms could be found")
    return candidates[mask][np.argsort(distances[mask])].tolist()


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

    * R1,R2 are further apart than 5 Angstroms, but less than half the box size
    * R2,R1,L1,L2 are not collinear
    * R2,R1,L1 angle not 'close' to 0 or 180 degrees
    * R2,R1,L1,L2 dihedral between -150 and 150 degrees
    """
    coords = trj.xyz

    if r1 == r2:
        return False

    if np.linalg.norm(coords[:, r1, :] - coords[:, r2, :], axis=-1).mean() < 0.5:
        return False

    # TBD: Determine if box size means any dimension
    box_dimensions = trj.unitcell_lengths.mean(axis=0)
    if np.linalg.norm(coords[:, r1, :] - coords[:, r2, :], axis=-1).mean() >= 0.5 * box_dimensions.min(axis=-1):
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


def select_receptor_atoms(
    trj: mdtraj.Trajectory,
    ligand_ref_idxs: list[int],
    rmsf: npt.NDArray,
) -> list[int]:
    """Select protein atoms for Boresch-style restraints.

    Follows the validity checks of Baumann et al [1], but takes the nearest valid
    R1 rather than the first in atom order.

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).

    Args:
        trj: The trj containing the receptor and ligands.
        ligand_ref_idxs: The indices of the three ligands atoms that will be restrained.
        rmsf: per-atom RMSF (nm) computed against the caller's chosen reference
            frame, indexed by full-system atom index.

    Returns:
        The indices of the three atoms to use for the restraint

    Raises:
        ValueError: if no suitable receptor atoms could be found
    """
    receptor_idxs = _filter_receptor_atoms(trj, ligand_ref_idxs[0], rmsf)

    l1, l2, l3 = ligand_ref_idxs

    valid_r1_idxs = (r1 for r1 in receptor_idxs if _is_valid_r1(trj, r1, l1, l2, l3))

    valid_r2_pairs = ((r1, r2) for r1 in valid_r1_idxs for r2 in receptor_idxs if _is_valid_r2(trj, r1, r2, l1, l2))
    # chosen to match the SepTop reference implementation at commit 3705ba5
    max_distance = 0.8 * (trj.unitcell_lengths.mean(axis=0).min(axis=-1) / 2)
    for found_r1, found_r2 in valid_r2_pairs:
        valid_r3_idxs = [r3 for r3 in receptor_idxs if _is_valid_r3(trj, found_r1, found_r2, r3, l1)]

        if len(valid_r3_idxs) == 0:
            continue

        # TBD: Determine if it matters that this is different from Fig 3 of https://chemrxiv.org/engage/chemrxiv/article-details/6408d4f9cc600523a3df8c87
        # This checks that all protein atoms are within max distance from L1, P1, P2, but paper only says has to be from P1/P2
        r3_distances_per_frame = []
        for frame in trj.xyz:
            r3_r_distances = scipy.spatial.distance.cdist(frame[valid_r3_idxs, :], frame[[found_r1, found_r2], :])
            r3_l_distances = scipy.spatial.distance.cdist(frame[valid_r3_idxs, :], frame[[ligand_ref_idxs[0]], :])

            r3_distances_per_frame.append(np.hstack([r3_r_distances, r3_l_distances]))

        r3_distances_avg = np.stack(r3_distances_per_frame).mean(axis=0)

        max_distance_mask = r3_distances_avg.max(axis=-1) < max_distance
        r3_distances_avg = r3_distances_avg[max_distance_mask]
        if len(r3_distances_avg) == 0:
            continue

        valid_r3_idxs = np.array(valid_r3_idxs)[max_distance_mask].tolist()

        r3_distances_prod = r3_distances_avg[:, 0] * r3_distances_avg[:, 1]
        found_r3 = valid_r3_idxs[r3_distances_prod.argmax()]

        return [found_r1, found_r2, found_r3]

    raise ValueError("could not find a valid R3 atom within max distance")
