# Copyright 2025 Forrest York
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

"""Construction of SHAKE/RATTLE distance constraints for bonds involving
hydrogens.

The constrained Langevin integrator (``custom_ops.ConstrainedLangevinIntegrator``)
operates on *clusters*: disjoint groups of atoms whose constraints only couple
atoms within the same group. Under the assumption that no hydrogen is shared
between alchemical end states, every hydrogen is bonded to exactly one heavy
atom, so each cluster is simply a heavy atom together with the hydrogens bonded
to it (a "star"), optionally plus a hydrogen-hydrogen constraint that makes water
rigid. Clusters are therefore mutually disjoint and can be solved independently
on the GPU without atomics.

Masses must be passed *before* hydrogen mass repartitioning (HMR); HMR inflates
hydrogen masses to ~3 amu and would defeat mass-based hydrogen detection. Apply
HMR to the masses only after building the constraints, then hand the
repartitioned masses to the integrator.
"""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Keep these in sync with CONSTRAINT_MAX_CLUSTER_ATOMS / _CONSTRAINTS in
# tmd/cpp/src/kernels/k_constraints.cuh.
MAX_CLUSTER_ATOMS = 5
MAX_CLUSTER_CONSTRAINTS = 6

# Mass-based element thresholds (amu).
_H_MASS = 1.007825
_HYDROGEN_MAX_MASS = 1.5
_OXYGEN_MIN_MASS = 15.0
_OXYGEN_MAX_MASS = 17.0

# Default SHAKE/RATTLE solver settings.
DEFAULT_POS_TOL = 1e-8
DEFAULT_VEL_TOL = 1e-8
DEFAULT_MAX_ITERS = 50


@dataclass(frozen=True)
class ConstraintClusters:
    """GPU-ready, CSR-encoded description of distance constraints.

    Attributes
    ----------
    cluster_atom_offsets, cluster_atoms:
        CSR encoding of the atoms belonging to each cluster. ``cluster_atoms``
        holds global atom indices.
    cluster_constraint_offsets, constraint_local_i, constraint_local_j, constraint_r0:
        CSR encoding of the constraints in each cluster. The ``local`` indices
        index into the owning cluster's atom list (i.e. into the slice of
        ``cluster_atoms`` for that cluster).
    constrained_bond_rows:
        Indices into the original ``bond_idxs`` array of the harmonic bonds that
        were turned into constraints; use :func:`remove_constrained_bonds` (or
        these indices directly) to drop them from the harmonic bond potential.
    """

    cluster_atom_offsets: NDArray
    cluster_atoms: NDArray
    cluster_constraint_offsets: NDArray
    constraint_local_i: NDArray
    constraint_local_j: NDArray
    constraint_r0: NDArray
    constrained_bond_rows: NDArray

    @property
    def num_clusters(self) -> int:
        return len(self.cluster_atom_offsets) - 1

    @property
    def num_constraints(self) -> int:
        return len(self.constraint_r0)

    def to_custom_ops(
        self,
        precision=np.float64,
        pos_tol: float = DEFAULT_POS_TOL,
        vel_tol: float = DEFAULT_VEL_TOL,
        max_iters: int = DEFAULT_MAX_ITERS,
    ):
        """Instantiate the corresponding ``custom_ops.Constraints_f{32,64}``."""
        from tmd.lib import custom_ops

        if precision == np.float64:
            klass = custom_ops.Constraints_f64
        elif precision == np.float32:
            klass = custom_ops.Constraints_f32
        else:
            raise ValueError(f"unsupported precision {precision}")

        return klass(
            self.cluster_atom_offsets.astype(np.int32),
            self.cluster_atoms.astype(np.int32),
            self.cluster_constraint_offsets.astype(np.int32),
            self.constraint_local_i.astype(np.int32),
            self.constraint_local_j.astype(np.int32),
            self.constraint_r0.astype(precision),
            pos_tol,
            vel_tol,
            max_iters,
        )


def is_hydrogen(masses: NDArray) -> NDArray:
    """Boolean mask of which atoms are hydrogens, by (pre-HMR) mass."""
    return np.asarray(masses, dtype=np.float64) < _HYDROGEN_MAX_MASS


def _is_oxygen(mass: float) -> bool:
    return _OXYGEN_MIN_MASS <= mass <= _OXYGEN_MAX_MASS


def build_constraints(
    bond_idxs,
    bond_params,
    masses,
    angle_idxs=None,
    angle_params=None,
    rigid_water: bool = True,
) -> ConstraintClusters:
    """Build hydrogen-bond constraint clusters from a system's bonded terms.

    Parameters
    ----------
    bond_idxs:
        ``(B, 2)`` integer array of harmonic bond atom pairs.
    bond_params:
        ``(B, 2)`` array of ``[force_constant, equilibrium_length]`` per bond.
        Equilibrium lengths become the constraint distances.
    masses:
        Per-atom masses *before* hydrogen mass repartitioning, used to identify
        hydrogens and water oxygens.
    angle_idxs, angle_params:
        Optional ``(A, 3)`` harmonic angle atom triples (center atom in the
        middle column) and ``[force_constant, equilibrium_angle, ...]`` params.
        Required when ``rigid_water`` is True and the system contains water, to
        derive the rigid H-H distance from the equilibrium H-O-H angle.
    rigid_water:
        If True, add a hydrogen-hydrogen distance constraint to each water so
        that the H-O-H angle (and thus the whole molecule) is rigid.

    Returns
    -------
    ConstraintClusters
    """
    bond_idxs = np.asarray(bond_idxs, dtype=np.int64).reshape(-1, 2)
    bond_params = np.asarray(bond_params, dtype=np.float64).reshape(len(bond_idxs), -1)
    masses = np.asarray(masses, dtype=np.float64)
    is_h = is_hydrogen(masses)

    # Adjacency over all bonds (used for water detection).
    neighbors: dict[int, list[int]] = defaultdict(list)
    for i, j in bond_idxs:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    # Clusters keyed by their central heavy atom. Each hydrogen is bonded to
    # exactly one heavy atom, so this keying is unambiguous.
    clusters: "OrderedDict[int, dict]" = OrderedDict()

    def get_cluster(heavy: int) -> dict:
        if heavy not in clusters:
            clusters[heavy] = {"atoms": [heavy], "atom_set": {heavy}, "cons": []}
        return clusters[heavy]

    constrained_bond_rows: list[int] = []

    for b, (i, j) in enumerate(bond_idxs):
        i, j = int(i), int(j)
        hi, hj = bool(is_h[i]), bool(is_h[j])
        if hi == hj:
            # H-H or heavy-heavy bond: not a hydrogen-stretch constraint.
            continue
        heavy, h = (j, i) if hi else (i, j)
        r0 = float(bond_params[b, 1])
        cluster = get_cluster(heavy)
        if h not in cluster["atom_set"]:
            cluster["atoms"].append(h)
            cluster["atom_set"].add(h)
        cluster["cons"].append((heavy, h, r0))
        constrained_bond_rows.append(b)

    if rigid_water:
        angle_lookup: dict[tuple[int, int, int], float] = {}
        if angle_idxs is not None:
            angle_idxs_arr = np.asarray(angle_idxs, dtype=np.int64).reshape(-1, 3)
            angle_params_arr = np.asarray(angle_params, dtype=np.float64).reshape(
                len(angle_idxs_arr), -1
            )
            for a, (p, q, r) in enumerate(angle_idxs_arr):
                p, q, r = int(p), int(q), int(r)
                angle_lookup[(min(p, r), q, max(p, r))] = float(angle_params_arr[a, 1])

        for heavy, cluster in clusters.items():
            nbrs = neighbors[heavy]
            if len(nbrs) != 2 or not all(is_h[n] for n in nbrs):
                continue
            if not _is_oxygen(masses[heavy]):
                continue
            h1, h2 = int(nbrs[0]), int(nbrs[1])
            key = (min(h1, h2), heavy, max(h1, h2))
            if key not in angle_lookup:
                raise ValueError(
                    "rigid_water=True requires H-O-H angle parameters for water "
                    f"oxygen {heavy}; pass angle_idxs and angle_params"
                )
            theta0 = angle_lookup[key]
            oh_lengths = {h: r for (_, h, r) in cluster["cons"]}
            r1 = oh_lengths[h1]
            r2 = oh_lengths[h2]
            d_hh = float(np.sqrt(r1 * r1 + r2 * r2 - 2.0 * r1 * r2 * np.cos(theta0)))
            cluster["cons"].append((h1, h2, d_hh))

    # Flatten clusters into CSR arrays with cluster-local constraint indices.
    cluster_atom_offsets = [0]
    cluster_atoms: list[int] = []
    cluster_constraint_offsets = [0]
    constraint_local_i: list[int] = []
    constraint_local_j: list[int] = []
    constraint_r0: list[float] = []

    for cluster in clusters.values():
        atoms = cluster["atoms"]
        cons = cluster["cons"]
        if len(atoms) > MAX_CLUSTER_ATOMS:
            raise ValueError(
                f"cluster has {len(atoms)} atoms, exceeding MAX_CLUSTER_ATOMS="
                f"{MAX_CLUSTER_ATOMS}"
            )
        if len(cons) > MAX_CLUSTER_CONSTRAINTS:
            raise ValueError(
                f"cluster has {len(cons)} constraints, exceeding "
                f"MAX_CLUSTER_CONSTRAINTS={MAX_CLUSTER_CONSTRAINTS}"
            )
        local = {g: k for k, g in enumerate(atoms)}
        cluster_atoms.extend(atoms)
        cluster_atom_offsets.append(len(cluster_atoms))
        for a, b, r0 in cons:
            constraint_local_i.append(local[a])
            constraint_local_j.append(local[b])
            constraint_r0.append(r0)
        cluster_constraint_offsets.append(len(constraint_r0))

    return ConstraintClusters(
        cluster_atom_offsets=np.array(cluster_atom_offsets, dtype=np.int32),
        cluster_atoms=np.array(cluster_atoms, dtype=np.int32),
        cluster_constraint_offsets=np.array(cluster_constraint_offsets, dtype=np.int32),
        constraint_local_i=np.array(constraint_local_i, dtype=np.int32),
        constraint_local_j=np.array(constraint_local_j, dtype=np.int32),
        constraint_r0=np.array(constraint_r0, dtype=np.float64),
        constrained_bond_rows=np.array(constrained_bond_rows, dtype=np.int64),
    )


def remove_constrained_bonds(bond_idxs, bond_params, constrained_bond_rows):
    """Drop the constrained bonds from a harmonic bond potential.

    Returns ``(bond_idxs, bond_params)`` with the rows in ``constrained_bond_rows``
    removed, matching the standard practice of replacing the harmonic stretch
    term with a rigid constraint.
    """
    bond_idxs = np.asarray(bond_idxs)
    bond_params = np.asarray(bond_params)
    keep = np.ones(len(bond_idxs), dtype=bool)
    keep[np.asarray(constrained_bond_rows, dtype=np.int64)] = False
    return bond_idxs[keep], bond_params[keep]
