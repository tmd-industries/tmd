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

import numpy as np
import pytest

from tmd.fe import constraints as cst
from tmd.lib import custom_ops
from tmd.potentials import HarmonicBond

pytestmark = [pytest.mark.memcheck]


def _build_test_system():
    """A rigid water (O + 2H) and a methane (C + 4H), well separated."""
    masses = np.array(
        [15.9994, 1.008, 1.008, 12.011, 1.008, 1.008, 1.008, 1.008], dtype=np.float64
    )

    r_oh = 0.09572
    theta = np.deg2rad(104.52)
    half = theta / 2.0
    O = np.array([0.0, 0.0, 0.0])
    H1 = r_oh * np.array([np.sin(half), np.cos(half), 0.0])
    H2 = r_oh * np.array([-np.sin(half), np.cos(half), 0.0])

    r_ch = 0.109
    C = np.array([2.0, 2.0, 2.0])
    dirs = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Hs = C + r_ch * dirs

    x0 = np.vstack([O, H1, H2, C, Hs])

    bond_idxs = np.array([[0, 1], [0, 2], [3, 4], [3, 5], [3, 6], [3, 7]], dtype=np.int32)
    bond_params = np.array(
        [
            [1000.0, r_oh],
            [1000.0, r_oh],
            [1000.0, r_ch],
            [1000.0, r_ch],
            [1000.0, r_ch],
            [1000.0, r_ch],
        ]
    )
    angle_idxs = np.array([[1, 0, 2]], dtype=np.int32)
    angle_params = np.array([[100.0, theta]])
    return masses, x0, bond_idxs, bond_params, angle_idxs, angle_params


def _global_constraint_pairs(clusters: cst.ConstraintClusters):
    """Return (pairs, r0) in global atom indices for verification."""
    pairs = []
    r0s = []
    cao = clusters.cluster_atom_offsets
    ca = clusters.cluster_atoms
    cco = clusters.cluster_constraint_offsets
    li = clusters.constraint_local_i
    lj = clusters.constraint_local_j
    r0 = clusters.constraint_r0
    for c in range(clusters.num_clusters):
        astart = cao[c]
        for k in range(cco[c], cco[c + 1]):
            pairs.append((ca[astart + li[k]], ca[astart + lj[k]]))
            r0s.append(r0[k])
    return np.array(pairs, dtype=int), np.array(r0s, dtype=float)


def test_build_constraints_clusters():
    masses, _, bond_idxs, bond_params, angle_idxs, angle_params = _build_test_system()
    clusters = cst.build_constraints(
        bond_idxs, bond_params, masses, angle_idxs, angle_params, rigid_water=True
    )
    # Two clusters: water and methane.
    assert clusters.num_clusters == 2
    # 2 O-H + 1 H-H (rigid water) + 4 C-H = 7 constraints.
    assert clusters.num_constraints == 7
    # The four C-H bonds and two O-H bonds become constraints (the H-H is not a
    # harmonic bond), so all six bonds are removed.
    assert len(clusters.constrained_bond_rows) == 6

    pairs, r0s = _global_constraint_pairs(clusters)
    assert len(pairs) == 7
    # A rigid-water H-H constraint between atoms 1 and 2 must be present.
    has_hh = any(set(p) == {1, 2} for p in pairs)
    assert has_hh


@pytest.mark.parametrize(
    "precision, constraints_cls, integrator_cls, context_cls, dist_atol, rv_atol",
    [
        (
            np.float64,
            custom_ops.Constraints_f64,
            custom_ops.ConstrainedLangevinIntegrator_f64,
            custom_ops.Context_f64,
            1e-7,
            1e-7,
        ),
        (
            np.float32,
            custom_ops.Constraints_f32,
            custom_ops.ConstrainedLangevinIntegrator_f32,
            custom_ops.Context_f32,
            1e-3,
            1e-3,
        ),
    ],
)
def test_constraints_hold_under_dynamics(
    precision, constraints_cls, integrator_cls, context_cls, dist_atol, rv_atol
):
    masses, x0, bond_idxs, bond_params, angle_idxs, angle_params = _build_test_system()
    clusters = cst.build_constraints(
        bond_idxs, bond_params, masses, angle_idxs, angle_params, rigid_water=True
    )
    pairs, targets = _global_constraint_pairs(clusters)

    con = clusters.to_custom_ops(precision=precision)
    assert con.num_clusters() == 2
    assert con.num_constraints() == 7

    # A zero-force harmonic bond between the two heavy atoms gives the Context a
    # potential (required) while leaving the dynamics purely thermostat +
    # constraints.
    hb = HarmonicBond(len(masses), np.array([[0, 3]], dtype=np.int32))
    bp_impl = hb.bind(np.array([[0.0, 1.0]])).to_gpu(precision=precision).bound_impl

    intg = integrator_cls(masses.astype(precision), 300.0, 2e-3, 1.0, 2025, con)

    box = (np.eye(3) * 5.0).astype(precision)
    v0 = np.zeros_like(x0).astype(precision)
    ctxt = context_cls(x0.astype(precision), v0, box, intg, [bp_impl])

    n_steps = 2000
    xs, _ = ctxt.multiple_steps(n_steps, n_steps // 20)
    assert len(xs) > 0

    # Constrained distances must be held across the whole trajectory.
    for frame in xs:
        d = np.linalg.norm(frame[pairs[:, 0]] - frame[pairs[:, 1]], axis=1)
        np.testing.assert_allclose(d, targets, atol=dist_atol)

    # RATTLE: relative velocity along each constraint must vanish.
    v = ctxt.get_v_t()
    final = xs[-1]
    rij = final[pairs[:, 0]] - final[pairs[:, 1]]
    vij = v[pairs[:, 0]] - v[pairs[:, 1]]
    rv = np.sum(rij * vij, axis=1)
    assert np.max(np.abs(rv)) < rv_atol

    # Sanity: the molecules actually moved (constraints are not just freezing
    # everything by zeroing velocities).
    assert not np.allclose(xs[0], xs[-1])


def test_lib_wrapper_runs():
    """The tmd.lib.ConstrainedLangevinIntegrator wrapper drives a Context."""
    from tmd import lib

    masses, x0, bond_idxs, bond_params, angle_idxs, angle_params = _build_test_system()
    clusters = cst.build_constraints(
        bond_idxs, bond_params, masses, angle_idxs, angle_params, rigid_water=True
    )
    pairs, targets = _global_constraint_pairs(clusters)

    precision = np.float64
    intg = lib.ConstrainedLangevinIntegrator(
        temperature=300.0,
        dt=2e-3,
        friction=1.0,
        masses=masses,
        seed=2025,
        constraints=clusters,
    ).impl(precision=precision)

    hb = HarmonicBond(len(masses), np.array([[0, 3]], dtype=np.int32))
    bp_impl = hb.bind(np.array([[0.0, 1.0]])).to_gpu(precision=precision).bound_impl

    box = (np.eye(3) * 5.0).astype(precision)
    v0 = np.zeros_like(x0).astype(precision)
    ctxt = lib.Context(x0, v0, box, intg, [bp_impl], precision=precision)

    xs, _ = ctxt.multiple_steps(500, 50)
    for frame in xs:
        d = np.linalg.norm(frame[pairs[:, 0]] - frame[pairs[:, 1]], axis=1)
        np.testing.assert_allclose(d, targets, atol=1e-7)
