# Copyright 2026, Forrest York
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

"""Tests for the softening of angles involving dummy hydrogens in :class:`SingleTopology`.

When running with constraints, angles that involve a hydrogen which is a dummy atom at one of the
end-states can lead to instabilities (e.g. when a nitrile carbon is mapped onto a methyl carbon).
The force constant of such angles is scaled by ``DUMMY_H_ANGLE_K_SCALE`` in the end-state where the
hydrogen is a dummy, while:

- the fully-interacting end-state of the same angle keeps its full force-field force constant,
- core angles and angles involving dummy (but non-hydrogen) atoms are left untouched,
- angles that belong to chiral volumes being turned on or off are softened as well, and are
  interpolated on the chiral-conversion schedules (``DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX``
  for angles of chiral volumes turning on, ``DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX`` for
  angles of chiral volumes turning off),
- the softened force constant is interpolated back up to full strength by the standard schedules.
"""

import contextlib

import numpy as np
import pytest
from common import ligand_from_smiles

from tmd.fe import single_topology, topology
from tmd.fe.dummy import canonicalize_bond
from tmd.fe.single_topology import (
    DUMMY_A_ANGLE_MIN_MAX,
    DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX,
    DUMMY_B_ANGLE_MIN_MAX,
    DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX,
    DUMMY_H_ANGLE_K_SCALE,
    AtomMapFlags,
    SingleTopology,
)
from tmd.ff import Forcefield

pytestmark = [pytest.mark.nogpu]

FF_FILE = "smirnoff_2_0_0_sc.py"

# propane <-> acetonitrile: the three core atoms are mapped C-C-C(triple-bond-N). The methyl hydrogens of
# propane are dummies at lambda=1 (MOL_A), the methyl hydrogens of acetonitrile are dummies at
# lambda=0 (MOL_B). The all-core angle C0-C1-C2 goes from ~tetrahedral (propane) to linear (nitrile).
PROPANE_ACETONITRILE = ("CCC", "CC#N", [[0, 0], [1, 1], [2, 2]])

# ethane <-> ethanol: the O (a dummy non-hydrogen atom) and its hydrogen (a dummy hydrogen) are
# dummies at lambda=0 (MOL_B), the ethane hydrogens are dummies at lambda=1 (MOL_A).
ETHANE_ETHANOL = ("CC", "CCO", [[0, 0], [1, 1]])

MOL_PAIRS = [PROPANE_ACETONITRILE, ETHANE_ETHANOL]


@contextlib.contextmanager
def patched_dummy_h_angle_k_scale(scale):
    """Temporarily override the module-level DUMMY_H_ANGLE_K_SCALE constant."""
    original = single_topology.DUMMY_H_ANGLE_K_SCALE
    single_topology.DUMMY_H_ANGLE_K_SCALE = scale
    try:
        yield
    finally:
        single_topology.DUMMY_H_ANGLE_K_SCALE = original


def build_single_topology(mol_a, mol_b, core, dummy_h_angle_k_scale=None):
    """Construct a SingleTopology, optionally with a modified DUMMY_H_ANGLE_K_SCALE."""
    ff = Forcefield.load_from_file(FF_FILE)
    if dummy_h_angle_k_scale is None:
        return SingleTopology(mol_a, mol_b, core, ff)
    with patched_dummy_h_angle_k_scale(dummy_h_angle_k_scale):
        return SingleTopology(mol_a, mol_b, core, ff)


def make_mols(smi_a, smi_b, core_rows):
    mol_a = ligand_from_smiles(smi_a, seed=2022)
    mol_b = ligand_from_smiles(smi_b, seed=2022)
    core = np.array(core_rows, dtype=np.int32)
    return mol_a, mol_b, core


def combined_atomic_num(st, c_idx):
    """Atomic number of a combined-topology atom (mol_a's value is used for core atoms)."""
    if st.c_flags[c_idx] == AtomMapFlags.MOL_B:
        return st.mol_b.GetAtomWithIdx(st.c_to_b[c_idx]).GetAtomicNum()
    return st.mol_a.GetAtomWithIdx(st.c_to_a[c_idx]).GetAtomicNum()


def expected_dummy_h_side(st, idxs):
    """Independent (test-side) determination of which endstate an angle has a dummy hydrogen in.

    Returns ``AtomMapFlags.MOL_B`` for angles with a (lone) MOL_B hydrogen (dummy at lambda=0),
    ``AtomMapFlags.MOL_A`` for angles with a (lone) MOL_A hydrogen (dummy at lambda=1), and
    ``None`` otherwise.
    """
    flags = [AtomMapFlags(st.c_flags[int(x)]) for x in idxs]
    if all(flag == AtomMapFlags.CORE for flag in flags):
        return None
    is_hydrogen = [combined_atomic_num(st, int(x)) == 1 for x in idxs]
    if not any(is_hydrogen):
        return None
    has_dummy_b_h = any(flag == AtomMapFlags.MOL_B and h for flag, h in zip(flags, is_hydrogen))
    has_dummy_a_h = any(flag == AtomMapFlags.MOL_A and h for flag, h in zip(flags, is_hydrogen))
    if has_dummy_b_h and not has_dummy_a_h:
        return AtomMapFlags.MOL_B
    elif has_dummy_a_h and not has_dummy_b_h:
        return AtomMapFlags.MOL_A
    return None


def angle_key(idxs, idx_map=None):
    """Canonicalized, hashable key for an angle (idxs given in mol space, mapped through idx_map)."""
    if idx_map is None:
        idx_map = range(len(idxs))
    return tuple(int(x) for x in canonicalize_bond(tuple(int(idx_map[x]) for x in idxs)))


def params_by_angle(params, idxs_arr):
    """Maps each aligned angle idx tuple to its parameter row."""
    return {tuple(int(x) for x in row): np.asarray(params)[i] for i, row in enumerate(idxs_arr)}


def get_ff_angle_params(mol, ff, idx_map=None):
    """Raw (unscaled) harmonic angle force-field parameters keyed by (canonicalized) idxs.

    Parameters
    ----------
    idx_map: array-like, optional
        Mapping from mol idxs to combined-topology idxs (e.g. ``st.b_to_c``). Defaults to the identity.

    Returns
    -------
    dict
        Maps each canonicalized angle idx tuple to its raw force-field parameter array.
    """
    top = topology.BaseTopology(mol, ff)
    angle_params, ha = top.parameterize_harmonic_angle(ff.ha_handle.params)
    if idx_map is None:
        idx_map = np.arange(mol.GetNumAtoms())
    return {angle_key(idxs, idx_map): np.asarray(params) for idxs, params in zip(ha.idxs, angle_params)}


def find_angle_row(idxs, key):
    """Row index of the aligned angle with the (canonicalized) idx tuple ``key``."""
    rows = [i for i, row in enumerate(idxs) if tuple(int(x) for x in row) == key]
    assert len(rows) == 1, f"angle {key} not found exactly once in aligned angle idxs"
    return rows[0]


def expected_interpolated_angle_k(src_k, dst_k, lamb, lamb_min, lamb_max, k_min=0.05):
    """Expected interpolated angle force constant.

    Mirrors the production harmonic-angle interpolation: log-linear in the window
    ``[lamb_min, lamb_max]`` (with force constants clipped below ``k_min``), pinned to the
    end-state values outside of it.
    """
    if lamb <= lamb_min:
        return src_k
    if lamb >= lamb_max:
        return dst_k
    t = (lamb - lamb_min) / (lamb_max - lamb_min)
    return float(np.exp((1 - t) * np.log(max(src_k, k_min)) + t * np.log(max(dst_k, k_min))))


@pytest.mark.parametrize("smi_a,smi_b,core_rows", MOL_PAIRS)
def test_dummy_h_angles_scaled_in_opposite_endstate(smi_a, smi_b, core_rows):
    """Only the k of angles with a dummy hydrogen is scaled, and only in the endstate where the hydrogen is a dummy."""
    mol_a, mol_b, core = make_mols(smi_a, smi_b, core_rows)

    st = build_single_topology(mol_a, mol_b, core)
    st_ref = build_single_topology(mol_a, mol_b, core, dummy_h_angle_k_scale=1.0)

    idxs = np.array(st.aligned_angle.idxs)
    np.testing.assert_array_equal(idxs, np.array(st_ref.aligned_angle.idxs))
    np.testing.assert_allclose(np.array(st.aligned_angle.mins), np.array(st_ref.aligned_angle.mins))
    np.testing.assert_allclose(np.array(st.aligned_angle.maxes), np.array(st_ref.aligned_angle.maxes))

    src_by_angle = params_by_angle(st.aligned_angle.src_params, idxs)
    dst_by_angle = params_by_angle(st.aligned_angle.dst_params, idxs)
    src_ref_by_angle = params_by_angle(st_ref.aligned_angle.src_params, idxs)
    dst_ref_by_angle = params_by_angle(st_ref.aligned_angle.dst_params, idxs)

    num_src_scaled = 0
    num_dst_scaled = 0
    for row in idxs:
        key = tuple(int(x) for x in row)
        side = expected_dummy_h_side(st, row)

        expected_src, expected_dst = src_ref_by_angle[key].copy(), dst_ref_by_angle[key].copy()
        if side == AtomMapFlags.MOL_B:
            # hydrogen is a dummy at lambda=0: only the src k is softened
            expected_src[0] *= DUMMY_H_ANGLE_K_SCALE
            num_src_scaled += 1
        elif side == AtomMapFlags.MOL_A:
            # hydrogen is a dummy at lambda=1: only the dst k is softened
            expected_dst[0] *= DUMMY_H_ANGLE_K_SCALE
            num_dst_scaled += 1

        err_msg = f"angle {key} src params modified: {src_by_angle[key]} != {expected_src}"
        np.testing.assert_allclose(src_by_angle[key], expected_src, err_msg=err_msg)
        err_msg = f"angle {key} dst params modified: {dst_by_angle[key]} != {expected_dst}"
        np.testing.assert_allclose(dst_by_angle[key], expected_dst, err_msg=err_msg)

    # sanity check: the test is not trivially passing with zero scaled angles on either side
    assert num_src_scaled > 0, "expected at least one angle with a dummy (MOL_B) hydrogen"
    assert num_dst_scaled > 0, "expected at least one angle with a dummy (MOL_A) hydrogen"


@pytest.mark.parametrize("smi_a,smi_b,core_rows", MOL_PAIRS)
def test_end_state_systems_are_not_modified(smi_a, smi_b, core_rows):
    """The softening is only applied to the interpolated (aligned) parameters, never to the end-state systems."""
    mol_a, mol_b, core = make_mols(smi_a, smi_b, core_rows)

    st = build_single_topology(mol_a, mol_b, core)
    st_ref = build_single_topology(mol_a, mol_b, core, dummy_h_angle_k_scale=1.0)

    for name in ["src_system", "dst_system"]:
        system = getattr(st, name)
        system_ref = getattr(st_ref, name)
        np.testing.assert_array_equal(
            system.angle.potential.idxs,
            system_ref.angle.potential.idxs,
            err_msg=f"{name} angle idxs changed",
        )
        np.testing.assert_allclose(
            np.array(system.angle.params),
            np.array(system_ref.angle.params),
            err_msg=f"{name} angle params should be unaffected by the dummy hydrogen softening",
        )


def test_lambda_endstates_match_forcefield_params():
    """Check the interpolated end-states against raw force-field parameters.

    Uses propane (mol_a) -> acetonitrile (mol_b):

    - the all-core angle C0-C1-C2 keeps the full mol_a force-field k at lambda=0 and the full mol_b
      force-field k (with a linear, pi, equilibrium angle) at lambda=1;
    - an H-C-C angle of the propane methyl (hydrogens are MOL_A dummies) keeps the full mol_a k at
      lambda=0 and is softened by DUMMY_H_ANGLE_K_SCALE at lambda=1;
    - an H-C-C angle of the acetonitrile methyl (hydrogens are MOL_B dummies) keeps the full mol_b k
      at lambda=1 and is softened by DUMMY_H_ANGLE_K_SCALE at lambda=0.
    """
    smi_a, smi_b, core_rows = PROPANE_ACETONITRILE
    mol_a, mol_b, core = make_mols(smi_a, smi_b, core_rows)
    ff = Forcefield.load_from_file(FF_FILE)

    st = build_single_topology(mol_a, mol_b, core)

    # combined idxs: 0, 1, 2 are the core carbons; 3-10 are mol_a (propane) hydrogens;
    # 11-13 are mol_b (acetonitrile) hydrogens.
    assert combined_atomic_num(st, 3) == 1
    assert combined_atomic_num(st, 8) == 1  # hydrogen on C2 (the methyl that becomes the nitrile)
    assert combined_atomic_num(st, 11) == 1

    idxs = np.array(st.aligned_angle.idxs)
    p0_by_angle = params_by_angle(st.setup_intermediate_state(0.0).angle.params, idxs)
    p1_by_angle = params_by_angle(st.setup_intermediate_state(1.0).angle.params, idxs)

    ff_angles_a = get_ff_angle_params(mol_a, ff, idx_map=st.a_to_c)
    ff_angles_b = get_ff_angle_params(mol_b, ff, idx_map=st.b_to_c)

    # 1) all-core angle: untouched at both ends and matches the raw force-field parameters
    core_angle_key = angle_key((0, 1, 2), st.a_to_c)
    np.testing.assert_allclose(
        p0_by_angle[core_angle_key][0], ff_angles_a[core_angle_key][0], err_msg="core angle src k modified"
    )
    np.testing.assert_allclose(
        p0_by_angle[core_angle_key][1],
        ff_angles_a[core_angle_key][1],
        err_msg="core angle src equilibrium angle modified",
    )
    core_angle_key_b = angle_key((0, 1, 2), st.b_to_c)
    np.testing.assert_allclose(
        p1_by_angle[core_angle_key][0], ff_angles_b[core_angle_key_b][0], err_msg="core angle dst k modified"
    )
    np.testing.assert_allclose(
        p1_by_angle[core_angle_key][1],
        ff_angles_b[core_angle_key_b][1],
        err_msg="core angle dst equilibrium angle modified",
    )
    np.testing.assert_allclose(
        p1_by_angle[core_angle_key][1], np.pi, atol=1e-6, err_msg="nitrile end-state angle should be linear"
    )

    # 2) H8-C2-C1: hydrogen on the methyl that becomes (part of) the nitrile
    h_a_key = angle_key((8, 2, 1), st.a_to_c)
    ff_k_a = ff_angles_a[h_a_key][0]
    np.testing.assert_allclose(
        p0_by_angle[h_a_key][0], ff_k_a, err_msg="MOL_A hydrogen angle k modified at the fully-interacting endstate"
    )
    np.testing.assert_allclose(
        p1_by_angle[h_a_key][0],
        DUMMY_H_ANGLE_K_SCALE * ff_k_a,
        err_msg="MOL_A hydrogen angle k not softened at the dummy endstate",
    )

    # 3) H11-C0-C1: hydrogen on the acetonitrile methyl (MOL_B hydrogen, dummy at lambda=0)
    # note: keys for mol_b angles are built from mol_b idxs mapped through b_to_c (c_to_b[11] is the
    # mol_b idx of the hydrogen at combined idx 11)
    h_b_key = angle_key((st.c_to_b[11], st.c_to_b[0], st.c_to_b[1]), st.b_to_c)
    ff_k_b = ff_angles_b[h_b_key][0]
    np.testing.assert_allclose(
        p1_by_angle[h_b_key][0], ff_k_b, err_msg="MOL_B hydrogen angle k modified at the fully-interacting endstate"
    )
    np.testing.assert_allclose(
        p0_by_angle[h_b_key][0],
        DUMMY_H_ANGLE_K_SCALE * ff_k_b,
        err_msg="MOL_B hydrogen angle k not softened at the dummy endstate",
    )


@pytest.mark.parametrize("smi_a,smi_b,core_rows", MOL_PAIRS)
def test_softened_angles_reach_full_strength_within_schedule(smi_a, smi_b, core_rows):
    """Softened force constants are interpolated back up to full strength by the standard schedules.

    Angles with MOL_B hydrogens follow DUMMY_B_ANGLE_MIN_MAX, so they are pinned to the (softened)
    src value at lambda=0 and must reach the full (unsoftened) dst value at lambda=DUMMY_B_ANGLE_MIN_MAX[1].
    Angles with MOL_A hydrogens follow DUMMY_A_ANGLE_MIN_MAX and behave symmetrically.
    """
    mol_a, mol_b, core = make_mols(smi_a, smi_b, core_rows)

    st = build_single_topology(mol_a, mol_b, core)
    st_ref = build_single_topology(mol_a, mol_b, core, dummy_h_angle_k_scale=1.0)

    idxs = np.array(st.aligned_angle.idxs)
    src_ref_by_angle = params_by_angle(st_ref.aligned_angle.src_params, idxs)
    dst_ref_by_angle = params_by_angle(st_ref.aligned_angle.dst_params, idxs)

    lamb_b = DUMMY_B_ANGLE_MIN_MAX[1]  # lamb at which dummy B angles are pinned to the dst params
    lamb_a = DUMMY_A_ANGLE_MIN_MAX[0]  # lamb at which dummy A angles are pinned to the src params
    p0_by_angle = params_by_angle(st.setup_intermediate_state(0.0).angle.params, idxs)
    p1_by_angle = params_by_angle(st.setup_intermediate_state(1.0).angle.params, idxs)
    pb_by_angle = params_by_angle(st.setup_intermediate_state(lamb_b).angle.params, idxs)
    pa_by_angle = params_by_angle(st.setup_intermediate_state(lamb_a).angle.params, idxs)

    num_b_checked = 0
    num_a_checked = 0
    for row in idxs:
        key = tuple(int(x) for x in row)
        side = expected_dummy_h_side(st, row)
        if side == AtomMapFlags.MOL_B and dst_ref_by_angle[key][0] > 0.0:
            # the dummy endstate k is strictly softened ...
            assert p0_by_angle[key][0] == pytest.approx(DUMMY_H_ANGLE_K_SCALE * src_ref_by_angle[key][0])
            # ... and is back at full strength at the end of the activation schedule
            np.testing.assert_allclose(
                pb_by_angle[key][0],
                dst_ref_by_angle[key][0],
                err_msg=f"angle {key} not restored to full strength at lamb={lamb_b}",
            )
            num_b_checked += 1
        elif side == AtomMapFlags.MOL_A and src_ref_by_angle[key][0] > 0.0:
            assert p1_by_angle[key][0] == pytest.approx(DUMMY_H_ANGLE_K_SCALE * dst_ref_by_angle[key][0])
            np.testing.assert_allclose(
                pa_by_angle[key][0],
                src_ref_by_angle[key][0],
                err_msg=f"angle {key} not restored to full strength at lamb={lamb_a}",
            )
            num_a_checked += 1

    assert num_b_checked > 0
    assert num_a_checked > 0


def test_angles_in_chiral_volumes_turning_on_are_softened_and_interpolated(monkeypatch):
    """Angles belonging to chiral volumes being turned on are softened and interpolated.

    Such angles are not exempt from the dummy hydrogen softening: the (softened) src force
    constant is interpolated up to the full dst force constant on the chiral-conversion
    schedule ``DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX``, i.e. it stays pinned to the
    softened src value until the schedule opens, follows the standard log-linear interpolation
    through the window, and reaches full strength at its upper edge.
    """
    smi_a, smi_b, core_rows = PROPANE_ACETONITRILE
    mol_a, mol_b, core = make_mols(smi_a, smi_b, core_rows)

    st = build_single_topology(mol_a, mol_b, core)
    st_ref = build_single_topology(mol_a, mol_b, core, dummy_h_angle_k_scale=1.0)
    idxs = np.array(st.aligned_angle.idxs)
    src_by_angle = params_by_angle(st.aligned_angle.src_params, idxs)
    src_ref_by_angle = params_by_angle(st_ref.aligned_angle.src_params, idxs)

    # pick angles with a dummy (MOL_B) hydrogen that are actually scaled (i.e. non-zero src k)
    mol_b_h_angles = [
        tuple(int(x) for x in row)
        for row in idxs
        if expected_dummy_h_side(st, row) == AtomMapFlags.MOL_B and src_by_angle[tuple(int(x) for x in row)][0] > 0.0
    ]
    assert len(mol_b_h_angles) >= 2
    target = mol_b_h_angles[0]
    other = mol_b_h_angles[1]

    monkeypatch.setattr(
        SingleTopology, "_angle_idxs_belong_to_chiral_volume_turning_on", lambda self, idxs: tuple(idxs) == target
    )

    st_chiral = build_single_topology(mol_a, mol_b, core)
    chiral_src_by_angle = params_by_angle(st_chiral.aligned_angle.src_params, idxs)
    chiral_dst_by_angle = params_by_angle(st_chiral.aligned_angle.dst_params, idxs)

    # the angle of the chiral volume being turned on is softened just like any other dummy hydrogen angle
    np.testing.assert_allclose(
        chiral_src_by_angle[target][0],
        DUMMY_H_ANGLE_K_SCALE * src_ref_by_angle[target][0],
        err_msg="angle belonging to a chiral volume being turned on should still be softened",
    )
    # and it is interpolated on the chiral-conversion schedule ...
    target_row = find_angle_row(idxs, target)
    np.testing.assert_allclose(
        st_chiral.aligned_angle.mins[target_row],
        DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX[0],
        err_msg="chiral volume angle should follow the converting-on schedule (min)",
    )
    np.testing.assert_allclose(
        st_chiral.aligned_angle.maxes[target_row],
        DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX[1],
        err_msg="chiral volume angle should follow the converting-on schedule (max)",
    )
    # ... while the other dummy hydrogen angles keep the standard schedule (the schedule is per-angle)
    other_row = find_angle_row(idxs, other)
    np.testing.assert_allclose(st_chiral.aligned_angle.mins[other_row], DUMMY_B_ANGLE_MIN_MAX[0])
    np.testing.assert_allclose(st_chiral.aligned_angle.maxes[other_row], DUMMY_B_ANGLE_MIN_MAX[1])

    # the softened force constant is interpolated up to full strength along the chiral schedule:
    # pinned to the softened src value before the window opens, log-linearly interpolated through
    # it, and back at the full dst value at (and beyond) its upper edge
    src_k = float(chiral_src_by_angle[target][0])
    dst_k = float(chiral_dst_by_angle[target][0])
    lamb_min, lamb_max = DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX
    assert src_k < dst_k
    for lamb in [0.0, 0.4, 0.5, 0.6, 0.65, lamb_max, 1.0]:
        p_by_angle = params_by_angle(st_chiral.setup_intermediate_state(lamb).angle.params, idxs)
        err_msg = f"angle {target} k not interpolated correctly at lamb={lamb}"
        np.testing.assert_allclose(
            p_by_angle[target][0],
            expected_interpolated_angle_k(src_k, dst_k, lamb, lamb_min, lamb_max),
            rtol=1e-6,
            atol=1e-9,
            err_msg=err_msg,
        )
    # in particular: pinned at the softened value in the middle of the pinned region, and back at
    # full (unsoftened) strength at the end of the activation schedule
    p_pin = params_by_angle(st_chiral.setup_intermediate_state(0.4).angle.params, idxs)
    assert p_pin[target][0] == pytest.approx(DUMMY_H_ANGLE_K_SCALE * src_ref_by_angle[target][0])
    p_full = params_by_angle(st_chiral.setup_intermediate_state(lamb_max).angle.params, idxs)
    np.testing.assert_allclose(
        p_full[target][0], src_ref_by_angle[target][0], err_msg="angle not restored to full strength"
    )


def test_angles_in_chiral_volumes_turning_off_are_softened_and_interpolated(monkeypatch):
    """Angles belonging to chiral volumes being turned off are softened and interpolated.

    Such angles are not exempt from the dummy hydrogen softening: the (softened) dst force
    constant is interpolated from the full src force constant on the chiral-conversion schedule
    ``DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX``, i.e. full strength is held until the
    schedule opens (at ``1 - DUMMY_B_CHIRAL_ANGLE_CONVERTING_ON_MIN_MAX[1]``), followed by the
    standard log-linear interpolation down to the softened dst value.
    """
    smi_a, smi_b, core_rows = PROPANE_ACETONITRILE
    mol_a, mol_b, core = make_mols(smi_a, smi_b, core_rows)

    st = build_single_topology(mol_a, mol_b, core)
    st_ref = build_single_topology(mol_a, mol_b, core, dummy_h_angle_k_scale=1.0)
    idxs = np.array(st.aligned_angle.idxs)
    dst_by_angle = params_by_angle(st.aligned_angle.dst_params, idxs)
    dst_ref_by_angle = params_by_angle(st_ref.aligned_angle.dst_params, idxs)

    # pick angles with dummy (MOL_A) hydrogens that are actually scaled (i.e. non-zero dst k)
    mol_a_h_angles = [
        tuple(int(x) for x in row)
        for row in idxs
        if expected_dummy_h_side(st, row) == AtomMapFlags.MOL_A and dst_by_angle[tuple(int(x) for x in row)][0] > 0.0
    ]
    assert len(mol_a_h_angles) >= 2
    target = mol_a_h_angles[0]
    other = mol_a_h_angles[1]

    monkeypatch.setattr(
        SingleTopology, "_angle_idxs_belong_to_chiral_volume_turning_off", lambda self, idxs: tuple(idxs) == target
    )

    st_chiral = build_single_topology(mol_a, mol_b, core)
    chiral_src_by_angle = params_by_angle(st_chiral.aligned_angle.src_params, idxs)
    chiral_dst_by_angle = params_by_angle(st_chiral.aligned_angle.dst_params, idxs)

    # the angle of the chiral volume being turned off is softened just like any other dummy hydrogen angle
    np.testing.assert_allclose(
        chiral_dst_by_angle[target][0],
        DUMMY_H_ANGLE_K_SCALE * dst_ref_by_angle[target][0],
        err_msg="angle belonging to a chiral volume being turned off should still be softened",
    )
    # and it is interpolated on the chiral-conversion schedule ...
    target_row = find_angle_row(idxs, target)
    np.testing.assert_allclose(
        st_chiral.aligned_angle.mins[target_row],
        DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX[0],
        err_msg="chiral volume angle should follow the converting-off schedule (min)",
    )
    np.testing.assert_allclose(
        st_chiral.aligned_angle.maxes[target_row],
        DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX[1],
        err_msg="chiral volume angle should follow the converting-off schedule (max)",
    )
    # ... while the other dummy hydrogen angles keep the standard schedule (the schedule is per-angle)
    other_row = find_angle_row(idxs, other)
    np.testing.assert_allclose(st_chiral.aligned_angle.mins[other_row], DUMMY_A_ANGLE_MIN_MAX[0])
    np.testing.assert_allclose(st_chiral.aligned_angle.maxes[other_row], DUMMY_A_ANGLE_MIN_MAX[1])

    # the force constant is interpolated down from full strength along the chiral schedule: full
    # src value while the window is closed, log-linearly interpolated through it, and pinned to
    # the softened dst value at (and beyond) its upper edge
    src_k = float(chiral_src_by_angle[target][0])
    dst_k = float(chiral_dst_by_angle[target][0])
    lamb_min, lamb_max = DUMMY_A_CHIRAL_ANGLE_CONVERTING_OFF_MIN_MAX
    assert src_k > dst_k
    for lamb in [0.0, lamb_min, 0.4, 0.55, 0.7, lamb_max, 1.0]:
        p_by_angle = params_by_angle(st_chiral.setup_intermediate_state(lamb).angle.params, idxs)
        err_msg = f"angle {target} k not interpolated correctly at lamb={lamb}"
        np.testing.assert_allclose(
            p_by_angle[target][0],
            expected_interpolated_angle_k(src_k, dst_k, lamb, lamb_min, lamb_max),
            rtol=1e-6,
            atol=1e-9,
            err_msg=err_msg,
        )
    # in particular: still at full (unsoftened) strength in the middle of the pinned region, and
    # fully softened at the end of the deactivation schedule
    p_full = params_by_angle(st_chiral.setup_intermediate_state(0.2).angle.params, idxs)
    assert p_full[target][0] == pytest.approx(dst_ref_by_angle[target][0])
    p_soft = params_by_angle(st_chiral.setup_intermediate_state(lamb_max).angle.params, idxs)
    np.testing.assert_allclose(
        p_soft[target][0], DUMMY_H_ANGLE_K_SCALE * dst_ref_by_angle[target][0], err_msg="angle not softened at lamb_max"
    )
