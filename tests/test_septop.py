# (C) 2026 Justin Gullingsrud
"""Tests for the SepTop free-energy library.

Covers, without a GPU:

* ``prepare_host_edge`` potential structure and the opposite lambda
  coupling of the two ligands on a water box;
* Boresch restraint addition/scaling, central-atom selection, and the
  solvent-leg zero-length bond;
* anchor selection and full restraint setup on a real protein + 2-ligand
  host (using a synthetic noise-perturbed trajectory, no MD);
* the ``DualTopology`` mol_a-mol_b nonbonded exclusions.

The single GPU test exercises complex-leg end-state minimization at the
alchemical endpoints, reporting which atom group carries residual large
forces on failure.
"""

import os

import numpy as np
import pytest

import tmd.testsystems.fep_benchmark as fep_benchmark
from tmd.constants import DEFAULT_PROTEIN_FF, DEFAULT_TEMP, DEFAULT_WATER_FF, NBParamIdx
from tmd.fe import utils
from tmd.fe.free_energy import Trajectory
from tmd.fe.septop import (
    RestraintParams,
    SepTopAnchors,
    SepTopFreeEnergy,
    get_septop_initial_state,
    select_central_atoms,
    select_septop_anchors,
)
from tmd.fe.stored_arrays import StoredArrays
from tmd.ff import Forcefield
from tmd.md import builders
from tmd.md.minimizer import MAX_FORCE_NORM
from tmd.potentials import (
    HarmonicAngle,
    HarmonicBond,
    Nonbonded,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)

# ---- fixtures ----


@pytest.fixture(scope="module")
def hif2a_pair_in_water():
    """Two hif2a ligands in a small water box (no protein), without restraints."""
    src = os.path.dirname(fep_benchmark.__file__) + "/hif2a/ligands.sdf"
    mols = utils.read_sdf(src)
    mol_a = mols[0]
    mol_b = mols[1]
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    host_config = builders.build_water_system(4.0, ff.water_ff, mols=[mol_a, mol_b])
    return mol_a, mol_b, ff, host_config


@pytest.fixture(scope="module")
def hif2a_complex():
    """Build a hif2a protein + 2 ligands host."""
    pdb = os.path.join(os.path.dirname(fep_benchmark.__file__), "hif2a", "5tbm_solv_equil.pdb")
    sdf = os.path.join(os.path.dirname(fep_benchmark.__file__), "hif2a", "ligands.sdf")
    mols = utils.read_sdf(sdf)
    mol_a, mol_b = mols[0], mols[1]
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    host_config = builders.build_protein_system(pdb, DEFAULT_PROTEIN_FF, DEFAULT_WATER_FF, mols=[mol_a, mol_b])
    return mol_a, mol_b, ff, host_config


def _make_synthetic_trajectory(host_config, mol_a, mol_b, n_frames: int = 10, noise: float = 1e-3) -> Trajectory:
    """Build a fake joint ``[host, mol_a, mol_b]`` ``Trajectory`` from the
    equilibrium pose."""
    rng = np.random.default_rng(0)
    coords = np.concatenate([host_config.conf, utils.get_romol_conf(mol_a), utils.get_romol_conf(mol_b)])
    frames = [coords + rng.normal(scale=noise, size=coords.shape) for _ in range(n_frames)]
    boxes = [host_config.box for _ in range(n_frames)]
    return Trajectory(
        frames=StoredArrays.from_chunks([frames]),
        boxes=boxes,
        final_velocities=None,
        final_barostat_volume_scale_factor=1.0,
    )


@pytest.fixture(scope="module")
def septop_potentials(hif2a_complex):
    """SepTopFreeEnergy potentials at lambda=0.5 (both mols partially coupled)."""
    mol_a, mol_b, _, host_config = hif2a_complex
    ff = Forcefield.load_default()
    afe = SepTopFreeEnergy(mol_a, mol_b, ff)
    ubps, params, masses = afe.prepare_host_edge(ff, host_config, 0.5)
    return mol_a, mol_b, host_config, ubps, params, masses


# ---- water-box unit tests ----


@pytest.mark.nogpu
def test_septop_prepare_host_edge_no_restraints(hif2a_pair_in_water):
    """Without anchors, prepare_host_edge produces standard ABFE potentials."""
    mol_a, mol_b, ff, host_config = hif2a_pair_in_water
    afe = SepTopFreeEnergy(mol_a, mol_b, ff)
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()
    n_total = n_host + n_a + n_b

    for lamb in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ubps, _params, masses = afe.prepare_host_edge(ff, host_config, lamb)
        assert len(masses) == n_total

        kinds = {type(p) for p in ubps}
        assert kinds == {
            HarmonicBond,
            HarmonicAngle,
            PeriodicTorsion,
            Nonbonded,
            NonbondedPairListPrecomputed,
        }


@pytest.mark.nogpu
def test_septop_lambda_drives_two_ligands_oppositely(hif2a_pair_in_water):
    """At lambda=0, mol_a is fully coupled and mol_b decoupled; reversed at 1."""
    mol_a, mol_b, ff, host_config = hif2a_pair_in_water
    afe = SepTopFreeEnergy(mol_a, mol_b, ff)

    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()
    sl_a = slice(n_host, n_host + n_a)
    sl_b = slice(n_host + n_a, n_host + n_a + n_b)

    def get_nb(lamb):
        ubps, params, _ = afe.prepare_host_edge(ff, host_config, lamb)
        nb = next(p for pot, p in zip(ubps, params) if isinstance(pot, Nonbonded))
        return np.asarray(nb)

    nb0 = get_nb(0.0)
    # mol_a fully coupled -> charges intact, W=0
    assert np.any(nb0[sl_a, NBParamIdx.Q_IDX] != 0.0)
    np.testing.assert_array_equal(nb0[sl_a, NBParamIdx.W_IDX], 0.0)
    # mol_b decoupled -> charges zero (past decharge_interval=(0.25, 0.75)), W>0
    np.testing.assert_array_equal(nb0[sl_b, NBParamIdx.Q_IDX], 0.0)
    assert np.all(nb0[sl_b, NBParamIdx.W_IDX] > 0)

    nb1 = get_nb(1.0)
    np.testing.assert_array_equal(nb1[sl_a, NBParamIdx.Q_IDX], 0.0)
    assert np.all(nb1[sl_a, NBParamIdx.W_IDX] > 0)
    assert np.any(nb1[sl_b, NBParamIdx.Q_IDX] != 0.0)
    np.testing.assert_array_equal(nb1[sl_b, NBParamIdx.W_IDX], 0.0)


@pytest.mark.nogpu
def test_septop_endpoint_symmetry(hif2a_pair_in_water):
    """NB params for mol_a at lamb=l match mol_b at lamb=1-l (mod per-mol params)."""
    mol_a, mol_b, ff, host_config = hif2a_pair_in_water
    afe = SepTopFreeEnergy(mol_a, mol_b, ff)
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()

    def get_nb(lamb):
        ubps, params, _ = afe.prepare_host_edge(ff, host_config, lamb)
        nb = next(p for pot, p in zip(ubps, params) if isinstance(pot, Nonbonded))
        return np.asarray(nb)

    # the W coordinate for mol_a at lambda=l must match mol_b at lambda=1-l
    for lamb in [0.1, 0.3, 0.7]:
        a_at_l = get_nb(lamb)[n_host : n_host + n_a, NBParamIdx.W_IDX]
        # at the corresponding mirror lambda, mol_b's W slice should match a's
        b_at_1ml = get_nb(1.0 - lamb)[n_host + n_a : n_host + n_a + n_b, NBParamIdx.W_IDX]
        # all atoms in a single ligand share the same W shift, so just compare scalar
        assert a_at_l.shape == (n_a,)
        assert b_at_1ml.shape == (n_b,)
        np.testing.assert_allclose(a_at_l[0], b_at_1ml[0], rtol=1e-6)


@pytest.mark.nogpu
def test_septop_restraints_added_when_anchors_set(hif2a_pair_in_water):
    """Restraints add extra HarmonicBond/HarmonicAngle/PeriodicTorsion potentials."""
    mol_a, mol_b, ff, host_config = hif2a_pair_in_water
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    mol_b.GetNumAtoms()
    a_off = n_host
    b_off = n_host + n_a

    # Pick arbitrary atoms; we're only checking the structure, not physics.
    anchors = SepTopAnchors(
        rec_atoms=[0, 1, 2],
        lig_atoms_a=[a_off + 0, a_off + 1, a_off + 2],
        lig_atoms_b=[b_off + 0, b_off + 1, b_off + 2],
    )

    # We need x0/box0 for restraint geometry math.
    x0 = np.concatenate([host_config.conf, utils.get_romol_conf(mol_a), utils.get_romol_conf(mol_b)])
    box0 = host_config.box

    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        anchors=anchors,
        x0=x0,
        box0=box0,
        rst_params=RestraintParams(),
    )

    # Without restraints (no anchors): count baseline potentials
    afe_no = SepTopFreeEnergy(mol_a, mol_b, ff)
    base_ubps, _, _ = afe_no.prepare_host_edge(ff, host_config, 0.5)
    base_counts = dict.fromkeys((HarmonicBond, HarmonicAngle, PeriodicTorsion), 0)
    for p in base_ubps:
        if type(p) in base_counts:
            base_counts[type(p)] += 1

    ubps, _, _ = afe.prepare_host_edge(ff, host_config, 0.5)
    counts = dict.fromkeys(base_counts, 0)
    for p in ubps:
        if type(p) in counts:
            counts[type(p)] += 1

    # SepTop adds 1 restraint of each type per ligand -> +2 of each
    assert counts[HarmonicBond] == base_counts[HarmonicBond] + 2
    assert counts[HarmonicAngle] == base_counts[HarmonicAngle] + 2
    assert counts[PeriodicTorsion] == base_counts[PeriodicTorsion] + 2


@pytest.mark.nogpu
def test_septop_restraint_scaling_endpoints(hif2a_pair_in_water):
    """At lambda=0 ligand A's restraint scale=0; at lambda=1 ligand A's scale=1.

    The bond restraint stores [fc, r0]; fc = kb * scale, so fc=0 means scale=0.
    """
    mol_a, mol_b, ff, host_config = hif2a_pair_in_water
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    a_off = n_host
    b_off = n_host + n_a

    anchors = SepTopAnchors(
        rec_atoms=[0, 1, 2],
        lig_atoms_a=[a_off + 0, a_off + 1, a_off + 2],
        lig_atoms_b=[b_off + 0, b_off + 1, b_off + 2],
    )
    x0 = np.concatenate([host_config.conf, utils.get_romol_conf(mol_a), utils.get_romol_conf(mol_b)])
    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        anchors=anchors,
        x0=x0,
        box0=host_config.box,
        rst_params=RestraintParams(),
    )

    def collect_bond_restraints(lamb):
        ubps, params, _ = afe.prepare_host_edge(ff, host_config, lamb)
        # The two ligand restraints are appended after baseline bonded
        # potentials. Identify them by single-pair idxs of length 1.
        out = []
        for pot, p in zip(ubps, params):
            if isinstance(pot, HarmonicBond) and len(pot.idxs) == 1:
                out.append((pot, np.asarray(p)))
        return out

    # at lamb=0: ligand A scale=0, ligand B scale=1
    bonds_lo = collect_bond_restraints(0.0)
    assert len(bonds_lo) == 2
    fc_a_lo, fc_b_lo = bonds_lo[0][1][0, 0], bonds_lo[1][1][0, 0]
    assert fc_a_lo == 0.0
    assert fc_b_lo > 0.0

    # at lamb=1: ligand A scale=1, ligand B scale=0
    bonds_hi = collect_bond_restraints(1.0)
    assert len(bonds_hi) == 2
    fc_a_hi, fc_b_hi = bonds_hi[0][1][0, 0], bonds_hi[1][1][0, 0]
    assert fc_a_hi > 0.0
    assert fc_b_hi == 0.0


@pytest.mark.nogpu
def test_select_central_atoms_in_range(hif2a_pair_in_water):
    """Central atoms land within each ligand's slice of the combined system."""
    mol_a, mol_b, _ff, host_config = hif2a_pair_in_water
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()

    central_a, central_b = select_central_atoms(host_config, mol_a, mol_b)
    assert n_host <= central_a < n_host + n_a
    assert n_host + n_a <= central_b < n_host + n_a + n_b


@pytest.mark.nogpu
def test_septop_solvent_leg_adds_one_zero_length_bond(hif2a_pair_in_water):
    """The solvent leg adds exactly one extra zero-length inter-ligand bond."""
    mol_a, mol_b, ff, host_config = hif2a_pair_in_water
    x0 = np.concatenate([host_config.conf, utils.get_romol_conf(mol_a), utils.get_romol_conf(mol_b)])
    central_atoms = select_central_atoms(host_config, mol_a, mol_b)
    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        x0=x0,
        box0=host_config.box,
        rst_params=RestraintParams(),
        central_atoms=central_atoms,
    )
    afe_no = SepTopFreeEnergy(mol_a, mol_b, ff)

    def count_single_pair_bonds(a):
        ubps, params, _ = a.prepare_host_edge(ff, host_config, 0.5)
        out = []
        for pot, p in zip(ubps, params):
            if isinstance(pot, HarmonicBond) and len(pot.idxs) == 1:
                out.append((pot, np.asarray(p)))
        return out

    base = count_single_pair_bonds(afe_no)
    with_bond = count_single_pair_bonds(afe)
    assert len(with_bond) == len(base) + 1

    # The extra bond is between the two central atoms with r0=0, fc=kb,
    # held constant across lambda.
    extra_pot, extra_params = with_bond[-1]
    np.testing.assert_array_equal(extra_pot.idxs[0], np.array(central_atoms))
    assert extra_params[0, 0] == RestraintParams().kb
    assert extra_params[0, 1] == 0.0


# ---- real protein + 2-ligand host tests ----


@pytest.mark.nogpu
def test_select_septop_anchors_real_complex(hif2a_complex):
    """Pick anchors using a synthetic joint dual-ligand trajectory."""
    mol_a, mol_b, _, host_config = hif2a_complex
    trj = _make_synthetic_trajectory(host_config, mol_a, mol_b)

    anchors = select_septop_anchors(host_config, mol_a, mol_b, trj)

    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()

    assert len(anchors.rec_atoms) == 3
    assert len(anchors.lig_atoms_a) == 3
    assert len(anchors.lig_atoms_b) == 3
    # rec atoms inside the host range
    for r in anchors.rec_atoms:
        assert 0 <= r < n_host
    # ligand A atoms in [n_host, n_host + n_a)
    for la in anchors.lig_atoms_a:
        assert n_host <= la < n_host + n_a
    # ligand B atoms in [n_host + n_a, n_host + n_a + n_b)
    for lb in anchors.lig_atoms_b:
        assert n_host + n_a <= lb < n_host + n_a + n_b


@pytest.mark.nogpu
def test_septop_full_prepare_host_edge_with_real_complex(hif2a_complex):
    """Build a fully restrained SepTopFreeEnergy and call prepare_host_edge."""
    mol_a, mol_b, ff, host_config = hif2a_complex
    trj = _make_synthetic_trajectory(host_config, mol_a, mol_b)
    anchors = select_septop_anchors(host_config, mol_a, mol_b, trj)

    # Joint x0 for restraint geometry: host + mol_a + mol_b.
    x0 = np.concatenate([host_config.conf, utils.get_romol_conf(mol_a), utils.get_romol_conf(mol_b)])
    box0 = host_config.box
    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        anchors=anchors,
        x0=x0,
        box0=box0,
        rst_params=RestraintParams(),
    )

    for lamb in [0.0, 0.5, 1.0]:
        ubps, _params, masses = afe.prepare_host_edge(ff, host_config, lamb)
        assert len(masses) == len(host_config.conf) + mol_a.GetNumAtoms() + mol_b.GetNumAtoms()
        kinds = {type(p) for p in ubps}
        # At minimum we expect bonds, angles, torsions, and Nonbonded to appear.
        for kind in (HarmonicBond, HarmonicAngle, PeriodicTorsion, Nonbonded):
            assert kind in kinds

    # Sanity: get_restraint_correction should return finite values.
    corr_a = afe.get_restraint_correction(anchors.lig_atoms_a, temperature=300.0)
    corr_b = afe.get_restraint_correction(anchors.lig_atoms_b, temperature=300.0)
    assert np.isfinite(corr_a)
    assert np.isfinite(corr_b)


@pytest.mark.nogpu
def test_mol_a_mol_b_nonbonded_excluded(septop_potentials):
    """All mol_a-mol_b NB pairs must be excluded.

    SepTop puts both ligands in the binding pocket simultaneously.
    DualTopology is supposed to exclude all guest-guest nonbonded
    interactions (via ``exclude_all_ligand_ligand_ixns``). Verify this
    holds for our ``SepTopFreeEnergy.prepare_host_edge`` output and is
    not silently dropped along the way.
    """
    mol_a, mol_b, host_config, ubps, _, _ = septop_potentials
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()
    a_lo, a_hi = n_host, n_host + n_a
    b_lo, b_hi = n_host + n_a, n_host + n_a + n_b

    nb = next(p for p in ubps if isinstance(p, Nonbonded))

    def _is_cross(i, j):
        return (a_lo <= i < a_hi and b_lo <= j < b_hi) or (b_lo <= i < b_hi and a_lo <= j < a_hi)

    cross_excl = sum(1 for i, j in nb.exclusion_idxs if _is_cross(i, j))
    assert cross_excl == n_a * n_b, (
        f"Nonbonded must exclude all {n_a * n_b} mol_a-mol_b pairs; only {cross_excl} are present."
    )

    pairlist = next(p for p in ubps if isinstance(p, NonbondedPairListPrecomputed))
    cross_pairs = sum(1 for i, j in pairlist.idxs if _is_cross(i, j))
    assert cross_pairs == 0, (
        f"NonbondedPairListPrecomputed must not include cross mol_a-mol_b pairs; got {cross_pairs}."
    )


# ---- GPU end-state preparation ----


@pytest.mark.parametrize("lamb", [0.0, 1.0])
def test_septop_endstate_minimizes(hif2a_complex, lamb):
    """``optimize_abfe_initial_state`` must succeed at SepTop endpoints.

    Reproduces the joint-frame assembly + per-window minimization that
    the complex leg performs via bisection's ``make_optimized``.
    Uses a synthetic joint dual-ligand trajectory so we don't pay for the
    pre-anchor equilibration; the failure mode this test guards against is
    in the joint-frame assembly and the endpoint minimization, not the eq
    MD itself.
    """
    from tmd.fe.absolute.abfe import optimize_abfe_initial_state
    from tmd.fe.rbfe import setup_optimized_host
    from tmd.fe.utils import get_romol_conf

    mol_a, mol_b, _, host_config = hif2a_complex
    ff = Forcefield.load_default()

    host_config = setup_optimized_host(host_config, [mol_a, mol_b], ff, equilibration_steps=200)

    def _joint_trj(mol_a, mol_b):
        rng = np.random.default_rng(0)
        coords = np.concatenate([host_config.conf, get_romol_conf(mol_a), get_romol_conf(mol_b)])
        frames = [coords + rng.normal(scale=1e-3, size=coords.shape) for _ in range(5)]
        boxes = [host_config.box for _ in range(5)]
        return Trajectory(
            frames=StoredArrays.from_chunks([frames]),
            boxes=boxes,
            final_velocities=None,
            final_barostat_volume_scale_factor=1.0,
        )

    trj = _joint_trj(mol_a, mol_b)
    anchors = select_septop_anchors(host_config, mol_a, mol_b, trj)

    n_host = len(host_config.conf)
    pos = trj.frames[-1]
    box = host_config.box

    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        anchors=anchors,
        x0=pos,
        box0=box,
        rst_params=RestraintParams(),
    )
    state = get_septop_initial_state(afe, ff, host_config, host_config.conf, DEFAULT_TEMP, seed=2026, lamb=lamb)

    try:
        optimize_abfe_initial_state(state)
    except Exception as exc:
        # Re-evaluate forces at x0 to surface where the residual large
        # forces are.
        from tmd.md.minimizer import get_val_and_grad_fn

        val_and_grad = get_val_and_grad_fn(state.potentials, state.box0)
        _, forces = val_and_grad(state.x0)
        force_norms = np.linalg.norm(forces, axis=-1)
        bad = np.where(force_norms > MAX_FORCE_NORM)[0]
        n_a = mol_a.GetNumAtoms()
        n_protein = n_host - host_config.num_water_atoms - host_config.num_membrane_atoms
        groups = {"protein": 0, "water/ion": 0, "mol_a": 0, "mol_b": 0}
        for i in bad:
            if i < n_protein:
                groups["protein"] += 1
            elif i < n_host:
                groups["water/ion"] += 1
            elif i < n_host + n_a:
                groups["mol_a"] += 1
            else:
                groups["mol_b"] += 1
        msg = (
            f"\nlambda={lamb}: minimization raised {type(exc).__name__}: {exc}\n"
            f"atoms with |F| > {MAX_FORCE_NORM}: {len(bad)} / {len(force_norms)}\n"
            f"  by group: {groups}\n"
        )
        pytest.fail(msg)
