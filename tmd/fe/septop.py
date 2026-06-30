# (C) 2026 Justin Gullingsrud

"""Separated topologies (SepTop) relative binding free energy.

Computes a relative binding free energy between two ligands ``mol_a`` and
``mol_b`` via two legs that share a single alchemical schedule. Along one
lambda coordinate, ``mol_a`` is decoupled as ``lambda`` goes 0 -> 1 while
``mol_b`` is simultaneously coupled (it follows ``1 - lambda``).

Both legs are driven by :func:`estimate_septop` via its ``phase`` argument:

* ``phase="complex"`` -- the two ligands share a single solvated receptor and
  each is held in place by Boresch-style restraints that turn on/off in
  opposite directions along lambda.
* ``phase="aqueous"`` -- the two ligands share a single water box (no
  receptor). Instead of receptor restraints, one near-central atom is chosen
  in each ligand (see :func:`select_central_atoms`) and a single constant
  zero-length harmonic bond is applied between them. The symmetric bond
  cancels between endpoints, so the solvent leg needs no standard-state
  correction.

The relative binding free energy is then
``ddG = dG_complex - dG_solvent - (corr_A - corr_B)``, where the per-ligand
restraint corrections ``corr_A``/``corr_B`` come from the complex leg.

References
----------
Rocklin, Mobley and Dill, "Separated topologies -- a method for relative
binding free energy calculations using orientational restraints",
J. Chem. Phys. 138, 085104 (2013).
https://pubmed.ncbi.nlm.nih.gov/23464180/

Notes
-----
This module deliberately re-implements pieces of
``tmd.fe.free_energy.AbsoluteFreeEnergy.prepare_host_edge`` and
``tmd.fe.absolute.abfe.get_initial_state`` so that we can apply the per-atom
lambda transforms (decharge, epsilon scale, 4D W shift) to the two ligands
independently. The TODOs flag what should eventually move upstream into tmd.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import mdtraj
import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from tmd.constants import DEFAULT_PRESSURE, DEFAULT_TEMP, NBParamIdx
from tmd.fe import model_utils
from tmd.fe.absolute.abfe import (
    optimize_abfe_initial_state,
    sample_for_restraints,
)
from tmd.fe.absolute.free_energy import RestraintParams
from tmd.fe.absolute.restraints import (
    select_ligand_atoms_baumann,
    select_receptor_atoms_baumann,
)
from tmd.fe.cif_writer import build_openmm_topology
from tmd.fe.free_energy import (
    AbsoluteFreeEnergy,
    HREXSimulationResult,
    InitialState,
    MDParams,
    SimulationResult,
    Trajectory,
    make_pair_bar_plots,
    run_sims_bisection,
)
from tmd.fe.interpolate import linear_interpolation, pad
from tmd.fe.lambda_schedule import (
    bisection_lambda_schedule,
)
from tmd.fe.rbfe import (
    HostConfig,
    estimate_relative_free_energy_bisection_hrex_impl,
    optimize_coordinates,
    optimize_initial_state_from_pre_optimized,
    setup_optimized_host,
)
from tmd.fe.topology import DualTopology
from tmd.fe.utils import set_romol_conf
from tmd.ff import Forcefield
from tmd.lib import LangevinIntegrator, MonteCarloBarostat
from tmd.md.barostat.utils import get_bond_list, get_group_indices
from tmd.md.thermostat.utils import sample_velocities
from tmd.potentials import HarmonicAngle, HarmonicBond, Nonbonded, PeriodicTorsion
from tmd.potentials.potential import get_potential_by_type

__all__ = (
    "RestraintParams",
    "SepTopAnchors",
    "SepTopFreeEnergy",
    "SepTopResult",
    "estimate_septop",
    "get_septop_initial_state",
    "select_central_atoms",
    "select_septop_anchors",
)

# Default schedule for the short equilibration used to select Boresch anchors.
# This run only needs to relax the bound pose and yield ligand RMSF, so it is
# much cheaper than the production schedule. Applied as overrides onto the
# production MDParams when no explicit ``eq_md_params`` is supplied.
DEFAULT_EQ_N_EQ_STEPS = 10000
DEFAULT_EQ_N_FRAMES = 100


@dataclass(frozen=True)
class SepTopAnchors:
    """Restraint atom indices for a SepTop calculation.

    Indices are into the combined ``[host, mol_a, mol_b]`` coordinate array.
    """

    rec_atoms: list[int]
    lig_atoms_a: list[int]
    lig_atoms_b: list[int]


@dataclass
class SepTopResult:
    """Result of a complex- or solvent-leg SepTop calculation."""

    sim_result: SimulationResult | HREXSimulationResult
    anchors: SepTopAnchors | None
    correction_a: float
    correction_b: float

    @property
    def correction(self) -> float:
        """Net restraint correction in kJ/mol.

        ``ddG_complex_corrected = ddG_complex_raw - (correction_a - correction_b)``
        """
        return self.correction_a - self.correction_b


def select_septop_anchors(
    host_config: HostConfig,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    trj: Trajectory,
) -> SepTopAnchors:
    """Pick Boresch anchor atoms for a two-ligand complex.

    Both ligands' anchors and the shared receptor anchors are picked from a
    single joint ``[host, mol_a, mol_b]`` equilibration trajectory in which both
    ligands are fully coupled (but do not interact with each other). Each
    ligand's three anchor atoms come from the Baumann RMSF-based selector
    applied to its own atom slice; the three receptor anchors are picked once
    using ligand A's atoms (both ligands bind the same pocket, so the chosen
    receptor atoms are usable for either).

    Parameters
    ----------
    host_config : HostConfig
        Host configuration providing the OpenMM topology.
    mol_a, mol_b : rdkit Mol
        Ligands. Their atom indices in the returned :class:`SepTopAnchors`
        are relative to the joint ``[host, mol_a, mol_b]`` ordering.
    trj : Trajectory
        Joint dual-ligand equilibration of ``[host, mol_a, mol_b]``
        (frames of size ``n_host + n_a + n_b``).

    Returns
    -------
    SepTopAnchors
        Anchors with indices into the joint ``[host, mol_a, mol_b]`` system.
    """
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    n_b = mol_b.GetNumAtoms()
    assert trj.frames[0].shape[0] == n_host + n_a + n_b, (
        f"trj has {trj.frames[0].shape[0]} atoms; expected {n_host + n_a + n_b}"
    )

    boxes = np.array(trj.boxes)
    omm = build_openmm_topology([host_config.omm_topology, mol_a, mol_b])
    top = mdtraj.Topology.from_openmm(omm)
    angles = [[90.0] * 3] * len(boxes)
    lengths = [np.diag(b) for b in boxes]
    full = mdtraj.Trajectory(trj.frames, top, unitcell_lengths=lengths, unitcell_angles=angles)
    full.image_molecules(inplace=True)
    ref = full[-1]
    full.superpose(ref, atom_indices=full.topology.select("backbone"))
    rmsf = mdtraj.rmsf(full, ref)

    lig_ids_a_local = select_ligand_atoms_baumann(mol_a, rmsf[n_host : n_host + n_a])
    lig_ids_b_local = select_ligand_atoms_baumann(mol_b, rmsf[n_host + n_a : n_host + n_a + n_b])

    # Receptor selection: use ligand A's atom indices in the joint frame; the
    # chosen receptor atoms apply to both ligands since they share the pocket.
    rec_ids = select_receptor_atoms_baumann(full, [i + n_host for i in lig_ids_a_local])

    return SepTopAnchors(
        rec_atoms=list(rec_ids),
        lig_atoms_a=[i + n_host for i in lig_ids_a_local],
        lig_atoms_b=[i + n_host + n_a for i in lig_ids_b_local],
    )


def select_central_atoms(
    host_config: HostConfig,
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
) -> tuple[int, int]:
    """Pick one central atom from each ligand for the solvent leg bond.

    For each ligand, choose the atom closest to its geometric center. The
    returned indices are into the combined ``[host, mol_a, mol_b]`` ordering
    so they can be used directly as ``HarmonicBond`` atom indices.
    """
    from tmd.fe.utils import get_romol_conf

    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()

    def _central_local(mol: Chem.Mol) -> int:
        conf = get_romol_conf(mol)
        center = conf.mean(axis=0)
        return int(np.argmin(np.linalg.norm(conf - center, axis=1)))

    central_a = n_host + _central_local(mol_a)
    central_b = n_host + n_a + _central_local(mol_b)
    return central_a, central_b


def _apply_lambda_transform_to_slice(
    nb_params: jnp.ndarray,
    slice_start: int,
    slice_end: int,
    lamb: float,
    decharge_interval: tuple[float, float],
    eps_scale_interval: tuple[float, float],
    cutoff: float,
    w_interval: tuple[float, float] = (0.0, 1.0),
) -> jnp.ndarray:
    """Apply the AbsoluteFreeEnergy lambda transform to a slice of NB params.

    Mirrors the per-ligand math in
    :py:meth:`tmd.fe.free_energy.AbsoluteFreeEnergy.prepare_host_edge`.

    TODO: upstream into tmd as a public helper so SepTop can share the
    implementation rather than duplicating it.
    """
    if lamb <= 0.0:
        return nb_params
    sl = slice(slice_start, slice_end)

    # Linearly decharge the slice
    nb_params = nb_params.at[sl, NBParamIdx.Q_IDX].set(
        pad(
            linear_interpolation,
            nb_params[sl, NBParamIdx.Q_IDX],
            jnp.zeros_like(nb_params[sl, NBParamIdx.Q_IDX]),
            lamb,
            decharge_interval[0],
            decharge_interval[1],
        )
    )
    # Scale down the epsilon term over ``eps_scale_interval``.
    dst_eps = nb_params[sl, NBParamIdx.LJ_EPS_IDX] / 3
    nb_params = nb_params.at[sl, NBParamIdx.LJ_EPS_IDX].set(
        pad(
            linear_interpolation,
            nb_params[sl, NBParamIdx.LJ_EPS_IDX],
            jnp.where(dst_eps > 0.02, dst_eps, 0.02),
            lamb,
            eps_scale_interval[0],
            eps_scale_interval[1],
        )
    )
    # Shift the W coordinate to improve efficiency. ``w_interval`` confines the
    # 4D decoupling to a sub-range of lambda: W stays 0 below ``w_interval[0]``,
    # ramps linearly to ``cutoff`` across the interval, and stays ``cutoff``
    # above ``w_interval[1]``. Confining it after the decharge interval realizes
    # a sequential "decharge in place, then lift the neutral core" schedule.
    w_start, w_end = w_interval
    if lamb <= w_start:
        w_lambda = 0.0
    elif lamb >= w_end:
        w_lambda = 1.0
    else:
        w_lambda = (lamb - w_start) / (w_end - w_start)
    nb_params = nb_params.at[sl, NBParamIdx.W_IDX].set(linear_interpolation(0.0, cutoff, w_lambda))
    return nb_params


class SepTopFreeEnergy(AbsoluteFreeEnergy):
    """Two-ligand absolute-difference binding free energy.

    Wraps a :class:`tmd.fe.topology.DualTopology` so that both ligands sit in
    the simulation simultaneously. ``prepare_host_edge`` couples ``mol_a`` on
    ``lambda`` and ``mol_b`` on ``1 - lambda``, and (when ``anchors`` is set)
    appends Boresch restraints for both ligands with appropriately staged
    on/off schedules.
    """

    def __init__(
        self,
        mol_a: Chem.Mol,
        mol_b: Chem.Mol,
        ff: Forcefield,
        anchors: SepTopAnchors | None = None,
        x0: NDArray | None = None,
        box0: NDArray | None = None,
        rst_params: RestraintParams | None = None,
        central_atoms: tuple[int, int] | None = None,
        decharge_interval: tuple[float, float] = (0.0, 0.2),
        eps_scale_interval: tuple[float, float] = (0.2, 0.4),
        w_interval: tuple[float, float] = (0.0, 1.0),
    ):
        # ``mol`` exists on the parent class; treat A as the canonical
        # AbsoluteFreeEnergy.mol so existing helpers (e.g. those that read
        # ``afe.mol``) operate on ligand A.
        top = DualTopology(mol_a, mol_b, ff)
        super().__init__(
            mol_a,
            top,
            decharge_interval=decharge_interval,
            eps_scale_interval=eps_scale_interval,
        )
        self.mol_a = mol_a
        self.mol_b = mol_b
        self.ff = ff
        self.anchors = anchors
        self.x0 = x0
        self.box0 = box0
        self.rst_params = rst_params
        self.central_atoms = central_atoms
        self.w_interval = w_interval

    # ---- coordinates ----
    def prepare_combined_coords(self, host_coords: NDArray | None = None) -> NDArray:
        """Concatenate ``[host, mol_a, mol_b]`` coordinates."""
        from tmd.fe.utils import get_romol_conf

        a_coords = get_romol_conf(self.mol_a)
        b_coords = get_romol_conf(self.mol_b)
        ligand_coords = np.concatenate([a_coords, b_coords])
        if host_coords is None:
            return ligand_coords
        return np.concatenate([host_coords, ligand_coords])

    # ---- atom-index helpers ----
    def _slices(self, host_config: HostConfig) -> tuple[slice, slice]:
        """Return the (mol_a, mol_b) atom slices in the combined system."""
        n_host = len(host_config.conf)
        n_a = self.mol_a.GetNumAtoms()
        n_b = self.mol_b.GetNumAtoms()
        return slice(n_host, n_host + n_a), slice(n_host + n_a, n_host + n_a + n_b)

    # ---- alchemical schedule ----
    def prepare_host_edge(self, ff: Forcefield, host_config: HostConfig, lamb: float) -> tuple[tuple, tuple, NDArray]:  # type: ignore[type-arg]
        """Build the host/dual-guest system at a particular lambda.

        ``mol_a`` follows ``lambda`` (coupled at 0, decoupled at 1); ``mol_b``
        follows ``1 - lambda`` (decoupled at 0, coupled at 1). Boresch
        restraints, if configured via ``self.anchors`` and ``self.rst_params``,
        are appended.
        """
        from tmd.fe import topology
        from tmd.fe.utils import get_mol_masses

        ff_params = ff.get_params()
        hgt = topology.HostGuestTopology(
            host_config.host_system.get_U_fns(),
            self.top,
            host_config.num_water_atoms,
            ff,
            host_config.omm_topology,
        )

        # Build with lamb=0 (no W-shift baked in) so we can drive the two
        # ligands independently below.
        combined_params, combined_potentials = self._get_system_params_and_potentials(ff_params, hgt, 0.0)
        combined_params = list(combined_params)

        # Locate the Nonbonded potential and apply per-ligand lambda transforms.
        nb_idx = next(i for i, pot in enumerate(combined_potentials) if isinstance(pot, Nonbonded))
        nb_params = combined_params[nb_idx]
        sl_a, sl_b = self._slices(host_config)
        cutoff = hgt.host_nonbonded.potential.cutoff
        nb_params = _apply_lambda_transform_to_slice(
            nb_params,
            sl_a.start,
            sl_a.stop,
            lamb,
            self.decharge_interval,
            self.eps_scale_interval,
            cutoff,
            self.w_interval,
        )
        nb_params = _apply_lambda_transform_to_slice(
            nb_params,
            sl_b.start,
            sl_b.stop,
            1.0 - lamb,
            self.decharge_interval,
            self.eps_scale_interval,
            cutoff,
            self.w_interval,
        )
        combined_params[nb_idx] = nb_params

        ligand_masses_a = get_mol_masses(self.mol_a)
        ligand_masses_b = get_mol_masses(self.mol_b)
        combined_masses = np.concatenate([np.array(host_config.masses), ligand_masses_a, ligand_masses_b])

        ubps = list(combined_potentials)
        params = list(combined_params)

        if self.anchors is not None and self.rst_params is not None:
            # Restraint scaling: each ligand follows the same single-ligand ABFE
            # schedule, applied to its own effective lambda.
            on = self.rst_params.on
            scale_a = min(1.0, lamb / on) if lamb > 0 else 0.0
            scale_b = min(1.0, (1.0 - lamb) / on) if lamb < 1 else 0.0
            for atoms_lig, scale in (
                (self.anchors.lig_atoms_a, scale_a),
                (self.anchors.lig_atoms_b, scale_b),
            ):
                hb, hbp = self._get_bond_restraint(host_config, atoms_lig, scale)
                ha, hap = self._get_angle_restraint(host_config, atoms_lig, scale)
                dr, drp = self._get_dihedral_restraint(host_config, atoms_lig, scale)
                ubps.extend([hb, ha, dr])
                params.extend([hbp, hap, drp])
        elif self.central_atoms is not None and self.rst_params is not None:
            # Solvent leg: tether the two ligands together with a single
            # zero-length harmonic bond between their central atoms, held at
            # constant full strength across the whole lambda schedule. The
            # bond is identical at both endpoints, so it cancels in the
            # solvent-leg free energy and needs no analytical correction.
            hb, hbp = self._get_inter_ligand_bond(host_config)
            ubps.append(hb)
            params.append(hbp)

        return tuple(ubps), tuple(params), combined_masses

    def prepare_equilibration_edge(self, ff: Forcefield, host_config: HostConfig) -> tuple[tuple, tuple, NDArray]:  # type: ignore[type-arg]
        """Build the joint system with both ligands fully coupled.

        Both ``mol_a`` and ``mol_b`` interact fully with the host and solvent but
        not with each other (the dual topology excludes all inter-ligand
        nonbonded interactions). No lambda transform and no restraints are
        applied. Used to equilibrate the bound complex prior to anchor
        selection: keeping both ligands fully coupled holds each in the pocket
        (a decoupled ligand would feel no host forces and drift away), so a
        single joint trajectory yields realistic RMSF for both ligands.
        """
        from tmd.fe import topology
        from tmd.fe.utils import get_mol_masses

        ff_params = ff.get_params()
        hgt = topology.HostGuestTopology(
            host_config.host_system.get_U_fns(),
            self.top,
            host_config.num_water_atoms,
            ff,
            host_config.omm_topology,
        )
        combined_params, combined_potentials = self._get_system_params_and_potentials(ff_params, hgt, 0.0)
        ligand_masses_a = get_mol_masses(self.mol_a)
        ligand_masses_b = get_mol_masses(self.mol_b)
        combined_masses = np.concatenate([np.array(host_config.masses), ligand_masses_a, ligand_masses_b])
        return tuple(combined_potentials), tuple(combined_params), combined_masses

    # ---- restraint helpers (per-ligand) ----
    #
    # We avoid constructing a full AbsoluteBindingFreeEnergy for each ligand
    # because that class assumes a single combined coordinate array spanning
    # one ligand only. Instead we replicate just the geometry math here using
    # the combined ``self.x0`` / ``self.box0``.
    def _bond_geometry(self, lig_atoms: list[int]) -> tuple[list[int], float]:
        from tmd.potentials.jax_utils import delta_r

        assert self.anchors is not None and self.x0 is not None
        i0 = [self.anchors.rec_atoms[0], lig_atoms[0]]
        a0, b0 = self.x0[i0]
        r0 = float(np.linalg.norm(delta_r(a0, b0, self.box0)))
        return i0, r0

    def _angle_geometry(self, lig_atoms: list[int]) -> list[tuple[list[int], float]]:
        from tmd.potentials.bonded import kahan_angle

        assert self.anchors is not None and self.x0 is not None
        rec = self.anchors.rec_atoms
        i0 = [rec[1], rec[0], lig_atoms[0]]
        i1 = [rec[0], lig_atoms[0], lig_atoms[1]]
        ai0, aj0, ak0 = self.x0[i0]
        ai1, aj1, ak1 = self.x0[i1]
        t0 = kahan_angle(ai0, aj0, ak0, 0.0, self.box0)
        t1 = kahan_angle(ai1, aj1, ak1, 0.0, self.box0)
        return [(i0, t0), (i1, t1)]

    def _dihedral_geometry(self, lig_atoms: list[int]) -> list[tuple[list[int], float]]:
        from tmd.potentials.bonded import signed_torsion_angle

        assert self.anchors is not None and self.x0 is not None
        rec = self.anchors.rec_atoms
        i0 = [rec[2], rec[1], rec[0], lig_atoms[0]]
        i1 = [rec[1], rec[0], lig_atoms[0], lig_atoms[1]]
        i2 = [rec[0], lig_atoms[0], lig_atoms[1], lig_atoms[2]]
        ci0, cj0, ck0, cl0 = self.x0[i0]
        ci1, cj1, ck1, cl1 = self.x0[i1]
        ci2, cj2, ck2, cl2 = self.x0[i2]
        p0 = signed_torsion_angle(ci0, cj0, ck0, cl0, self.box0)
        p1 = signed_torsion_angle(ci1, cj1, ck1, cl1, self.box0)
        p2 = signed_torsion_angle(ci2, cj2, ck2, cl2, self.box0)
        return [(i0, p0), (i1, p1), (i2, p2)]

    def _get_bond_restraint(
        self, host_config: HostConfig, lig_atoms: list[int], scale: float
    ) -> tuple[HarmonicBond, NDArray]:
        assert self.rst_params is not None
        i0, r0 = self._bond_geometry(lig_atoms)
        fc = self.rst_params.kb * scale
        n_total = len(host_config.conf) + self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()
        idxs = np.array([i0], dtype=np.int32)
        params = np.array([[fc, r0]], dtype=np.float32)
        return HarmonicBond(n_total, idxs), params

    def _get_inter_ligand_bond(self, host_config: HostConfig) -> tuple[HarmonicBond, NDArray]:
        """Zero-length bond between the two ligands' central atoms.

        Used in the solvent leg in place of the per-ligand Boresch
        restraints: a single harmonic bond at constant full strength
        (``rst_params.kb``) keeps the two ligands co-located so the dual
        topology stays well-defined as each ligand is (de)coupled.
        """
        assert self.rst_params is not None and self.central_atoms is not None
        n_total = len(host_config.conf) + self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()
        idxs = np.array([list(self.central_atoms)], dtype=np.int32)
        params = np.array([[self.rst_params.kb, 0.0]], dtype=np.float32)
        return HarmonicBond(n_total, idxs), params

    def _get_angle_restraint(
        self, host_config: HostConfig, lig_atoms: list[int], scale: float
    ) -> tuple[HarmonicAngle, NDArray]:
        assert self.rst_params is not None
        ((i0, t0), (i1, t1)) = self._angle_geometry(lig_atoms)
        fc = self.rst_params.ka * scale
        n_total = len(host_config.conf) + self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()
        idxs = np.array([i0, i1], dtype=np.int32)
        params = np.array([[fc, t0, 0], [fc, t1, 0]], dtype=np.float32)
        return HarmonicAngle(n_total, idxs), params

    def _get_dihedral_restraint(
        self, host_config: HostConfig, lig_atoms: list[int], scale: float
    ) -> tuple[PeriodicTorsion, NDArray]:
        assert self.rst_params is not None
        idxs_ = []
        params_ = []
        k = self.rst_params.kd * scale
        const = 0.0
        last_ids: list[int] = []
        for ids, phi in self._dihedral_geometry(lig_atoms):
            for period in range(1, 6 + 1):
                phase = period * phi
                fc = -k / period
                params_.append([fc, phase, period])
                idxs_.append(ids)
                const += fc
            last_ids = ids
        params_.append([-const, 0, 0])
        idxs_.append(last_ids)
        n_total = len(host_config.conf) + self.mol_a.GetNumAtoms() + self.mol_b.GetNumAtoms()
        idxs = np.array(idxs_, dtype=np.int32)
        params = np.array(params_, dtype=np.float32)
        return PeriodicTorsion(n_total, idxs), params

    def get_restraint_correction(self, lig_atoms: list[int], temperature: float) -> float:
        """Single-ligand Boresch correction (kJ/mol) using ``lig_atoms``.

        Uses the same formula as
        :meth:`AbsoluteBindingFreeEnergy.get_restraint_correction`.
        """
        from tmd.constants import BOLTZ

        assert self.rst_params is not None
        _, r0 = self._bond_geometry(lig_atoms)
        ((_, t0), (_, t1)) = self._angle_geometry(lig_atoms)
        s0 = np.sin(t0)
        s1 = np.sin(t1)
        V0 = 1660 * (0.1**3)
        kT = BOLTZ * temperature
        kb = self.rst_params.kb
        ka = self.rst_params.ka
        kd = self.rst_params.kd * sum(range(1, 7))
        return -kT * np.log(8 * np.pi**2 * V0 * (kb * ka**2 * kd**3) ** 0.5 / (r0**2 * s0 * s1 * (2 * np.pi * kT) ** 3))


# ---- initial-state construction ----


def get_septop_initial_state(
    afe: SepTopFreeEnergy,
    ff: Forcefield,
    host_config: HostConfig,
    host_conf: NDArray,
    temperature: float,
    seed: int,
    lamb: float,
) -> InitialState:
    """Build an InitialState for a SepTop window at ``lamb``.

    Mirrors :func:`tmd.fe.absolute.abfe.get_initial_state` but tracks two
    ligand index ranges and chooses ``interacting_atoms`` based on which
    ligand is fully coupled at the endpoint (A at lamb=0, B at lamb=1).
    """
    from tmd.potentials import HarmonicBond as _HarmonicBond

    ubps, params, masses = afe.prepare_host_edge(ff, host_config, lamb)
    x0 = afe.prepare_combined_coords(host_coords=host_conf)
    bps = [ubp.bind(p) for ubp, p in zip(ubps, params)]

    bond_potential = get_potential_by_type(ubps, _HarmonicBond)
    hmr_masses = model_utils.apply_hmr(masses, bond_potential.idxs)
    group_idxs = get_group_indices(get_bond_list(bond_potential), len(masses))
    baro = MonteCarloBarostat(len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, 25, seed)
    box0 = host_config.box

    v0 = sample_velocities(hmr_masses, temperature, seed)

    n_host = len(host_config.conf)
    n_a = afe.mol_a.GetNumAtoms()
    n_b = afe.mol_b.GetNumAtoms()
    lig_a_idxs = np.arange(n_host, n_host + n_a, dtype=np.int32)
    lig_b_idxs = np.arange(n_host + n_a, n_host + n_a + n_b, dtype=np.int32)
    ligand_idxs = np.concatenate([lig_a_idxs, lig_b_idxs])

    dt = 2.5e-3
    friction = 1.0
    intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)
    protein_idxs = np.arange(
        n_host - host_config.num_water_atoms - host_config.num_membrane_atoms,
        dtype=np.int32,
    )

    # interacting_atoms: at endpoints, exactly one ligand is fully coupled
    # (W==0). At interior lambdas both ligands have nonzero W shifts, so
    # leave it None to disable the optimization.
    if lamb == 0.0:
        interacting_atoms = lig_a_idxs
    elif lamb == 1.0:
        interacting_atoms = lig_b_idxs
    else:
        interacting_atoms = None

    return InitialState(
        bps,
        intg,
        baro,
        x0,
        v0,
        box0,
        lamb,
        ligand_idxs,
        protein_idxs,
        interacting_atoms=interacting_atoms,
    )


def get_septop_equilibration_state(
    afe: SepTopFreeEnergy,
    ff: Forcefield,
    host_config: HostConfig,
    host_conf: NDArray,
    temperature: float,
    seed: int,
) -> InitialState:
    """Build an InitialState with both ligands fully coupled for equilibration.

    Mirrors :func:`get_septop_initial_state` but uses
    :meth:`SepTopFreeEnergy.prepare_equilibration_edge` so both ligands are
    fully coupled (and mutually non-interacting) with no restraints. Used to
    equilibrate the bound complex before anchor selection.
    """
    from tmd.potentials import HarmonicBond as _HarmonicBond

    ubps, params, masses = afe.prepare_equilibration_edge(ff, host_config)
    x0 = afe.prepare_combined_coords(host_coords=host_conf)
    bps = [ubp.bind(p) for ubp, p in zip(ubps, params)]

    bond_potential = get_potential_by_type(ubps, _HarmonicBond)
    hmr_masses = model_utils.apply_hmr(masses, bond_potential.idxs)
    group_idxs = get_group_indices(get_bond_list(bond_potential), len(masses))
    baro = MonteCarloBarostat(len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, 25, seed)
    box0 = host_config.box

    v0 = sample_velocities(hmr_masses, temperature, seed)

    n_host = len(host_config.conf)
    n_a = afe.mol_a.GetNumAtoms()
    n_b = afe.mol_b.GetNumAtoms()
    lig_a_idxs = np.arange(n_host, n_host + n_a, dtype=np.int32)
    lig_b_idxs = np.arange(n_host + n_a, n_host + n_a + n_b, dtype=np.int32)
    ligand_idxs = np.concatenate([lig_a_idxs, lig_b_idxs])

    dt = 2.5e-3
    friction = 1.0
    intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, seed)
    protein_idxs = np.arange(
        n_host - host_config.num_water_atoms - host_config.num_membrane_atoms,
        dtype=np.int32,
    )

    # Both ligands are fully coupled (W==0), so both are valid interacting atoms.
    return InitialState(
        bps,
        intg,
        baro,
        x0,
        v0,
        box0,
        0.0,
        ligand_idxs,
        protein_idxs,
        interacting_atoms=ligand_idxs,
    )


# ---- top-level estimator ----


def _equilibrate_joint(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    ff: Forcefield,
    host_config: HostConfig,
    eq_md_params: MDParams,
    seed: int,
) -> Trajectory:
    """Equilibrate both ligands together in the bound pocket.

    Builds a joint ``[host, mol_a, mol_b]`` system in which both ligands are
    fully coupled to the host and solvent but excluded from interacting with
    each other, then runs a short MD trajectory (``eq_md_params``) used for
    anchor selection. The rdkit conformers of both mols are updated in place to
    the equilibrated poses so the caller can read them back via
    ``get_romol_conf``.
    """
    afe = SepTopFreeEnergy(mol_a, mol_b, ff)
    state = get_septop_equilibration_state(afe, ff, host_config, host_config.conf, DEFAULT_TEMP, seed)
    state = optimize_abfe_initial_state(state)
    trj = sample_for_restraints(state, eq_md_params, replicas=1)
    n_host = len(host_config.conf)
    n_a = mol_a.GetNumAtoms()
    frame = trj.frames[-1]
    set_romol_conf(mol_a, frame[n_host : n_host + n_a])
    set_romol_conf(mol_b, frame[n_host + n_a :])
    return trj


def estimate_septop(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    ff: Forcefield,
    host_config: HostConfig,
    prefix: str,
    md_params: MDParams,
    n_windows: int,
    min_overlap: float,
    rst_params: RestraintParams,
    eq_md_params: MDParams | None = None,
    decharge_lambda: float = 0.25,
    eps_scale_lambda: float = 0.25,
    w_lambda: float = 0.5,
    enable_batching: bool = False,
    phase: str = "complex",
) -> SepTopResult:
    """Run one leg of a SepTop calculation.

    Mirrors the structure of the ABFE estimator:
    build a dual-ligand system, then run a bisection (optionally HREX) sampler
    over lambda in ``[0, 1]`` where ``mol_a`` is decoupled as ``mol_b`` is
    coupled.

    Parameters
    ----------
    phase
        ``"complex"`` (default) runs the receptor-bound leg with per-ligand
        Boresch restraints and returns the analytical restraint corrections.
        ``"aqueous"`` runs the solvent leg: the two ligands share a water box
        and are tethered by a single zero-length bond between their central
        atoms, so no anchors are picked and the corrections are zero.

    Returns
    -------
    SepTopResult
        Wraps the underlying :class:`SimulationResult` /
        :class:`HREXSimulationResult`. For the complex leg it also carries the
        chosen anchors and per-ligand analytical restraint corrections; for the
        solvent leg ``anchors`` is ``None`` and the corrections are ``0``.
    """
    if phase not in ("complex", "aqueous"):
        raise ValueError(f"unsupported SepTop phase: {phase!r}")

    # Build the per-ligand decoupling intervals from the scalar knobs: decharge
    # over the symmetric ``[decharge_lambda, 1 - decharge_lambda]``, epsilon
    # scaling over ``[0, eps_scale_lambda]``, and the W-coordinate shift over
    # ``[w_lambda, 1.0]``.
    decharge_interval = (decharge_lambda, 1.0 - decharge_lambda)
    eps_scale_interval = (0.0, eps_scale_lambda)
    w_interval = (w_lambda, 1.0)

    # The anchor-selection equilibration reuses the production MDParams by
    # default, shortened to a cheap burn-in plus a handful of frames. Callers
    # can override it wholesale via ``eq_md_params``.
    if eq_md_params is None:
        eq_md_params = replace(
            md_params,
            n_eq_steps=DEFAULT_EQ_N_EQ_STEPS,
            n_frames=DEFAULT_EQ_N_FRAMES,
        )

    host_config = setup_optimized_host(host_config, [mol_a, mol_b], ff)
    temperature = DEFAULT_TEMP

    anchors: SepTopAnchors | None
    if phase == "complex":
        afe, host_config, host_conf_eq, anchors = _setup_complex_leg(
            mol_a,
            mol_b,
            ff,
            host_config,
            eq_md_params,
            rst_params,
            decharge_interval,
            eps_scale_interval,
            w_interval,
        )
    else:
        afe, host_config, host_conf_eq, anchors = _setup_solvent_leg(
            mol_a,
            mol_b,
            ff,
            host_config,
            rst_params,
            decharge_interval,
            eps_scale_interval,
            w_interval,
        )

    def make_state(lamb: float) -> InitialState:
        return get_septop_initial_state(afe, ff, host_config, host_conf_eq, temperature, md_params.seed, lamb)

    # Pre-optimize a grid of initial states the way RBFE does
    # (tmd.fe.rbfe.setup_initial_states): minimize an evenly spaced lambda
    # schedule from the endstates inward (lambda < 0.5 chained from the
    # lambda=0 endstate, lambda >= 0.5 from the lambda=1 endstate). Each
    # bisection window is then warm-started from the nearest pre-optimized
    # state on the same side of lambda=0.5 via
    # optimize_initial_state_from_pre_optimized. Shared by both the HREX and
    # non-HREX paths below.
    lambda_grid = bisection_lambda_schedule(n_windows)
    pre_optimized_states = [make_state(lamb) for lamb in lambda_grid]
    optimized_x0s = optimize_coordinates(pre_optimized_states, min_cutoff=None)
    for state, x0 in zip(pre_optimized_states, optimized_x0s):
        state.x0 = x0

    def optimize_state(state: InitialState) -> InitialState:
        return optimize_initial_state_from_pre_optimized(state, pre_optimized_states)

    def make_optimized(lamb: float) -> InitialState:
        return optimize_state(make_state(lamb))

    if md_params.hrex_params is None:
        bisection_params = md_params

        initial_lambdas = [0.0, 1.0]
        batch_size = 8 if enable_batching else 1
        list_of_results, trjs = run_sims_bisection(
            initial_lambdas,
            make_optimized,
            bisection_params,
            n_bisections=n_windows - len(initial_lambdas),
            temperature=temperature,
            min_overlap=min_overlap,
            batch_size=batch_size,
        )
        results = list_of_results[-1]
        plots = make_pair_bar_plots(results, temperature, prefix)
        sim_result = SimulationResult(results, plots, trjs, md_params, list_of_results)
    else:
        sim_result = estimate_relative_free_energy_bisection_hrex_impl(
            temperature,
            0.0,
            1.0,
            md_params,
            n_windows,
            make_state,
            optimize_state,
            combined_prefix=prefix,
            min_overlap=min_overlap,
        )

    if anchors is None:
        return SepTopResult(
            sim_result=sim_result,
            anchors=None,
            correction_a=0.0,
            correction_b=0.0,
        )

    correction_a = float(afe.get_restraint_correction(anchors.lig_atoms_a, temperature))
    correction_b = float(afe.get_restraint_correction(anchors.lig_atoms_b, temperature))

    return SepTopResult(
        sim_result=sim_result,
        anchors=anchors,
        correction_a=correction_a,
        correction_b=correction_b,
    )


def _setup_complex_leg(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    ff: Forcefield,
    host_config: HostConfig,
    eq_md_params: MDParams,
    rst_params: RestraintParams,
    decharge_interval: tuple[float, float] = (0.0, 0.2),
    eps_scale_interval: tuple[float, float] = (0.2, 0.4),
    w_interval: tuple[float, float] = (0.0, 1.0),
) -> tuple[SepTopFreeEnergy, HostConfig, NDArray, SepTopAnchors]:
    """Equilibrate, pick Boresch anchors, and build the complex-leg system."""
    # Equilibrate both ligands together in a single joint [host, mol_a, mol_b]
    # system. Both ligands are fully coupled to the host and solvent but
    # excluded from interacting with each other, so each stays bound in the
    # pocket (a decoupled ligand would feel no host forces and drift away). The
    # resulting joint trajectory is used to pick anchors for both ligands and
    # provides the relaxed starting frame for production.
    trj = _equilibrate_joint(mol_a, mol_b, ff, host_config, eq_md_params, eq_md_params.seed)

    anchors = select_septop_anchors(host_config, mol_a, mol_b, trj)

    # Build the joint production starting frame directly from the final frame of
    # the joint equilibration: host, mol_a, and mol_b are all mutually
    # consistent, so the receptor and ligand anchors line up with x0.
    n_host = len(host_config.conf)
    frame = trj.frames[-1]
    host_conf_eq = frame[:n_host]
    pos = frame
    box = trj.boxes[-1]
    host_config = replace(host_config, conf=host_conf_eq, box=box)

    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        anchors=anchors,
        x0=pos,
        box0=box,
        rst_params=rst_params,
        decharge_interval=decharge_interval,
        eps_scale_interval=eps_scale_interval,
        w_interval=w_interval,
    )
    return afe, host_config, host_conf_eq, anchors


def _setup_solvent_leg(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    ff: Forcefield,
    host_config: HostConfig,
    rst_params: RestraintParams,
    decharge_interval: tuple[float, float] = (0.0, 0.2),
    eps_scale_interval: tuple[float, float] = (0.2, 0.4),
    w_interval: tuple[float, float] = (0.0, 1.0),
) -> tuple[SepTopFreeEnergy, HostConfig, NDArray, None]:
    """Build the dual-ligand solvent-leg system (no anchors, central bond)."""
    from tmd.fe.utils import get_romol_conf

    host_conf = host_config.conf
    box = host_config.box
    pos = np.concatenate([host_conf, get_romol_conf(mol_a), get_romol_conf(mol_b)])
    central_atoms = select_central_atoms(host_config, mol_a, mol_b)

    afe = SepTopFreeEnergy(
        mol_a,
        mol_b,
        ff,
        x0=pos,
        box0=box,
        rst_params=rst_params,
        central_atoms=central_atoms,
        decharge_interval=decharge_interval,
        eps_scale_interval=eps_scale_interval,
        w_interval=w_interval,
    )
    return afe, host_config, host_conf, None
