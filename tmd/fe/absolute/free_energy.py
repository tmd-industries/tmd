# Copyright 2025 Justin Gullingsrud
# Modifications Copyright 2026, Forrest York
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


from dataclasses import dataclass
from typing import Self

import mdtraj
import numpy as np
from numpy.typing import NDArray

from tmd.constants import BOLTZ
from tmd.fe.absolute.restraints import (
    select_ligand_atoms_baumann,
    select_receptor_atoms_baumann,
)
from tmd.fe.cif_writer import build_openmm_topology
from tmd.fe.free_energy import (
    AbsoluteFreeEnergy,
)
from tmd.fe.rbfe import (
    HostConfig,
)
from tmd.ff import Forcefield
from tmd.potentials import HarmonicAngle, HarmonicBond, PeriodicTorsion
from tmd.potentials.bonded import kahan_angle, signed_torsion_angle
from tmd.potentials.jax_utils import delta_r

Snapshot = tuple[NDArray, NDArray]


def get_abfe_endstate_trajectory(topology, mol, snapshots: Snapshot) -> mdtraj.Trajectory:
    """Construct a trajectory from endstate snapshots for ABFE.

    Parameters
    ----------
    topology
        OpenMM topology of the host (receptor + solvent)
    mol
        RDKit molecule object for the ligand
    snapshots
        tuple of (frames, boxes) arrays from the simulation

    Returns
    -------
    traj
        mdtraj Trajectory with combined receptor+ligand topology
    """
    combined_top = build_openmm_topology([topology, mol])
    top = mdtraj.Topology.from_openmm(combined_top)
    frames, boxes = snapshots
    angles = [[90.0] * 3] * len(boxes)
    lengths = [np.diag(box) for box in boxes]
    traj = mdtraj.Trajectory(frames, top, unitcell_lengths=lengths, unitcell_angles=angles)
    return traj


@dataclass(frozen=True)
class RestraintParams:
    """Parameters for harmonic restraints used in absolute binding free energy.

    Attributes
    ----------
    kb : float
        Bond restraint force constant (kcal/mol/nm^2), default 500
    ka : float
        Angle restraint force constant (kcal/mol/rad^2), default 200
    kd : float
        Dihedral restraint force constant (kcal/mol), default 10
    on : float
        Lambda value at which restraints reach full strength, default 0.0625
    """

    kb: float = 500  # bond restraint strength
    ka: float = 200  # angle restraint
    kd: float = 10  # dihedral restraint
    on: float = 0.0625  # where in lambda schedule the restraints go to full strength


class AbsoluteBindingFreeEnergy(AbsoluteFreeEnergy):
    """Absolute binding free energy calculation with boresch-style restraints.

    This class implements the Baumann et al. [1] approach for absolute binding
    free energy calculations, using harmonic restraints to define a binding
    site reference frame based on receptor-ligand atom pairs.

    The restraint geometry is defined by 3 receptor atoms and 3 ligand atoms,
    forming a bond, 2 angles, and 3 dihedrals that collectively restrain the
    ligand orientation and position relative to the receptor.

    References
    ----------
    [1] [Baumann et al., 2019] A rigorous framework for relative and absolute
        host-guest binding free energy calculations.
        J. Chem. Phys. 151:104104, 2019.
        https://doi.org/10.1063/1.5100842
    """

    def __init__(
        self,
        rec_atoms: list[int],
        lig_atoms: list[int],
        x0: NDArray,
        box0: NDArray,
        params: RestraintParams,
        *args,
        **kwargs,
    ):
        """Initialize with restraint atom indices and reference geometry.

        Parameters
        ----------
        rec_atoms : list of int
            Indices of 3 receptor atoms defining the restraint atoms.
            rec_atoms[0] is the reference receptor atom bonded to the ligand.
        lig_atoms : list of int
            Indices of 3 ligand atoms defining the restraint atoms.
            lig_atoms[0] is the reference ligand atom bonded to the receptor.
        x0 : NDArray
            Initial system coordinates, should include the ligand
        box0 : NDArray
            Initial box
        params : RestraintParams
            Restraint force constants and lambda schedule parameters.
        *args
            Positional arguments to AbsoluteFreeEnergy. Refer to tmd.fe.free_energy.AbsoluteFreeEnergy for details
        *kwargs
            Keyword arguments to AbsoluteFreeEnergy. Refer to tmd.fe.free_energy.AbsoluteFreeEnergy for details
        """
        super().__init__(*args, **kwargs)
        self.rec_atoms = rec_atoms
        self.lig_atoms = lig_atoms
        self.x0 = x0
        self.box0 = box0
        self.params = params

    @classmethod
    def create(cls, bt, host_config: HostConfig, tmtrj, restraint_params: RestraintParams) -> Self:
        """Construct an AbsoluteBindingFreeEnergy instance from a trajectory.

        Selects restraint atoms using RMSF filtering (Baumann method) and
        builds the restraint geometry from the final frame of the trajectory.

        Parameters
        ----------
        bt
            BaseTopology object containing the ligand and setup information.
        host_config
            HostConfig with receptor and simulation parameters.
        tmtrj
            MDTraj Trajectory object with frames and boxes from the simulation.
        restraint_params
            RestraintParams specifying force constants and schedule.


        Returns
        -------
        AbsoluteBindingFreeEnergy
            Initialized instance with selected restraint atoms and geometry.
        """
        mol = bt.mol

        # construct trajectory from snapshots aligned to final frame
        # TODO: toss out initial 20% of frames?
        snapshots = (tmtrj.frames, np.array(tmtrj.boxes))
        trj = get_abfe_endstate_trajectory(host_config.omm_topology, mol, snapshots)
        trj.image_molecules(inplace=True)
        ref = trj[-1]
        """
        In the original implementation, rmsf was computed after aligning the
        ligand-only trajectory to the initial ligand coordinates.  Here we
        instead align the protein backbone using the final frame of the trajectory.
        The motivation here is that we want ligand restraint points which are stable
        relative to the protein starting point.
        """
        trj.superpose(ref, atom_indices=trj.topology.select("backbone"))
        # use rmsf to filter restraint atoms
        rmsf = mdtraj.rmsf(trj, ref)

        # select ligand atoms
        ligand_offset = trj.n_atoms - mol.GetNumAtoms()
        lig_ids = select_ligand_atoms_baumann(mol, rmsf[ligand_offset:])
        lig_ids = [i + ligand_offset for i in lig_ids]

        # select receptor atoms
        # FIXME: reuse the rmsf calculation instead of redoing it in this routine
        rec_ids = select_receptor_atoms_baumann(trj, lig_ids)

        pos = tmtrj.frames[-1]
        box = tmtrj.boxes[-1]
        return cls(rec_ids, lig_ids, pos, box, restraint_params, mol, bt)

    def get_bond_geometry(self) -> tuple[list[int], float]:
        """Get the bond restraint atom indices and equilibrium distance.

        The bond is defined between the first receptor restrained atom and the first
        ligand restrained atom.

        Returns
        -------
        atom_indices : list of int
            [receptor_atom, ligand_atom]
        equilibrium_distance : float
            Equilibrium bond distance r0 (nm)
        """
        i0 = [self.rec_atoms[0], self.lig_atoms[0]]
        a0, b0 = self.x0[i0]
        r0 = np.linalg.norm(delta_r(a0, b0, self.box0))
        return (i0, float(r0))

    def get_bond_restraint(self, scale: float) -> tuple[HarmonicBond, NDArray]:
        """Get the bond restraint potential and its parameters.

        Parameters
        ----------
        scale : float
            Scale factor for the force constant (0 to 1).

        Returns
        -------
        potential : HarmonicBond
            HarmonicBond potential with the restraint atom indices.
        params : NDArray
            Array of shape (1, 2) with [force_constant, equilibrium_distance].
        """
        i0, r0 = self.get_bond_geometry()
        fc = self.params.kb * scale

        idxs = np.array([i0], dtype=np.int32)
        params = np.array([[fc, r0]], dtype=np.float32)

        return HarmonicBond(self.x0.shape[0], idxs), params

    def get_angle_geometry(self) -> list[tuple[list[int], float]]:
        """Get the angle restraint atom indices and equilibrium angles.

        Two angles are defined:
        * Angle 1: rec[1] - rec[0] - lig[0]
        * Angle 2: rec[0] - lig[0] - lig[1]

        Returns
        -------
        angles : list of (atom_indices, angle) tuples
            Each entry is ([rec[i], rec[j], lig[k]], equilibrium_angle_rad)
        """
        rec = self.rec_atoms
        lig = self.lig_atoms
        i0 = [rec[1], rec[0], lig[0]]
        i1 = [rec[0], lig[0], lig[1]]

        pos = self.x0
        t0 = kahan_angle(*pos[i0], 0.0, self.box0)  # type: ignore
        t1 = kahan_angle(*pos[i1], 0.0, self.box0)  # type: ignore
        return [(i0, t0), (i1, t1)]

    def get_angle_restraint(self, scale: float) -> tuple[HarmonicAngle, NDArray]:
        """Get the angle restraint potential and its parameters.

        Parameters
        ----------
        scale : float
            Scale factor for the force constant (0 to 1).

        Returns
        -------
        potential : HarmonicAngle
            HarmonicAngle potential with the restraint atom indices.
        params : NDArray
            Array of shape (2, 3) with [force_constant, equilibrium_angle, period]
            for each of the two angles.
        """
        ((i0, t0), (i1, t1)) = self.get_angle_geometry()
        fc = self.params.ka * scale

        idxs = np.array([i0, i1], dtype=np.int32)
        params = np.array([[fc, t0, 0], [fc, t1, 0]], dtype=np.float32)
        return HarmonicAngle(self.x0.shape[0], idxs), params

    def get_dihedral_geometry(self) -> list[tuple[list[int], float]]:
        """Get the dihedral restraint atom indices and equilibrium angles.

        Three dihedrals are defined:
        * Dihedral 1: rec[2] - rec[1] - rec[0] - lig[0]
        * Dihedral 2: rec[1] - rec[0] - lig[0] - lig[1]
        * Dihedral 3: rec[0] - lig[0] - lig[1] - lig[2]

        Returns
        -------
        dihedrals : list of (atom_indices, angle) tuples
            Each entry is ([atom0, atom1, atom2, atom3], equilibrium_angle_rad)
        """
        rec = self.rec_atoms
        lig = self.lig_atoms
        i0 = [rec[2], rec[1], rec[0], lig[0]]
        i1 = [rec[1], rec[0], lig[0], lig[1]]
        i2 = [rec[0], lig[0], lig[1], lig[2]]

        pos = self.x0
        p0 = signed_torsion_angle(*pos[i0], self.box0)  # type: ignore
        p1 = signed_torsion_angle(*pos[i1], self.box0)  # type: ignore
        p2 = signed_torsion_angle(*pos[i2], self.box0)  # type: ignore
        return [(i0, p0), (i1, p1), (i2, p2)]

    def get_dihedral_restraint(self, scale: float) -> tuple[PeriodicTorsion, NDArray]:
        """Get the dihedral restraint potential and its parameters.

        A constant offset is added so that the restraint minimum has E=0.

        Parameters
        ----------
        scale : float
            Scale factor for the force constant (0 to 1).

        Returns
        -------
        potential : PeriodicTorsion
            PeriodicTorsion potential with the restraint atom indices.
        params : NDArray
            Array of shape (19, 3) with [force_constant, phase, period]
        """
        idxs_ = []
        params_ = []
        k = self.params.kd * scale
        const = 0.0
        for ids, phi in self.get_dihedral_geometry():
            for period in range(1, 6 + 1):
                phase = period * phi
                fc = -k / period
                params_.append([fc, phase, period])
                idxs_.append(ids)
                const += fc
        # add a constant term to make restraint minimum have E=0
        params_.append([-const, 0, 0])
        idxs_.append(ids)

        idxs = np.array(idxs_, dtype=np.int32)
        params = np.array(params_, dtype=np.float32)
        return PeriodicTorsion(self.x0.shape[0], idxs), params

    def get_restraint_correction(self, temperature: float) -> float:
        """Compute analytical correction to free energy from restraints.

        Corrects for the fact that the ligand is restrained to the receptor
        during the decoupled state. The correction corresponds to the
        free energy difference between the restrained and unrestrained states.

        Parameters
        ----------
        temperature : float
            Simulation temperature in Kelvin.

        Returns
        -------
        correction : float
            Free energy correction (kcal/mol) to add to the binding free energy.

        Notes
        -----
        Based on Equation 32 of [1], which gives the analytical correction for
        harmonic restraints. The formula accounts for the standard state volume
        (1 M = 1660 A^3) and the force constants of all restraints.

        References
        ----------

        [1] Boresch, Stefan, et al. "Absolute binding free energies: a quantitative
            approach for their calculation." The Journal of Physical Chemistry B 107.35
            (2003): 9535-9551.
            https://pubs.acs.org/doi/10.1021/jp0217839
        """
        _, r0 = self.get_bond_geometry()
        ((_, t0), (_, t1)) = self.get_angle_geometry()
        s0 = np.sin(t0)
        s1 = np.sin(t1)
        V0 = 1660 * (0.1**3)  # 1M standard state in nm^3
        kT = BOLTZ * temperature

        kb = self.params.kb
        ka = self.params.ka
        # Account for multiple terms in the dihedral restraint
        kd = self.params.kd * sum(range(1, 7))

        return -kT * np.log(8 * np.pi**2 * V0 * (kb * ka**2 * kd**3) ** 0.5 / (r0**2 * s0 * s1 * (2 * np.pi * kT) ** 3))

    def prepare_host_edge(self, ff: Forcefield, host_config: HostConfig, lamb: float):
        """Construct the alchemical schedule with restraints for host-edge ABFE.

        The lambda schedule works as follows:

        * lambda=0: fully coupled state, no restraints
        * lambda=restraint_on (~0.0625): fully coupled state, full restraints
        * lambda=1: decoupled/intermolecular-decharged state, full restraints

        Between lambda=0 and lambda=restraint_on, restraints are linearly
        scaled from 0 to full strength. Between lambda=restraint_on and
        lambda=1, the alchemical decoupling proceeds with full restraints.

        Parameters
        ----------
        ff : Forcefield
            Forcefield for building potentials.
        host_config : HostConfig
            Configuration for the host (receptor + solvent) system.
        lamb : float
            Current lambda value in [0, 1].

        Returns
        -------
        ubps : tuple
            Tuple of BoundPotential objects including restraint potentials.
        params : tuple
            Tuple of parameter arrays corresponding to each potential.
        masses : tuple
            Tuple of masses for the system.
        """
        restraint_on = self.params.on
        m = 1 / (1 - restraint_on)
        b = 1 - m
        hglamb = max(0, m * lamb + b)
        ubps, params, masses = super().prepare_host_edge(ff, host_config, hglamb)

        scale = min(1, lamb / restraint_on)
        # must use the same set of potentials in all windows, even if scale is zero
        hb, hbp = self.get_bond_restraint(scale)
        ha, hap = self.get_angle_restraint(scale)
        dr, drp = self.get_dihedral_restraint(scale)

        ubps = (*list(ubps), hb, ha, dr)
        params = (*list(params), hbp, hap, drp)

        return ubps, params, masses
