# Copyright 2019-2025, Relay Therapeutics
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

import multiprocessing
from abc import ABC
from dataclasses import dataclass, fields

import jax
import numpy as np
import scipy

from tmd.integrator import simulate
from tmd.potentials import (
    BoundPotential,
    ChiralAtomRestraint,
    ChiralBondRestraint,
    HarmonicAngle,
    HarmonicBond,
    Nonbonded,
    NonbondedPairListPrecomputed,
    PeriodicTorsion,
)

# Chiral bond restraints are disabled until checks are added (see GH #815)
# from tmd.potentials import bonded, chiral_restraints, nonbonded


def minimize_scipy(U_fn, x0, return_traj=False, seed=2024):
    shape = x0.shape

    @jax.jit
    def U_flat(x_flat):
        x_full = x_flat.reshape(*shape)
        return U_fn(x_full)

    grad_bfgs_fn = jax.jit(jax.grad(U_flat))

    traj = []

    def callback_fn(x):
        if return_traj:
            traj.append(x.reshape(*shape))

    minimizer_kwargs = {"jac": grad_bfgs_fn, "callback": callback_fn}
    res = scipy.optimize.basinhopping(U_flat, x0.reshape(-1), minimizer_kwargs=minimizer_kwargs, seed=seed)
    xi = res.x.reshape(*shape)

    if return_traj:
        return traj
    else:
        return xi


def simulate_system(U_fn, x0, num_samples=20000, steps_per_batch=500, num_workers=None, minimize=True):
    # this functions runs in 64bit since it's calling into jax.
    x0 = x0.astype(np.float64)
    num_atoms = x0.shape[0]

    seed = 2023

    if minimize:
        x_min = minimize_scipy(U_fn, x0, seed=seed)
    else:
        x_min = x0

    num_workers = num_workers or multiprocessing.cpu_count()
    samples_per_worker = int(np.ceil(num_samples / num_workers))

    burn_in_batches = num_samples // 10
    frames, _ = simulate(
        x_min,
        U_fn,
        300.0,
        np.ones(num_atoms) * 4.0,
        steps_per_batch,
        samples_per_worker + burn_in_batches,
        num_workers,
        seed=seed,
    )
    # (ytz): discard burn in batches
    frames = frames[:, burn_in_batches:, :, :]
    # collect over all workers
    frames = frames.reshape(-1, num_atoms, 3)[:num_samples]
    # sanity check that we didn't undersample
    assert len(frames) == num_samples
    return frames


@dataclass
class AbstractSystem(ABC):
    def get_U_fn(self):
        """
        Return a jax function that evaluates the potential energy of a set of coordinates.
        """
        U_fns = self.get_U_fns()

        def U_fn(x):
            return sum(U(x, box=None) for U in U_fns)

        return U_fn

    def get_U_fns(self) -> list[BoundPotential]:
        """
        Return a list of bound potential"""
        potentials: list[BoundPotential] = []
        for f in fields(self):
            bp = getattr(self, f.name)
            # (TODO): chiral_bonds currently disabled
            if f.name != "chiral_bond":
                potentials.append(bp)

        return potentials


@dataclass
class HostSystem(AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    nonbonded_all_pairs: BoundPotential[Nonbonded]

    # def __post_init__(self):
    #     self.bond.params = self.bond.params.astype(np.float32)
    #     self.angle.params = self.angle.params.astype(np.float32)
    #     self.proper.params = self.proper.params.astype(np.float32)
    #     self.improper.params = self.improper.params.astype(np.float32)
    #     self.nonbonded_all_pairs.params = self.nonbonded_all_pairs.params.astype(np.float32)


@dataclass
class GuestSystem(AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_pair_list: BoundPotential[NonbondedPairListPrecomputed]

    # def __post_init__(self):
    #     self.bond.params = self.bond.params.astype(np.float32)
    #     self.angle.params = self.angle.params.astype(np.float32)
    #     self.proper.params = self.proper.params.astype(np.float32)
    #     self.improper.params = self.improper.params.astype(np.float32)
    #     self.chiral_atom.params = self.chiral_atom.params.astype(np.float32)
    #     self.chiral_bond.params = self.chiral_bond.params.astype(np.float32)
    #     self.nonbonded_pair_list.params = self.nonbonded_pair_list.params.astype(np.float32)


@dataclass
class HostGuestSystem(AbstractSystem):
    # utility system container
    bond: BoundPotential[HarmonicBond]
    angle: BoundPotential[HarmonicAngle]
    proper: BoundPotential[PeriodicTorsion]
    improper: BoundPotential[PeriodicTorsion]
    chiral_atom: BoundPotential[ChiralAtomRestraint]
    chiral_bond: BoundPotential[ChiralBondRestraint]
    nonbonded_pair_list: BoundPotential[NonbondedPairListPrecomputed]
    nonbonded_all_pairs: BoundPotential[Nonbonded]

    # def __post_init__(self):
    #     self.bond.params = self.bond.params.astype(np.float32)
    #     self.angle.params = self.angle.params.astype(np.float32)
    #     self.proper.params = self.proper.params.astype(np.float32)
    #     self.improper.params = self.improper.params.astype(np.float32)
    #     self.chiral_atom.params = self.chiral_atom.params.astype(np.float32)
    #     self.chiral_bond.params = self.chiral_bond.params.astype(np.float32)
    #     self.nonbonded_pair_list.params = self.nonbonded_pair_list.params.astype(np.float32)
    #     self.nonbonded_all_pairs.params = self.nonbonded_all_pairs.params.astype(np.float32)
