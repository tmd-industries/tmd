# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025 Forrest York
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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, cast, overload

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from tmd.lib import custom_ops

from . import bonded, chiral_restraints, jax_interface, nonbonded, summed
from .potential import (
    BoundGpuImplWrapper_f32,
    BoundGpuImplWrapper_f64,
    BoundPotential,
    GpuImplWrapper_f32,
    GpuImplWrapper_f64,
    Potential,
    Precision,
)
from .types import Box, Conf, Params


@dataclass
class HarmonicBond(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return bonded.harmonic_bond(conf, params, box, self.idxs)


@dataclass
class HarmonicAngle(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return bonded.harmonic_angle(conf, params, box, self.idxs)


@dataclass
class CentroidRestraint(Potential):
    group_a_idxs: NDArray[np.int32]
    group_b_idxs: NDArray[np.int32]
    kb: float
    b0: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return bonded.centroid_restraint(conf, params, box, self.group_a_idxs, self.group_b_idxs, self.kb, self.b0)


@dataclass
class ChiralAtomRestraint(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return chiral_restraints.chiral_atom_restraint(conf, params, box, self.idxs)


@dataclass
class ChiralBondRestraint(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]
    signs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return chiral_restraints.chiral_bond_restraint(conf, params, box, self.idxs, self.signs)


@dataclass
class FlatBottomBond(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return bonded.flat_bottom_bond(conf, params, box, self.idxs)


@dataclass
class LogFlatBottomBond(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]
    beta: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return bonded.log_flat_bottom_bond(conf, params, box, self.idxs, self.beta)


@dataclass
class PeriodicTorsion(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return bonded.periodic_torsion(conf, params, box, self.idxs)


@dataclass
class Nonbonded(Potential):
    num_atoms: int
    exclusion_idxs: NDArray[np.int32]
    scale_factors: NDArray[np.float64]
    beta: float
    cutoff: float
    atom_idxs: Optional[NDArray[np.int32]] = None
    disable_hilbert_sort: bool = False
    nblist_padding: float = 0.1

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return nonbonded.nonbonded(
            conf,
            params,
            box,
            self.exclusion_idxs,
            self.scale_factors,
            self.beta,
            self.cutoff,
            runtime_validate=False,  # needed for this to be JAX-transformable
            atom_idxs=self.atom_idxs,
        )

    @overload
    def to_gpu(self, precision: type[np.float32]) -> "GpuImplWrapper_f32": ...

    @overload
    def to_gpu(self, precision: type[np.float64]) -> "GpuImplWrapper_f64": ...

    def to_gpu(self, precision: Precision) -> GpuImplWrapper_f32 | GpuImplWrapper_f64:
        atom_idxs = self.atom_idxs if self.atom_idxs is not None else np.arange(self.num_atoms, dtype=np.int32)
        all_pairs = NonbondedInteractionGroup(
            self.num_atoms,
            atom_idxs,
            self.beta,
            self.cutoff,
            col_atom_idxs=atom_idxs,
            disable_hilbert_sort=self.disable_hilbert_sort,
            nblist_padding=self.nblist_padding,
        )
        exclusion_idxs, scale_factors = nonbonded.filter_exclusions(atom_idxs, self.exclusion_idxs, self.scale_factors)
        exclusions = NonbondedExclusions(
            self.num_atoms,
            exclusion_idxs,
            scale_factors.astype(precision),
            precision(self.beta),
            precision(self.cutoff),
        )
        return FanoutSummedPotential([exclusions, all_pairs]).to_gpu(precision)


@dataclass
class NonbondedInteractionGroup(Potential):
    num_atoms: int
    row_atom_idxs: NDArray[np.int32]
    beta: float
    cutoff: float
    col_atom_idxs: Optional[NDArray[np.int32]] = None
    disable_hilbert_sort: bool = False
    nblist_padding: float = 0.1

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        num_atoms, _ = jnp.array(conf).shape

        vdW, electrostatics = nonbonded.nonbonded_interaction_groups(
            conf,
            params,
            box,
            self.row_atom_idxs,
            self.col_atom_idxs,
            self.beta,
            self.cutoff,
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)


@dataclass
class NonbondedPairList(Potential):
    num_atoms: int
    idxs: NDArray[np.int32] | list[NDArray[np.int32]]
    rescale_mask: NDArray[np.float64] | list[NDArray[np.float64]]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        vdW, electrostatics = nonbonded.nonbonded_on_specific_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff, self.rescale_mask
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)


@dataclass
class NonbondedExclusions(Potential):
    num_atoms: int
    idxs: NDArray[np.int32] | list[NDArray[np.int32]]
    rescale_mask: NDArray[np.float64] | list[NDArray[np.float64]]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        vdW, electrostatics = nonbonded.nonbonded_on_specific_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff, self.rescale_mask
        )
        U = jnp.sum(vdW) + jnp.sum(electrostatics)
        return -U


@dataclass
class NonbondedPairListPrecomputed(Potential):
    """
    This implements a pairlist with precomputed parameters. It differs from the regular NonbondedPairlist in that it
    expects params of the form s0*q_ij, s_ij, s1*e_ij, and w_offsets_ij, where s are the scaling factors and combining
    rules have already been applied.

    Note that you should not use this class to implement exclusions (that are later cancelled out by AllPairs) since the
    floating point operations are different in python vs C++.
    """

    num_atoms: int
    idxs: NDArray[np.int32] | list[NDArray[np.int32]]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        vdW, electrostatics = nonbonded.nonbonded_on_precomputed_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)


@dataclass
class SummedPotential(Potential):
    potentials: Sequence[Potential]
    params_init: Sequence[Params]
    parallel: bool = True

    def __post_init__(self):
        if len(self.potentials) != len(self.params_init):
            raise ValueError("number of potentials != number of parameter arrays")

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return summed.summed_potential(conf, params, box, self.potentials, self.params_shapes)

    @overload
    def to_gpu(self, precision: type[np.float32]) -> "SummedPotentialGpuImplWrapper_f32": ...

    @overload
    def to_gpu(self, precision: type[np.float64]) -> "SummedPotentialGpuImplWrapper_f64": ...

    def to_gpu(self, precision: Precision) -> "SummedPotentialGpuImplWrapper_f32 | SummedPotentialGpuImplWrapper_f64":
        sizes = [ps.size for ps in self.params_init]
        if precision == np.float32:
            impls_f32: list[custom_ops.Potential_f32] = [p.to_gpu(precision).unbound_impl for p in self.potentials]
            return SummedPotentialGpuImplWrapper_f32(custom_ops.SummedPotential_f32(impls_f32, sizes, self.parallel))
        elif precision == np.float64:
            impls_f64: list[custom_ops.Potential_f64] = [p.to_gpu(precision).unbound_impl for p in self.potentials]
            return SummedPotentialGpuImplWrapper_f64(custom_ops.SummedPotential_f64(impls_f64, sizes, self.parallel))
        else:
            assert 0

    def call_with_params_list(self, conf: Conf, params: Sequence[Params], box: Box) -> float | Array:
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])
        return self(conf, params_flat, box)

    def bind_params_list(self, params: Sequence[Params]) -> BoundPotential["SummedPotential"]:
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])
        return BoundPotential(self, params_flat)

    @property
    def params_shapes(self):
        return [ps.shape for ps in self.params_init]

    def unflatten_params(self, params: Params) -> list[Params]:
        return summed.unflatten_params(params, self.params_shapes)


def make_summed_potential(bps: Sequence[BoundPotential]):
    potentials = [bp.potential for bp in bps]
    params = [bp.params for bp in bps]
    return SummedPotential(potentials, params).bind_params_list(params)


@dataclass
class SummedPotentialGpuImplWrapper_f32(GpuImplWrapper_f32):
    """Handles flattening parameters before passing to kernel to provide a nicer interface"""

    def call_with_params_list(self, conf: Conf, params: Sequence[Params], box: Box) -> float:
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params_flat, box)
        return cast(float, res)

    def bind_params_list(self, params: Sequence[Params]) -> BoundGpuImplWrapper_f32:
        params_flat = np.concatenate([ps.reshape(-1) for ps in params])
        return BoundGpuImplWrapper_f32(custom_ops.BoundPotential_f32(self.unbound_impl, params_flat))


@dataclass
class SummedPotentialGpuImplWrapper_f64(GpuImplWrapper_f64):
    """Handles flattening parameters before passing to kernel to provide a nicer interface"""

    def call_with_params_list(self, conf: Conf, params: Sequence[Params], box: Box) -> float:
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params_flat, box)
        return cast(float, res)

    def bind_params_list(self, params: Sequence[Params]) -> BoundGpuImplWrapper_f64:
        params_flat = np.concatenate([ps.reshape(-1) for ps in params])
        return BoundGpuImplWrapper_f64(custom_ops.BoundPotential_f64(self.unbound_impl, params_flat))


@dataclass
class FanoutSummedPotential(Potential):
    potentials: Sequence[Potential]
    parallel: bool = True

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        return summed.fanout_summed_potential(conf, params, box, self.potentials)

    @overload
    def to_gpu(self, precision: type[np.float32]) -> "GpuImplWrapper_f32": ...

    @overload
    def to_gpu(self, precision: type[np.float64]) -> "GpuImplWrapper_f64": ...

    def to_gpu(self, precision: Precision) -> GpuImplWrapper_f32 | GpuImplWrapper_f64:
        if precision == np.float32:
            impls_f32: list[custom_ops.Potential_f32] = [p.to_gpu(precision).unbound_impl for p in self.potentials]
            return GpuImplWrapper_f32(custom_ops.FanoutSummedPotential_f32(impls_f32, self.parallel))
        elif precision == np.float64:
            impls_f64: list[custom_ops.Potential_f64] = [p.to_gpu(precision).unbound_impl for p in self.potentials]
            return GpuImplWrapper_f64(custom_ops.FanoutSummedPotential_f64(impls_f64, self.parallel))
        else:
            assert 0
