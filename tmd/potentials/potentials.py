# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2025-2026, Forrest York
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
    combine_pot_params,
)
from .types import Box, Conf, Params


@dataclass
class _BondBase(Potential):
    num_atoms: int
    idxs: NDArray[np.int32]

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        return self.__class__(self.num_atoms, combine_pot_params(self.idxs, other_pot.idxs))


@dataclass
class HarmonicBond(_BondBase):
    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return bonded.harmonic_bond(conf, params, box, self.idxs)


@dataclass
class HarmonicAngle(_BondBase):
    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return bonded.harmonic_angle(conf, params, box, self.idxs)


@dataclass
class CentroidRestraint(Potential):
    group_a_idxs: NDArray[np.int32]
    group_b_idxs: NDArray[np.int32]
    kb: float
    b0: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return bonded.centroid_restraint(conf, params, box, self.group_a_idxs, self.group_b_idxs, self.kb, self.b0)

    def combine(self, other_pot):
        raise NotImplementedError("Can't combine")


@dataclass
class ChiralAtomRestraint(_BondBase):
    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return chiral_restraints.chiral_atom_restraint(conf, params, box, self.idxs)


@dataclass
class ChiralBondRestraint(_BondBase):
    signs: NDArray[np.int32]

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return chiral_restraints.chiral_bond_restraint(conf, params, box, self.idxs, self.signs)


@dataclass
class FlatBottomBond(_BondBase):
    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return bonded.flat_bottom_bond(conf, params, box, self.idxs)


@dataclass
class FlatBottomRestraint(_BondBase):
    restraint_coords: NDArray

    def __call__(self, conf: Conf, params: Params, box: Optional[Box]) -> float | Array:
        return bonded.flat_bottom_restraint(conf, params, box, self.idxs, self.restraint_coords)

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        return self.__class__(
            self.num_atoms,
            combine_pot_params(self.idxs, other_pot.idxs),
            combine_pot_params(self.restraint_coords, other_pot.restraint_coords),
        )


@dataclass
class LogFlatBottomBond(_BondBase):
    idxs: NDArray[np.int32]
    beta: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return bonded.log_flat_bottom_bond(conf, params, box, self.idxs, self.beta)

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        if self.beta != other_pot.beta:
            raise ValueError(f"LogFlatBottomBond beta must match: {self.beta} != {other_pot.beta}")
        return self.__class__(self.num_atoms, combine_pot_params(self.idxs, other_pot.idxs), self.beta)


@dataclass
class PeriodicTorsion(_BondBase):
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
        assert isinstance(params, (np.ndarray, Array))
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
        if isinstance(self.exclusion_idxs, list):
            assert isinstance(self.scale_factors, list), "scale factors must be a list of scale factors"
            assert self.atom_idxs is None or isinstance(self.atom_idxs, list), (
                "atom indices must be a list of atom indices"
            )
            assert len(self.exclusion_idxs) == len(self.scale_factors)
            atom_idxs = (
                self.atom_idxs
                if self.atom_idxs is not None
                else [np.arange(self.num_atoms, dtype=np.int32) for _ in range(len(self.exclusion_idxs))]
            )
        else:
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
        if isinstance(self.exclusion_idxs, list):
            exclusion_idxs = []
            scale_factors = []
            for idxs, exc, fac in zip(atom_idxs, self.exclusion_idxs, self.scale_factors):
                ex, factor = nonbonded.filter_exclusions(idxs, exc, fac)
                exclusion_idxs.append(ex)
                scale_factors.append(factor.astype(precision))
        else:
            exclusion_idxs, scale_factors = nonbonded.filter_exclusions(
                atom_idxs, self.exclusion_idxs, self.scale_factors
            )
            scale_factors = scale_factors.astype(precision)
        exclusions = NonbondedExclusions(
            self.num_atoms,
            exclusion_idxs,
            scale_factors,
            precision(self.beta),
            precision(self.cutoff),
        )
        return FanoutSummedPotential([exclusions, all_pairs]).to_gpu(precision)

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        if self.beta != other_pot.beta:
            raise ValueError(f"Nonbonded beta must match: {self.beta} != {other_pot.beta}")
        if self.cutoff != other_pot.cutoff:
            raise ValueError(f"Nonbonded cutoff must match: {self.cutoff} != {other_pot.cutoff}")
        if self.disable_hilbert_sort != other_pot.disable_hilbert_sort:
            raise ValueError(
                f"Nonbonded disable_hilbert_sort must match: {self.disable_hilbert_sort} != {other_pot.disable_hilbert_sort}"
            )
        if self.nblist_padding != other_pot.nblist_padding:
            raise ValueError(
                f"Nonbonded nblist_padding must match: {self.nblist_padding} != {other_pot.nblist_padding}"
            )
        if type(self.atom_idxs) is not type(other_pot.atom_idxs):
            raise ValueError("Nonbonded must either provide atom idxs or both have None")
        atom_idxs: None | list = None
        if self.atom_idxs is not None:
            atom_idxs = combine_pot_params(self.atom_idxs, other_pot.atom_idxs)
        return self.__class__(
            self.num_atoms,
            combine_pot_params(self.exclusion_idxs, other_pot.exclusion_idxs),
            combine_pot_params(self.scale_factors, other_pot.scale_factors),
            self.beta,
            self.cutoff,
            atom_idxs,
            disable_hilbert_sort=self.disable_hilbert_sort,
            nblist_padding=self.nblist_padding,
        )


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
        assert isinstance(params, (np.ndarray, Array))
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

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        if self.beta != other_pot.beta:
            raise ValueError(f"NonbondedInteractionGroup beta must match: {self.beta} != {other_pot.beta}")
        if self.cutoff != other_pot.cutoff:
            raise ValueError(f"NonbondedInteractionGroup cutoff must match: {self.cutoff} != {other_pot.cutoff}")
        if self.disable_hilbert_sort != other_pot.disable_hilbert_sort:
            raise ValueError(
                f"NonbondedInteractionGroup disable_hilbert_sort must match: {self.disable_hilbert_sort} != {other_pot.disable_hilbert_sort}"
            )
        if self.nblist_padding != other_pot.nblist_padding:
            raise ValueError(
                f"NonbondedInteractionGroup nblist_padding must match: {self.nblist_padding} != {other_pot.nblist_padding}"
            )
        if type(self.col_atom_idxs) is not type(other_pot.col_atom_idxs):
            raise ValueError("NonbondedInteractionGroup must either provide col_atom_idxs or both have None")
        col_atom_idxs: None | list = None
        if self.col_atom_idxs is not None:
            col_atom_idxs = combine_pot_params(self.col_atom_idxs, other_pot.col_atom_idxs)
        return self.__class__(
            self.num_atoms,
            combine_pot_params(self.row_atom_idxs, other_pot.row_atom_idxs),
            self.beta,
            self.cutoff,
            col_atom_idxs,
            disable_hilbert_sort=self.disable_hilbert_sort,
            nblist_padding=self.nblist_padding,
        )


@dataclass
class NonbondedPairList(Potential):
    num_atoms: int
    idxs: NDArray[np.int32] | list[NDArray[np.int32]]
    rescale_mask: NDArray | list[NDArray]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        vdW, electrostatics = nonbonded.nonbonded_on_specific_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff, self.rescale_mask
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        if self.beta != other_pot.beta:
            raise ValueError(f"NonbondedPairList beta must match: {self.beta} != {other_pot.beta}")
        if self.cutoff != other_pot.cutoff:
            raise ValueError(f"NonbondedPairList cutoff must match: {self.cutoff} != {other_pot.cutoff}")
        return self.__class__(
            self.num_atoms,
            combine_pot_params(self.idxs, other_pot.idxs),
            combine_pot_params(self.rescale_mask, other_pot.rescale_mask),
            self.beta,
            self.cutoff,
        )


@dataclass
class NonbondedExclusions(Potential):
    num_atoms: int
    idxs: NDArray[np.int32] | list[NDArray[np.int32]]
    rescale_mask: NDArray[np.float64] | list[NDArray[np.float64]]
    beta: float
    cutoff: float

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        vdW, electrostatics = nonbonded.nonbonded_on_specific_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff, self.rescale_mask
        )
        U = jnp.sum(vdW) + jnp.sum(electrostatics)
        return -U

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        if self.beta != other_pot.beta:
            raise ValueError(f"NonbondedExclusions beta must match: {self.beta} != {other_pot.beta}")
        if self.cutoff != other_pot.cutoff:
            raise ValueError(f"NonbondedExclusions cutoff must match: {self.cutoff} != {other_pot.cutoff}")
        return self.__class__(
            self.num_atoms,
            combine_pot_params(self.idxs, other_pot.idxs),
            combine_pot_params(self.rescale_mask, other_pot.rescale_mask),
            self.beta,
            self.cutoff,
        )


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
        assert isinstance(params, (np.ndarray, Array))
        vdW, electrostatics = nonbonded.nonbonded_on_precomputed_pairs(
            conf, params, box, self.idxs, self.beta, self.cutoff
        )
        return jnp.sum(vdW) + jnp.sum(electrostatics)

    def combine(self, other_pot):
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Other potential does not match type")
        if self.num_atoms != other_pot.num_atoms:
            raise ValueError(f"Potentials must have same number of atoms: {self.num_atoms} != {other_pot.num_atoms}")
        if self.beta != other_pot.beta:
            raise ValueError(f"NonbondedExclusions beta must match: {self.beta} != {other_pot.beta}")
        if self.cutoff != other_pot.cutoff:
            raise ValueError(f"NonbondedExclusions cutoff must match: {self.cutoff} != {other_pot.cutoff}")
        return self.__class__(
            self.num_atoms,
            combine_pot_params(self.idxs, other_pot.idxs),
            self.beta,
            self.cutoff,
        )


@dataclass
class SummedPotential(Potential):
    potentials: Sequence[Potential]
    params_init: Sequence[Params]
    parallel: bool = True

    def __post_init__(self):
        if len(self.potentials) != len(self.params_init):
            raise ValueError("number of potentials != number of parameter arrays")

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
        return summed.summed_potential(conf, params, box, self.potentials, self.params_shapes)

    @overload
    def to_gpu(self, precision: type[np.float32]) -> "SummedPotentialGpuImplWrapper_f32": ...

    @overload
    def to_gpu(self, precision: type[np.float64]) -> "SummedPotentialGpuImplWrapper_f64": ...

    def to_gpu(self, precision: Precision) -> "SummedPotentialGpuImplWrapper_f32 | SummedPotentialGpuImplWrapper_f64":
        assert all([isinstance(ps, (np.ndarray, Array)) for ps in self.params_init])
        sizes = [ps.size for ps in self.params_init]  # type: ignore
        if precision == np.float32:
            impls_f32: list[custom_ops.Potential_f32] = [p.to_gpu(precision).unbound_impl for p in self.potentials]
            return SummedPotentialGpuImplWrapper_f32(custom_ops.SummedPotential_f32(impls_f32, sizes, self.parallel))
        elif precision == np.float64:
            impls_f64: list[custom_ops.Potential_f64] = [p.to_gpu(precision).unbound_impl for p in self.potentials]
            return SummedPotentialGpuImplWrapper_f64(custom_ops.SummedPotential_f64(impls_f64, sizes, self.parallel))
        else:
            assert 0

    def call_with_params_list(self, conf: Conf, params: Sequence[Params], box: Box) -> float | Array:
        assert all([isinstance(ps, (np.ndarray, Array)) for ps in self.params_init])
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])  # type: ignore
        return self(conf, params_flat, box)

    def bind_params_list(self, params: Sequence[Params]) -> BoundPotential["SummedPotential"]:
        assert all([isinstance(ps, (np.ndarray, Array)) for ps in self.params_init])
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])  # type: ignore
        return BoundPotential(self, params_flat)

    @property
    def params_shapes(self):
        assert all([isinstance(ps, (np.ndarray, Array)) for ps in self.params_init])
        return [ps.shape for ps in self.params_init]  # type: ignore

    def unflatten_params(self, params: Params) -> list[Params]:
        return summed.unflatten_params(params, self.params_shapes)

    def combine(self, other_pot):
        raise NotImplementedError("Can't combine")


def make_summed_potential(bps: Sequence[BoundPotential]):
    potentials = [bp.potential for bp in bps]
    params = [bp.params for bp in bps]
    return SummedPotential(potentials, params).bind_params_list(params)


@dataclass
class SummedPotentialGpuImplWrapper_f32(GpuImplWrapper_f32):
    """Handles flattening parameters before passing to kernel to provide a nicer interface"""

    def call_with_params_list(self, conf: Conf, params: Sequence[Params], box: Box) -> float:
        assert len(params) == 0 or isinstance(params[0], (np.ndarray, Array))
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])  # type: ignore
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params_flat, box)
        return cast(float, res)

    def bind_params_list(self, params: Sequence[Params]) -> BoundGpuImplWrapper_f32:
        assert len(params) == 0 or isinstance(params[0], (np.ndarray, Array))
        params_flat = np.concatenate([ps.reshape(-1) for ps in params])  # type: ignore
        return BoundGpuImplWrapper_f32(custom_ops.BoundPotential_f32(self.unbound_impl, params_flat))


@dataclass
class SummedPotentialGpuImplWrapper_f64(GpuImplWrapper_f64):
    """Handles flattening parameters before passing to kernel to provide a nicer interface"""

    def call_with_params_list(self, conf: Conf, params: Sequence[Params], box: Box) -> float:
        assert len(params) == 0 or isinstance(params[0], (np.ndarray, Array))
        params_flat = jnp.concatenate([ps.reshape(-1) for ps in params])  # type: ignore
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params_flat, box)
        return cast(float, res)

    def bind_params_list(self, params: Sequence[Params]) -> BoundGpuImplWrapper_f64:
        assert len(params) == 0 or isinstance(params[0], (np.ndarray, Array))
        params_flat = np.concatenate([ps.reshape(-1) for ps in params])  # type: ignore
        return BoundGpuImplWrapper_f64(custom_ops.BoundPotential_f64(self.unbound_impl, params_flat))


@dataclass
class FanoutSummedPotential(Potential):
    potentials: Sequence[Potential]
    parallel: bool = True

    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array:
        assert isinstance(params, (np.ndarray, Array))
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

    def combine(self, other_pot):
        raise NotImplementedError("Can't combine")
