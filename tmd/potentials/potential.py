# Copyright 2019-2025, Relay Therapeutics
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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from typing import Any, Generic, Optional, TypeVar, cast, overload

import numpy as np
from jax import Array
from numpy.typing import NDArray

from tmd.lib import custom_ops

from . import jax_interface
from .types import Box, Conf, Params

Precision = Any

_P = TypeVar("_P", bound="Potential", covariant=True)


def combine_pot_params(param_a, param_b) -> list:
    if isinstance(param_a, (np.ndarray, Array)):
        param_a = [param_a]
    if isinstance(param_b, (np.ndarray, Array)):
        param_b = [param_b]
    return param_a + param_b


@dataclass
class Potential(ABC):
    @abstractmethod
    def __call__(self, conf: Conf, params: Params, box: Box) -> float | Array: ...

    def bind(self: _P, params: Params) -> "BoundPotential[_P]":
        return BoundPotential(self, params)

    @overload
    def to_gpu(self, precision: type[np.float32]) -> "GpuImplWrapper_f32": ...

    @overload
    def to_gpu(self, precision: type[np.float64]) -> "GpuImplWrapper_f64": ...

    def to_gpu(self, precision: Precision) -> "GpuImplWrapper_f32 | GpuImplWrapper_f64":
        ctor = getattr(custom_ops, self._custom_ops_class_name(precision))
        args = astuple(self)
        impl = ctor(*args)
        if precision == np.float32:
            return GpuImplWrapper_f32(impl)
        elif precision == np.float64:
            return GpuImplWrapper_f64(impl)
        else:
            assert 0

    @classmethod
    def _custom_ops_class_name(cls, precision: Precision) -> str:
        suffix = get_custom_ops_class_name_suffix(precision)
        return f"{cls.__name__}_{suffix}"

    @abstractmethod
    def combine(self, other_pot: _P) -> _P:
        raise NotImplementedError()


@dataclass
class BoundPotential(Generic[_P]):
    potential: _P
    params: Params

    def __call__(self, conf: Conf, box: Box) -> float | Array:
        assert isinstance(self.params, (np.ndarray, Array))
        return self.potential(conf, self.params, box)

    @overload
    def to_gpu(self, precision: type[np.float32]) -> "BoundGpuImplWrapper_f32": ...

    @overload
    def to_gpu(self, precision: type[np.float64]) -> "BoundGpuImplWrapper_f64": ...

    def to_gpu(self, precision: Precision) -> "BoundGpuImplWrapper_f32 | BoundGpuImplWrapper_f64":
        return self.potential.to_gpu(precision).bind(np.asarray(self.params, dtype=precision))

    def combine(self, other_pot: "BoundPotential[_P]") -> "BoundPotential[_P]":
        if not isinstance(other_pot, self.__class__):
            raise TypeError("Must combine with bound potential")
        combined_pot = self.potential.combine(other_pot.potential)
        combined_params = combine_pot_params(self.params, other_pot.params)
        return BoundPotential(combined_pot, combined_params)


@dataclass
class GpuImplWrapper_f32:
    unbound_impl: custom_ops.Potential_f32

    def __call__(self, conf: NDArray, params: NDArray, box: NDArray) -> float:
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params, box)
        return cast(float, res)

    def bind(self, params: NDArray) -> "BoundGpuImplWrapper_f32 | BoundGpuImplWrapper_f64":
        return BoundGpuImplWrapper_f32(custom_ops.BoundPotential_f32(self.unbound_impl, params.astype(np.float32)))


@dataclass
class GpuImplWrapper_f64:
    unbound_impl: custom_ops.Potential_f64

    def __call__(self, conf: NDArray, params: NDArray, box: NDArray) -> float:
        res = jax_interface.call_unbound_impl(self.unbound_impl, conf, params, box)
        return cast(float, res)

    def bind(self, params: NDArray) -> "BoundGpuImplWrapper_f32 | BoundGpuImplWrapper_f64":
        return BoundGpuImplWrapper_f64(custom_ops.BoundPotential_f64(self.unbound_impl, params.astype(np.float64)))


@dataclass
class BoundGpuImplWrapper_f32:
    bound_impl: custom_ops.BoundPotential_f32

    def __call__(self, conf: NDArray, box: NDArray) -> float:
        res = jax_interface.call_bound_impl(self.bound_impl, conf, box)
        return cast(float, res)


@dataclass
class BoundGpuImplWrapper_f64:
    bound_impl: custom_ops.BoundPotential_f64

    def __call__(self, conf: NDArray, box: NDArray) -> float:
        res = jax_interface.call_bound_impl(self.bound_impl, conf, box)
        return cast(float, res)


def get_custom_ops_class_name_suffix(precision: Precision):
    if precision == np.float32:
        return "f32"
    elif precision == np.float64:
        return "f64"
    else:
        raise ValueError("invalid precision")


def get_bound_potential_by_type(bps: Sequence[BoundPotential[_P]], pot_type: type[_P]) -> BoundPotential[_P]:
    """Given a list of bound potentials return the first bound potential with the matching potential type.

    Raises
    ------
        ValueError:
            Unable to find potential with the expected type
    """
    result: Optional[BoundPotential[_P]] = None
    for bp in bps:
        if isinstance(bp.potential, pot_type):
            result = bp
            break
    if result is None:
        raise ValueError(f"Unable to find potential of type: {pot_type}")
    return result


def get_potential_by_type(pots: Sequence[Potential], pot_type: type[_P]) -> _P:
    """Given a list of potentials return the first potential with the matching type.

    Raises
    ------
        ValueError:
            Unable to find potential with the expected type
    """
    result: Optional[_P] = None
    for pot in pots:
        if isinstance(pot, pot_type):
            result = pot
            break
    if result is None:
        raise ValueError(f"Unable to find potential of type: {pot_type}")
    return result
