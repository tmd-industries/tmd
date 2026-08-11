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

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from tmd.lib import custom_ops

# safe to pickle!


@dataclass
class LangevinIntegrator:
    temperature: float
    dt: float
    friction: float
    masses: NDArray[np.float64]
    seed: int

    def impl(self, precision=np.float32):
        klass: type[custom_ops.LangevinIntegrator_f32] | type[custom_ops.LangevinIntegrator_f64] = (
            custom_ops.LangevinIntegrator_f32
        )
        if precision == np.float64:
            klass = custom_ops.LangevinIntegrator_f64
        return klass(
            np.array(self.masses, dtype=precision),
            self.temperature,
            self.dt,
            self.friction,
            self.seed,
        )


@dataclass
class ConstraintGroups:
    groups: list[list[int]]
    distances: list[list[float]]
    water_group_indices: NDArray[np.int_]
    tolerance: float = 1e-5
    max_iter: int = 15

    def __post_init__(self):
        assert len(self.groups) == len(self.distances)
        assert len(set(self.water_group_indices)) == len(self.water_group_indices)
        if len(self.water_group_indices) > 0:
            assert np.min(self.water_group_indices) >= 0
            assert np.max(self.water_group_indices) < len(self.groups)
        for group, dists in zip(self.groups, self.distances):
            assert len(group) > 1
            assert len(group) - 1 == len(dists)

    def sort(self) -> "ConstraintGroups":
        """Sort the constraint groups such that the water groups are at moved to the front of the constraint groups"""
        n_groups = len(self.groups)
        if n_groups == 0:
            return ConstraintGroups([], [], np.array([], dtype=np.int_))

        non_water_indices = np.delete(np.arange(n_groups, dtype=np.int_), self.water_group_indices)

        # Sort the water indices, probably redundant
        water_indices = np.sort(self.water_group_indices)

        sorted_indices = np.concatenate([water_indices, non_water_indices], dtype=np.int_)

        sorted_groups = [self.groups[i] for i in sorted_indices]
        sorted_distances = [self.distances[i] for i in sorted_indices]
        sorted_water_indices = np.arange(len(water_indices), dtype=np.int_)

        return ConstraintGroups(sorted_groups, sorted_distances, sorted_water_indices, self.tolerance, self.max_iter)

    def concatenate(self, other: "ConstraintGroups") -> "ConstraintGroups":
        if not isinstance(other, ConstraintGroups):
            raise TypeError("Can only concatenate other constraint groups")
        num_starting_groups = len(self.groups)
        new_group = ConstraintGroups(
            self.groups + other.groups,
            self.distances + other.distances,
            np.concatenate([self.water_group_indices, other.water_group_indices + num_starting_groups]),
            tolerance=max(self.tolerance, other.tolerance),
            max_iter=max(self.max_iter, other.max_iter),
        )
        return new_group.sort()

    def impl(self, masses: NDArray, precision=np.float32):
        klass: type[custom_ops.ConstraintGroups_f32] | type[custom_ops.ConstraintGroups_f64] = (
            custom_ops.ConstraintGroups_f32
        )
        if precision == np.float64:
            klass = custom_ops.ConstraintGroups_f64
        # Water indices are currently dropped
        return klass(
            np.array(masses, dtype=precision),
            self.groups,
            self.distances,
            self.max_iter,
            self.tolerance,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ConstraintGroups):
            raise TypeError("Can't only compare to other constraint groups")

        def group_sets(obj):
            heavy_to_sets = {}
            for group in obj.groups:
                heavy_to_sets[group[0]] = set(group[1:])
            return heavy_to_sets

        self_groups = group_sets(self)
        comp_groups = group_sets(other)
        return all(comp_groups.get(anchor, set()) == light_group for anchor, light_group in self_groups.items())


@dataclass
class ConstrainedLangevinIntegrator:
    temperature: float
    dt: float
    friction: float
    masses: NDArray[np.float64]
    seed: int
    constraints: ConstraintGroups

    def impl(self, precision=np.float32):
        klass: (
            type[custom_ops.ConstrainedLangevinIntegrator_f32] | type[custom_ops.ConstrainedLangevinIntegrator_f64]
        ) = custom_ops.ConstrainedLangevinIntegrator_f32
        if precision == np.float64:
            klass = custom_ops.ConstrainedLangevinIntegrator_f64
        return klass(
            np.array(self.masses, dtype=precision),
            self.temperature,
            self.dt,
            self.friction,
            self.seed,
            self.constraints.impl(self.masses, precision),
        )


@dataclass
class VelocityVerletIntegrator:
    dt: float
    masses: NDArray[np.float64]

    cbs: NDArray[np.float64] = field(init=False)

    def __post_init__(self):
        cb = self.dt / np.asarray(self.masses)
        cb *= -1
        self.cbs = cb

    def impl(self, precision=np.float32):
        klass: type[custom_ops.VelocityVerletIntegrator_f32] | type[custom_ops.VelocityVerletIntegrator_f64] = (
            custom_ops.VelocityVerletIntegrator_f32
        )
        if precision == np.float64:
            klass = custom_ops.VelocityVerletIntegrator_f64
        return klass(self.dt, self.cbs.astype(precision))


@dataclass
class MonteCarloBarostat:
    N: int
    pressure: float
    temperature: float
    group_idxs: Any  # TODO: address mixed convention for type of group_idxs
    interval: int
    seed: int
    adaptive_scaling_enabled: bool = True
    initial_volume_scale_factor: Optional[float] = None

    def impl(self, bound_potentials, precision=np.float32):
        klass: type[custom_ops.MonteCarloBarostat_f32] | type[custom_ops.MonteCarloBarostat_f64] = (
            custom_ops.MonteCarloBarostat_f32
        )
        if precision == np.float64:
            klass = custom_ops.MonteCarloBarostat_f64
        return klass(
            self.N,
            self.pressure,
            self.temperature,
            self.group_idxs,
            self.interval,
            bound_potentials,
            self.seed,
            self.adaptive_scaling_enabled,
            self.initial_volume_scale_factor or 0.0,  # 0.0 is a special value meaning "use 1% of initial box volume"
        )


# wrapper to do automatic casting
def Context(
    x0, v0, box, integrator, bps, movers=None, precision=np.float32
) -> custom_ops.Context_f32 | custom_ops.Context_f64:
    x0 = x0.astype(precision)
    v0 = v0.astype(precision)
    box = box.astype(precision)

    klass: type[custom_ops.Context_f32] | type[custom_ops.Context_f64] = custom_ops.Context_f32
    if precision == np.float64:
        klass = custom_ops.Context_f64
    return klass(x0, v0, box, integrator, bps, movers)
