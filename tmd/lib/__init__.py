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
