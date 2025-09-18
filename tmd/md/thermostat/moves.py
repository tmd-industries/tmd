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

from tmd.lib import custom_ops
from tmd.md.moves import Move
from tmd.md.states import CoordsVelBox


class UnadjustedLangevinMove(Move):
    def __init__(self, integrator_impl, bound_impls, n_steps=5):
        self.integrator_impl = integrator_impl
        self.bound_impls = bound_impls
        self.n_steps = n_steps

    def move(self, x: CoordsVelBox):
        # note: context creation overhead here is actually very small!
        ctxt = custom_ops.Context_f32(
            x.coords,
            x.velocities,
            x.box,
            self.integrator_impl,
            self.bound_impls,
        )

        # arguments: lambda_schedule, du_dl_interval, x_interval
        _ = ctxt.multiple_steps(self.n_steps, 0)
        x_t = ctxt.get_x_t()
        v_t = ctxt.get_v_t()

        after_nvt = CoordsVelBox(x_t, v_t, x.box.copy())

        return after_nvt
