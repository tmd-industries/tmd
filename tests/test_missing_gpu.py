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

"""Tests related to tmd when there are no GPUs"""

import numpy as np
import pytest

from tmd.ff import Forcefield
from tmd.lib import custom_ops
from tmd.md import builders

# Run tests in the no-gpu
pytestmark = [pytest.mark.nogpu]


def test_no_gpu_raises_exception():
    ff = Forcefield.load_from_file("smirnoff_2_0_0_sc.py")
    host_config = builders.build_water_system(3.0, ff.water_ff)

    host_fns = host_config.host_system.get_U_fns()

    with pytest.raises(custom_ops.InvalidHardware, match="Invalid Hardware - Code "):
        host_fns[0].to_gpu(np.float32)
