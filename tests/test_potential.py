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

import numpy as np
import pytest

from tmd.potentials import BoundPotential, HarmonicAngle, HarmonicBond, Potential
from tmd.potentials.potential import get_bound_potential_by_type, get_potential_by_type

pytestmark = [pytest.mark.nogpu]


def test_get_potential_by_type():
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_potential_by_type([], HarmonicBond)

    pots = [HarmonicAngle(idxs=np.array([[0, 1, 2]], dtype=np.int32))]
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_potential_by_type(pots, HarmonicBond)

    pots.append(HarmonicBond(2, idxs=np.array([[0, 1]], dtype=np.int32)))
    bonded = get_potential_by_type(pots, HarmonicBond)
    assert isinstance(bonded, Potential)
    assert isinstance(bonded, HarmonicBond)


def test_get_bound_potential_by_type():
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_bound_potential_by_type([], HarmonicBond)

    bps = [
        BoundPotential(
            potential=HarmonicAngle(idxs=np.array([[0, 1, 2]], dtype=np.int32)), params=np.array([[0.0, 0.0]])
        )
    ]
    with pytest.raises(ValueError, match="Unable to find potential of type"):
        get_bound_potential_by_type(bps, HarmonicBond)

    bps.append(
        BoundPotential(
            potential=HarmonicBond(2, idxs=np.array([[0, 1]], dtype=np.int32)), params=np.array([[0.0, 0.0]])
        )
    )

    bonded = get_bound_potential_by_type(bps, HarmonicBond)
    assert isinstance(bonded, BoundPotential)
    assert isinstance(bonded.potential, HarmonicBond)
