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

import numpy as np
from numpy.typing import NDArray

from tmd.constants import BOLTZ


def sample_velocities(masses: NDArray, temperature: float, seed: int) -> NDArray:
    """Sample Maxwell-Boltzmann velocities ~ N(0, sqrt(kB T / m)

    Parameters
    ----------

    masses:
        Array of masses

    temperature:
        float representing temperature in kelvin

    seed:
        integer to use to use as seed

    Returns
    -------
    (N, 3) velocities array, where N is the length of masses
    """
    n_particles = len(masses)
    spatial_dim = 3

    rng = np.random.default_rng(seed)
    v_unscaled = rng.standard_normal(size=(n_particles, spatial_dim))

    # intended to be consistent with tmd.integrator:langevin_coefficients
    sigma = np.sqrt(BOLTZ * temperature) * np.sqrt(1 / masses)
    v_scaled = v_unscaled * np.expand_dims(sigma, axis=1)

    assert v_scaled.shape == (n_particles, spatial_dim)

    return v_scaled.astype(np.float32)
