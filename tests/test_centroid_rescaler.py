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
import pytest

from tmd.md.barostat.moves import CentroidRescaler
from tmd.md.barostat.utils import compute_intramolecular_distances

np.random.seed(2021)

pytestmark = [pytest.mark.nocuda]


def _generate_random_instance():
    # randomly generate point set of size between 50 and 1000
    n_particles = np.random.randint(50, 1000)
    particle_inds = np.arange(n_particles)

    # randomly generate group_inds with group sizes between 1 and 10
    group_inds = []
    np.random.shuffle(particle_inds)
    i = 0
    while i < len(particle_inds):
        j = min(n_particles, i + np.random.randint(1, 10))
        group_inds.append(np.array(particle_inds[i:j]))
        i = j

    # randomly generate coords
    coords = np.array(np.random.randn(n_particles, 3))

    return coords, group_inds


def test_null_rescaling(n_instances=10):
    """scaling by a factor of 1.0x shouldn't change coordinates"""
    for _ in range(n_instances):
        coords, group_inds = _generate_random_instance()
        center = np.random.randn(3)

        rescaler = CentroidRescaler(group_inds)
        coords_prime = rescaler.scale_centroids(coords, center, 1.0)

        np.testing.assert_allclose(coords_prime, coords)


def test_intramolecular_distance(n_instances=10):
    """Test that applying a rescaling doesn't change intramolecular distances"""
    for _ in range(n_instances):
        coords, group_inds = _generate_random_instance()
        distances = compute_intramolecular_distances(coords, group_inds)

        center = np.random.randn(3)
        scale = np.random.rand() + 0.5

        rescaler = CentroidRescaler(group_inds)
        coords_prime = rescaler.scale_centroids(coords, center, scale)
        distances_prime = compute_intramolecular_distances(coords_prime, group_inds)

        np.testing.assert_allclose(np.hstack(distances_prime), np.hstack(distances))


def test_compute_centroids(n_instances=10):
    """test that CentroidRescaler's compute_centroids agrees with _slow_compute_centroids
    on random instances of varying size"""

    for _ in range(n_instances):
        coords, group_inds = _generate_random_instance()

        # assert compute_centroids agrees with _slow_compute_centroids
        rescaler = CentroidRescaler(group_inds)
        fast_centroids = rescaler.compute_centroids(coords)
        slow_centroids = rescaler._slow_compute_centroids(coords)
        np.testing.assert_array_almost_equal(slow_centroids, fast_centroids)
