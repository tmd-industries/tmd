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

# test the custom_op for rmsd alignment
import numpy as np
from scipy.stats import ortho_group, special_ortho_group

from tmd.lib import custom_ops
from tmd.potentials import rmsd


def test_rmsd_align_proper():
    """Assert that the same optimal alignment is recovered by both
    reference (`rmsd.align_x2_unto_x1`) and CPU (`custom_ops.rmsd_align`),
    for cases where `x1` and `x2` differ by a rigid transformation

    This is specialized to proper rotations sampled from SO(3)
    """

    N = 25

    np.random.seed(2021)

    for _ in range(1000):
        x1 = np.random.rand(N, 3)
        random_t = np.random.rand(3)
        random_R = special_ortho_group.rvs(3)
        x2 = x1 @ random_R + random_t

        assert np.linalg.norm(x2 - x1) > 1e-6

        x2_aligned_reference = rmsd.align_x2_unto_x1(x1, x2)

        # com2 should equal to that of com1
        np.testing.assert_almost_equal(np.mean(x1, axis=0), np.mean(x2_aligned_reference, axis=0))

        assert np.linalg.norm(x2_aligned_reference - x1) < 1e-6

        x2_aligned_test = custom_ops.rmsd_align(x1, x2)

        np.testing.assert_almost_equal(np.mean(x1, axis=0), np.mean(x2_aligned_test, axis=0))

        assert np.linalg.norm(x2_aligned_test - x1) < 1e-6

        np.testing.assert_almost_equal(x2_aligned_reference, x2_aligned_test)


def test_rmsd_align_improper():
    """
    Similar to the _proper() case except that we sample from O(3) and we don't check for exact
    RMSD of zero.
    """
    N = 25

    np.random.seed(2021)

    for _ in range(1000):
        x1 = np.random.rand(N, 3)
        random_t = np.random.rand(3)
        random_R = ortho_group.rvs(3)
        x2 = x1 @ random_R + random_t

        # compute initial rmsd when recentered

        x1_recentered = x1 - np.mean(x1, axis=0)
        x2_recentered = x2 - np.mean(x2, axis=0)

        initial_rmsd = np.linalg.norm(x1_recentered - x2_recentered)

        x2_aligned_reference = rmsd.align_x2_unto_x1(x1, x2)

        # com2 should be equal to that of com1
        np.testing.assert_almost_equal(np.mean(x1, axis=0), np.mean(x2_aligned_reference, axis=0))

        # test that rmsd has decreased

        assert np.linalg.norm(x2_aligned_reference - x1) < initial_rmsd

        # repeat for x2

        x2_aligned_test = custom_ops.rmsd_align(x1, x2)

        np.testing.assert_almost_equal(np.mean(x1, axis=0), np.mean(x2_aligned_test, axis=0))

        assert np.linalg.norm(x2_aligned_test - x1) < initial_rmsd

        np.testing.assert_almost_equal(x2_aligned_reference, x2_aligned_test)
