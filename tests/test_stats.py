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

from tmd.stats import mean_unsigned_error, r_squared, root_mean_square_error

# Plotting code should not depend on CUDA
pytestmark = [pytest.mark.nocuda]


def test_mean_unsigned_error():
    rng = np.random.default_rng(2025)
    x = rng.normal(0, 0.5, size=10000)

    y = rng.normal(1, 0.5, size=10000)

    np.testing.assert_allclose(mean_unsigned_error(x, y), 1.0, atol=0.1)


def test_root_mean_square_error():
    rng = np.random.default_rng(2025)
    x = rng.normal(0, 0.5, size=10000)

    y = rng.normal(1, 0.5, size=10000)

    np.testing.assert_allclose(root_mean_square_error(x, y), 1.0, atol=0.3)

    # Should be strictly greater than mean unsigned error, by being more sensitive to outliers
    assert mean_unsigned_error(x, y) < root_mean_square_error(x, y)


def test_r_squared():
    x = np.array([0, 1])
    y = np.array([0, 1])
    assert r_squared(x, y) == 1.0

    x = np.array([1, 0])
    y = np.array([0, 1])
    assert r_squared(x, y) == 1.0

    x = np.array([0, 0])
    y = np.array([0, 1])
    assert not np.isfinite(r_squared(x, y))

    rng = np.random.default_rng(2025)
    x = rng.uniform(0, 100, size=100)
    y = rng.uniform(0, 100, size=100)
    assert 0.0 <= r_squared(x, y) <= 1.0
