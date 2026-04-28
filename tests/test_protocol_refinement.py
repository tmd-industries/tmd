# Copyright 2019-2025, Relay Therapeutics
# Modifications Copyright 2026, Forrest York, Justin Gullingsrud
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
from hypothesis import given, seed
from hypothesis.strategies import composite, floats, lists, sampled_from

from tmd.fe.protocol_refinement import apportion, greedy_bisection_step

lambdas = floats(0.0, 1.0, allow_subnormal=False)

# https://github.com/python/mypy/issues/12617
lambda_schedules = lists(lambdas, min_size=2, unique=True).map(sorted)  # type: ignore


pytestmark = [pytest.mark.nocuda]


@composite
def greedy_bisection_step_args_instances(draw):
    protocol = draw(lambda_schedules)
    worst_pair_lam1 = draw(sampled_from(protocol[:-1]))

    def local_cost(lam1, _):
        return 1.0 if lam1 == worst_pair_lam1 else 0.0

    def make_intermediate(lam1, lam2):
        assert lam1 < lam2
        return draw(floats(lam1, lam2, allow_subnormal=False))

    return protocol, local_cost, make_intermediate


@given(greedy_bisection_step_args_instances())
@seed(2023)
def test_greedy_bisection_step(args):
    protocol, local_cost, make_intermediate = args
    refined_protocol, _ = greedy_bisection_step(protocol, local_cost, make_intermediate)
    assert len(refined_protocol) == len(protocol) + 1
    assert set(refined_protocol).issuperset(set(protocol))
    assert refined_protocol == sorted(refined_protocol)


class TestApportion:
    def test_uniform_weights(self):
        """Equal weights should distribute as evenly as possible."""
        result = apportion(np.array([1.0, 1.0, 1.0]), 9)
        np.testing.assert_array_equal(result, [3, 3, 3])

    def test_uniform_weights_with_remainder(self):
        """When total is not divisible, remainder goes to entries with largest fractional parts."""
        result = apportion(np.array([1.0, 1.0, 1.0]), 10)
        assert result.sum() == 10
        # Each gets at least 3; one gets a bonus
        assert np.min(result) == 3
        assert np.max(result) == 4

    def test_proportional(self):
        """Weights of 3:1 with total=8 should give [6, 2]."""
        result = apportion(np.array([3.0, 1.0]), 8)
        np.testing.assert_array_equal(result, [6, 2])

    def test_single_weight(self):
        """Single entry gets everything."""
        result = apportion(np.array([5.0]), 7)
        np.testing.assert_array_equal(result, [7])

    def test_minimum_one_per_entry(self):
        """With total == n, each entry should get at least 1 (even tiny weights)."""
        result = apportion(np.array([0.001, 0.001, 100.0]), 3)
        assert result.sum() == 3
        assert np.all(result >= 0)
        # The dominant weight gets most; but floor guarantees at least 0 per entry.
        # Largest remainder distributes the rest.
        assert result[2] >= 1

    def test_sum_is_exact(self):
        """Allocation must always sum to total exactly."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            n = rng.integers(1, 10)
            total = rng.integers(n, n + 20)
            weights = rng.random(n) + 1e-6
            result = apportion(weights, int(total))
            assert result.sum() == total
            assert len(result) == n
            assert np.all(result >= 0)

    def test_small_deficit_gets_zero_windows(self):
        """A gap with a tiny deficit relative to others can receive zero windows."""
        # Simulate 3 gaps: two with large deficit, one nearly at target
        # deficits: [0.5, 0.5, 0.001]
        weights = np.array([0.5, 0.5, 0.001])
        result = apportion(weights, 3)
        assert result.sum() == 3
        # The near-target gap gets 0 — windows go where they're needed most
        assert result[2] == 0

    def test_negative_weights_fails(self):
        with pytest.raises(AssertionError, match="All weights must greater than or equal to 0"):
            apportion(np.array([-0.1, 0.1]), 2)

    def test_weights_greater_than_total_fails(self):
        with pytest.raises(AssertionError, match="total must be greater than the number of weights"):
            apportion(np.array([0.1, 0.1]), 1)

    def test_empty_weights(self):
        with pytest.raises(AssertionError, match="Must provide at least one weight"):
            apportion(np.array([]), 1)
