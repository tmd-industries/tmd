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

from tmd.fe.lambda_schedule import (
    bisection_lambda_schedule,
    interpolate_pre_optimized_protocol,
    validate_lambda_schedule,
)

pytestmark = [pytest.mark.nocuda]


def test_validate_lambda_schedule():
    """check that assertions fail when they should"""

    # Want to test 2 sizes, the latter is the one currently used in RABFE
    for K in [50, 64]:
        good_lambda_schedule = np.linspace(0, 1, K)
        reversed_schedule = good_lambda_schedule[::-1]
        truncated_schedule = good_lambda_schedule[1:]

        validate_lambda_schedule(good_lambda_schedule, K)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(reversed_schedule, K)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(truncated_schedule, K - 1)

        with pytest.raises(AssertionError):
            validate_lambda_schedule(truncated_schedule, K)


def test_interpolate_pre_optimized_protocol():
    linear = np.linspace(0, 1, 50)
    nonlinear = np.linspace(0, 1, 64) ** 2

    for sched in [linear, nonlinear]:
        # recover ~exactly the initial schedule
        K = len(sched)
        sched_prime = interpolate_pre_optimized_protocol(sched, K)
        assert np.allclose(sched, sched_prime)

        # produce valid protocols when downsampling
        reduced = interpolate_pre_optimized_protocol(sched, K // 2)
        validate_lambda_schedule(reduced, K // 2)


@pytest.mark.parametrize("interval", [(0.0, 1.0), [0.25, 0.5]])
@pytest.mark.parametrize("n_windows", [1, 2, 3, 4, 8, 9, 16, 32, 48, 49])
def test_bisection_lambda_schedule(interval, n_windows):
    if n_windows < 2:
        with pytest.raises(AssertionError):
            bisection_lambda_schedule(n_windows)
        return

    schedule = bisection_lambda_schedule(n_windows, lambda_interval=interval)
    assert schedule[0] == interval[0]
    assert schedule[-1] == interval[1]
    assert len(schedule) <= n_windows + 1
    if len(schedule) >= 3:
        mid_point = interval[0] + (interval[1] - interval[0]) / 2
        assert np.any(np.isclose(schedule, mid_point))
    differences = np.diff(schedule)
    assert np.allclose(differences, differences[0])
