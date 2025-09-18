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

from typing import Callable

import numpy as np
import pytest
from scipy.stats import ks_2samp

from tmd.md.moves import MonteCarloMove

pytestmark = [pytest.mark.nocuda]


class RWMH1D(MonteCarloMove[float]):
    def __init__(self, log_q: Callable[[float], float], proposal_radius: float):
        super().__init__()
        self.log_q = log_q
        self.proposal_radius = proposal_radius

    def propose(self, x: float) -> tuple[float, float]:
        x_prop = np.random.normal(x, self.proposal_radius)
        log_q_diff = self.log_q(x_prop) - self.log_q(x)
        log_acceptance_probability = np.minimum(log_q_diff, 0.0)
        return x_prop, log_acceptance_probability


@pytest.mark.parametrize("seed", [2023, 2024, 2025])
@pytest.mark.parametrize("dist", ["uniform", "normal"])
def test_random_walk_metropolis_hastings(dist, seed):
    """Test sampling from a 1-d normal distribution using Random Walk Metropolis-Hastings"""

    np.random.seed(seed)
    n_samples = 100_000
    dx = 0.1

    # estimate autocorrelation time, number of independent samples
    tau = round(1 / dx**2)
    n_independent_samples = n_samples // tau - 1

    log_q_offset = np.random.uniform(-1.0, 1.0)  # arbitrary offset added to log_q

    if dist == "normal":
        log_q = lambda x: -(x**2) / 2 + log_q_offset
        target_samples = np.random.normal(0, 1, size=(n_independent_samples,))
    else:
        log_q = lambda x: log_q_offset if -1 < x < 1 else -float("inf")
        target_samples = np.random.uniform(-1, 1, size=(n_independent_samples,))

    sampler = RWMH1D(log_q, dx)
    x_0 = np.random.uniform(-1.0, 1.0)
    rw_samples = sampler.sample_chain(x_0, n_samples)

    decorrelated_rw_samples = rw_samples[tau::tau]

    _, pvalue = ks_2samp(decorrelated_rw_samples, target_samples)

    assert pvalue >= 0.01
