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
from numpy.typing import NDArray
from scipy.stats import kendalltau, pearsonr, spearmanr


def mean_unsigned_error(x: NDArray, y: NDArray) -> float:
    return np.mean(np.abs(x - y))


def root_mean_square_error(x: NDArray, y: NDArray) -> float:
    return np.sqrt(np.mean((x - y) ** 2))


def r_squared(x: NDArray, y: NDArray) -> float:
    return pearsonr(x, y).statistic ** 2


def spearman_rho(x: NDArray, y: NDArray) -> float:
    return spearmanr(x, y).statistic


def kendall_tau(x: NDArray, y: NDArray) -> float:
    return kendalltau(x, y).statistic


def bootstrap_statistic(
    x: NDArray, y: NDArray, statistic: Callable[[NDArray, NDArray], float], n_bootstrap: int = 10_000, seed: int = 2025
) -> NDArray:
    rng = np.random.default_rng(seed)

    indices = rng.integers(0, len(x), (n_bootstrap, len(x)))

    samples = [statistic(sampled_x, sampled_y) for sampled_x, sampled_y in zip(x[indices], y[indices])]
    return np.array(samples)
