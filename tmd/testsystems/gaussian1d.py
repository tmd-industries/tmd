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

import jax.numpy as jnp
from jax.scipy.stats.norm import logpdf as norm_logpdf
from scipy.stats import norm


def make_gaussian_testsystem():
    """normalized/unnormalized 1D Gaussian with a dependence on lambda and params"""

    def annealed_gaussian_def(lam, params):
        initial_mean, initial_log_sigma = 0.0, 0.0
        target_mean, target_log_sigma = params

        # lam = 0 -> (mean = 0, stddev = 1)
        # lam = 1 -> (mean = target_mean, stddev = target_sigma)
        mean = lam * target_mean - (1 - lam) * initial_mean
        stddev = jnp.exp(lam * target_log_sigma + (1 - lam) * initial_log_sigma)

        return mean, stddev

    def sample(lam, params, n_samples):
        mean, stddev = annealed_gaussian_def(lam, params)
        return norm.rvs(loc=mean, scale=stddev, size=(n_samples, 1))

    def logpdf(x, lam, params):
        mean, stddev = annealed_gaussian_def(lam, params)
        return jnp.sum(norm_logpdf(x, loc=mean, scale=stddev))

    def u_fxn(x, lam, params):
        """unnormalized version of -logpdf"""
        mean, stddev = annealed_gaussian_def(lam, params)
        return jnp.sum(0.5 * ((x - mean) / stddev) ** 2)

    def normalized_u_fxn(x, lam, params):
        return -logpdf(x, lam, params)

    def reduced_free_energy(lam, params):
        mean, stddev = annealed_gaussian_def(lam, params)
        log_z = jnp.log(stddev * jnp.sqrt(2 * jnp.pi))
        return -log_z

    return u_fxn, normalized_u_fxn, sample, reduced_free_energy, annealed_gaussian_def
