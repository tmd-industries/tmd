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
from jax import numpy as jnp

from tmd.potentials.jax_utils import bernoulli_logpdf


def local_resampling_move(
    x,
    target_logpdf_fxn,
    particle_selection_log_prob_fxn,
    mcmc_move,
):
    x = jnp.array(x)
    n_particles = len(x)

    # select particles to be updated
    selection_probs = np.exp(particle_selection_log_prob_fxn(x))
    assert np.min(selection_probs) >= 0 and np.max(selection_probs) <= 1, "selection_probs must be in [0,1]"
    assert selection_probs.shape == (n_particles,), "must compute per-particle selection_probs"
    selection_mask = np.random.rand(n_particles) < selection_probs  # TODO: factor out dependence on global numpy rng?

    # construct restrained version of target
    def restrained_logpdf_fxn(x) -> float:
        log_p_i = particle_selection_log_prob_fxn(x)
        return target_logpdf_fxn(x) + bernoulli_logpdf(log_p_i, selection_mask)

    # construct smaller sampling problem, defined only on selected particles
    def subproblem_logpdf(x_sub) -> float:
        x_full = x.at[selection_mask].set(x_sub)
        return restrained_logpdf_fxn(x_full)

    # apply any valid MCMC move to this subproblem
    x_sub = x[selection_mask]
    x_next_sub, aux = mcmc_move(x_sub, subproblem_logpdf)
    x_next = x.at[selection_mask].set(x_next_sub)

    return x_next, aux
