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

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from .types import Box, Conf, Params, PotentialFxn


def summed_potential(
    conf: Conf,
    params: Params,
    box: Box,
    U_fns: Sequence[PotentialFxn],
    shapes: Sequence[tuple],
):
    """Reference implementation of the custom_ops SummedPotential.

    Parameters
    ----------
    conf: array (N, 3)
        conformation

    params: array (P,)
        flattened array of parameters for all potential terms

    box: array (3, 3)
        periodic box

    U_fns: list of functions with signature (conf, params, box) -> energy
        potential terms

    shapes: list of tuple
        shapes of the parameter array input for each potential term (must be same length as U_fns)
    """
    assert isinstance(params, (np.ndarray, Array))
    assert len(U_fns) == len(shapes)
    unflattened_params = unflatten_params(params, shapes)
    return jnp.sum(jnp.array([U_fn(conf, ps, box) for U_fn, ps in zip(U_fns, unflattened_params)]))


def unflatten_params(params: Params, shapes: Sequence[tuple[int, ...]]) -> list[Params]:
    assert isinstance(params, (np.ndarray, Array))
    sizes = [int(np.prod(shape)) for shape in shapes]
    assert params.shape == (sum(sizes),)
    split_indices = np.cumsum(sizes)
    return [ps.reshape(shape) for ps, shape in zip(np.split(params, split_indices[:-1]), shapes)]


def fanout_summed_potential(
    conf: Conf,
    params: Params,
    box: Box,
    U_fns: Sequence[PotentialFxn],
):
    """Reference implementation of the custom_ops FanoutSummedPotential.

    Parameters
    ----------
    conf: array (N, 3)
        conformation

    params: array (P,)
        flattened array of parameters shared by each potential term

    box: array (3, 3)
        periodic box

    U_fns: list of functions with signature (conf, params, box) -> energy
        potential terms
    """
    assert isinstance(params, (np.ndarray, Array))
    return jnp.sum(jnp.array([U_fn(conf, ps, box) for U_fn, ps in zip(U_fns, jnp.array(params))]))
