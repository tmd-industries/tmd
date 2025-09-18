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
from jax import Array
from jax.typing import ArrayLike

from tmd.lib import custom_ops


def fixed_to_float(v: ArrayLike) -> Array:
    """Meant to imitate the logic of tmd/cpp/src/fixed_point.hpp::FIXED_TO_FLOAT"""
    return jnp.float64(jnp.int64(jnp.uint64(v))) / custom_ops.FIXED_EXPONENT


def float_to_fixed(v: ArrayLike) -> Array:
    """Meant to imitate the logic of tmd/cpp/src/kernels/k_fixed_point.cuh::FLOAT_TO_FIXED"""
    return jnp.uint64(jnp.int64(v * custom_ops.FIXED_EXPONENT))
