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

from typing import Callable, TypeAlias

import jax
import numpy as np

Array: TypeAlias = jax.Array | np.ndarray
Conf: TypeAlias = Array
# A single batch of parameters
ParameterSet: TypeAlias = Array
# One or more batches of parameters
Params: TypeAlias = ParameterSet | list[ParameterSet]
Box: TypeAlias = Array

PotentialFxn = Callable[[Conf, Params, Box], float | jax.Array]
BoundPotentialFxn = Callable[[Conf, Box], float | jax.Array]
