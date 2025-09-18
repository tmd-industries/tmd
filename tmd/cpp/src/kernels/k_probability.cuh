// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025 Forrest York
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "k_logsumexp.cuh"
#include <cmath>

namespace tmd {

template <typename RealType>
void __global__ k_compute_log_weights_from_energies(
    const int N, const RealType beta, const __int128 *__restrict__ energies,
    RealType *log_probabilities) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __int128 energy;
  while (idx < N) {
    energy = energies[idx];
    // If the energy is invalid, set the log probability to inf
    log_probabilities[idx] =
        !fixed_point_overflow(energy)
            ? FIXED_ENERGY_TO_FLOAT<RealType>(FLOAT_TO_FIXED_ENERGY<RealType>(
                  beta * FIXED_ENERGY_TO_FLOAT<RealType>(energy)))
            : INFINITY;
    idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
