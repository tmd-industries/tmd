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

#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"

namespace tmd {

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_harmonic_bond(
    const int N,                              // Atoms in each system
    const int B,                              // number of bonds
    const RealType *__restrict__ coords,      // [K, N, 3]
    const RealType *__restrict__ box,         // [K, 3, 3]
    const RealType *__restrict__ params,      // [B, 2]
    const int *__restrict__ bond_idxs,        // [B, 2]
    const int *__restrict__ bond_system_idxs, // [B],
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  int b_idx = blockDim.x * blockIdx.x + threadIdx.x;

  while (b_idx < B) {
    const int system_idx = bond_system_idxs[b_idx];
    const int coord_offset = system_idx * N;

    int src_idx = bond_idxs[b_idx * 2 + 0];
    int dst_idx = bond_idxs[b_idx * 2 + 1];

    src_idx += coord_offset;
    dst_idx += coord_offset;
    RealType dx[3];
    RealType d2ij = 0;
#pragma unroll
    for (int d = 0; d < 3; d++) {
      RealType delta = coords[src_idx * 3 + d] - coords[dst_idx * 3 + d];
      delta -= box[system_idx * 9 + d * 3 + d] *
               nearbyint(delta / box[system_idx * 9 + d * 3 + d]);
      dx[d] = delta;
      d2ij += delta * delta;
    }

    int kb_idx = b_idx * 2 + 0;
    int b0_idx = b_idx * 2 + 1;

    RealType kb = params[kb_idx];
    RealType b0 = params[b0_idx];

    RealType dij = sqrt(d2ij);
    RealType db = dij - b0;

    if (COMPUTE_DU_DX) {
      const RealType inv_dij = 1 / dij;
#pragma unroll
      for (int d = 0; d < 3; d++) {
        RealType grad_delta = b0 != 0 ? kb * db * dx[d] * inv_dij : kb * dx[d];
        unsigned long long ull_grad =
            FLOAT_TO_FIXED_BONDED<RealType>(grad_delta);
        atomicAdd(du_dx + src_idx * 3 + d, ull_grad);
        atomicAdd(du_dx + dst_idx * 3 + d, -ull_grad);
      }
    }

    if (COMPUTE_DU_DP) {
      atomicAdd(du_dp + kb_idx, FLOAT_TO_FIXED_BONDED(0.5 * db * db));
      atomicAdd(du_dp + b0_idx, FLOAT_TO_FIXED_BONDED(-kb * db));
    }

    if (COMPUTE_U) {
      u[b_idx] = FLOAT_TO_FIXED_ENERGY<RealType>(kb / 2 * db * db);
    }
    b_idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
