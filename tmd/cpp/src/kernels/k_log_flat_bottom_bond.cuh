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
#include "k_flat_bottom_bond.cuh"

namespace tmd {

template <typename RealType>
RealType __device__ __forceinline__ stable_log_1_exp_neg(RealType x) {
  const RealType LOG_2 = 0.693147180559945309417232121;
  return x < LOG_2 ? log(-expm1(-x)) : log1p(-exp(-x));
}

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_log_flat_bottom_bond(
    const int N,
    const int B, // number of bonds
    const RealType *__restrict__ coords, const RealType *__restrict__ box,
    const RealType *__restrict__ params, // [B, 3]
    const int *__restrict__ bond_idxs,   // [B, 2]
    const int *__restrict__ system_idxs, // [B]
    const RealType beta, unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  // which bond
  const auto b_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (b_idx >= B) {
    return;
  }

  const int system_idx = system_idxs[b_idx];
  const int coord_offset = system_idx * N;
  const int box_offset = system_idx * 9;

  // which atoms
  const int atoms_per_bond = 2;
  int atoms_idx = b_idx * atoms_per_bond;
  const int src_idx = bond_idxs[atoms_idx + 0] + coord_offset;
  const int dst_idx = bond_idxs[atoms_idx + 1] + coord_offset;

  // look up params
  const int num_params = 3;
  int params_idx = b_idx * num_params;
  int k_idx = params_idx + 0;
  int rmin_idx = params_idx + 1;
  int rmax_idx = params_idx + 2;

  RealType k = params[k_idx];
  RealType rmin = params[rmin_idx];
  RealType rmax = params[rmax_idx];

  // compute common subexpressions involving distance, displacements
  RealType dx[3];
  RealType r2 = 0;
#pragma unroll
  for (int d = 0; d < 3; d++) {
    RealType delta = coords[src_idx * 3 + d] - coords[dst_idx * 3 + d];
    delta -= box[box_offset + (d * 3 + d)] *
             nearbyint(delta / box[box_offset + (d * 3 + d)]);
    dx[d] = delta;
    r2 += delta * delta;
  }
  RealType r = sqrt(r2);

  // branches -> masks
  RealType r_gt_rmax = static_cast<RealType>(r > rmax);
  RealType r_lt_rmin = static_cast<RealType>(r < rmin);

  RealType nrg = compute_flat_bottom_energy(k, r, rmin, rmax);

  if (COMPUTE_U) {
    // RealType u_real = -log(1 - exp(-beta * nrg)) / beta;
    RealType u_real = -stable_log_1_exp_neg(beta * nrg) / beta;
    // Always set the energy buffer value to ensure buffer is initialized
    u[b_idx] = FLOAT_TO_FIXED_ENERGY<RealType>(u_real);
  }

  // If in the flat bottom region, exit after computing energies
  if (r_gt_rmax == 0 && r_lt_rmin == 0) {
    return;
  }
  if (COMPUTE_DU_DP || COMPUTE_DU_DX) {
    RealType prefactor = -exp(-beta * nrg);
    prefactor = prefactor / (static_cast<RealType>(1) + prefactor);

    RealType d_r_min = r - rmin;
    RealType d_r_max = r - rmax;
    RealType d_rmin_3 = d_r_min * d_r_min * d_r_min;
    RealType d_rmax_3 = d_r_max * d_r_max * d_r_max;
    if (COMPUTE_DU_DP) {
      // compute parameter derivatives
      RealType du_dk_real =
          (r_gt_rmax * ((d_rmax_3 * d_r_max) * static_cast<RealType>(0.25))) +
          (r_lt_rmin * ((d_rmin_3 * d_r_min) * static_cast<RealType>(0.25)));
      RealType du_drmin_real = r_lt_rmin * (-k * d_rmin_3);
      RealType du_drmax_real = r_gt_rmax * (-k * d_rmax_3);

      // cast float -> fixed
      unsigned long long du_dk =
          FLOAT_TO_FIXED_BONDED<RealType>(du_dk_real * prefactor);
      unsigned long long du_drmin =
          FLOAT_TO_FIXED_BONDED<RealType>(du_drmin_real * prefactor);
      unsigned long long du_drmax =
          FLOAT_TO_FIXED_BONDED<RealType>(du_drmax_real * prefactor);

      // increment du_dp array
      atomicAdd(du_dp + k_idx, du_dk);
      atomicAdd(du_dp + rmin_idx, du_drmin);
      atomicAdd(du_dp + rmax_idx, du_drmax);
    }

    if (COMPUTE_DU_DX) {
      RealType du_dr = k * ((r_gt_rmax * d_rmax_3) + (r_lt_rmin * d_rmin_3));
      RealType inv_r = 1 / r;
#pragma unroll
      for (int d = 0; d < 3; d++) {
        // compute du/dcoords
        RealType du_dsrc_real = du_dr * dx[d] * inv_r;

        // cast float -> fixed
        unsigned long long du_dsrc =
            FLOAT_TO_FIXED_BONDED<RealType>(prefactor * du_dsrc_real);

        // increment du_dx array
        atomicAdd(du_dx + src_idx * 3 + d, du_dsrc);
        atomicAdd(du_dx + dst_idx * 3 + d, -du_dsrc);
      }
    }
  }
}

} // namespace tmd
