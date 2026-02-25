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
#include "../gpu_utils.cuh"
#include "k_fixed_point.cuh"

namespace tmd {

template <typename RealType>
inline __device__ RealType dot_product(const RealType a[3],
                                       const RealType b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename RealType>
inline __device__ void cross_product(const RealType a[3], const RealType b[3],
                                     RealType c[3]) {
  // these extra __dmul_rn calls are needed to preserve bitwise
  // anticommutativity i.e. cross(a,b) is bitwise identical to -cross(b,a)
  // except in the sign-bit
  c[0] = rmul_rn(a[1], b[2]) - rmul_rn(a[2], b[1]);
  c[1] = rmul_rn(a[2], b[0]) - rmul_rn(a[0], b[2]);
  c[2] = rmul_rn(a[0], b[1]) - rmul_rn(a[1], b[0]);
}

template <typename RealType, int D, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_periodic_torsion(
    const int N,                          // Atoms in system
    const int T,                          // number of bonds
    const RealType *__restrict__ coords,  // [n, 3]
    const RealType *__restrict__ box,     // [3, 3]
    const RealType *__restrict__ params,  // [p, 3]
    const int *__restrict__ torsion_idxs, // [b, 4]
    const int *__restrict__ system_idxs,  // [b],
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  const auto t_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (t_idx >= T) {
    return;
  }

  const int system_idx = system_idxs[t_idx];
  const int coord_offset = system_idx * N;
  const int box_offset = system_idx * 9;

  int i_idx = torsion_idxs[t_idx * 4 + 0] + coord_offset;
  int j_idx = torsion_idxs[t_idx * 4 + 1] + coord_offset;
  int k_idx = torsion_idxs[t_idx * 4 + 2] + coord_offset;
  int l_idx = torsion_idxs[t_idx * 4 + 3] + coord_offset;

  RealType rij[D];
  RealType rkj[D];
  RealType rkl[D];

  RealType rkj_norm_square = 0;

// (todo) cap to three dims, while keeping stride at 4
#pragma unroll D
  for (int d = 0; d < D; d++) {
    RealType box_dim = box[box_offset + d * 3 + d];
    RealType vij = coords[j_idx * D + d] - coords[i_idx * D + d];
    RealType vkj = coords[j_idx * D + d] - coords[k_idx * D + d];
    RealType vkl = coords[l_idx * D + d] - coords[k_idx * D + d];
    vij -= box_dim * nearbyint(vij / box_dim);
    vkj -= box_dim * nearbyint(vkj / box_dim);
    vkl -= box_dim * nearbyint(vkl / box_dim);
    rij[d] = vij;
    rkj[d] = vkj;
    rkl[d] = vkl;
    rkj_norm_square += vkj * vkj;
  }

  RealType n1[D], n2[D];

  cross_product(rij, rkj, n1);
  cross_product(rkj, rkl, n2);

  RealType n1_norm_square = dot_product(n1, n1);
  RealType n2_norm_square = dot_product(n2, n2);

  RealType n3[D];
  cross_product(n1, n2, n3);

  RealType rij_dot_rkj = dot_product(rij, rkj);
  RealType rkl_dot_rkj = dot_product(rkl, rkj);

  RealType rkj_norm = sqrt(rkj_norm_square);

#pragma unroll D
  for (int d = 0; d < D; d++) {
    rkj[d] /= rkj_norm;
  }

  RealType y = dot_product(n3, rkj);
  RealType x = dot_product(n1, n2);
  RealType angle = atan2(y, x);

  int kt_idx = t_idx * D + 0;
  int phase_idx = t_idx * D + 1;
  int period_idx = t_idx * D + 2;

  RealType kt = params[kt_idx];
  RealType phase = params[phase_idx];
  RealType period = params[period_idx];

  if (COMPUTE_DU_DX) {
    RealType prefactor = kt * sin(period * angle - phase) * period;

    for (int d = 0; d < D; d++) {
      RealType d_angle_dR0 = rkj_norm / n1_norm_square * n1[d];
      RealType d_angle_dR3 = -rkj_norm / n2_norm_square * n2[d];
      // no fma allowed here, breaks commutativity.
      RealType d_angle_dR1 =
          rmul_rn(rij_dot_rkj / rkj_norm_square, d_angle_dR0) -
          rmul_rn(d_angle_dR3, rkl_dot_rkj / rkj_norm_square);

      unsigned long long dangle_di =
          FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR0 * prefactor);
      unsigned long long dangle_dl =
          FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR3 * prefactor);
      atomicAdd(du_dx + i_idx * D + d, dangle_di);
      atomicAdd(du_dx + l_idx * D + d, dangle_dl);
      unsigned long long dangle_dj =
          FLOAT_TO_FIXED_BONDED<RealType>(d_angle_dR1 * prefactor);
      atomicAdd(du_dx + j_idx * D + d, dangle_dj - dangle_di);
      atomicAdd(du_dx + k_idx * D + d, -dangle_dj - dangle_dl);
    }
  }

  if (COMPUTE_DU_DP) {
    RealType du_dkt = 1 + cos(period * angle - phase);
    RealType du_dphase = kt * sin(period * angle - phase);
    RealType du_dperiod = -kt * sin(period * angle - phase) * angle;

    atomicAdd(du_dp + kt_idx, FLOAT_TO_FIXED_BONDED(du_dkt));
    atomicAdd(du_dp + phase_idx, FLOAT_TO_FIXED_BONDED(du_dphase));
    atomicAdd(du_dp + period_idx, FLOAT_TO_FIXED_BONDED(du_dperiod));
  }

  if (COMPUTE_U) {
    u[t_idx] =
        FLOAT_TO_FIXED_ENERGY<RealType>(kt * (1 + cos(period * angle - phase)));
  }
}

} // namespace tmd
