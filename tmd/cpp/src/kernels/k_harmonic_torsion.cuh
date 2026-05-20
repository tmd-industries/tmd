// Copyright 2026, Justin Gullingsrud
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
#include "k_periodic_torsion.cuh" // reuse dot_product / cross_product

namespace tmd {

// Harmonic dihedral potential:
//
//     u(phi) = 0.5 * k * delta^2,    delta = wrap(phi - phi0)  in (-pi, pi]
//
// The geometric chain rule for d phi / d x is identical to that used by
// k_periodic_torsion; only the prefactor and the energy/parameter math
// differ.
template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_harmonic_torsion(
    const int N,                          // atoms per system
    const int T,                          // number of dihedrals
    const RealType *__restrict__ coords,  // [K, N, 3]
    const RealType *__restrict__ box,     // [K, 3, 3]
    const RealType *__restrict__ params,  // [T, 2]  (k, phi0)
    const int *__restrict__ torsion_idxs, // [T, 4]
    const int *__restrict__ system_idxs,  // [T]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  constexpr int D = 3;

  const auto t_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (t_idx >= T) {
    return;
  }

  const int system_idx = system_idxs[t_idx];
  const int coord_offset = system_idx * N;
  const int box_offset = system_idx * 9;

  const int i_idx = torsion_idxs[t_idx * 4 + 0] + coord_offset;
  const int j_idx = torsion_idxs[t_idx * 4 + 1] + coord_offset;
  const int k_idx = torsion_idxs[t_idx * 4 + 2] + coord_offset;
  const int l_idx = torsion_idxs[t_idx * 4 + 3] + coord_offset;

  RealType rij[D];
  RealType rkj[D];
  RealType rkl[D];

  RealType rkj_norm_square = 0;

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

  const int k_param_idx = t_idx * 2 + 0;
  const int phi0_idx = t_idx * 2 + 1;

  RealType k = params[k_param_idx];
  RealType phi0 = params[phi0_idx];

  // wrap (angle - phi0) into (-pi, pi]
  RealType diff = angle - phi0;
  RealType delta = atan2(sin(diff), cos(diff));

  if (COMPUTE_DU_DX) {
    // The `d_angle_dR{0,1,3}` quantities below carry the same sign convention
    // as in k_periodic_torsion. For the periodic form,
    //   u = kt * (1 + cos(period*phi - phase)),  du/dphi = -kt*period*sin(...),
    // and the kernel uses `prefactor = +kt*sin*period` together with
    // `du_dx += prefactor * d_angle_dR0`. Consistency therefore requires
    //   prefactor = -du/dphi.
    // For the harmonic form, du/dphi = k * delta, so prefactor = -k * delta.
    RealType prefactor = -k * delta;

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
    RealType du_dk = static_cast<RealType>(0.5) * delta * delta;
    RealType du_dphi0 = -k * delta;
    atomicAdd(du_dp + k_param_idx, FLOAT_TO_FIXED_BONDED(du_dk));
    atomicAdd(du_dp + phi0_idx, FLOAT_TO_FIXED_BONDED(du_dphi0));
  }

  if (COMPUTE_U) {
    u[t_idx] = FLOAT_TO_FIXED_ENERGY<RealType>(static_cast<RealType>(0.5) * k *
                                               delta * delta);
  }
}

} // namespace tmd
