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

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_harmonic_angle(const int N,
                                 const int A, // number of angles
                                 const RealType *__restrict__ coords, // [N, 3]
                                 const RealType *__restrict__ box,    // [3, 3]
                                 const RealType *__restrict__ params, // [P, 3]
                                 const int *__restrict__ angle_idxs,  // [A, 3]
                                 const int *__restrict__ system_idxs, // [A]
                                 unsigned long long *__restrict__ du_dx,
                                 unsigned long long *__restrict__ du_dp,
                                 __int128 *__restrict__ u) {

  const auto a_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (a_idx >= A) {
    return;
  }

  const int system_idx = system_idxs[a_idx];
  const int coord_offset = system_idx * N;

  const int i_idx = angle_idxs[a_idx * 3 + 0] + coord_offset;
  const int j_idx = angle_idxs[a_idx * 3 + 1] + coord_offset;
  const int k_idx = angle_idxs[a_idx * 3 + 2] + coord_offset;

  const int ka_idx = a_idx * 3 + 0;
  const int a0_idx = a_idx * 3 + 1;
  const int eps_idx = a_idx * 3 + 2;

  RealType ka = params[ka_idx];
  RealType a0 = params[a0_idx];
  RealType eps = params[eps_idx];

  RealType rji[4];  // vector from j to i
  RealType rjk[4];  // vector from j to k
  RealType nji = 0; // initialize your summed variables!
  RealType njk = 0; // initialize your summed variables!
  // first pass, compute the norms
  for (int d = 0; d < 3; d++) {
    RealType box_dim = box[system_idx * 9 + d * 3 + d];
    RealType vji = coords[i_idx * 3 + d] - coords[j_idx * 3 + d];
    RealType vjk = coords[k_idx * 3 + d] - coords[j_idx * 3 + d];
    vji -= box_dim * nearbyint(vji / box_dim);
    vjk -= box_dim * nearbyint(vjk / box_dim);
    rji[d] = vji;
    rjk[d] = vjk;
    nji += vji * vji;
    njk += vjk * vjk;
  }

  rji[3] = eps;
  rjk[3] = eps;

  nji = sqrt(nji + eps * eps);
  njk = sqrt(njk + eps * eps);

  RealType hi = 0;
  RealType lo = 0;

  // second pass, compute the hi/lo values
  for (int d = 0; d < 4; d++) {
    RealType vji = rji[d];
    RealType vjk = rjk[d];
    // rsub/radds are used to maintain bitwise reversibility wrt i and k
    RealType a = rsub_rn(njk * vji, nji * vjk);
    RealType b = radd_rn(njk * vji, nji * vjk);
    hi += a * a;
    lo += b * b;
  }

  RealType y = sqrt(hi);
  RealType x = sqrt(lo);
  RealType angle = 2 * atan2(y, x);
  RealType delta = angle - a0;

  if (COMPUTE_DU_DX || COMPUTE_DU_DP) {
    auto a = rji;
    auto b = rjk;
    RealType a_norm = nji;
    RealType b_norm = njk;

    // use the identity: a x (b x c) = b(a.c) - c(a.b)
    RealType a_dot_b = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    RealType a_dot_a = a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3];
    RealType b_dot_b = b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3];

    RealType aab[4];
    RealType bba[4];

    for (int d = 0; d < 4; d++) {
      aab[d] = a[d] * a_dot_b - b[d] * a_dot_a;
      bba[d] = b[d] * a_dot_b - a[d] * b_dot_b;
    }

    RealType aab_norm = sqrt(aab[0] * aab[0] + aab[1] * aab[1] +
                             aab[2] * aab[2] + aab[3] * aab[3]);
    RealType bba_norm = sqrt(bba[0] * bba[0] + bba[1] * bba[1] +
                             bba[2] * bba[2] + bba[3] * bba[3]);
    RealType prefactor = ka * delta;

    RealType coeff_i = prefactor * (1 / a_norm);
    RealType coeff_k = prefactor * (1 / b_norm);

    if (COMPUTE_DU_DX) {
      RealType grad_i[3];
      RealType grad_k[3];

      for (int d = 0; d < 3; d++) {
        grad_i[d] = (aab_norm == 0) ? 0 : aab[d] / aab_norm;
        grad_k[d] = (bba_norm == 0) ? 0 : bba[d] / bba_norm;
      }

      RealType dx_i = coeff_i * grad_i[0];
      RealType dy_i = coeff_i * grad_i[1];
      RealType dz_i = coeff_i * grad_i[2];

      unsigned long long gx_i = FLOAT_TO_FIXED_BONDED<RealType>(dx_i);
      unsigned long long gy_i = FLOAT_TO_FIXED_BONDED<RealType>(dy_i);
      unsigned long long gz_i = FLOAT_TO_FIXED_BONDED<RealType>(dz_i);

      atomicAdd(du_dx + i_idx * 3 + 0, gx_i);
      atomicAdd(du_dx + i_idx * 3 + 1, gy_i);
      atomicAdd(du_dx + i_idx * 3 + 2, gz_i);

      RealType dx_k = coeff_k * grad_k[0];
      RealType dy_k = coeff_k * grad_k[1];
      RealType dz_k = coeff_k * grad_k[2];

      unsigned long long gx_k = FLOAT_TO_FIXED_BONDED<RealType>(dx_k);
      unsigned long long gy_k = FLOAT_TO_FIXED_BONDED<RealType>(dy_k);
      unsigned long long gz_k = FLOAT_TO_FIXED_BONDED<RealType>(dz_k);

      atomicAdd(du_dx + k_idx * 3 + 0, gx_k);
      atomicAdd(du_dx + k_idx * 3 + 1, gy_k);
      atomicAdd(du_dx + k_idx * 3 + 2, gz_k);

      atomicAdd(du_dx + j_idx * 3 + 0, -gx_i - gx_k);
      atomicAdd(du_dx + j_idx * 3 + 1, -gy_i - gy_k);
      atomicAdd(du_dx + j_idx * 3 + 2, -gz_i - gz_k);
    }

    if (COMPUTE_DU_DP) {
      RealType dka_grad = delta * delta / 2;
      atomicAdd(du_dp + ka_idx, FLOAT_TO_FIXED_BONDED(dka_grad));

      RealType da0_grad = -delta * ka;
      atomicAdd(du_dp + a0_idx, FLOAT_TO_FIXED_BONDED(da0_grad));

      RealType eps0_grad = (aab_norm == 0) ? 0 : coeff_i * aab[3] / aab_norm;
      RealType eps1_grad = (bba_norm == 0) ? 0 : coeff_k * bba[3] / bba_norm;
      atomicAdd(du_dp + eps_idx, FLOAT_TO_FIXED_BONDED(eps0_grad + eps1_grad));
    }
  }

  if (COMPUTE_U) {
    u[a_idx] = FLOAT_TO_FIXED_ENERGY<RealType>((ka / 2) * delta * delta);
  }
}

} // namespace tmd
