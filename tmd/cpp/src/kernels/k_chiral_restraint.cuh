// Copyright 2019-2025, Relay Therapeutics
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
#include "chiral_utils.cuh"
#include "k_fixed_point.cuh"

namespace tmd {

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_chiral_atom_restraint(
    const int N,
    const int R,                         // number of restraints
    const RealType *__restrict__ coords, // [N, 3]
    const RealType *__restrict__ params, // [R]
    const int *__restrict__ idxs,        // [R, 4]
    const int *__restrict__ system_idxs, // [R]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  const auto r_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (r_idx >= R) {
    return;
  }

  const int coord_offset = system_idxs[r_idx] * N;

  const int xc_idx = idxs[r_idx * 4 + 0] + coord_offset;
  const int x1_idx = idxs[r_idx * 4 + 1] + coord_offset;
  const int x2_idx = idxs[r_idx * 4 + 2] + coord_offset;
  const int x3_idx = idxs[r_idx * 4 + 3] + coord_offset;

  // static_casts are needed to prevent compiler from complaining from
  // double->float
  Vector<RealType> xc(
      {coords[xc_idx * 3 + 0], coords[xc_idx * 3 + 1], coords[xc_idx * 3 + 2]});
  Vector<RealType> x1(
      {coords[x1_idx * 3 + 0], coords[x1_idx * 3 + 1], coords[x1_idx * 3 + 2]});
  Vector<RealType> x2(
      {coords[x2_idx * 3 + 0], coords[x2_idx * 3 + 1], coords[x2_idx * 3 + 2]});
  Vector<RealType> x3(
      {coords[x3_idx * 3 + 0], coords[x3_idx * 3 + 1], coords[x3_idx * 3 + 2]});

  Vector<RealType> xc_grad;
  Vector<RealType> x1_grad;
  Vector<RealType> x2_grad;
  Vector<RealType> x3_grad;

  RealType vol;
  pyramidal_vol_and_grad(xc, x1, x2, x3, vol, xc_grad, x1_grad, x2_grad,
                         x3_grad);
  RealType k_restr = params[r_idx];

  // Always set the energies to avoid getting uninitialized values
  if (COMPUTE_U) {
    u[r_idx] = vol > static_cast<RealType>(0.0)
                   ? FLOAT_TO_FIXED_BONDED<RealType>(k_restr * vol * vol)
                   : 0;
  }

  if (k_restr == 0) {
    return;
  }

  if (vol <= 0) {
    return;
  }

  if (COMPUTE_DU_DX) {
    RealType prefactor = 2 * k_restr * vol;

    atomicAdd(du_dx + xc_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(xc_grad.x * prefactor));
    atomicAdd(du_dx + xc_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(xc_grad.y * prefactor));
    atomicAdd(du_dx + xc_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(xc_grad.z * prefactor));

    atomicAdd(du_dx + x1_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x1_grad.x * prefactor));
    atomicAdd(du_dx + x1_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x1_grad.y * prefactor));
    atomicAdd(du_dx + x1_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x1_grad.z * prefactor));

    atomicAdd(du_dx + x2_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x2_grad.x * prefactor));
    atomicAdd(du_dx + x2_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x2_grad.y * prefactor));
    atomicAdd(du_dx + x2_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x2_grad.z * prefactor));

    atomicAdd(du_dx + x3_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x3_grad.x * prefactor));
    atomicAdd(du_dx + x3_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x3_grad.y * prefactor));
    atomicAdd(du_dx + x3_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x3_grad.z * prefactor));
  }

  if (COMPUTE_DU_DP) {
    atomicAdd(du_dp + r_idx, FLOAT_TO_FIXED_BONDED<RealType>(vol * vol));
  }
}

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_chiral_bond_restraint(
    const int R, // number of restraints
    const RealType *__restrict__ coords,
    const RealType *__restrict__ params, // [R]
    const int *__restrict__ idxs,        // [R, 2]
    const int *__restrict__ signs,       // [R]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  const auto r_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (r_idx >= R) {
    return;
  }

  int x0_idx = idxs[r_idx * 4 + 0];
  int x1_idx = idxs[r_idx * 4 + 1];
  int x2_idx = idxs[r_idx * 4 + 2];
  int x3_idx = idxs[r_idx * 4 + 3];

  // static_casts are needed to prevent compiler from complaining from
  // double->float
  Vector<RealType> x0(
      {coords[x0_idx * 3 + 0], coords[x0_idx * 3 + 1], coords[x0_idx * 3 + 2]});
  Vector<RealType> x1(
      {coords[x1_idx * 3 + 0], coords[x1_idx * 3 + 1], coords[x1_idx * 3 + 2]});
  Vector<RealType> x2(
      {coords[x2_idx * 3 + 0], coords[x2_idx * 3 + 1], coords[x2_idx * 3 + 2]});
  Vector<RealType> x3(
      {coords[x3_idx * 3 + 0], coords[x3_idx * 3 + 1], coords[x3_idx * 3 + 2]});

  Vector<RealType> x0_grad;
  Vector<RealType> x1_grad;
  Vector<RealType> x2_grad;
  Vector<RealType> x3_grad;

  RealType vol;
  RealType k_restr = params[r_idx];

  int sign = signs[r_idx];

  torsion_vol_and_grad(x0, x1, x2, x3, vol, x0_grad, x1_grad, x2_grad, x3_grad);

  // Always set the energies to avoid having to memset
  if (COMPUTE_U) {
    u[r_idx] = sign * vol > 0
                   ? FLOAT_TO_FIXED_BONDED<RealType>(k_restr * vol * vol)
                   : 0;
  }

  if (k_restr == 0) {
    return;
  }

  if (sign * vol <= 0) {
    return;
  }

  if (COMPUTE_DU_DX) {
    RealType prefactor = 2 * k_restr * vol;

    atomicAdd(du_dx + x0_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x0_grad.x * prefactor));
    atomicAdd(du_dx + x0_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x0_grad.y * prefactor));
    atomicAdd(du_dx + x0_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x0_grad.z * prefactor));

    atomicAdd(du_dx + x1_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x1_grad.x * prefactor));
    atomicAdd(du_dx + x1_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x1_grad.y * prefactor));
    atomicAdd(du_dx + x1_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x1_grad.z * prefactor));

    atomicAdd(du_dx + x2_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x2_grad.x * prefactor));
    atomicAdd(du_dx + x2_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x2_grad.y * prefactor));
    atomicAdd(du_dx + x2_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x2_grad.z * prefactor));

    atomicAdd(du_dx + x3_idx * 3 + 0,
              FLOAT_TO_FIXED_BONDED<RealType>(x3_grad.x * prefactor));
    atomicAdd(du_dx + x3_idx * 3 + 1,
              FLOAT_TO_FIXED_BONDED<RealType>(x3_grad.y * prefactor));
    atomicAdd(du_dx + x3_idx * 3 + 2,
              FLOAT_TO_FIXED_BONDED<RealType>(x3_grad.z * prefactor));
  }

  if (COMPUTE_DU_DP) {
    atomicAdd(du_dp + r_idx, FLOAT_TO_FIXED_BONDED<RealType>(vol * vol));
  }
}

} // namespace tmd
