// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025-2026 Forrest York, Justin Gullingsrud
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

#pragma once

#include "../nonbonded_common.hpp"
#include "k_fixed_point.cuh"

namespace tmd {

// Heuristic derived from DHFR with cutoff=1.2, where ixn_count ~= NR * 2.4
static const double NONBONDED_BLOCKS_TO_ROW_ATOMS_RATIO = 1.2;
static const int NONBONDED_KERNEL_THREADS_PER_BLOCK = 64;

#define PI 3.141592653589793115997963468544185161
#define TWO_OVER_SQRT_PI 1.128379167095512595889238330988549829708

// Reaction field electrostatics:
//   V(r) = q_ij * (1/r + r^2/(2*rc^3) - 3/(2*rc))
// damping_factor = r * V(r) / q_ij = 1 + r^3/(2*rc^3) - 3*r/(2*rc)
// returns dV/dr / q_ij = -1/r^2 + r/rc^3
double __device__ __forceinline__ real_es_factor(double cutoff, double dij,
                                                 double inv_d2ij,
                                                 double &damping_factor) {
  double rc3 = cutoff * cutoff * cutoff;
  double inv_rc3 = rcp_rn(rc3);
  double inv_rc = rcp_rn(cutoff);
  double dij3 = dij * dij * dij;
  damping_factor = 1.0 + 0.5 * dij3 * inv_rc3 - 1.5 * dij * inv_rc;
  return -inv_d2ij + dij * inv_rc3;
}

float __device__ __forceinline__ real_es_factor(float cutoff, float dij,
                                                float inv_d2ij,
                                                float &damping_factor) {
  float rc3 = cutoff * cutoff * cutoff;
  float inv_rc3 = rcp_rn(rc3);
  float inv_rc = rcp_rn(cutoff);
  float dij3 = dij * dij * dij;
  damping_factor = 1.0f + 0.5f * dij3 * inv_rc3 - 1.5f * dij * inv_rc;
  return -inv_d2ij + dij * inv_rc3;
}

// Compute the terms associated with electrostatics.
// This is pulled out into a function to ensure that the same bit values
// are computed to ensure that that the fixed point values are exactly the same
// regardless of where the values are computed.
template <typename RealType, bool COMPUTE_U>
void __device__ __forceinline__ compute_electrostatics(
    const RealType charge_scale, const RealType qi, const RealType qj,
    const RealType d2ij, const RealType cutoff, RealType &dij,
    RealType &inv_dij, RealType &inv_d2ij, RealType &damping_factor,
    RealType &es_prefactor, RealType &u) {
  inv_dij = rsqrt(d2ij);

  dij = d2ij * inv_dij;
  inv_d2ij = inv_dij * inv_dij;

  RealType qij = qi * qj;
  es_prefactor = charge_scale * qij * inv_dij *
                 real_es_factor(cutoff, dij, inv_d2ij, damping_factor);

  if (COMPUTE_U) {
    u = charge_scale * qij * inv_dij * damping_factor;
  }
}

// Handles the computation related to the LJ terms.
// This is pulled out into a function to ensure that the same bit values
// are computed to ensure that that the fixed point values are exactly the same
// regardless of where the values are computed.
template <typename RealType, bool COMPUTE_U>
void __device__ __forceinline__
compute_lj(RealType lj_scale, RealType eps_i, RealType eps_j, RealType sig_i,
           RealType sig_j, RealType inv_dij, RealType inv_d2ij, RealType &u,
           RealType &delta_prefactor, RealType &sig_grad, RealType &eps_grad) {
  RealType eps_ij = eps_i * eps_j;
  RealType sig_ij = sig_i + sig_j;

  RealType sig_inv_dij = sig_ij * inv_dij;
  RealType sig2_inv_d2ij = sig_inv_dij * sig_inv_dij;
  RealType sig4_inv_d4ij = sig2_inv_d2ij * sig2_inv_d2ij;
  RealType sig6_inv_d6ij = sig4_inv_d4ij * sig2_inv_d2ij;
  RealType sig6_inv_d8ij = sig6_inv_d6ij * inv_d2ij;
  RealType sig5_inv_d6ij = sig_ij * sig4_inv_d4ij * inv_d2ij;

  RealType lj_prefactor =
      lj_scale * eps_ij * sig6_inv_d8ij * (sig6_inv_d6ij * 48 - 24);
  if (COMPUTE_U) {
    u += lj_scale * 4 * eps_ij * (sig6_inv_d6ij - 1) * sig6_inv_d6ij;
  }

  delta_prefactor -= lj_prefactor;

  sig_grad = lj_scale * 24 * eps_ij * sig5_inv_d6ij * (2 * sig6_inv_d6ij - 1);
  eps_grad = lj_scale * 4 * (sig6_inv_d6ij - 1) * sig6_inv_d6ij;
}

} // namespace tmd
