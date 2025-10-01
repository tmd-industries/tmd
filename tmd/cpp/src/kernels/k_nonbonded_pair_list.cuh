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

#pragma once

// NOTE: The order of associative operations performed by this kernel
// should be identical to that of k_nonbonded. This is to ensure that
// we get exact cancellation when subtracting exclusions computed
// using this kernel.

#include "../fixed_point.hpp"
#include "k_nonbonded_common.cuh"

namespace tmd {

template <bool Negated>
void __device__ __forceinline__ accumulate(unsigned long long *__restrict acc,
                                           unsigned long long val) {
  atomicAdd(acc, Negated ? -val : val);
}

template <typename RealType, bool Negated, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_nonbonded_pair_list(
    const int N,
    const int M, // number of pairs
    const RealType *__restrict__ coords, const RealType *__restrict__ params,
    const RealType *__restrict__ box,
    const int *__restrict__ pair_idxs,   // [M, 2] pair-list of atoms
    const int *__restrict__ system_idxs, // [M]
    const RealType *__restrict__ scales, // [M]
    const RealType beta, const RealType cutoff,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u_buffer) {
  // (ytz): oddly enough the order of atom_i and atom_j
  // seem to not matter. I think this is because distance calculations
  // are bitwise identical in both dij(i, j) and dij(j, i) . However we
  // do need the calculation done for exclusions to perfectly mirror
  // that of the nonbonded kernel itself. Remember that floating points
  // commute but are not associative.

  const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= M) {
    return;
  }

  const int system_idx = system_idxs[pair_idx];
  const int coord_offset = system_idx * N;
  const int box_offset = system_idx * 9;

  const int atom_i_idx = pair_idxs[pair_idx * 2 + 0] + coord_offset;

  RealType ci_x = coords[atom_i_idx * 3 + 0];
  RealType ci_y = coords[atom_i_idx * 3 + 1];
  RealType ci_z = coords[atom_i_idx * 3 + 2];

  unsigned long long gi_x = 0;
  unsigned long long gi_y = 0;
  unsigned long long gi_z = 0;

  int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
  int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
  int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
  int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
  int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

  RealType qi = params[charge_param_idx_i];
  RealType sig_i = params[lj_param_idx_sig_i];
  RealType eps_i = params[lj_param_idx_eps_i];
  RealType w_i = params[w_param_idx_i];

  unsigned long long g_qi = 0;
  unsigned long long g_sigi = 0;
  unsigned long long g_epsi = 0;
  unsigned long long g_wi = 0;

  const int atom_j_idx = pair_idxs[pair_idx * 2 + 1] + coord_offset;

  RealType cj_x = coords[atom_j_idx * 3 + 0];
  RealType cj_y = coords[atom_j_idx * 3 + 1];
  RealType cj_z = coords[atom_j_idx * 3 + 2];

  unsigned long long gj_x = 0;
  unsigned long long gj_y = 0;
  unsigned long long gj_z = 0;

  int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
  int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
  int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
  int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
  int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

  RealType qj = params[charge_param_idx_j];
  RealType sig_j = params[lj_param_idx_sig_j];
  RealType eps_j = params[lj_param_idx_eps_j];
  RealType w_j = params[w_param_idx_j];

  unsigned long long g_qj = 0;
  unsigned long long g_sigj = 0;
  unsigned long long g_epsj = 0;
  unsigned long long g_wj = 0;

  RealType cutoff_squared = cutoff * cutoff;

  RealType charge_scale = scales[pair_idx * 2 + 0];
  RealType lj_scale = scales[pair_idx * 2 + 1];

  RealType box_x = box[box_offset + (0 * 3 + 0)];
  RealType box_y = box[box_offset + (1 * 3 + 1)];
  RealType box_z = box[box_offset + (2 * 3 + 2)];

  RealType inv_box_x = rcp_rn(box_x);
  RealType inv_box_y = rcp_rn(box_y);
  RealType inv_box_z = rcp_rn(box_z);

  RealType delta_x = ci_x - cj_x;
  RealType delta_y = ci_y - cj_y;
  RealType delta_z = ci_z - cj_z;

  delta_x -= box_x * nearbyint(delta_x * inv_box_x);
  delta_y -= box_y * nearbyint(delta_y * inv_box_y);
  delta_z -= box_z * nearbyint(delta_z * inv_box_z);

  RealType delta_w = w_i - w_j;
  RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z +
                  delta_w * delta_w;

  RealType u = 0.0;
  // see note: this must be strictly less than
  if (d2ij < cutoff_squared) {

    RealType ebd;
    RealType delta_prefactor;
    RealType dij;
    RealType inv_dij;
    RealType inv_d2ij;
    compute_electrostatics<RealType, COMPUTE_U>(charge_scale, qi, qj, d2ij,
                                                beta, dij, inv_dij, inv_d2ij,
                                                ebd, delta_prefactor, u);

    // lennard jones force
    if (eps_i != 0 && eps_j != 0) {
      RealType sig_grad;
      RealType eps_grad;
      compute_lj<RealType, COMPUTE_U>(lj_scale, eps_i, eps_j, sig_i, sig_j,
                                      inv_dij, inv_d2ij, u, delta_prefactor,
                                      sig_grad, eps_grad);

      if (COMPUTE_DU_DP) {
        g_sigi +=
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
        g_sigj +=
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
        g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(
            eps_grad * eps_j);
        g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(
            eps_grad * eps_i);
      }
    }

    unsigned long long ull_gx =
        FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_x);
    unsigned long long ull_gy =
        FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_y);
    unsigned long long ull_gz =
        FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_z);
    gi_x += ull_gx;
    gi_y += ull_gy;
    gi_z += ull_gz;
    gj_x -= ull_gx;
    gj_y -= ull_gy;
    gj_z -= ull_gz;

    if (COMPUTE_DU_DX) {
      accumulate<Negated>(du_dx + atom_i_idx * 3 + 0, gi_x);
      accumulate<Negated>(du_dx + atom_i_idx * 3 + 1, gi_y);
      accumulate<Negated>(du_dx + atom_i_idx * 3 + 2, gi_z);

      accumulate<Negated>(du_dx + atom_j_idx * 3 + 0, gj_x);
      accumulate<Negated>(du_dx + atom_j_idx * 3 + 1, gj_y);
      accumulate<Negated>(du_dx + atom_j_idx * 3 + 2, gj_z);
    }

    if (COMPUTE_DU_DP) {
      g_qi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(
          charge_scale * qj * inv_dij * ebd);
      g_qj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(
          charge_scale * qi * inv_dij * ebd);

      g_wi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(
          delta_prefactor * delta_w);
      g_wj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(
          -delta_prefactor * delta_w);

      accumulate<Negated>(du_dp + charge_param_idx_i, g_qi);
      accumulate<Negated>(du_dp + charge_param_idx_j, g_qj);

      accumulate<Negated>(du_dp + lj_param_idx_sig_i, g_sigi);
      accumulate<Negated>(du_dp + lj_param_idx_eps_i, g_epsi);

      accumulate<Negated>(du_dp + lj_param_idx_sig_j, g_sigj);
      accumulate<Negated>(du_dp + lj_param_idx_eps_j, g_epsj);

      accumulate<Negated>(du_dp + w_param_idx_i, g_wi);
      accumulate<Negated>(du_dp + w_param_idx_j, g_wj);
    }
  }
  // Always accumulate into the energy buffers even if there is no interaction
  // to ensure buffer is zeroed out, avoids having to memset every call
  if (COMPUTE_U) {
    // Do not do `FLOAT_TO_FIXED_ENERGY(Negated ? -u : u)` as that can produce a
    // positive fixed point value
    u_buffer[pair_idx] = Negated ? -FLOAT_TO_FIXED_ENERGY<RealType>(u)
                                 : FLOAT_TO_FIXED_ENERGY<RealType>(u);
  }
}

} // namespace tmd
