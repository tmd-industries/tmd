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

#include "k_nonbonded_common.cuh"

namespace tmd {

// Shape of parameter array is identical to other nonbonded variants
// except that rows map to pairs instead of individual atoms
static const int PARAMS_PER_PAIR = PARAMS_PER_ATOM;

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_nonbonded_precomputed(
    const int N,                         // Number of atoms in each system
    const int M,                         // number of pairs
    const RealType *__restrict__ coords, // [N, 3] coordinates
    const RealType *__restrict__ params, // [M, 4] q_ij, s_ij, e_ij, w_offset_ij
    const RealType *__restrict__ box,    // box vectors
    const int *__restrict__ pair_idxs,   // [M, 2] pair-list of atoms
    const int *__restrict__ system_idxs, // [M] Which system the pair is from
    const RealType beta, const RealType cutoff_squared,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u_buffer) {

  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (pair_idx < M) {

    const int system_idx = system_idxs[pair_idx];
    const int coord_offset = system_idx * N;
    const int box_offset = system_idx * 9;

    const RealType box_x = box[box_offset + 0 * 3 + 0];
    const RealType box_y = box[box_offset + 1 * 3 + 1];
    const RealType box_z = box[box_offset + 2 * 3 + 2];

    const RealType inv_box_x = rcp_rn(box_x);
    const RealType inv_box_y = rcp_rn(box_y);
    const RealType inv_box_z = rcp_rn(box_z);

    int params_ij_idx = pair_idx * PARAMS_PER_PAIR;
    RealType q_ij = params[params_ij_idx + PARAM_OFFSET_CHARGE];
    RealType sig_ij = params[params_ij_idx + PARAM_OFFSET_SIG];
    RealType eps_ij = params[params_ij_idx + PARAM_OFFSET_EPS];
    RealType delta_w = params[params_ij_idx + PARAM_OFFSET_W];

    RealType g_q_ij = 0;
    RealType g_sig_ij = 0;
    RealType g_eps_ij = 0;
    RealType g_dw_ij = 0;

    const int atom_i_idx = pair_idxs[pair_idx * 2 + 0] + coord_offset;
    const int atom_j_idx = pair_idxs[pair_idx * 2 + 1] + coord_offset;

    RealType ci_x = coords[atom_i_idx * 3 + 0];
    RealType ci_y = coords[atom_i_idx * 3 + 1];
    RealType ci_z = coords[atom_i_idx * 3 + 2];

    RealType gi_x = 0;
    RealType gi_y = 0;
    RealType gi_z = 0;

    RealType cj_x = coords[atom_j_idx * 3 + 0];
    RealType cj_y = coords[atom_j_idx * 3 + 1];
    RealType cj_z = coords[atom_j_idx * 3 + 2];

    RealType gj_x = 0;
    RealType gj_y = 0;
    RealType gj_z = 0;

    RealType delta_x = ci_x - cj_x;
    RealType delta_y = ci_y - cj_y;
    RealType delta_z = ci_z - cj_z;

    delta_x -= box_x * nearbyint(delta_x * inv_box_x);
    delta_y -= box_y * nearbyint(delta_y * inv_box_y);
    delta_z -= box_z * nearbyint(delta_z * inv_box_z);

    __int128 energy = 0;

    RealType d2_ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z +
                     delta_w * delta_w;

    if (d2_ij < cutoff_squared) {

      RealType d_ij = sqrt(d2_ij);

      RealType inv_dij = 1 / d_ij;

      if (q_ij != 0) {

        RealType damping_factor;
        RealType es_factor = real_es_factor(beta, d_ij, inv_dij,
                                            inv_dij * inv_dij, damping_factor);

        if (COMPUTE_U) {
          // energies
          RealType coulomb = q_ij * inv_dij;
          RealType nrg = damping_factor * coulomb;
          energy += FLOAT_TO_FIXED_ENERGY<RealType>(nrg);
        }

        if (COMPUTE_DU_DX || COMPUTE_DU_DP) {
          RealType du_dr = q_ij * es_factor;

          RealType force_prefactor = du_dr * inv_dij;
          if (du_dx) {
            // forces
            gi_x += delta_x * force_prefactor;
            gi_y += delta_y * force_prefactor;
            gi_z += delta_z * force_prefactor;

            gj_x += -delta_x * force_prefactor;
            gj_y += -delta_y * force_prefactor;
            gj_z += -delta_z * force_prefactor;
          }

          if (du_dp) {
            // du/dp
            g_q_ij = damping_factor * inv_dij;
            g_dw_ij += delta_w * force_prefactor;
          }
        }
      }

      if (eps_ij != 0 && sig_ij != 0) {
        RealType d4_ij = d2_ij * d2_ij;
        RealType d6_ij = d4_ij * d2_ij;

        RealType sig2_ij = sig_ij * sig_ij;
        RealType sig4_ij = sig2_ij * sig2_ij;
        RealType sig6_ij = sig4_ij * sig2_ij;
        RealType du_de;
        if (COMPUTE_U || COMPUTE_DU_DP) {
          RealType sig2_inv_d2ij = (sig_ij * inv_dij) * (sig_ij * inv_dij);
          RealType sig4_inv_d4ij = sig2_inv_d2ij * sig2_inv_d2ij;
          RealType sig6_inv_d6ij = sig4_inv_d4ij * sig2_inv_d2ij;
          du_de =
              static_cast<RealType>(4.0) * (sig6_inv_d6ij - 1) * sig6_inv_d6ij;
          if (COMPUTE_U) {
            // energies
            RealType nrg = eps_ij * du_de;
            energy += FLOAT_TO_FIXED_ENERGY<RealType>(nrg);
          }
        }

        if (COMPUTE_DU_DX || COMPUTE_DU_DP) {
          RealType d12_ij = d6_ij * d6_ij;
          RealType du_dr = eps_ij * static_cast<RealType>(24.0) * sig6_ij *
                           (d6_ij - static_cast<RealType>(2.0) * sig6_ij) /
                           (d12_ij * d_ij);

          RealType force_prefactor = du_dr * inv_dij;
          if (COMPUTE_DU_DX) {
            gi_x += delta_x * force_prefactor;
            gi_y += delta_y * force_prefactor;
            gi_z += delta_z * force_prefactor;

            gj_x += -delta_x * force_prefactor;
            gj_y += -delta_y * force_prefactor;
            gj_z += -delta_z * force_prefactor;
          }

          if (du_dp) {
            RealType du_ds =
                static_cast<RealType>(-24.0) * eps_ij * (sig4_ij * sig_ij) *
                (d6_ij - static_cast<RealType>(2.0) * sig6_ij) / d12_ij;
            g_eps_ij = du_de;
            g_sig_ij = du_ds;
            g_dw_ij += delta_w * force_prefactor;
          }
        }
      }

      if (COMPUTE_DU_DX) {
        unsigned long long gx = FLOAT_TO_FIXED_NONBONDED(gi_x);
        unsigned long long gy = FLOAT_TO_FIXED_NONBONDED(gi_y);
        unsigned long long gz = FLOAT_TO_FIXED_NONBONDED(gi_z);

        atomicAdd(du_dx + atom_i_idx * 3 + 0, gx);
        atomicAdd(du_dx + atom_i_idx * 3 + 1, gy);
        atomicAdd(du_dx + atom_i_idx * 3 + 2, gz);

        atomicAdd(du_dx + atom_j_idx * 3 + 0, -gx);
        atomicAdd(du_dx + atom_j_idx * 3 + 1, -gy);
        atomicAdd(du_dx + atom_j_idx * 3 + 2, -gz);
      }

      if (COMPUTE_DU_DP) {
        atomicAdd(
            du_dp + params_ij_idx + PARAM_OFFSET_CHARGE,
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(g_q_ij));
        atomicAdd(
            du_dp + params_ij_idx + PARAM_OFFSET_SIG,
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(g_sig_ij));
        atomicAdd(
            du_dp + params_ij_idx + PARAM_OFFSET_EPS,
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(g_eps_ij));
        atomicAdd(
            du_dp + params_ij_idx + PARAM_OFFSET_W,
            FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(g_dw_ij));
      }
    }
    // Always set the energy to avoid having to running memset on energy buffer
    if (COMPUTE_U) {
      u_buffer[pair_idx] = energy;
    }
    pair_idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
