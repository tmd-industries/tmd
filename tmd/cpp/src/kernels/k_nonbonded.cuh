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

#include "../fixed_point.hpp"
#include "../gpu_utils.cuh"
#include "k_nonbonded_common.cuh"
#include "kernel_utils.cuh"
#include <cub/cub.cuh>

namespace tmd {

template <typename RealType>
void __global__ k_check_rebuild_coords_and_box_gather(
    const int N, const unsigned int *__restrict__ atom_idxs,
    const RealType *__restrict__ new_coords,
    const RealType *__restrict__ old_coords,
    const RealType *__restrict__ new_box, const RealType *__restrict__ old_box,
    const RealType padding, int *rebuild_flag) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < 9) {
    // (ytz): box vectors have exactly 9 components
    // we can probably derive a looser bound later on.
    if (old_box[idx] != new_box[idx]) {
      rebuild_flag[0] = 1;
    }
  }

  if (idx >= N) {
    return;
  }

  const int atom_idx = atom_idxs[idx];
  // cast coords
  RealType xi = old_coords[atom_idx * 3 + 0];
  RealType yi = old_coords[atom_idx * 3 + 1];
  RealType zi = old_coords[atom_idx * 3 + 2];
  RealType xj = new_coords[atom_idx * 3 + 0];
  RealType yj = new_coords[atom_idx * 3 + 1];
  RealType zj = new_coords[atom_idx * 3 + 2];
  RealType dx = xi - xj;
  RealType dy = yi - yj;
  RealType dz = zi - zj;
  RealType d2ij = dx * dx + dy * dy + dz * dz;
  if (d2ij > static_cast<RealType>(0.25) * padding * padding) {
    // (ytz): this is *safe* but technically is a race condition
    rebuild_flag[0] = 1;
  }
}

template <typename RealType, int COORDS_DIM, int PARAMS_DIM>
void __global__ k_gather_coords_and_params(
    const int N, const unsigned int *__restrict__ idxs,
    const RealType *__restrict__ coords, const RealType *__restrict__ params,
    RealType *__restrict__ gathered_coords,
    RealType *__restrict__ gathered_params) {
  static_assert(COORDS_DIM == 3);
  static_assert(PARAMS_DIM == PARAMS_PER_ATOM);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  const unsigned int atom_idx = idxs[idx];
  // Coords have 3 dimensions, params have 4
#pragma unroll COORDS_DIM
  for (int i = 0; i < COORDS_DIM; i++) {
    gathered_coords[idx * COORDS_DIM + i] = coords[atom_idx * COORDS_DIM + i];
  }
#pragma unroll PARAMS_DIM
  for (int i = 0; i < PARAMS_DIM; i++) {
    gathered_params[idx * PARAMS_DIM + i] = params[atom_idx * PARAMS_DIM + i];
  }
}

// ALCHEMICAL == false guarantees that the tile's atoms are such that
// 1. src_param and dst_params are equal for every i in R and j in C
// 2. w_i and w_j are identical for every (i,j) in (RxC)
template <typename RealType, bool ALCHEMICAL, bool COMPUTE_U,
          bool COMPUTE_DU_DX, bool COMPUTE_DU_DP, bool COMPUTE_UPPER_TRIANGLE,
          bool COMPUTE_J_GRADS>
// void __device__ __forceinline__ v_nonbonded_unified(
void __device__ v_nonbonded_unified(
    const int tile_idx, const int N, const int NR,
    const unsigned int
        *__restrict__ output_permutation, // [N] Permutation from atom idx ->
                                          // output buffer idx idx
    const RealType *__restrict__ coords,  // [N * 3]
    const RealType *__restrict__ params,  // [N * PARAMS_PER_ATOM]
    const RealType *__restrict__ box, __int128 &energy_accumulator,
    const RealType beta, const RealType cutoff_squared,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp) {

  const RealType box_x = box[0 * 3 + 0];
  const RealType box_y = box[1 * 3 + 1];
  const RealType box_z = box[2 * 3 + 2];
  const RealType inv_box_x = rcp_rn(box_x);
  const RealType inv_box_y = rcp_rn(box_y);
  const RealType inv_box_z = rcp_rn(box_z);

  int row_block_idx = ixn_tiles[tile_idx];

  const int warp_idx = threadIdx.x % WARP_SIZE;
  const int index = row_block_idx * WARP_SIZE + warp_idx;

  const unsigned int atom_i_idx = index < NR ? index : N;
  const unsigned int dest_i_idx =
      atom_i_idx < N ? output_permutation[atom_i_idx] : N;

  RealType ci_x = atom_i_idx < N ? coords[atom_i_idx * 3 + 0] : 0;
  RealType ci_y = atom_i_idx < N ? coords[atom_i_idx * 3 + 1] : 0;
  RealType ci_z = atom_i_idx < N ? coords[atom_i_idx * 3 + 2] : 0;

  unsigned long long gi_x = 0;
  unsigned long long gi_y = 0;
  unsigned long long gi_z = 0;

  RealType qi = atom_i_idx < N
                    ? params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_CHARGE]
                    : 0;
  RealType sig_i = atom_i_idx < N
                       ? params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_SIG]
                       : 0;
  RealType eps_i = atom_i_idx < N
                       ? params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_EPS]
                       : 0;
  RealType w_i = atom_i_idx < N
                     ? params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W]
                     : 0;

  unsigned long long g_qi = 0;
  unsigned long long g_sigi = 0;
  unsigned long long g_epsi = 0;
  unsigned long long g_wi = 0;

  int atom_j_idx = ixn_atoms[tile_idx * WARP_SIZE + warp_idx];
  int dest_j_idx = atom_j_idx < N ? output_permutation[atom_j_idx] : N;

  RealType cj_x = atom_j_idx < N ? coords[atom_j_idx * 3 + 0] : 0;
  RealType cj_y = atom_j_idx < N ? coords[atom_j_idx * 3 + 1] : 0;
  RealType cj_z = atom_j_idx < N ? coords[atom_j_idx * 3 + 2] : 0;

  // compiler should optimize these away if they're not used.
  unsigned long long gj_x = 0;
  unsigned long long gj_y = 0;
  unsigned long long gj_z = 0;

  RealType qj = atom_j_idx < N
                    ? params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_CHARGE]
                    : 0;
  RealType sig_j = atom_j_idx < N
                       ? params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_SIG]
                       : 0;
  RealType eps_j = atom_j_idx < N
                       ? params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_EPS]
                       : 0;
  RealType w_j = atom_j_idx < N
                     ? params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W]
                     : 0;

  // compiler should optimize these variables if they're not used.
  unsigned long long g_qj = 0;
  unsigned long long g_sigj = 0;
  unsigned long long g_epsj = 0;
  unsigned long long g_wj = 0;

  const int src_lane = (warp_idx + 1) % WARP_SIZE; // fixed
  // #pragma unroll
  for (int round = 0; round < WARP_SIZE; round++) {

    RealType delta_x = ci_x - cj_x;
    RealType delta_y = ci_y - cj_y;
    RealType delta_z = ci_z - cj_z;

    delta_x -= box_x * nearbyint(delta_x * inv_box_x);
    delta_y -= box_y * nearbyint(delta_y * inv_box_y);
    delta_z -= box_z * nearbyint(delta_z * inv_box_z);

    RealType d2ij = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

    RealType delta_w;

    if (ALCHEMICAL) {
      // (ytz): we are guaranteed that delta_w is zero if ALCHEMICAL == false
      delta_w = w_i - w_j;
      d2ij += delta_w * delta_w;
    }

    // All idxs must be smaller than N and if N == NR then we are doing upper
    // triangle and thus atom_i_idx must be less than atom_j_idx
    bool valid_ij = (atom_i_idx < N) && (atom_j_idx < N);
    if (COMPUTE_UPPER_TRIANGLE && atom_i_idx >= atom_j_idx) {
      valid_ij = false;
    }

    // (ytz): note that d2ij must be *strictly* less than cutoff_squared. This
    // is because we set the non-interacting atoms to exactly
    // real_cutoff*real_cutoff. This ensures that atoms who's 4th dimension is
    // set to cutoff are non-interacting.
    if (valid_ij && d2ij < cutoff_squared) {
      // electrostatics
      RealType u;
      RealType es_prefactor;
      RealType ebd;
      RealType dij;
      RealType inv_dij;
      RealType inv_d2ij;
      compute_electrostatics<RealType, COMPUTE_U>(1.0, qi, qj, d2ij, beta, dij,
                                                  inv_dij, inv_d2ij, ebd,
                                                  es_prefactor, u);

      RealType delta_prefactor = es_prefactor;

      // lennard jones force
      if (eps_i != 0 && eps_j != 0) {
        RealType sig_grad;
        RealType eps_grad;
        compute_lj<RealType, COMPUTE_U>(1.0, eps_i, eps_j, sig_i, sig_j,
                                        inv_dij, inv_d2ij, u, delta_prefactor,
                                        sig_grad, eps_grad);

        // do chain rule inside loop
        if (COMPUTE_DU_DP) {
          g_sigi +=
              FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(sig_grad);
          g_epsi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(
              eps_grad * eps_j);
          if (COMPUTE_J_GRADS) {
            g_sigj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(
                sig_grad);
            g_epsj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(
                eps_grad * eps_i);
          }
        }
      }

      if (COMPUTE_DU_DX) {
        unsigned long long ull_gx =
            FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_x);
        unsigned long long ull_gy =
            FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_y);
        unsigned long long ull_gz =
            FLOAT_TO_FIXED_NONBONDED(delta_prefactor * delta_z);
        gi_x += ull_gx;
        gi_y += ull_gy;
        gi_z += ull_gz;
        if (COMPUTE_J_GRADS) {
          gj_x -= ull_gx;
          gj_y -= ull_gy;
          gj_z -= ull_gz;
        }
      }

      if (COMPUTE_DU_DP) {
        g_qi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(
            qj * inv_dij * ebd);

        if (COMPUTE_J_GRADS) {
          g_qj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(
              qi * inv_dij * ebd);
        }

        if (ALCHEMICAL) {
          g_wi += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(
              delta_prefactor * delta_w);
          if (COMPUTE_J_GRADS) {
            g_wj += FLOAT_TO_FIXED_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(
                -delta_prefactor * delta_w);
          }
        }
      }

      if (COMPUTE_U) {
        energy_accumulator += FLOAT_TO_FIXED_ENERGY<RealType>(u);
      }
    }

    // j_idxs are non-contiguous, so we can't easily pre-compute this
    atom_j_idx = __shfl_sync(0xffffffff, atom_j_idx, src_lane);
    qj = __shfl_sync(0xffffffff, qj, src_lane);
    eps_j = __shfl_sync(0xffffffff, eps_j, src_lane);
    sig_j = __shfl_sync(0xffffffff, sig_j, src_lane);

    cj_x = __shfl_sync(0xffffffff, cj_x, src_lane);
    cj_y = __shfl_sync(0xffffffff, cj_y, src_lane);
    cj_z = __shfl_sync(0xffffffff, cj_z, src_lane);

    if (ALCHEMICAL) {
      w_j = __shfl_sync(0xffffffff, w_j,
                        src_lane); // this also can be optimized away
    }

    if (COMPUTE_J_GRADS) {
      if (COMPUTE_DU_DX) {
        gj_x = __shfl_sync(0xffffffff, gj_x, src_lane);
        gj_y = __shfl_sync(0xffffffff, gj_y, src_lane);
        gj_z = __shfl_sync(0xffffffff, gj_z, src_lane);
      }
      if (COMPUTE_DU_DP) {
        g_qj = __shfl_sync(0xffffffff, g_qj, src_lane);
        g_sigj = __shfl_sync(0xffffffff, g_sigj, src_lane);
        g_epsj = __shfl_sync(0xffffffff, g_epsj, src_lane);
        g_wj = __shfl_sync(0xffffffff, g_wj, src_lane);
      }
    }
  }

  // Note as long as TILE_SIZE == WARP_SIZE, don't have to shuffle sync
  // dest_j_idx
  if (COMPUTE_DU_DX) {
    if (atom_i_idx < N) {
      atomicAdd(du_dx + dest_i_idx * 3 + 0, gi_x);
      atomicAdd(du_dx + dest_i_idx * 3 + 1, gi_y);
      atomicAdd(du_dx + dest_i_idx * 3 + 2, gi_z);
    }
    if (COMPUTE_J_GRADS && atom_j_idx < N) {
      atomicAdd(du_dx + dest_j_idx * 3 + 0, gj_x);
      atomicAdd(du_dx + dest_j_idx * 3 + 1, gj_y);
      atomicAdd(du_dx + dest_j_idx * 3 + 2, gj_z);
    }
  }

  if (COMPUTE_DU_DP) {
    if (atom_i_idx < N) {
      atomicAdd(du_dp + dest_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_CHARGE,
                g_qi);
      atomicAdd(du_dp + dest_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_SIG,
                g_sigi);
      atomicAdd(du_dp + dest_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_EPS,
                g_epsi);
      atomicAdd(du_dp + dest_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W, g_wi);
    }

    if (COMPUTE_J_GRADS && atom_j_idx < N) {
      atomicAdd(du_dp + dest_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_CHARGE,
                g_qj);
      atomicAdd(du_dp + dest_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_SIG,
                g_sigj);
      atomicAdd(du_dp + dest_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_EPS,
                g_epsj);
      atomicAdd(du_dp + dest_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W, g_wj);
    }
  }
}

template <typename RealType, int THREADS_PER_BLOCK, bool COMPUTE_U,
          bool COMPUTE_DU_DX, bool COMPUTE_DU_DP, bool COMPUTE_UPPER_TRIANGLE,
          bool COMPUTE_COL_GRADS>
void __global__ k_nonbonded_unified(
    const int N,  // Number of atoms involved in the interaction group
    const int NR, // Number of row indices
    const unsigned int *__restrict__ ixn_count,
    const unsigned int
        *__restrict__ output_permutation, // [N] Permutation from atom idx ->
                                          // output buffer idx
    const RealType *__restrict__ coords,  // [N, 3]
    const RealType *__restrict__ params,  // [N, PARAMS_PER_ATOM]
    const RealType *__restrict__ box,     // [3, 3]
    const RealType beta, const RealType cutoff,
    const int *__restrict__ ixn_tiles,
    const unsigned int *__restrict__ ixn_atoms,
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp,
    __int128 *__restrict__ u_buffer // [blockDim.x]
) {
  static_assert(THREADS_PER_BLOCK <= 256 &&
                (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);
  __int128 energy_accumulator = 0;

  // Tile size is the same as warp size but it doesn't have to be.
  // Can be used interchangably at the moment, but in the future we may have
  // different tile sizes.
  const int tile_size = WARP_SIZE;

  const int tiles_per_block = blockDim.x / tile_size;
  const int stride = gridDim.x * tiles_per_block;
  int tile_idx = blockIdx.x * tiles_per_block + (threadIdx.x / tile_size);

  const int tile_offset = threadIdx.x % tile_size;

  const RealType cutoff_squared = cutoff * cutoff;

  const unsigned int interactions = *ixn_count;

  while (tile_idx < interactions) {

    int row_block_idx = ixn_tiles[tile_idx];
    int index = row_block_idx * tile_size + tile_offset;
    int atom_i_idx = index < NR ? index : N;
    int atom_j_idx = ixn_atoms[tile_idx * tile_size + tile_offset];

    // if any atom_j_idx is less than num_row_atoms, or if COMPUTE_COL_GRADS is
    // True, we always compute_j_grads for this tile.
    bool compute_j_grads =
        COMPUTE_COL_GRADS || __any_sync(0xffffffff, atom_j_idx < NR);

    RealType w_i = atom_i_idx < N
                       ? params[atom_i_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W]
                       : 0;
    RealType w_j = atom_j_idx < N
                       ? params[atom_j_idx * PARAMS_PER_ATOM + PARAM_OFFSET_W]
                       : 0;

    int is_vanilla = w_i == 0 && w_j == 0;

    bool tile_is_vanilla = __all_sync(0xffffffff, is_vanilla);

    if (compute_j_grads) {
      if (tile_is_vanilla) {
        v_nonbonded_unified<RealType, 0, COMPUTE_U, COMPUTE_DU_DX,
                            COMPUTE_DU_DP, COMPUTE_UPPER_TRIANGLE, 1>(
            tile_idx, N, NR, output_permutation, coords, params, box,
            energy_accumulator, beta, cutoff_squared, ixn_tiles, ixn_atoms,
            du_dx, du_dp);
      } else {
        v_nonbonded_unified<RealType, 1, COMPUTE_U, COMPUTE_DU_DX,
                            COMPUTE_DU_DP, COMPUTE_UPPER_TRIANGLE, 1>(
            tile_idx, N, NR, output_permutation, coords, params, box,
            energy_accumulator, beta, cutoff_squared, ixn_tiles, ixn_atoms,
            du_dx, du_dp);
      };
    } else {
      if (tile_is_vanilla) {
        v_nonbonded_unified<RealType, 0, COMPUTE_U, COMPUTE_DU_DX,
                            COMPUTE_DU_DP, COMPUTE_UPPER_TRIANGLE, 0>(
            tile_idx, N, NR, output_permutation, coords, params, box,
            energy_accumulator, beta, cutoff_squared, ixn_tiles, ixn_atoms,
            du_dx, du_dp);
      } else {
        v_nonbonded_unified<RealType, 1, COMPUTE_U, COMPUTE_DU_DX,
                            COMPUTE_DU_DP, COMPUTE_UPPER_TRIANGLE, 0>(
            tile_idx, N, NR, output_permutation, coords, params, box,
            energy_accumulator, beta, cutoff_squared, ixn_tiles, ixn_atoms,
            du_dx, du_dp);
      };
    }
    tile_idx += stride;
  }
  if (COMPUTE_U) {
    using BlockReduce = cub::BlockReduce<__int128, THREADS_PER_BLOCK>;

    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Sum's return value is only valid in thread 0
    __int128 aggregate = BlockReduce(temp_storage).Sum(energy_accumulator);

    if (threadIdx.x == 0) {
      u_buffer[blockIdx.x] = aggregate;
    }
  }
}

template <typename RealType, int THREADS_PER_BLOCK>
void __global__ k_compute_nonbonded_target_atom_energies(
    const int N, const int num_target_atoms,
    const int *__restrict__ target_atoms,        // [num_target_atoms]
    const int *__restrict__ target_mols,         // [num_target_atoms]
    const int *__restrict__ target_mols_offsets, // [num_mols + 1]
    const RealType *__restrict__ coords,         // [N, 3]
    const RealType *__restrict__ params,         // [N, PARAMS_PER_ATOM]
    const RealType *__restrict__ box,            // [3, 3],
    const RealType beta, const RealType cutoff_squared,
    __int128 *__restrict__ output_energies // [num_target_atoms, gridDim.x]
) {
  static_assert(THREADS_PER_BLOCK <= 256 &&
                (THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0);
  __int128 energy_accumulator;

  using BlockReduce = cub::BlockReduce<__int128, THREADS_PER_BLOCK>;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const RealType bx = box[0 * 3 + 0];
  const RealType by = box[1 * 3 + 1];
  const RealType bz = box[2 * 3 + 2];

  const RealType inv_bx = rcp_rn(bx);
  const RealType inv_by = rcp_rn(by);
  const RealType inv_bz = rcp_rn(bz);
  int row_idx = blockIdx.y;
  while (row_idx < num_target_atoms) {

    int atom_i_idx = target_atoms[row_idx];
    int mol_i_idx = target_mols[row_idx];

    int min_mol_offset = target_mols_offsets[mol_i_idx];
    int max_mol_offset = target_mols_offsets[mol_i_idx + 1];
    int min_atom_idx = target_atoms[min_mol_offset];
    int max_atom_idx = target_atoms[max_mol_offset - 1];

    int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
    int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
    int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];
    RealType w_i = params[w_param_idx_i];

    RealType ci_x = coords[atom_i_idx * 3 + 0];
    RealType ci_y = coords[atom_i_idx * 3 + 1];
    RealType ci_z = coords[atom_i_idx * 3 + 2];

    int atom_j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // All threads in the threadblock must loop to allow for __syncthreads() and
    // the row accumulation.
    while (atom_j_idx - threadIdx.x < N) {
      // Zero out the energy buffer
      energy_accumulator = 0;
      // The two atoms are in the same molecule, don't compute the energies
      // requires that the atom indices in each target mol is consecutive
      if (atom_j_idx < N &&
          (atom_j_idx < min_atom_idx || atom_j_idx > max_atom_idx)) {

        int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
        int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
        int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
        int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
        int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

        RealType qj = params[charge_param_idx_j];
        RealType sig_j = params[lj_param_idx_sig_j];
        RealType eps_j = params[lj_param_idx_eps_j];
        RealType w_j = params[w_param_idx_j];

        RealType cj_x = coords[atom_j_idx * 3 + 0];
        RealType cj_y = coords[atom_j_idx * 3 + 1];
        RealType cj_z = coords[atom_j_idx * 3 + 2];

        RealType delta_x = ci_x - cj_x;
        RealType delta_y = ci_y - cj_y;
        RealType delta_z = ci_z - cj_z;
        RealType delta_w = w_i - w_j;

        delta_x -= bx * nearbyint(delta_x * inv_bx);
        delta_y -= by * nearbyint(delta_y * inv_by);
        delta_z -= bz * nearbyint(delta_z * inv_bz);

        RealType d2ij = delta_x * delta_x + delta_y * delta_y +
                        delta_z * delta_z + delta_w * delta_w;

        if (d2ij < cutoff_squared) {
          RealType u;
          RealType delta_prefactor;
          RealType ebd;
          RealType dij;
          RealType inv_dij;
          RealType inv_d2ij;
          compute_electrostatics<RealType, true>(1.0, qi, qj, d2ij, beta, dij,
                                                 inv_dij, inv_d2ij, ebd,
                                                 delta_prefactor, u);
          // lennard jones energy
          if (eps_i != 0 && eps_j != 0) {
            RealType sig_grad;
            RealType eps_grad;
            compute_lj<RealType, true>(1.0, eps_i, eps_j, sig_i, sig_j, inv_dij,
                                       inv_d2ij, u, delta_prefactor, sig_grad,
                                       eps_grad);
          }
          // Store the atom by atom energy
          energy_accumulator = FLOAT_TO_FIXED_ENERGY<RealType>(u);
        }
      }

      // Sum's return value is only valid in thread 0
      __int128 aggregate = BlockReduce(temp_storage).Sum(energy_accumulator);
      // Call sync threads to ensure temp storage can be re-used
      __syncthreads();

      if (threadIdx.x == 0) {
        output_energies[row_idx * gridDim.x + blockIdx.x] += aggregate;
      }

      atom_j_idx += gridDim.x * blockDim.x;
    }
    row_idx += gridDim.y * blockDim.y;
  }
}

// NUM_BLOCKS is the number of blocks that
// k_compute_nonbonded_target_atom_energies is run with. Decides the number of
// values that need to be accumulated per atom. Also dictates the number of
// threads
template <typename RealType, int NUM_BLOCKS>
void __global__ k_accumulate_atom_energies_to_per_mol_energies(
    const int target_mols,
    const int *__restrict__ mol_offsets, // [target_mols + 1]
    const __int128
        *__restrict__ per_atom_energies, // [target_atoms, NUM_BLOCKS]
    __int128 *__restrict__ per_mol_energies) {

  int mol_idx = blockIdx.x;

  static_assert(NUM_BLOCKS <= 256 && (NUM_BLOCKS & (NUM_BLOCKS - 1)) == 0);
  __int128 local_accumulator;

  using BlockReduce = cub::BlockReduce<__int128, NUM_BLOCKS>;

  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  while (mol_idx < target_mols) {
    local_accumulator = 0;
    const int mol_start = mol_offsets[mol_idx];
    const int mol_end = mol_offsets[mol_idx + 1];

    int idx = (mol_start * NUM_BLOCKS) + threadIdx.x;
    while (idx < mol_end * NUM_BLOCKS) {
      local_accumulator += per_atom_energies[idx];

      idx += blockDim.x;
    }

    // Sum's return value is only valid in thread 0
    __int128 aggregate = BlockReduce(temp_storage).Sum(local_accumulator);
    // Call sync threads to ensure temp storage can be re-used
    __syncthreads();

    if (threadIdx.x == 0) {
      per_mol_energies[mol_idx] = aggregate;
    }

    mol_idx += gridDim.x;
  }
}

// k_atom_by_atom_energies is intended to be used for computing the energies of
// a subset of atoms against all other atoms. The kernel allows changing the
// positions of the target atoms by passing in an array for target_coords, if a
// nullptr is provided it will use the coords array for the positions of the
// target atoms. This allows modification of the positions of a subset of atoms,
// avoiding the need to duplicating all of the coordinates.
template <typename RealType>
void __global__ k_atom_by_atom_energies(
    const int N, const int num_target_atoms,
    const int *__restrict__ target_atoms, // [num_target_atoms]
    const RealType
        *__restrict__ target_coords, // [num_target_atoms, 3] Can be nullptr if
                                     // should use coords for the target atoms
    const RealType *__restrict__ coords, // [N, 3]
    const RealType *__restrict__ params, // [N, PARAMS_PER_ATOM]
    const RealType *__restrict__ box,    // [3, 3],
    const RealType beta, const RealType cutoff_squared,
    RealType *__restrict__ output_energies // [num_target_atoms, N]
) {
  const RealType bx = box[0 * 3 + 0];
  const RealType by = box[1 * 3 + 1];
  const RealType bz = box[2 * 3 + 2];

  const RealType inv_bx = rcp_rn(bx);
  const RealType inv_by = rcp_rn(by);
  const RealType inv_bz = rcp_rn(bz);
  int row_idx = blockIdx.y;
  while (row_idx < num_target_atoms) {

    int atom_i_idx = target_atoms[row_idx];

    RealType ci_x = target_coords != nullptr ? target_coords[row_idx * 3 + 0]
                                             : coords[atom_i_idx * 3 + 0];
    RealType ci_y = target_coords != nullptr ? target_coords[row_idx * 3 + 1]
                                             : coords[atom_i_idx * 3 + 1];
    RealType ci_z = target_coords != nullptr ? target_coords[row_idx * 3 + 2]
                                             : coords[atom_i_idx * 3 + 2];

    int params_i_idx = atom_i_idx * PARAMS_PER_ATOM;
    int charge_param_idx_i = params_i_idx + PARAM_OFFSET_CHARGE;
    int lj_param_idx_sig_i = params_i_idx + PARAM_OFFSET_SIG;
    int lj_param_idx_eps_i = params_i_idx + PARAM_OFFSET_EPS;
    int w_param_idx_i = params_i_idx + PARAM_OFFSET_W;

    RealType qi = params[charge_param_idx_i];
    RealType sig_i = params[lj_param_idx_sig_i];
    RealType eps_i = params[lj_param_idx_eps_i];
    RealType w_i = params[w_param_idx_i];

    int atom_j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (atom_j_idx < N) {
      RealType u = 0.0;

      int params_j_idx = atom_j_idx * PARAMS_PER_ATOM;
      int charge_param_idx_j = params_j_idx + PARAM_OFFSET_CHARGE;
      int lj_param_idx_sig_j = params_j_idx + PARAM_OFFSET_SIG;
      int lj_param_idx_eps_j = params_j_idx + PARAM_OFFSET_EPS;
      int w_param_idx_j = params_j_idx + PARAM_OFFSET_W;

      RealType qj = params[charge_param_idx_j];
      RealType sig_j = params[lj_param_idx_sig_j];
      RealType eps_j = params[lj_param_idx_eps_j];
      RealType w_j = params[w_param_idx_j];

      RealType cj_x = coords[atom_j_idx * 3 + 0];
      RealType cj_y = coords[atom_j_idx * 3 + 1];
      RealType cj_z = coords[atom_j_idx * 3 + 2];

      RealType delta_x = ci_x - cj_x;
      RealType delta_y = ci_y - cj_y;
      RealType delta_z = ci_z - cj_z;
      RealType delta_w = w_i - w_j;

      delta_x -= bx * nearbyint(delta_x * inv_bx);
      delta_y -= by * nearbyint(delta_y * inv_by);
      delta_z -= bz * nearbyint(delta_z * inv_bz);

      RealType d2ij = delta_x * delta_x + delta_y * delta_y +
                      delta_z * delta_z + delta_w * delta_w;

      if (d2ij < cutoff_squared) {
        RealType delta_prefactor;
        RealType ebd;
        RealType dij;
        RealType inv_dij;
        RealType inv_d2ij;
        compute_electrostatics<RealType, true>(1.0, qi, qj, d2ij, beta, dij,
                                               inv_dij, inv_d2ij, ebd,
                                               delta_prefactor, u);

        // lennard jones force
        if (eps_i != 0 && eps_j != 0) {
          RealType sig_grad;
          RealType eps_grad;
          compute_lj<RealType, true>(1.0, eps_i, eps_j, sig_i, sig_j, inv_dij,
                                     inv_d2ij, u, delta_prefactor, sig_grad,
                                     eps_grad);
        }
      }
      // Store the atom by atom energy, can be in floating point since there is
      // no accumulation in the kernel
      output_energies[row_idx * N + atom_j_idx] = u;
      atom_j_idx += gridDim.x * blockDim.x;
    }
    row_idx += gridDim.y * blockDim.y;
  }
}

} // namespace tmd
