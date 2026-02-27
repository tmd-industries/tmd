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

#include "curand.h"
#include "k_fixed_point.cuh"

namespace tmd {

// k_rescale_positions scales the box and the centroids of groups to evaluate a
// potential barostat move
template <typename RealType, bool SCALE_X, bool SCALE_Y, bool SCALE_Z>
void __global__ k_rescale_positions(
    const int num_systems, const int atoms_per_system, const int num_mols,
    const int N,                                     // Number of atoms to shift
    RealType *__restrict__ coords,                   // Coordinates
    const RealType *__restrict__ length_scale,       // [num_systems]
    const RealType *__restrict__ box,                // [num_systems, 9]
    RealType *__restrict__ scaled_box,               // [num_systems, 9]
    const int *__restrict__ atom_idxs,               // [num_systems, N]
    const int *__restrict__ mol_idxs,                // [num_systems, N]
    const int *__restrict__ mol_offsets,             // [num_systems, N]
    const unsigned long long *__restrict__ centroids // [num_systems, N*3]
) {
  static_assert(SCALE_X | SCALE_Y | SCALE_Z);

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  RealType center_x =
      box[system_idx * 9 + 0 * 3 + 0] * static_cast<RealType>(0.5);
  RealType center_y =
      box[system_idx * 9 + 1 * 3 + 1] * static_cast<RealType>(0.5);
  RealType center_z =
      box[system_idx * 9 + 2 * 3 + 2] * static_cast<RealType>(0.5);

  RealType scale = *length_scale;
  if (idx == 0) {
    if (SCALE_X) {
      scaled_box[system_idx * 9 + 0 * 3 + 0] =
          box[system_idx * 9 + 0 * 3 + 0] * scale;
    } else {
      // Perform copy
      scaled_box[system_idx * 9 + 0 * 3 + 0] = box[system_idx * 9 + 0 * 3 + 0];
    }
    if (SCALE_Y) {
      scaled_box[system_idx * 9 + 1 * 3 + 1] =
          box[system_idx * 9 + 1 * 3 + 1] * scale;
    } else {
      // Perform copy
      scaled_box[system_idx * 9 + 1 * 3 + 1] = box[system_idx * 9 + 1 * 3 + 1];
    }
    if (SCALE_Z) {
      scaled_box[system_idx * 9 + 2 * 3 + 2] =
          box[system_idx * 9 + 2 * 3 + 2] * scale;
    } else {
      // Perform copy
      scaled_box[system_idx * 9 + 2 * 3 + 2] = box[system_idx * 9 + 2 * 3 + 2];
    }
  }
  while (idx < N) {
    int atom_idx = atom_idxs[idx];
    int mol_idx = mol_idxs[idx];

    // Handle the mol offsets right
    RealType num_atoms =
        static_cast<RealType>(mol_offsets[mol_idx + 1] - mol_offsets[mol_idx]);

    RealType centroid_x =
        FIXED_TO_FLOAT<RealType>(
            centroids[system_idx * num_mols * 3 + mol_idx * 3 + 0]) /
        num_atoms;
    RealType centroid_y =
        FIXED_TO_FLOAT<RealType>(
            centroids[system_idx * num_mols * 3 + mol_idx * 3 + 1]) /
        num_atoms;
    RealType centroid_z =
        FIXED_TO_FLOAT<RealType>(
            centroids[system_idx * num_mols * 3 + mol_idx * 3 + 2]) /
        num_atoms;

    // compute displacement needed to shift centroid back into the scaled
    // homebox
    if (SCALE_X) {
      RealType displacement_x =
          ((centroid_x - center_x) * scale) + center_x - centroid_x;
      centroid_x += displacement_x;
      RealType scaled_box_x = box[system_idx * 9 + 0 * 3 + 0] * scale;
      RealType new_center_x = scaled_box_x * floor(centroid_x / scaled_box_x);
      coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 0] +=
          displacement_x - new_center_x;
    } else {
      // Still need to image to avoid drift
      coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 0] -=
          box[system_idx * 9 + 0 * 3 + 0] *
          floor(centroid_x / box[system_idx * 9 + 0 * 3 + 0]);
    }

    if (SCALE_Y) {
      RealType displacement_y =
          ((centroid_y - center_y) * scale) + center_y - centroid_y;
      centroid_y += displacement_y;
      RealType scaled_box_y = box[system_idx * 9 + 1 * 3 + 1] * scale;
      RealType new_center_y = scaled_box_y * floor(centroid_y / scaled_box_y);
      coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 1] +=
          displacement_y - new_center_y;
    } else {
      // Still need to image to avoid drift
      coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 1] -=
          box[system_idx * 9 + 1 * 3 + 1] *
          floor(centroid_y / box[system_idx * 9 + 1 * 3 + 1]);
    }

    if (SCALE_Z) {
      RealType displacement_z =
          ((centroid_z - center_z) * scale) + center_z - centroid_z;
      centroid_z += displacement_z;
      RealType scaled_box_z = box[system_idx * 9 + 2 * 3 + 2] * scale;
      RealType new_center_z = scaled_box_z * floor(centroid_z / scaled_box_z);
      coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 2] +=
          displacement_z - new_center_z;
    } else {
      // Still need to image to avoid drift
      coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 2] -=
          box[system_idx * 9 + 2 * 3 + 2] *
          floor(centroid_z / box[system_idx * 9 + 2 * 3 + 2]);
    }

    idx += gridDim.x * blockDim.x;
  }
}

// k_find_group_centroids computes the centroids of a group of atoms.
template <typename RealType>
void __global__ k_find_group_centroids(
    const int num_systems, const int atoms_per_system, const int num_mols,
    const int N,                         // Number of atoms to shift
    const RealType *__restrict__ coords, // Coordinates [num_systems, N, 3]
    const int *__restrict__ atom_idxs,   // [num_systems, N]
    const int *__restrict__ mol_idxs,    // [num_systems, N]
    unsigned long long
        *__restrict__ centroids // [num_systems, num_molecules, 3]
) {
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < N) {
    int atom_idx = atom_idxs[idx];
    int mol_idx = mol_idxs[idx];
    atomicAdd(
        centroids + system_idx * num_mols * 3 + mol_idx * 3 + 0,
        FLOAT_TO_FIXED<RealType>(
            coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 0]));
    atomicAdd(
        centroids + system_idx * num_mols * 3 + mol_idx * 3 + 1,
        FLOAT_TO_FIXED<RealType>(
            coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 1]));
    atomicAdd(
        centroids + system_idx * num_mols * 3 + mol_idx * 3 + 2,
        FLOAT_TO_FIXED<RealType>(
            coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 2]));
    idx += gridDim.x * blockDim.x;
  }
}

// k_setup_barostat_move performs the initialization for a barostat move. It
// determines what the the proposed volume will be and sets up d_length_scale
// and d_volume_delta for use in k_decide_move.
template <typename RealType, bool SCALE_X, bool SCALE_Y, bool SCALE_Z>
void __global__ k_setup_barostat_move(
    const int num_systems, const bool adaptive,
    curandState_t *__restrict__ rng,                  // [num_systems]
    const RealType *__restrict__ d_box,               // [num_systems, 3, 3]
    RealType *__restrict__ d_volume_delta,            // [num_systems]
    RealType *__restrict__ d_volume_scale,            // [num_systems]
    RealType *__restrict__ d_length_scale,            // [num_systems]
    RealType *__restrict__ d_volume,                  // [num_systems]
    RealType *__restrict__ d_metropolis_hastings_rand // [num_systems]
) {
  static_assert(SCALE_X | SCALE_Y | SCALE_Z);

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) {
    return; // Only a single thread needs to perform this operation
  }
  curandState_t local_rng = rng[system_idx];

  RealType rand_scale = template_curand_uniform<RealType>(&local_rng);
  RealType metropolis_hastings = template_curand_uniform<RealType>(&local_rng);
  // Need to store this in global memory to avoid race condition in
  // k_decide_move
  d_metropolis_hastings_rand[system_idx] = metropolis_hastings;

  // Only safe so long as there is a single thread
  rng[system_idx] = local_rng;

  const RealType volume = d_box[system_idx * 9 + 0 * 3 + 0] *
                          d_box[system_idx * 9 + 1 * 3 + 1] *
                          d_box[system_idx * 9 + 2 * 3 + 2];
  if (adaptive && d_volume_scale[system_idx] == 0.0) {
    d_volume_scale[system_idx] = 0.01 * volume;
  }
  const RealType delta_volume =
      d_volume_scale[system_idx] * 2 * (rand_scale - 0.5);
  const RealType new_volume = volume + delta_volume;
  d_volume[system_idx] = volume;
  d_volume_delta[system_idx] = delta_volume;

  constexpr int dimensions_scaled =
      (SCALE_X ? 1 : 0) + (SCALE_Y ? 1 : 0) + (SCALE_Z ? 1 : 0);
  if (dimensions_scaled == 3) {
    d_length_scale[system_idx] = cbrt(new_volume / volume);
  } else if (dimensions_scaled == 2) {
    d_length_scale[system_idx] = sqrt(new_volume / volume);
  } else if (dimensions_scaled == 1) {
    d_length_scale[system_idx] = new_volume / volume;
  }
}

// k_decide_move handles the metropolis check for whether or not to accept a
// barostat move that scales the box volume. If the move is accepted then the
// box will be scaled as well as all of the coordinates. It also handles the
// bookkeeping for the acceptance counters.
template <typename RealType>
void __global__ k_decide_move(
    const int num_systems, const int N, const bool adaptive,
    const int num_molecules, const RealType kt, const RealType pressure,
    const RealType *__restrict__ rand,           // [num_systems]
    const RealType *__restrict__ d_volume,       // [num_systems]
    const RealType *__restrict__ d_volume_delta, // [num_systems]
    RealType *__restrict__ d_volume_scale,       // [num_systems]
    const __int128 *__restrict__ d_init_u,       // [num_systems]
    const __int128 *__restrict__ d_final_u,      // [num_systems]
    RealType *__restrict__ d_box,                // [num_systems, 3, 3]
    const RealType *__restrict__ d_box_proposed, // [num_systems, 3, 3]
    RealType *__restrict__ d_x,                  // [num_systems, N, 3]
    const RealType *__restrict__ d_x_proposed,   // [num_systems, N, 3]
    int *__restrict__ num_accepted,              // [num_systems]
    int *__restrict__ num_attempted              // [num_systems]
) {
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Don't compute volume from the box. It leads to a race condition since
  // `d_box` is updated in this kernel
  const RealType volume = d_volume[system_idx];
  const RealType volume_delta = d_volume_delta[system_idx];
  const RealType new_volume = volume + volume_delta;
  RealType energy_delta = INFINITY;
  if (!fixed_point_overflow(d_final_u[system_idx]) &&
      !fixed_point_overflow(d_init_u[system_idx])) {
    energy_delta = FIXED_ENERGY_TO_FLOAT<RealType>(d_final_u[system_idx] -
                                                   d_init_u[system_idx]);
  }

  const RealType local_rand = rand[system_idx];

  const RealType w = energy_delta + pressure * volume_delta -
                     num_molecules * kt * log(new_volume / volume);

  const bool rejected = w > 0 && local_rand > exp(-w / kt);

  while (idx < N) {
    if (idx == 0) {
      if (!rejected) {
        num_accepted[system_idx]++;
      }
      num_attempted[system_idx]++;
      if (adaptive && num_attempted[system_idx] >= 10) {
        if (num_accepted[system_idx] < 0.25 * num_attempted[system_idx]) {
          d_volume_scale[system_idx] /= 1.1;
          // Reset the counters
          num_attempted[system_idx] = 0;
          num_accepted[system_idx] = 0;
        } else if (num_accepted[system_idx] >
                   0.75 * num_attempted[system_idx]) {
          d_volume_scale[system_idx] =
              min(d_volume_scale[system_idx] * 1.1, volume * 0.3);
          // Reset the counters
          num_attempted[system_idx] = 0;
          num_accepted[system_idx] = 0;
        }
      }
    }
    if (rejected) {
      return;
    }
    // If the mc move was accepted copy all of the data into place

    if (idx < 9) {
      d_box[system_idx * 9 + idx] = d_box_proposed[system_idx * 9 + idx];
    }

#pragma unroll 3
    for (int i = 0; i < 3; i++) {
      d_x[system_idx * N * 3 + idx * 3 + i] =
          d_x_proposed[system_idx * N * 3 + idx * 3 + i];
    }
    idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
