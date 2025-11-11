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

#include "../gpu_utils.cuh"
#include "k_hilbert.cuh"
#include "stdio.h"

namespace tmd {

// k_coords_to_kv_gather assigns coordinates to the relevant hilbert curve bin.
// Note that this kernel requires the use of double precision as imaging into
// the home box with float precision can result in the final coordinates being
// outside of the home box for coordinates with large magnitudes.
template <typename RealType>
void __global__ k_coords_to_kv_gather(
    const int num_systems, const int atoms_per_system,
    const unsigned int *__restrict__ system_counts, // [num_systems]
    const unsigned int
        *__restrict__ atom_idxs,         // [num_systems, atoms_per_system]
    const RealType *__restrict__ coords, // [num_systems, atoms_per_system, 3]
    const RealType *__restrict__ box,    // [num_systems, 3, 3]
    const unsigned int
        *__restrict__ bin_to_idx,    // [HILBERT_GRID_DIM, HILBERT_GRID_DIM,
                                     // HILBERT_GRID_DIM]
    unsigned int *__restrict__ keys, // [num_systems, atoms_per_system]
    unsigned int *__restrict__ vals  // [num_systems, atoms_per_system]
) {

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  const unsigned int N = system_counts[system_idx];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const RealType bx = box[system_idx * 9 + 0 * 3 + 0];
  const RealType by = box[system_idx * 9 + 1 * 3 + 1];
  const RealType bz = box[system_idx * 9 + 2 * 3 + 2];

  const RealType inv_bx = rcp_rn(bx);
  const RealType inv_by = rcp_rn(by);
  const RealType inv_bz = rcp_rn(bz);

  const RealType inv_bin_width =
      min(min(inv_bx, inv_by), inv_bz) * (HILBERT_GRID_DIM - 1.0);

  while (idx < N) {
    int atom_idx = atom_idxs[system_idx * atoms_per_system + idx];

    RealType x = coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 0];
    RealType y = coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 1];
    RealType z = coords[system_idx * atoms_per_system * 3 + atom_idx * 3 + 2];

    // floor is used in place of nearbyint here to ensure all particles are
    // imaged into the home box. This differs from distances calculations where
    // the nearest possible image is calculated rather than imaging into the
    // home box.
    x -= bx * floor(x * inv_bx);
    y -= by * floor(y * inv_by);
    z -= bz * floor(z * inv_bz);

    // for extremely large coordinates (during a simulation blow-up)
    // loss of precision can put the atoms outside of hilbert grid dim
    unsigned int bin_x =
        min(static_cast<unsigned int>(x * inv_bin_width), HILBERT_GRID_DIM - 1);
    unsigned int bin_y =
        min(static_cast<unsigned int>(y * inv_bin_width), HILBERT_GRID_DIM - 1);
    unsigned int bin_z =
        min(static_cast<unsigned int>(z * inv_bin_width), HILBERT_GRID_DIM - 1);

    keys[system_idx * atoms_per_system + idx] =
        bin_to_idx[bin_x * HILBERT_GRID_DIM * HILBERT_GRID_DIM +
                   bin_y * HILBERT_GRID_DIM + bin_z];
    // uncomment below if you want to preserve the atom ordering
    // keys[system_idx * atoms_per_system + idx] = atom_idx;
    vals[system_idx * atoms_per_system + idx] = atom_idx;

    idx += gridDim.x * blockDim.x;
  }
}

template void __global__ k_coords_to_kv_gather<float>(
    const int num_systems, const int atoms_per_system,
    const unsigned int *__restrict__ system_counts,
    const unsigned int *__restrict__ atom_idxs,
    const float *__restrict__ coords, const float *__restrict__ box,
    const unsigned int *__restrict__ bin_to_idx,
    unsigned int *__restrict__ keys, unsigned int *__restrict__ vals);

template void __global__ k_coords_to_kv_gather<double>(
    const int num_systems, const int atoms_per_system,
    const unsigned int *__restrict__ system_counts,
    const unsigned int *__restrict__ atom_idxs,
    const double *__restrict__ coords, const double *__restrict__ box,
    const unsigned int *__restrict__ bin_to_idx,
    unsigned int *__restrict__ keys, unsigned int *__restrict__ vals);

} // namespace tmd
