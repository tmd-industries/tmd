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

#pragma once

namespace tmd {

// Divide [0,1]^3 box into HILBERT_GRID_DIM^3 voxels for Hilbert sort
static const int HILBERT_GRID_DIM = 128;

// Encode grid index along each dimension using HILBERT_N_BITS
static const int HILBERT_N_BITS = 8;
static const int HILBERT_MAX_GRID_DIM = 1 << HILBERT_N_BITS;
static_assert(HILBERT_N_BITS == 8);
static_assert(HILBERT_GRID_DIM <= HILBERT_MAX_GRID_DIM);

// generate kv values from coordinates to be radix sorted allowing the selection
// of a subset of coordinates
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
);

} // namespace tmd
