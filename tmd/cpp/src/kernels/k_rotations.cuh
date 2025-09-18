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

template <typename RealType>
void __device__ __forceinline__
hamilton_product(const RealType *__restrict__ q1,
                 const RealType *__restrict__ q2, RealType *__restrict__ out);

template <typename RealType>
void __device__ __forceinline__ rotate_coordinates_by_quaternion(
    RealType *__restrict__ local_coords, RealType *__restrict__ quaternion);

// k_rotate_coordinates rotates coordinates by quaternions. Does *NOT* modify
// the coordinates centroid. This method is for validating rotations by
// quaternions.
template <typename RealType>
void __global__ k_rotate_coordinates(
    const int N,                              // Number of coordinates
    const int n_rotations,                    // Number of quaternions
    const RealType *__restrict__ coords,      // [N, 3]
    const RealType *__restrict__ quaternions, // [n_rotations, 4]
    RealType *__restrict__ rotated_coords     // [N * n_rotations, 3]
);

// k_rotate_and_translate_mols rotates coordinates about its centroid given a
// quaternion. Places the molecule's centroid at the translation as the final
// step, if SCALE=true then the translation that is provided will be scaled by
// the box vectors.
template <typename RealType, bool SCALE>
void __global__ k_rotate_and_translate_mols(
    const int total_proposals, const int batch_size,
    const int *__restrict__ offset,
    const RealType *__restrict__ coords,       // [N, 3]
    const RealType *__restrict__ box,          // [3, 3]
    const int *__restrict__ samples,           // [batch_size]
    const int *__restrict__ mol_offsets,       // [num_mols + 1]
    const RealType *__restrict__ quaternions,  // [batch_size, 4]
    const RealType *__restrict__ translations, // [batch_size, 3]
    RealType *__restrict__ coords_out          // [batch_size, num_atoms, 3]
);

} // namespace tmd
