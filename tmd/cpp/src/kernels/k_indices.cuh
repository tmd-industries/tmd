// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025, Forrest York
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
// Takes a source and destination array. Assumes K <= N with values in the src
// are less than or equal to K. The value of the src is used as the index and
// the value in the destination array. Allows combining a series of indices to
// get a unique set of values.
void __global__ k_set_value_to_idx(const int N, // Number of values in src
                                   const int K, // Number of values in dest
                                   const unsigned int *__restrict__ src,
                                   unsigned int *__restrict__ dest);

// Any value that is >=N becomes the idx and any value that is an idx becomes N.
// Assumes that the array is made up of indices that correspond to their index
// in the array, otherwise the inversion may contain values that were in the
// input.
void __global__ k_invert_indices(const int N, unsigned int *__restrict__ arr);

template <typename T>
void __global__ k_arange(const size_t N, T *__restrict__ arr);

template <typename T>
void __global__ k_fill(const size_t N, T *__restrict__ arr, const T val);

template <typename T>
void __global__ k_segment_arange(const size_t num_segments,
                                 const size_t elements_per_segment,
                                 T *__restrict__ arr);

} // namespace tmd
