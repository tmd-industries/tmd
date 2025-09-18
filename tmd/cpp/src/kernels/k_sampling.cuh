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

#include <cub/cub.cuh>

namespace tmd {

// References:
// https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
// https://lips.cs.princeton.edu/the-gumbel-max-trick-for-discrete-distributions/
template <typename RealType>
void __global__
k_setup_gumbel_max_trick(const int N, const RealType *__restrict__ log_weights,
                         const RealType *__restrict__ gumbel_noise,
                         RealType *__restrict__ prepared_gumbel);

template <typename RealType>
void __global__ k_setup_gumbel_max_trick_with_offset(
    const int num_segments, const int total_values, const int max_offset,
    const int *__restrict__ noise_offset,     // [1]
    const int *__restrict__ segment_offsets,  // [blockDim.y]
    const RealType *__restrict__ log_weights, // [max_offset, num_segments]
    const RealType *__restrict__ gumbel_noise,
    RealType *__restrict__ prepared_gumbel);

template <typename RealType>
void __global__ k_setup_gumbel_max_trick_targeted_insertion(
    const int num_segments, const int vals_per_segment, const int max_offset,
    const int *__restrict__ noise_offset,      // [1]
    const int *__restrict__ segment_offsets,   // [num_segments]
    const RealType *__restrict__ log_weights,  // [total_values]
    const RealType *__restrict__ gumbel_noise, // [max_offset]
    RealType *__restrict__ prepared_gumbel     // [total_values]
);

template <typename T>
void __global__ k_copy_kv_key(
    const int N, const cub::KeyValuePair<int, T> *__restrict__ kv_pairs,
    int *__restrict__ out);

} // namespace tmd
