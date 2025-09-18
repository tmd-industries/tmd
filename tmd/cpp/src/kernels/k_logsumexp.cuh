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

// Convert the outputs of LogSumExp kernels into the final logsumexp value
template <typename RealType>
RealType __host__ __device__ __forceinline__
compute_logsumexp_final(const RealType max_val, const RealType sum) {
  return max_val + log(sum);
}

template <typename RealType>
void __global__ k_segmented_exp_sub_max(
    const int num_segments,
    const int *__restrict__ d_segment_offsets, // [num_segments]
    const RealType *__restrict__ max,          // [num_segments]
    const RealType *__restrict__ vals,         // [num_segments, K]
    RealType *__restrict__ out                 // [num_segments, K]
);

} // namespace tmd
