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

#include "k_logsumexp.cuh"

namespace tmd {

template <typename RealType>
void __global__ k_segmented_exp_sub_max(
    const int num_segments,
    const int *__restrict__ d_segment_offsets, // [num_segments + 1]
    const RealType *__restrict__ max,          // [num_segments]
    const RealType *__restrict__ vals,         // [num_segments, K]
    RealType *__restrict__ out                 // [num_segments, K]
) {
  int segment = blockIdx.y;
  while (segment < num_segments) {
    const int start = d_segment_offsets[segment];
    const int end = d_segment_offsets[segment + 1];
    const RealType max_val = max[segment];
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) + start;
    while (idx < end) {
      out[idx] = exp(vals[idx] - max_val);

      idx += gridDim.x * blockDim.x;
    }
    segment += gridDim.y * blockDim.y;
  }
}

template void __global__ k_segmented_exp_sub_max<float>(const int, const int *,
                                                        const float *,
                                                        const float *, float *);
template void __global__ k_segmented_exp_sub_max<double>(const int, const int *,
                                                         const double *,
                                                         const double *,
                                                         double *);

} // namespace tmd
