// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2026, Forrest York
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
#include "../gpu_utils.cuh"

namespace tmd {

template <typename RealType>
inline __device__ RealType dot_product(const RealType a[3],
                                       const RealType b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename RealType>
inline __device__ void cross_product(const RealType a[3], const RealType b[3],
                                     RealType c[3]) {
  // these extra __dmul_rn calls are needed to preserve bitwise
  // anticommutativity i.e. cross(a,b) is bitwise identical to -cross(b,a)
  // except in the sign-bit
  c[0] = rmul_rn(a[1], b[2]) - rmul_rn(a[2], b[1]);
  c[1] = rmul_rn(a[2], b[0]) - rmul_rn(a[0], b[2]);
  c[2] = rmul_rn(a[0], b[1]) - rmul_rn(a[1], b[0]);
}

template <typename RealType>
__global__ void k_invert_array(const size_t N, RealType *__restrict__ arr) {
  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < N) {
    arr[kernel_idx] = 1 / arr[kernel_idx];

    kernel_idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
