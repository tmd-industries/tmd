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

#include "k_indices.cuh"
#include "stdio.h"

namespace tmd {
// Takes a source and destination array.
// The value of the src is used as the index and the value in the destination
// array. Allows combining a series of indices to get a unique set of values.
void __global__ k_set_value_to_idx(const int N, // Number of values in src
                                   const int K, // Number of values in dest
                                   const unsigned int *__restrict__ src,
                                   unsigned int *__restrict__ dest) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  const unsigned int val = src[idx];
  if (val >= K) {
    return;
  }
  dest[val] = val;
}

// Any value that is >=N becomes the idx and any value that is an idx becomes N
void __global__ k_invert_indices(const int N, unsigned int *__restrict__ arr) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  arr[idx] = arr[idx] >= N ? idx : N;
}

template <typename T>
void __global__ k_arange(const size_t N, T *__restrict__ arr) {
  const T atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom_idx >= N) {
    return;
  }
  arr[atom_idx] = atom_idx;
}

template void __global__ k_arange<int>(const size_t, int *__restrict__ arr);
template void __global__ k_arange<unsigned int>(const size_t,
                                                unsigned int *__restrict__ arr);

template <typename T>
void __global__ k_fill(const size_t N, T *__restrict__ arr, const T val) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < N) {
    arr[idx] = val;
    idx += gridDim.x * blockDim.x;
  }
}

template void __global__ k_fill<int>(const size_t, int *__restrict__ arr,
                                     const int);
template void __global__ k_fill<unsigned int>(const size_t,
                                              unsigned int *__restrict__ arr,
                                              const unsigned int);
template void __global__ k_fill<float>(const size_t, float *__restrict__ arr,
                                       const float);
template void __global__ k_fill<double>(const size_t, double *__restrict__ arr,
                                        const double);

template <typename T>
void __global__ k_segment_arange(const size_t num_segments,
                                 const size_t elements_per_segment,
                                 T *__restrict__ arr) {
  int segment_idx = blockIdx.y;
  while (segment_idx < num_segments) {
    T arr_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (arr_idx < elements_per_segment) {
      arr[elements_per_segment * segment_idx + arr_idx] = arr_idx;
      arr_idx += gridDim.x * blockDim.x;
    }
    segment_idx += gridDim.y * blockDim.y;
  }
}

template void __global__ k_segment_arange<int>(const size_t, const size_t,
                                               int *__restrict__ arr);
template void __global__ k_segment_arange<unsigned int>(
    const size_t, const size_t, unsigned int *__restrict__ arr);

} // namespace tmd
