// Copyright 2025, Forrest York
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
#include "assert.h"
#include "cub_utils.cuh"
#include "energy_accum.hpp"
#include "gpu_utils.cuh"
#include <cub/cub.cuh>

#include "device_buffer.hpp"
#include "fixed_point.hpp"
#include <vector>

namespace tmd {

// An incredibly bad implementation to handle reducing by key. Can be improved
// by using shared memory to accumulate within threads
void __global__ k_reduce_energies_by_system(const size_t num_systems,
                                            const size_t N,
                                            const __int128 *__restrict__ nrg_in,
                                            const int *__restrict__ system_idxs,
                                            __int128 *__restrict__ nrg_out) {

  assert(gridDim.x * blockDim.x == 1);
  const size_t sys_idx = blockIdx.y;
  if (sys_idx >= num_systems) {
    return;
  }
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  __int128 accum = 0;
  while (idx < N) {
    accum += system_idxs[idx] == sys_idx ? nrg_in[idx] : 0;

    idx += 1;
  }

  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    nrg_out[sys_idx] = accum;
  }
}

EnergyAccumulator::EnergyAccumulator(const int batches, const int total_size)
    : batches_(batches), max_buffer_size_(total_size), temp_storage_bytes_(0) {
  assert(batches_ >= 1);
  __int128 *dummy_nrg_buffer = nullptr;
  size_t reduce_bytes = 0;
  gpuErrchk(cub::DeviceReduce::Sum(nullptr, reduce_bytes, dummy_nrg_buffer,
                                   dummy_nrg_buffer, max_buffer_size_));
  size_t sort_bytes = 0;
  // if (batches_ > 1) {
  //   // This is safe as long as the number of batches is accurate to the idxs
  //   cudaSafeMalloc(&d_sorted_nrgs_,
  //                  batches_ * max_buffer_size_ * sizeof(*d_sorted_nrgs_));
  //   cudaSafeMalloc(&d_sort_keys_out_,
  //                  batches_ * max_buffer_size_ * sizeof(*d_sort_keys_out_));
  //   // The reduction op is buggy: https://github.com/NVIDIA/cccl/issues/3890
  //   gpuErrchk(cub::DeviceRadixSort::SortPairs(
  //       nullptr, sort_bytes, d_sort_keys_out_, d_sort_keys_out_,
  //       d_sorted_nrgs_, d_sorted_nrgs_, batches_ * max_buffer_size_));
  //   // CUBSumOp reduction_op;
  //   // gpuErrchk(cub::DeviceReduce::ReduceByKey(
  //   //     nullptr, temp_storage_bytes_, dummy_idxs, d_reductions_out_,
  //   //     dummy_nrg_buffer, dummy_nrg_buffer, d_reductions_out_,
  //   reduction_op,
  //   //     batches * max_buffer_size_));
  // }
  temp_storage_bytes_ = max(reduce_bytes, sort_bytes);
  gpuErrchk(cudaMalloc(&d_sum_temp_storage_, temp_storage_bytes_));
}

EnergyAccumulator::~EnergyAccumulator() {
  gpuErrchk(cudaFree(d_sum_temp_storage_));
  // if (batches_ > 1) {
  //   // gpuErrchk(cudaFree(d_reductions_out_));
  //   // gpuErrchk(cudaFree(d_idxs_unique_));
  //   gpuErrchk(cudaFree(d_sort_keys_out_));
  //   gpuErrchk(cudaFree(d_sorted_nrgs_));
  // }
};

void EnergyAccumulator::sum_device(const int num_vals, const __int128 *d_nrg_in,
                                   const int *d_system_idxs,
                                   __int128 *d_nrg_out, cudaStream_t stream) {

  assert(num_vals <= max_buffer_size_ * batches_);
  if (batches_ == 1) {
    gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, temp_storage_bytes_,
                                     d_nrg_in, d_nrg_out, num_vals, stream));
  } else {
    // JANK

    gpuErrchk(cudaDeviceSynchronize());
    printf("Here\n");
    k_reduce_energies_by_system<<<dim3(1, batches_, 1), 1, 0, stream>>>(
        batches_, num_vals, d_nrg_in, d_system_idxs, d_nrg_out);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // gpuErrchk(cub::DeviceRadixSort::SortPairs(
    //     d_sum_temp_storage_, temp_storage_bytes_, d_system_idxs,
    //     d_sort_keys_out_, d_nrg_in, d_sorted_nrgs_, num_vals, 0,
    //     sizeof(*d_system_idxs) * 8, stream));
    // for (int i = 0; i < batches_; i++) {
    //   gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_,
    //   temp_storage_bytes_,
    //                                    d_sorted_nrgs_ + i * stride,
    //                                    d_nrg_out + i, stride, stream));
    // }
    // TBD: HANDLE THE PERMUTATION IMPLIED BY LOCAL MD
  }
}

} // namespace tmd
