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

namespace tmd {

void __global__ k_scatter_accumulate_system_energies(
    const int *__restrict__ reductions, // [1]
    const int *__restrict__ indices, // [number of reductions] The replica each
                                     // energy is associated with
    const __int128 *__restrict__ in_nrgs, // [number of reductions]
    __int128 *__restrict__ out_nrgs) {
  const int num_reductions = *reductions;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < num_reductions) {
    const int output_idx = indices[idx];
    out_nrgs[output_idx] += in_nrgs[idx];

    idx += gridDim.x * blockDim.x;
  }
}

EnergyAccumulator::EnergyAccumulator(const int batches, const int total_size)
    : batches_(batches), max_buffer_size_(total_size), temp_storage_bytes_(0) {
  assert(batches_ >= 1);
  __int128 *dummy_nrg_buffer = nullptr;
  if (max_buffer_size_ > 0 && max_buffer_size_ < batches_) {
    throw std::runtime_error(
        "Max buffer size must be larger than the number of batches, got " +
        std::to_string(max_buffer_size_) + " max buffer size and " +
        std::to_string(batches_) + "batches");
  }
  if (batches_ == 1) {
    gpuErrchk(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_,
                                     dummy_nrg_buffer, dummy_nrg_buffer,
                                     max_buffer_size_));
  } else {
    cudaSafeMalloc(&d_reductions_out_, sizeof(*d_reductions_out_));
    cudaSafeMalloc(&d_sorted_idxs_, max_buffer_size_ * sizeof(*d_sorted_idxs_));
    cudaSafeMalloc(&d_idxs_unique_, max_buffer_size_ * sizeof(*d_idxs_unique_));
    cudaSafeMalloc(&d_u_intermediate_,
                   max_buffer_size_ * sizeof(*d_u_intermediate_));
    cudaSafeMalloc(&d_sorted_nrgs_, max_buffer_size_ * sizeof(*d_sorted_nrgs_));

    size_t sort_bytes = 0;
    gpuErrchk(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, d_idxs_unique_, d_idxs_unique_, dummy_nrg_buffer,
        dummy_nrg_buffer, max_buffer_size_));

    size_t reduce_bytes = 0;
    CUBSumOp reduction_op;
    gpuErrchk(cub::DeviceReduce::ReduceByKey(
        nullptr, reduce_bytes, d_idxs_unique_, d_idxs_unique_, dummy_nrg_buffer,
        dummy_nrg_buffer, d_reductions_out_, reduction_op, max_buffer_size_));
    temp_storage_bytes_ = max(reduce_bytes, sort_bytes);
  }

  gpuErrchk(cudaMalloc(&d_temp_storage_buffer_, temp_storage_bytes_));
}

EnergyAccumulator::~EnergyAccumulator() {
  gpuErrchk(cudaFree(d_temp_storage_buffer_));
  if (batches_ > 1) {
    gpuErrchk(cudaFree(d_reductions_out_));
    gpuErrchk(cudaFree(d_idxs_unique_));
    gpuErrchk(cudaFree(d_u_intermediate_));
    gpuErrchk(cudaFree(d_sorted_nrgs_));
    gpuErrchk(cudaFree(d_sorted_idxs_));
  }
};

void EnergyAccumulator::sum_device(const int num_vals, const __int128 *d_nrg_in,
                                   const int *d_system_idxs,
                                   __int128 *d_nrg_out, cudaStream_t stream) {
  if (num_vals > max_buffer_size_) {
    throw std::runtime_error(
        "Number of values larger than the maximum buffer size");
  }
  if (batches_ == 1) {
    gpuErrchk(cub::DeviceReduce::Sum(d_temp_storage_buffer_,
                                     temp_storage_bytes_, d_nrg_in, d_nrg_out,
                                     num_vals, stream));
  } else {
    CUBSumOp reduction_op;

    // Clear the buffer
    gpuErrchk(cudaMemsetAsync(d_u_intermediate_, 0,
                              max_buffer_size_ * sizeof(*d_nrg_out), stream));
    // Copy any values written to the batch
    gpuErrchk(cudaMemcpyAsync(d_u_intermediate_, d_nrg_out,
                              batches_ * sizeof(*d_nrg_out),
                              cudaMemcpyDeviceToDevice, stream));

    gpuErrchk(cub::DeviceRadixSort::SortPairs(
        d_temp_storage_buffer_, temp_storage_bytes_, d_system_idxs,
        d_sorted_idxs_, d_nrg_in, d_sorted_nrgs_, num_vals,
        0,                           // begin key bit
        sizeof(*d_sorted_idxs_) * 8, // end key bit
        stream));

    gpuErrchk(cub::DeviceReduce::ReduceByKey(
        d_temp_storage_buffer_, temp_storage_bytes_, d_sorted_idxs_,
        d_idxs_unique_, d_sorted_nrgs_, d_u_intermediate_, d_reductions_out_,
        reduction_op, num_vals, stream));

    // Copy the values back to the output in the correct order
    k_scatter_accumulate_system_energies<<<
        ceil_divide(batches_, DEFAULT_THREADS_PER_BLOCK),
        DEFAULT_THREADS_PER_BLOCK, 0, stream>>>(
        d_reductions_out_, d_idxs_unique_, d_u_intermediate_, d_nrg_out);
    gpuErrchk(cudaPeekAtLastError());
  }
}

} // namespace tmd
