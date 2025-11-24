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

EnergyAccumulator::EnergyAccumulator(const int batches, const int total_size)
    : batches_(batches), max_buffer_size_(total_size), temp_storage_bytes_(0) {
  assert(batches_ >= 1);
  __int128 *dummy_nrg_buffer = nullptr;
  if (batches_ == 1) {
    gpuErrchk(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes_,
                                     dummy_nrg_buffer, dummy_nrg_buffer,
                                     max_buffer_size_));
  } else {
    // Allocates more memory than strictly needed, but would rather be safe
    // if indices are greater than batches, the results will be hosed
    cudaSafeMalloc(&d_reductions_out_,
                   max_buffer_size_ * batches_ * sizeof(*d_reductions_out_));
    cudaSafeMalloc(&d_idxs_unique_,
                   max_buffer_size_ * batches_ * sizeof(*d_idxs_unique_));
    // If this isn't large enough can hit invalid memory writes
    cudaSafeMalloc(&d_u_intermediate_,
                   max_buffer_size_ * batches_ * sizeof(*d_u_intermediate_));
    int *dummy_idxs = nullptr;
    CUBSumOp reduction_op;
    gpuErrchk(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_bytes_, dummy_idxs, d_idxs_unique_,
        dummy_nrg_buffer, dummy_nrg_buffer, d_reductions_out_, reduction_op,
        max_buffer_size_ * batches_));
  }

  gpuErrchk(cudaMalloc(&d_sum_temp_storage_, temp_storage_bytes_));
}

EnergyAccumulator::~EnergyAccumulator() {
  gpuErrchk(cudaFree(d_sum_temp_storage_));
  if (batches_ > 1) {
    gpuErrchk(cudaFree(d_reductions_out_));
    gpuErrchk(cudaFree(d_idxs_unique_));
    gpuErrchk(cudaFree(d_u_intermediate_));
  }
};

void EnergyAccumulator::sum_device(const int num_vals, const __int128 *d_nrg_in,
                                   const int *d_system_idxs,
                                   __int128 *d_nrg_out, cudaStream_t stream) {

  if (batches_ == 1) {
    gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, temp_storage_bytes_,
                                     d_nrg_in, d_nrg_out, num_vals, stream));
  } else {
    CUBSumOp reduction_op;

    // Clear the buffer
    gpuErrchk(cudaMemsetAsync(d_u_intermediate_, 0,
                              batches_ * sizeof(*d_nrg_out), stream));
    // Copy any values written to the batch
    gpuErrchk(cudaMemcpyAsync(d_u_intermediate_, d_nrg_out,
                              batches_ * sizeof(*d_nrg_out),
                              cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cub::DeviceReduce::ReduceByKey(
        d_sum_temp_storage_, temp_storage_bytes_, d_system_idxs, d_idxs_unique_,
        d_nrg_in, d_u_intermediate_, d_reductions_out_, reduction_op, num_vals,
        stream));

    // Copy the values back to the output
    gpuErrchk(cudaMemcpyAsync(d_nrg_out, d_u_intermediate_,
                              batches_ * sizeof(*d_nrg_out),
                              cudaMemcpyDeviceToDevice, stream));
    // TBD: HANDLE THE PERMUTATION IMPLIED BY LOCAL MD
  }
}

} // namespace tmd
