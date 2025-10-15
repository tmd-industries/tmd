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
#pragma once
#include "cuda_runtime.h"

namespace tmd {

// Utility class for accumulating energies of potentials
class EnergyAccumulator {

private:
  const int batches_;         // Number out of output energies
  const int max_buffer_size_; // Number of energies to accumulate

  // Buffers for finding the CUB call
  std::size_t temp_storage_bytes_;
  char *d_sum_temp_storage_;

  int *d_idxs_unique_;    // [batches]
  int *d_reductions_out_; // [batches]

public:
  EnergyAccumulator(const int batches, const int total_size);

  ~EnergyAccumulator();

  void sum_device(const int num_vals, const __int128 *d_nrg_in,
                  const int *d_system_idxs, __int128 *d_nrg_out,
                  cudaStream_t stream);
};

} // namespace tmd
