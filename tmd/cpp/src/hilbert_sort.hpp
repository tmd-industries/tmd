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

#pragma once
#include "cuda_runtime.h"
#include "device_buffer.hpp"
#include "math_utils.cuh"
#include <vector>

namespace tmd {

template <typename RealType> class HilbertSort {

private:
  const int num_systems_;
  const int N_;
  // used for hilbert sorting
  DeviceBuffer<unsigned int>
      d_bin_to_idx_; // mapping from
                     // HILBERT_GRID_DIMxHILBERT_GRID_DIMxHILBERT_GRID_DIM
                     // grid to hilbert curve index
  DeviceBuffer<unsigned int> d_sort_keys_in_;
  DeviceBuffer<unsigned int> d_sort_keys_out_;
  DeviceBuffer<unsigned int> d_sort_vals_in_;
  DeviceBuffer<char> d_sort_storage_;
  size_t d_sort_storage_bytes_;

public:
  HilbertSort(const int num_systems,
              const int N // number of atoms
  );

  ~HilbertSort();

  void sort_device(const int N, const unsigned int *d_atom_idxs,
                   const RealType *d_coords, const RealType *d_box,
                   unsigned int *d_output_perm, cudaStream_t stream);

  std::vector<unsigned int> sort_host(const int N, const RealType *h_coords,
                                      const RealType *h_box);
};

} // namespace tmd
