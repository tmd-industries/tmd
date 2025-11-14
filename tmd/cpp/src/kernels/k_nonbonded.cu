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

#include "k_nonbonded.cuh"

namespace tmd {

void __global__ k_setup_nblist_row_and_column_indices(
    const int num_systems, const int N, const int *__restrict__ row_idx_counts,
    const int *__restrict__ col_idx_counts, const bool is_disjoint,
    unsigned int *__restrict__ row_idxs, unsigned int *__restrict__ col_idxs) {

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  const int row_count = row_idx_counts[system_idx];
  const int col_count = col_idx_counts[system_idx];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < row_count || idx < col_count) {

    if (idx < row_count) {
      row_idxs[system_idx * N + idx] = idx;
    }
    if (idx < col_count) {
      if (is_disjoint) {
        col_idxs[system_idx * N + idx] = idx + row_count;
      } else {
        col_idxs[system_idx * N + idx] = idx;
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
