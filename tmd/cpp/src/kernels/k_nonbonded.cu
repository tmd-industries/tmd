// Copyright 2025-2026, Forrest York
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
    const int num_systems, const int N,
    const int *__restrict__ row_idx_counts, // [num_systems]
    const int *__restrict__ col_idx_counts, // [num_systems]
    const bool is_disjoint,
    unsigned int *__restrict__ row_idxs, // [num_systems, N]
    unsigned int *__restrict__ col_idxs  // [num_systems, N]
) {

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  const int row_count = row_idx_counts[system_idx];
  const int col_count = col_idx_counts[system_idx];
  const int idx_offset = system_idx * N;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < row_count || idx < col_count) {

    if (idx < row_count) {
      row_idxs[idx_offset + idx] = idx;
    }
    if (idx < col_count) {
      col_idxs[idx_offset + idx] = is_disjoint ? idx + row_count : idx;
    }

    idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
