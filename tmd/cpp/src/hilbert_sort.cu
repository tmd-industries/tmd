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

#include "gpu_utils.cuh"
#include "hilbert_sort.hpp"
#include "kernels/k_hilbert.cuh"
#include "vendored/hilbert.h"
#include <cub/cub.cuh>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmd {

template <typename RealType>
HilbertSort<RealType>::HilbertSort(const int num_systems, const int N)
    : num_systems_(num_systems), N_(N),
      d_bin_to_idx_(HILBERT_GRID_DIM * HILBERT_GRID_DIM * HILBERT_GRID_DIM),
      d_sort_keys_in_(num_systems_ * N), d_sort_keys_out_(num_systems_ * N), d_sort_vals_in_(num_systems_ * N),
      d_sort_storage_(0), d_sort_storage_bytes_(0) {
  // initialize hilbert curve which maps each of the HILBERT_GRID_DIM x
  // HILBERT_GRID_DIM x HILBERT_GRID_DIM cells into an index.
  std::vector<unsigned int> bin_to_idx(HILBERT_GRID_DIM * HILBERT_GRID_DIM *
                                       HILBERT_GRID_DIM);
  for (int i = 0; i < HILBERT_GRID_DIM; i++) {
    for (int j = 0; j < HILBERT_GRID_DIM; j++) {
      for (int k = 0; k < HILBERT_GRID_DIM; k++) {

        bitmask_t hilbert_coords[3];
        hilbert_coords[0] = i;
        hilbert_coords[1] = j;
        hilbert_coords[2] = k;

        unsigned int bin = static_cast<unsigned int>(
            hilbert_c2i(3, HILBERT_N_BITS, hilbert_coords));
        bin_to_idx[i * HILBERT_GRID_DIM * HILBERT_GRID_DIM +
                   j * HILBERT_GRID_DIM + k] = bin;
      }
    }
  }

  d_bin_to_idx_.copy_from(&bin_to_idx[0]);

  // estimate size needed to do radix sorting
  // reuse d_sort_keys_in_ rather than constructing a dummy output idxs buffer
  gpuErrchk(cub::DeviceRadixSort::SortPairs(
      nullptr, d_sort_storage_bytes_, d_sort_keys_in_.data,
      d_sort_keys_out_.data, d_sort_vals_in_.data, d_sort_keys_in_.data,
      num_items N_));

  d_sort_storage_.realloc(d_sort_storage_bytes_);
}

template <typename RealType> HilbertSort<RealType>::~HilbertSort(){};

template <typename RealType>
void HilbertSort<RealType>::sort_device(
    const int num_systems, const int N, const unsigned int *system_counts, const unsigned int *d_atom_idxs, const RealType *d_coords,
    const RealType *d_box, unsigned int *d_output_perm, cudaStream_t stream) {
  if (num_systems != num_systems_) {
    throw std::runtime_error("number of systems doesn't match");
  }
  if (N > N_) {
    throw std::runtime_error(
        "number of idxs to sort must be less than or equal to N");
  }

  if (N == 0) {
    return;
  }

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 dimGrid(ceil_divide(N, tpb), num_systems_, 1);

  k_coords_to_kv_gather<RealType><<<dimGrid, tpb, 0, stream>>>(
      num_systems, N, system_counts, d_atom_idxs, d_coords, d_box, d_bin_to_idx_.data, d_sort_keys_in_.data,
      d_sort_vals_in_.data);
  gpuErrchk(cudaPeekAtLastError());

  // DeviceSegmentedRadixSort doesn't do quite the task, since compacts the sort
  for (int i = 0; i < num_systems; i++) {
    gpuErrchk(cub::DeviceRadixSort::SortPairs(
        d_sort_storage_.data, d_sort_storage_bytes_, d_sort_keys_in_.data + i * N_,
        d_sort_keys_out_.data + i * N_, d_sort_vals_in_.data + i * N_, d_output_perm + i * N_, N_,
        0,                                 // begin bit
        sizeof(*d_sort_keys_in_.data) * 8, // end bit
        stream                             // cudaStream
        ));
  }
}

template <typename RealType>
std::vector<unsigned int>
HilbertSort<RealType>::sort_host(const int num_systems, const int N, const RealType *h_coords,
                                 const RealType *h_box) {

  std::vector<unsigned int> h_atom_idxs(num_systems * N);
  for (int i = 0; i < num_systems; i++ ) {
    std::iota(h_atom_idxs.begin() + i * N, h_atom_idxs.begin() + (i + 1) * N, 0);
  }

  std::vector<unsigned int> h_system_counts(num_systems, static_cast<unsigned int>(N));

  DeviceBuffer<RealType> d_coords(num_systems * N * 3);
  DeviceBuffer<RealType> d_box(num_systems * 3 * 3);
  DeviceBuffer<unsigned int> d_system_counts(h_system_counts);
  DeviceBuffer<unsigned int> d_atom_idxs(h_atom_idxs);
  DeviceBuffer<unsigned int> d_perm(num_systems * N);

  d_coords.copy_from(h_coords);
  d_box.copy_from(h_box);

  cudaStream_t stream = static_cast<cudaStream_t>(0);
  this->sort_device(num_systems, N, d_atom_idxs.data, d_coords.data, d_box.data, d_perm.data,
                    stream);
  gpuErrchk(cudaStreamSynchronize(stream));

  d_perm.copy_to(&h_atom_idxs[0]);
  return h_atom_idxs;
}

template class HilbertSort<double>;
template class HilbertSort<float>;

} // namespace tmd
