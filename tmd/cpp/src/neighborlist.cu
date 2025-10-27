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

#include <algorithm>
#include <numeric>

#include "assert.h"
#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_neighborlist.cuh"
#include "kernels/k_nonbonded_common.cuh"
#include "neighborlist.hpp"
#include "set_utils.hpp"

namespace tmd {

static int get_idx_total_blocks(const std::vector<int> &idx_counts,
                                const int tile_size) {
  int total_blocks = 0;
  // Need to round up for each set of columns to ensure correct padding
  for (int i = 0; i < idx_counts.size(); i++) {
    total_blocks += ceil_divide(idx_counts[i], tile_size);
  }
  return total_blocks;
}

template <typename RealType>
Neighborlist<RealType>::Neighborlist(const int num_systems, const int N,
                                     const bool compute_upper_triangular)
    : num_systems_(num_systems), max_system_size_(N), N_(N),
      column_idx_counts_(num_systems_), row_idx_counts_(num_systems_),
      compute_upper_triangular_(compute_upper_triangular) {

  if (num_systems == 0) {
    throw std::runtime_error("Neighborlist num_systems must be at least 1");
  }
  if (N == 0) {
    throw std::runtime_error("Neighborlist N must be at least 1");
  }
  // Populate the initial row/column indices
  std::fill(column_idx_counts_.begin(), column_idx_counts_.end(), N_);
  std::fill(row_idx_counts_.begin(), row_idx_counts_.end(), N_);
  const int column_blocks = this->num_column_blocks();
  const int row_blocks = this->num_row_blocks();
  const int Y = this->Y();

  const unsigned long long MAX_TILE_BUFFER =
      num_systems_ * row_blocks * column_blocks;
  const unsigned long long MAX_ATOM_BUFFER =
      num_systems_ * this->max_ixn_count();

  // interaction buffers
  // TBD: Should the ixn count be a single value since use case for this
  // downstream is a single kernel (batched nonbonded kernel)
  cudaSafeMalloc(&d_ixn_count_, num_systems_ * sizeof(*d_ixn_count_));
  cudaSafeMalloc(&d_ixn_tiles_, MAX_TILE_BUFFER * sizeof(*d_ixn_tiles_));
  cudaSafeMalloc(&d_ixn_atoms_, MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_));
  cudaSafeMalloc(&d_trim_atoms_, num_systems_ * column_blocks * Y * TILE_SIZE *
                                     sizeof(*d_trim_atoms_));

  // bounding box buffers
  cudaSafeMalloc(&d_row_block_bounds_ctr_,
                 num_systems_ * row_blocks * 3 *
                     sizeof(*d_row_block_bounds_ctr_));
  cudaSafeMalloc(&d_row_block_bounds_ext_,
                 num_systems_ * row_blocks * 3 *
                     sizeof(*d_row_block_bounds_ext_));
  cudaSafeMalloc(&d_column_block_bounds_ctr_,
                 num_systems_ * column_blocks * 3 *
                     sizeof(*d_column_block_bounds_ctr_));
  cudaSafeMalloc(&d_column_block_bounds_ext_,
                 num_systems_ * column_blocks * 3 *
                     sizeof(*d_column_block_bounds_ext_));

  // Row and column indice arrays
  cudaSafeMalloc(&d_column_idxs_,
                 num_systems_ * max_system_size_ * sizeof(*d_column_idxs_));
  cudaSafeMalloc(&d_column_system_idxs_, num_systems_ * max_system_size_ *
                                             sizeof(*d_column_system_idxs_));
  cudaSafeMalloc(&d_column_idx_counts_,
                 num_systems_ * sizeof(*d_column_idx_counts_));
  // // Increment by 1 as the first value will be zero
  // cudaSafeMalloc(&d_column_idx_offsets_,
  //                (num_systems_ + 1) * sizeof(*d_column_idx_counts_));
  // gpuErrchk(
  //     cudaMemset(d_column_idx_offsets_, 0, (num_systems_ + 1) *
  //     sizeof(*d_column_idx_offsets_)));

  cudaSafeMalloc(&d_row_idxs_,
                 num_systems_ * max_system_size_ * sizeof(*d_row_idxs_));
  cudaSafeMalloc(&d_row_system_idxs_,
                 num_systems_ * max_system_size_ * sizeof(*d_row_system_idxs_));
  cudaSafeMalloc(&d_row_idx_counts_, num_systems_ * sizeof(*d_row_idx_counts_));
  // Increment by 1 as the first value will be zero
  // cudaSafeMalloc(&d_row_idx_offsets_,
  //                (num_systems_ + 1) * sizeof(*d_row_idx_counts_));
  // gpuErrchk(
  //     cudaMemset(d_row_idx_offsets_, 0, (num_systems_ + 1) *
  //     sizeof(*d_row_idx_offsets_)));

  // Since d_column_idx_counts_ and d_row_idx_counts_ are the same length can be
  // used for both gpuErrchk(cub::DeviceScan::InclusiveSum(
  //     nullptr, temp_sum_bytes_, d_column_idx_counts_,
  //     d_column_idxs_offsets_, num_systems_));

  // cudaSafeMalloc(&d_temp_storage_buffer_, temp_sum_bytes_);

  this->reset_row_idxs();
}

template <typename RealType> Neighborlist<RealType>::~Neighborlist() {
  gpuErrchk(cudaFree(d_column_idxs_));
  gpuErrchk(cudaFree(d_column_system_idxs_));
  gpuErrchk(cudaFree(d_column_idx_counts_));
  gpuErrchk(cudaFree(d_row_idxs_));
  gpuErrchk(cudaFree(d_row_system_idxs_));
  gpuErrchk(cudaFree(d_row_idx_counts_));

  gpuErrchk(cudaFree(d_ixn_count_));
  gpuErrchk(cudaFree(d_ixn_tiles_));
  gpuErrchk(cudaFree(d_ixn_atoms_));
  gpuErrchk(cudaFree(d_trim_atoms_));

  gpuErrchk(cudaFree(d_row_block_bounds_ctr_));
  gpuErrchk(cudaFree(d_row_block_bounds_ext_));
  gpuErrchk(cudaFree(d_column_block_bounds_ctr_));
  gpuErrchk(cudaFree(d_column_block_bounds_ext_));

  // gpuErrchk(cudaFree(d_temp_storage_buffer_));
}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_host(const int N,
                                                       const RealType *h_coords,
                                                       const RealType *h_box,
                                                       RealType *h_bb_ctrs,
                                                       RealType *h_bb_exts) {

  const int D = 3;
  DeviceBuffer<RealType> d_coords(num_systems_ * N * D);
  DeviceBuffer<RealType> d_box(num_systems_ * D * D);

  d_coords.copy_from(h_coords);
  d_box.copy_from(h_box);

  this->compute_block_bounds_device(N, D, d_coords.data, d_box.data,
                                    static_cast<cudaStream_t>(0));
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(h_bb_ctrs, d_column_block_bounds_ctr_,
                       this->num_column_blocks() * 3 *
                           sizeof(*d_column_block_bounds_ctr_),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_bb_exts, d_column_block_bounds_ext_,
                       this->num_column_blocks() * 3 *
                           sizeof(*d_column_block_bounds_ext_),
                       cudaMemcpyDeviceToHost));
}

// Return the number of tiles that interact
template <typename RealType>
unsigned int Neighborlist<RealType>::num_tile_ixns() {
  unsigned int h_ixn_count;
  gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count_, 1 * sizeof(*d_ixn_count_),
                       cudaMemcpyDeviceToHost));
  return h_ixn_count;
}

template <typename RealType>
std::vector<std::vector<int>> Neighborlist<RealType>::get_nblist_host(
    int N, const RealType *h_coords, const RealType *h_box,
    const RealType cutoff, const RealType padding) {

  if (N != N_) {
    throw std::runtime_error("N != N_");
  }

  DeviceBuffer<RealType> d_coords(N * 3);
  DeviceBuffer<RealType> d_box(3 * 3);
  d_coords.copy_from(h_coords);
  d_box.copy_from(h_box);

  this->build_nblist_device(N, d_coords.data, d_box.data, cutoff, padding,
                            static_cast<cudaStream_t>(0));

  gpuErrchk(cudaDeviceSynchronize());
  const int column_blocks = this->num_column_blocks();
  const int row_blocks = this->num_row_blocks();

  unsigned long long MAX_TILE_BUFFER = row_blocks * column_blocks;
  unsigned long long MAX_ATOM_BUFFER = this->max_ixn_count();

  unsigned int h_ixn_count;
  gpuErrchk(cudaMemcpy(&h_ixn_count, d_ixn_count_, 1 * sizeof(*d_ixn_count_),
                       cudaMemcpyDeviceToHost));
  std::vector<int> h_ixn_tiles(MAX_TILE_BUFFER);
  std::vector<unsigned int> h_ixn_atoms(MAX_ATOM_BUFFER);
  gpuErrchk(cudaMemcpy(&h_ixn_tiles[0], d_ixn_tiles_,
                       MAX_TILE_BUFFER * sizeof(*d_ixn_tiles_),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&h_ixn_atoms[0], d_ixn_atoms_,
                       MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_),
                       cudaMemcpyDeviceToHost));

  std::vector<std::vector<int>> ixn_list(row_blocks, std::vector<int>());
  for (int i = 0; i < h_ixn_count; i++) {
    int tile_idx = h_ixn_tiles[i];
    for (int j = 0; j < TILE_SIZE; j++) {
      int atom_j_idx = h_ixn_atoms[i * TILE_SIZE + j];
      if (atom_j_idx < N) {
        ixn_list[tile_idx].push_back(atom_j_idx);
      }
    }
  }

  return ixn_list;
}

template <typename RealType>
void Neighborlist<RealType>::build_nblist_device(
    const int N, const RealType *d_coords, const RealType *d_box,
    const RealType cutoff, const RealType padding, const cudaStream_t stream) {

  const int D = 3;
  this->compute_block_bounds_device(N, D, d_coords, d_box, stream);
  const int tpb = TILE_SIZE;
  const int row_blocks = this->num_row_blocks();
  const int Y = this->Y();

  dim3 dimGrid(row_blocks, Y, 1); // block x, y, z dims

  // (ytz): TBD shared memory, stream
  if (this->compute_upper_triangular()) {
    k_find_blocks_with_ixns<RealType, true><<<dimGrid, tpb, 0, stream>>>(
        num_systems_, N_, d_column_idx_counts_, d_row_idx_counts_,
        d_column_idxs_, d_row_idxs_, d_column_block_bounds_ctr_,
        d_column_block_bounds_ext_, d_column_block_bounds_ctr_,
        d_column_block_bounds_ext_, d_coords, d_box, d_ixn_count_, d_ixn_tiles_,
        d_ixn_atoms_, d_trim_atoms_, cutoff, padding);
  } else {
    k_find_blocks_with_ixns<RealType, false><<<dimGrid, tpb, 0, stream>>>(
        num_systems_, N_, d_column_idx_counts_, d_row_idx_counts_,
        d_column_idxs_, d_row_idxs_, d_column_block_bounds_ctr_,
        d_column_block_bounds_ext_, d_row_block_bounds_ctr_,
        d_row_block_bounds_ext_, d_coords, d_box, d_ixn_count_, d_ixn_tiles_,
        d_ixn_atoms_, d_trim_atoms_, cutoff, padding);
  }

  gpuErrchk(cudaPeekAtLastError());
  k_compact_trim_atoms<<<row_blocks, tpb, 0, stream>>>(
      N_, Y, d_trim_atoms_, d_ixn_count_, d_ixn_tiles_, d_ixn_atoms_);

  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Neighborlist<RealType>::compute_block_bounds_device(
    const int N,              // Number of atoms
    const int D,              // Box dimensions
    const RealType *d_coords, // [N*3]
    const RealType *d_box,    // [D*3]
    const cudaStream_t stream) {

  if (D != 3) {
    throw std::runtime_error("D != 3");
  }

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const int max_col_idxs =
      *std::max_element(column_idx_counts_.begin(), column_idx_counts_.end());
  dim3 col_dim_grid(ceil_divide(max_col_idxs, tpb), num_systems_, 1);
  printf("Max col idxs %d\n", max_col_idxs);
  k_find_block_bounds<RealType><<<col_dim_grid, tpb, 0, stream>>>(
      num_systems_, N_, d_column_idx_counts_, d_column_idxs_, d_coords, d_box,
      d_column_block_bounds_ctr_, d_column_block_bounds_ext_, d_ixn_count_);
  gpuErrchk(cudaPeekAtLastError());
  // In the case of upper triangle of the matrix, the column and row indices are
  // the same, so only compute block ixns for both when they are different

  // We have three possible scenarios:
  // If we compute upper triangular:
  // - we're in the all-pairs case, row_idxs are equal to col_idxs.
  // - we're in the overlapping rectangular case, row_idxs are a subset of
  // col_idxs. If we do not compute upper triangular:
  // - we're in the disjoint rectangular case, row_idxs need to be processed as
  // well.
  if (!this->compute_upper_triangular()) {
    const int max_row_idxs =
        *std::max_element(row_idx_counts_.begin(), row_idx_counts_.end());
    dim3 row_dim_grid(ceil_divide(max_row_idxs, tpb), num_systems_, 1);

    k_find_block_bounds<RealType><<<row_dim_grid, tpb, 0, stream>>>(
        num_systems_, N_, d_row_idx_counts_, d_row_idxs_, d_coords, d_box,
        d_row_block_bounds_ctr_, d_row_block_bounds_ext_, d_ixn_count_);
    gpuErrchk(cudaPeekAtLastError());
  }
};

template <typename RealType>
void Neighborlist<RealType>::set_compute_upper_triangular(bool val) {
  compute_upper_triangular_ = val;
}

template <typename RealType>
void Neighborlist<RealType>::set_row_idxs(std::vector<unsigned int> &row_idxs) {
  std::set<unsigned int> unique_row_idxs(row_idxs.begin(), row_idxs.end());
  std::vector<unsigned int> col_idxs =
      get_indices_difference<unsigned int>(N_, unique_row_idxs);
  this->set_row_idxs_and_col_idxs(row_idxs, col_idxs);
}

template <typename RealType>
void Neighborlist<RealType>::set_row_idxs_and_col_idxs(
    std::vector<unsigned int> &row_idxs, std::vector<unsigned int> &col_idxs) {
  if (row_idxs.size() == 0) {
    throw std::runtime_error("idxs can't be empty");
  }
  std::set<unsigned int> unique_row_idxs(row_idxs.begin(), row_idxs.end());
  if (unique_row_idxs.size() != row_idxs.size()) {
    throw std::runtime_error("atom indices must be unique");
  }
  if (row_idxs.size() >= N_) {
    throw std::runtime_error("number of idxs must be less than N");
  }
  if (*std::max_element(row_idxs.begin(), row_idxs.end()) >= N_) {
    throw std::runtime_error("indices values must be less than N");
  }
  if (col_idxs.size() == 0) {
    throw std::runtime_error("idxs can't be empty");
  }
  std::set<unsigned int> unique_col_idxs(col_idxs.begin(), col_idxs.end());
  if (unique_col_idxs.size() != col_idxs.size()) {
    throw std::runtime_error("atom indices must be unique");
  }
  if (col_idxs.size() > N_) {
    throw std::runtime_error("number of col idxs must be <= N");
  }
  if (*std::max_element(col_idxs.begin(), col_idxs.end()) >= N_) {
    throw std::runtime_error("indices values must be less than N");
  }

  const size_t row_count = row_idxs.size();
  const size_t col_count = col_idxs.size();

  DeviceBuffer<unsigned int> row_idx_buffer(row_count);
  DeviceBuffer<unsigned int> col_idx_buffer(col_count);

  row_idx_buffer.copy_from(&row_idxs[0]);
  col_idx_buffer.copy_from(&col_idxs[0]);
  // note, this passes in col/row as opposed to row/col
  this->set_idxs_device(col_count, row_count, col_idx_buffer.data,
                        row_idx_buffer.data, static_cast<cudaStream_t>(0));
  gpuErrchk(cudaDeviceSynchronize());
}

template <typename RealType> void Neighborlist<RealType>::reset_row_idxs() {
  const cudaStream_t stream = static_cast<cudaStream_t>(0);
  this->reset_row_idxs_device(stream);
  gpuErrchk(cudaStreamSynchronize(stream));
}

template <typename RealType>
void Neighborlist<RealType>::reset_row_idxs_device(const cudaStream_t stream) {
  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  dim3 dimGrid(ceil_divide(N_, tpb), num_systems_, 1); // block x, y, z dims
  // Fill the indices with the 0 to N-1 indices, indicating 'normal'
  // neighborlist operation
  // printf("Num Systems %d\n", num_systems_);
  k_segment_arange<unsigned int>
      <<<dimGrid, tpb, 0, stream>>>(num_systems_, N_, d_column_idxs_);
  gpuErrchk(cudaPeekAtLastError());
  k_segment_arange<unsigned int>
      <<<dimGrid, tpb, 0, stream>>>(num_systems_, N_, d_row_idxs_);
  gpuErrchk(cudaPeekAtLastError());

  k_reset_system_idxs<<<dimGrid, tpb, 0, stream>>>(
      num_systems_, N_, d_column_system_idxs_, d_row_system_idxs_);
  gpuErrchk(cudaPeekAtLastError());

  k_fill<<<ceil_divide(num_systems_, tpb), tpb, 0, stream>>>(
      num_systems_, d_column_idx_counts_, static_cast<unsigned int>(N_));
  gpuErrchk(cudaPeekAtLastError());
  k_fill<<<ceil_divide(num_systems_, tpb), tpb, 0, stream>>>(
      num_systems_, d_row_idx_counts_, static_cast<unsigned int>(N_));
  gpuErrchk(cudaPeekAtLastError());

  // // Setup the offsets
  // // Offset output buffers by 1 since the first value is always zero
  // gpuErrchk(cub::DeviceScan::InclusiveSum(
  //     d_temp_storage_buffer_, temp_sum_bytes_, d_column_idx_counts_,
  //     d_column_idxs_offsets_ + 1, num_systems_));

  // gpuErrchk(cub::DeviceScan::InclusiveSum(
  //     d_temp_storage_buffer_, temp_sum_bytes_, d_row_idx_counts_,
  //     d_row_idxs_offsets_ + 1, num_systems_));
}

template <typename RealType>
void Neighborlist<RealType>::resize(const int size) {
  const cudaStream_t stream = static_cast<cudaStream_t>(0);
  this->resize_device(size, stream);
  gpuErrchk(cudaStreamSynchronize(stream));
}

// Resize the Neighborlist to function on a different size. Note that this only
// allows finding interactions on a smaller set of the system, will not increase
// the size of the underlying buffers.
template <typename RealType>
void Neighborlist<RealType>::resize_device(const int size,
                                           const cudaStream_t stream) {
  if (size <= 0) {
    throw std::runtime_error("size is must be at least 1");
  }
  if (size > max_system_size_) {
    throw std::runtime_error(
        "size is greater than max size: " + std::to_string(size) + " > " +
        std::to_string(max_system_size_));
  }
  this->N_ = size;
  this->reset_row_idxs_device(stream);
}

// set_idxs_device is for use when idxs exist on the GPU already and are used as
// the new idxs to compute the neighborlist on.
template <typename RealType>
void Neighborlist<RealType>::set_idxs_device(const int NC, const int NR,
                                             unsigned int *d_in_column_idxs,
                                             unsigned int *d_in_row_idxs,
                                             const cudaStream_t stream) {
  // TBD: Change the input to be vectors
  assert(num_systems_ == 1);
  if (NC > N_) {
    throw std::runtime_error("NC > N_ ");
  }
  if (NR > N_) {
    throw std::runtime_error("NR > N_ ");
  }
  if (NC == 0 || NR == 0) {
    throw std::runtime_error(
        "Number of column and row indices must be non-zero");
  }
  const size_t tpb = DEFAULT_THREADS_PER_BLOCK;

  // initialize both to N_
  k_initialize_array<unsigned int>
      <<<ceil_divide(max_system_size_, tpb), tpb, 0, stream>>>(
          max_system_size_, d_column_idxs_, N_);
  gpuErrchk(cudaPeekAtLastError());
  k_initialize_array<unsigned int>
      <<<ceil_divide(max_system_size_, tpb), tpb, 0, stream>>>(max_system_size_,
                                                               d_row_idxs_, N_);
  gpuErrchk(cudaPeekAtLastError());

  // The indices must already be on the GPU and are copied into the neighborlist
  // buffers.
  gpuErrchk(cudaMemcpyAsync(d_column_idxs_, d_in_column_idxs,
                            num_systems_ * NC * sizeof(*d_column_idxs_),
                            cudaMemcpyDeviceToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_row_idxs_, d_in_row_idxs,
                            num_systems_ * NR * sizeof(*d_row_idxs_),
                            cudaMemcpyDeviceToDevice, stream));

  // Update the row and column counts
  // TBD: Write this to the GPU
  std::fill(row_idx_counts_.begin(), row_idx_counts_.end(), NR);
  std::fill(column_idx_counts_.begin(), column_idx_counts_.end(), NC);

  // TBD: Avoid the memcpy from host to device (should be easily done, but then
  // the host copies need to be updated)
  gpuErrchk(cudaMemcpyAsync(d_column_idx_counts_, &column_idx_counts_[0],
                            num_systems_ * sizeof(*d_column_idx_counts_),
                            cudaMemcpyHostToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_row_idx_counts_, &row_idx_counts_[0],
                            num_systems_ * sizeof(*d_row_idx_counts_),
                            cudaMemcpyHostToDevice, stream));

  // TBD: Decide jank of where num_systems_ gets applied
  const unsigned long long MAX_ATOM_BUFFER =
      num_systems_ * this->max_ixn_count();
  // Clear the atom ixns, to avoid reuse
  // Set to max value, ie greater than N. Note that Memset is on bytes, which is
  // why it is UCHAR_MAX
  gpuErrchk(cudaMemsetAsync(d_ixn_atoms_, UCHAR_MAX,
                            MAX_ATOM_BUFFER * sizeof(*d_ixn_atoms_), stream));
}

template <typename RealType>
bool Neighborlist<RealType>::compute_upper_triangular() const {
  return compute_upper_triangular_;
};

template <typename RealType>
int Neighborlist<RealType>::num_column_blocks() const {
  return get_idx_total_blocks(column_idx_counts_, TILE_SIZE);
  int total_col_blocks = 0;
  // Need to round up for each set of columns to ensure correct padding
  for (int i = 0; i < column_idx_counts_.size(); i++) {
    total_col_blocks += ceil_divide(column_idx_counts_[i], TILE_SIZE);
  }
  return total_col_blocks;
};

template <typename RealType> int Neighborlist<RealType>::Y() const {
  const int max_column_blocks =
      *std::max_element(column_idx_counts_.begin(), column_idx_counts_.end());
  // Doesn't scale with the number of systems
  return ceil_divide(max_column_blocks, WARP_SIZE);
};

template <typename RealType>
int Neighborlist<RealType>::num_row_blocks() const {
  return get_idx_total_blocks(row_idx_counts_, TILE_SIZE);
}

template <typename RealType>
int Neighborlist<RealType>::total_column_idxs() const {
  return std::reduce(column_idx_counts_.begin(), column_idx_counts_.end());
}

template <typename RealType>
int Neighborlist<RealType>::total_row_idxs() const {
  return std::reduce(row_idx_counts_.begin(), row_idx_counts_.end());
}

// max_ixn_count determines the number of tile-atom interaction counts. For each
// tile that interacts with another it can have TILE_SIZE tile-atom
// interactions. Note that d_ixn_count_ is only the number of tile-tile
// interactions, and differs by a factor of TILE_SIZE
template <typename RealType> int Neighborlist<RealType>::max_ixn_count() const {
  // The maximum number of tile-atom interactions, equal to # tile-tile
  // interactions multiplied by TILE_SIZE (typically 32). Use the maximum value
  // of N to compute the size int of the upper triangular matrix to support any
  // set of row indices.
  const int n_blocks = ceil_divide(max_system_size_, TILE_SIZE);
  int max_tile_tile_interactions = (n_blocks * (n_blocks + 1)) / 2;
  // Each tile-tile interaction can have TILE_SIZE tile-atom interactions
  return max_tile_tile_interactions * TILE_SIZE;
}

template class Neighborlist<double>;
template class Neighborlist<float>;

} // namespace tmd
