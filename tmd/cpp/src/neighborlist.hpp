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
#include "math_utils.cuh"
#include <vector>

namespace tmd {

// enable 64bit stuff later
template <typename RealType> class Neighborlist {

private:
  const int num_systems_;     // Number of systems neighborlist runs over
  const int max_system_size_; // Max number of atoms in a system
  int N_;                     // Number of atoms in each system
  std::vector<int> column_idx_counts_; // [num_systems_] Number of atoms in
                                       // column, N_ for each system by default
  std::vector<int> row_idx_counts_; // [num_systems_] Number of atoms in row, N_
                                    // for each system by default

  const bool compute_upper_triangular_;

  RealType *d_row_block_bounds_ctr_;
  RealType *d_row_block_bounds_ext_;
  RealType *d_column_block_bounds_ctr_;
  RealType *d_column_block_bounds_ext_;

  unsigned int *d_row_idxs_;
  unsigned int *d_row_idx_counts_;

  unsigned int *d_column_idxs_;
  unsigned int *d_column_idx_counts_;

  unsigned int *d_ixn_count_;
  int *d_ixn_tiles_;
  unsigned int *d_ixn_atoms_;
  unsigned int *d_trim_atoms_;

public:
  // N - number of atoms
  Neighborlist(const int num_systems, const int N,
               bool compute_upper_triangular);

  ~Neighborlist();

  void set_row_idxs(std::vector<unsigned int> &idxs);

  void reset_row_idxs();

  void reset_row_idxs_device(const cudaStream_t stream);

  void resize(const int size);

  void resize_device(const int size, const cudaStream_t stream);

  void set_row_idxs_and_col_idxs(std::vector<unsigned int> &row_idxs,
                                 std::vector<unsigned int> &col_idxs);

  // Override to handle batches of row/column indices
  void
  set_row_idxs_and_col_idxs(std::vector<std::vector<unsigned int>> &row_idxs,
                            std::vector<std::vector<unsigned int>> &col_idxs);

  void set_idxs_device(const int NR, const int NC, unsigned int *row_idxs,
                       unsigned int *column_idxs, const cudaStream_t stream);

  void set_idxs_device(
      const int *d_row_counts, const int *d_col_counts,
      unsigned int *d_in_row_idxs, // [num_systems, max_system_size] expected to
                                   // be padded with max_system_size
      unsigned int *d_in_column_idxs, // [num_systems, max_system_size] expected
                                      // to be padded with max_system_size
      const cudaStream_t stream);

  std::vector<unsigned int> num_tile_ixns();

  std::vector<std::vector<std::vector<int>>>
  get_nblist_host(const int num_systems, const int N, const RealType *h_coords,
                  const RealType *h_box, const RealType cutoff,
                  const RealType padding);

  void build_nblist_device(const int N, const RealType *d_coords,
                           const RealType *d_box, const RealType cutoff,
                           const RealType padding, const cudaStream_t stream);

  void compute_block_bounds_host(const int num_systems, const int N,
                                 const RealType *h_coords,
                                 const RealType *h_box, RealType *h_bb_ctrs,
                                 RealType *h_bb_exts);

  unsigned int *get_ixn_atoms() { return d_ixn_atoms_; };

  int *get_ixn_tiles() { return d_ixn_tiles_; };

  unsigned int *get_ixn_count() { return d_ixn_count_; };

  unsigned int *get_row_idxs() { return d_row_idxs_; };

  unsigned int *get_num_row_idxs() { return d_row_idx_counts_; };

  int get_num_col_idxs() { return N_; };

  // get max number of row blocks
  int num_row_blocks() const;
  // get max number of column blocks
  int num_column_blocks() const;

  int get_num_systems() const { return num_systems_; };

  // get max number of interactions
  int max_ixn_count() const;

private:
  // Sum of all row idx sizes
  int total_row_idxs() const;

  // Sum of all column idx sizes
  int total_column_idxs() const;

  // Indicates that should only compute the upper triangle of the interactions
  // matrix, otherwise will compute the entire matrix.
  bool compute_upper_triangular() const;

  // The number of column blocks divided by warp size. Each thread handles a
  // block
  int Y() const;

  void compute_block_bounds_device(const int N, const int D,
                                   const RealType *d_coords,
                                   const RealType *d_box, cudaStream_t stream);
};

} // namespace tmd
