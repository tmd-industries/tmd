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

#include "hilbert_sort.hpp"
#include "neighborlist.hpp"
#include "nonbonded_common.hpp"
#include "potential.hpp"
#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace tmd {

enum class NonbondedInteractionType { DISJOINT, OVERLAPPING };

template <typename RealType>
class NonbondedInteractionGroup : public Potential<RealType> {

  typedef void (*k_nonbonded_fn)(const int N, const int NR,
                                 const unsigned int *__restrict__ ixn_count,
                                 const unsigned int *__restrict__ d_atom_idxs,
                                 const RealType *__restrict__ coords,
                                 const RealType *__restrict__ params, // [N]
                                 const RealType *__restrict__ box,
                                 const RealType beta, const RealType cutoff,
                                 const int *__restrict__ ixn_tiles,
                                 const unsigned int *__restrict__ ixn_atoms,
                                 unsigned long long *__restrict__ du_dx,
                                 unsigned long long *__restrict__ du_dp,
                                 __int128 *__restrict__ u_buffer);

private:
  const int
      N_; // total number of atoms, i.e. first dimension of input coords, params
  int NR_; // number of row atoms
  int NC_; // number of column atoms

  const NonbondedInteractionType interaction_type_;
  bool compute_col_grads_;

  size_t sum_storage_bytes_;
  void *d_sum_temp_storage_;

  std::array<k_nonbonded_fn, 32> kernel_ptrs_;

  unsigned int *d_col_atom_idxs_;
  unsigned int *d_row_atom_idxs_;

  unsigned int *d_arange_buffer_;

  int *p_ixn_count_; // pinned memory

  const RealType beta_;
  const RealType cutoff_;
  // This is safe to overflow, either reset to 0 or increment
  unsigned int steps_since_last_sort_;
  Neighborlist<RealType> nblist_;

  RealType nblist_padding_;
  __int128 *d_u_buffer_;   // [NONBONDED_KERNEL_BLOCKS]
  RealType *d_nblist_x_;   // coords which were used to compute the nblist
  RealType *d_nblist_box_; // box which was used to rebuild the nblist
  int *m_rebuild_nblist_;  // mapped, zero-copy memory
  int *d_rebuild_nblist_;  // device version

  unsigned int *d_perm_; // hilbert curve permutation

  // "sorted" means
  // - if hilbert sorting enabled, atoms are sorted into contiguous
  //   blocks by interaction group, and each block is hilbert-sorted
  //   independently
  // - otherwise, atoms are sorted into contiguous blocks by
  //   interaction group, with arbitrary ordering within each block
  RealType *d_sorted_x_; // sorted coordinates
  RealType *d_sorted_p_; // sorted parameters

  std::unique_ptr<HilbertSort<RealType>> hilbert_sort_;

  cudaEvent_t nblist_flag_sync_event_; // Event to synchronize rebuild flag on

  const bool disable_hilbert_;

  bool needs_sort();

  void sort(const RealType *d_x, const RealType *d_box, cudaStream_t stream);

  void validate_idxs(const int N, const std::vector<int> &row_atom_idxs,
                     const std::vector<int> &col_atom_idxs,
                     const bool allow_empty);

  int get_max_nonbonded_kernel_blocks() const;

  int get_cur_nonbonded_kernel_blocks() const;

public:
  RealType get_cutoff() const { return cutoff_; };

  void set_nblist_padding(const RealType padding);
  RealType get_nblist_padding() const { return nblist_padding_; };

  void set_compute_col_grads(bool value);
  bool get_compute_col_grads() const { return compute_col_grads_; };

  RealType get_beta() const { return beta_; };

  int get_num_col_idxs() const { return NC_; };
  int get_num_row_idxs() const { return NR_; };

  std::vector<int> get_row_idxs() const;
  std::vector<int> get_col_idxs() const;

  void set_atom_idxs_device(const int NR, const int NC,
                            unsigned int *d_row_idxs,
                            unsigned int *d_column_idxs,
                            const cudaStream_t stream);

  void set_atom_idxs(const std::vector<int> &row_atom_idxs,
                     const std::vector<int> &col_atom_idxs);

  NonbondedInteractionGroup(const int N, const std::vector<int> &row_atom_idxs,
                            const std::vector<int> &col_atom_idxs,
                            const RealType beta, const RealType cutoff,
                            const bool disable_hilbert_sort = false,
                            const RealType nblist_padding = 0.1);

  ~NonbondedInteractionGroup();

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx,
                              unsigned long long *d_du_dp, __int128 *d_u,
                              cudaStream_t stream) override;

  void du_dp_fixed_to_float(const int N, const int P,
                            const unsigned long long *du_dp,
                            RealType *du_dp_float) override;
};

} // namespace tmd
