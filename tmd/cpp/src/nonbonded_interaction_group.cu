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

#include <complex>
#include <string>
#include <vector>

#include "assert.h"
#include "device_buffer.hpp"
#include "energy_accum.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "kernels/k_indices.cuh"
#include "nonbonded_common.hpp"
#include "nonbonded_interaction_group.hpp"
#include "set_utils.hpp"

#include "k_nonbonded.cuh"

static const int STEPS_PER_SORT = 200;

static const int MAX_KERNEL_BLOCKS = 4096;

namespace tmd {

static NonbondedInteractionType get_nonbonded_interaction_type(
    const std::vector<std::vector<int>> &row_atom_idxs,
    const std::vector<std::vector<int>> &col_atom_idxs) {

  // row and col idxs must be either:
  // 1) disjoint: row_atom_idxs \intersection overlapping = empty set
  // 2) overlapping: row_atom_idxs == col_atom_idxs[:len(row_atom_idxs)]
  std::optional<NonbondedInteractionType> last_ixn_type;
  if (row_atom_idxs.size() != col_atom_idxs.size()) {
    throw std::runtime_error(
        "row atom batches and column atom batches don't match");
  }
  for (int batch_idx = 0; batch_idx < row_atom_idxs.size(); batch_idx++) {
    bool is_disjoint = true;
    auto row_atoms_batch = row_atom_idxs[batch_idx];
    auto col_atoms_batch = col_atom_idxs[batch_idx];
    std::set<int> unique_row_idxs(row_atoms_batch.begin(),
                                  row_atoms_batch.end());
    for (int col_atom_idx : col_atoms_batch) {
      if (unique_row_idxs.find(col_atom_idx) != unique_row_idxs.end()) {
        is_disjoint = false;
        break;
      }
    }
    if (is_disjoint) {
      if (last_ixn_type &&
          last_ixn_type.value() != NonbondedInteractionType::DISJOINT) {
        throw std::runtime_error(
            "batches of atom indices are not of the same type");
      } else {
        last_ixn_type = NonbondedInteractionType::DISJOINT;
      }
    } else {
      if (row_atoms_batch.size() > col_atoms_batch.size()) {
        throw std::runtime_error(
            "num row atoms(" + std::to_string(row_atoms_batch.size()) +
            ") must be <= num col atoms(" +
            std::to_string(col_atoms_batch.size()) + ") if non-disjoint");
      }
      bool is_overlapping = true;
      for (int i = 0; i < row_atoms_batch.size(); i++) {
        if (row_atoms_batch[i] != col_atoms_batch[i]) {
          is_overlapping = false;
          break;
        }
      }
      if (is_overlapping) {
        if (last_ixn_type &&
            last_ixn_type.value() != NonbondedInteractionType::OVERLAPPING) {
          throw std::runtime_error(
              "batches of atom indices are not of the same type");
        } else {
          last_ixn_type = NonbondedInteractionType::OVERLAPPING;
        }
      }
    }
  }
  if (!last_ixn_type) {
    throw std::runtime_error(
        "row and col indices are neither disjoint nor overlapping");
  }
  return last_ixn_type.value();
}

bool is_upper_triangular(NonbondedInteractionType ixn_type) {
  if (ixn_type == NonbondedInteractionType::DISJOINT) {
    return false;
  } else if (ixn_type == NonbondedInteractionType::OVERLAPPING) {
    return true;
  } else {
    throw std::runtime_error(
        "unknown NonbondedInteractionType for nblist construction");
  }
}

static int max_vector_int(const std::vector<int> &vec) {
  return *std::max_element(vec.begin(), vec.end());
}

// max_vector_int_sum finds maximum sum between two vectors at each index.
// Vectors must be the same length.
static int max_vector_int_sum(const std::vector<int> &vec,
                              const std::vector<int> &vec2) {
  assert(vec.size() == vec2.size());
  int max_sum = 0;
  for (auto i = 0; i < vec.size(); i++) {
    max_sum = max(max_sum, vec[i] + vec2[i]);
  }
  return max_sum;
}

static int
max_atoms_from_row_and_columns(const std::vector<int> &row_idx_counts,
                               const std::vector<int> &col_idx_counts,
                               NonbondedInteractionType ixn_type) {
  int K; // number of atoms involved in the interaction group
  if (ixn_type == NonbondedInteractionType::DISJOINT) {
    // If disjoint find the maximum row + column indices
    K = max_vector_int_sum(col_idx_counts, row_idx_counts);
  } else {
    // NC contains NR already, since they're overlapping
    K = max_vector_int(col_idx_counts);
  }
  return K;
}

template <typename RealType>
NonbondedInteractionGroup<RealType>::NonbondedInteractionGroup(
    const int num_systems, const int N,
    const std::vector<std::vector<int>> &row_atom_idxs,
    const std::vector<std::vector<int>> &col_atom_idxs, const RealType beta,
    const RealType cutoff, const bool disable_hilbert_sort,
    const RealType nblist_padding)
    : num_systems_(num_systems), N_(N),
      interaction_type_(
          get_nonbonded_interaction_type(row_atom_idxs, col_atom_idxs)),
      compute_col_grads_(true),
      nrg_accum_(num_systems_,
                 this->get_max_nonbonded_kernel_blocks() * num_systems_),
      kernel_ptrs_(
          {// enumerate over every possible kernel combination
           // Set threads to 1 if not computing energy to reduced unused shared
           // memory U: Compute U X: Compute DU_DX P: Compute DU_DP T: Compute
           // UPPER_TRIANGLE J: Compute COL_GRADS (LocalMD) U  X  P  T  J
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 0, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 0, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 0, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 0, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 1, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 1, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 1, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                0, 1, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 0, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 0, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 0, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 0, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 1, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 1, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 1, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0,
                                1, 1, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 0, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 0, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 0, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 0, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 1, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 1, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 1, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                0, 1, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 0, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 0, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 0, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 0, 1, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 1, 0, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 1, 0, 1>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 1, 1, 0>,
           &k_nonbonded_unified<RealType, NONBONDED_KERNEL_THREADS_PER_BLOCK, 1,
                                1, 1, 1, 1>}),
      column_idx_counts_(num_systems_), row_idx_counts_(num_systems_),
      beta_(beta), cutoff_(cutoff), steps_since_last_sort_(0),
      nblist_(num_systems_, N_, is_upper_triangular(interaction_type_)),
      nblist_padding_(nblist_padding), hilbert_sort_(nullptr),
      disable_hilbert_(disable_hilbert_sort) {

  this->validate_idxs(N_, row_atom_idxs, col_atom_idxs, false);
  for (int i = 0; i < num_systems_; i++) {
    column_idx_counts_[i] = col_atom_idxs[i].size();
    row_idx_counts_[i] = row_atom_idxs[i].size();
  }

  cudaSafeMalloc(&d_col_atom_idxs_,
                 num_systems_ * N_ * sizeof(*d_col_atom_idxs_));
  cudaSafeMalloc(&d_col_atom_idxs_counts_,
                 num_systems_ * sizeof(*d_col_atom_idxs_counts_));
  gpuErrchk(
      cudaMemcpy(d_col_atom_idxs_counts_, &column_idx_counts_[0],
                 column_idx_counts_.size() * sizeof(*d_col_atom_idxs_counts_),
                 cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_row_atom_idxs_,
                 num_systems_ * N_ * sizeof(*d_row_atom_idxs_));
  cudaSafeMalloc(&d_row_atom_idxs_counts_,
                 num_systems_ * sizeof(*d_row_atom_idxs_counts_));
  gpuErrchk(
      cudaMemcpy(d_row_atom_idxs_counts_, &row_idx_counts_[0],
                 row_idx_counts_.size() * sizeof(*d_row_atom_idxs_counts_),
                 cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_nblist_row_idxs_,
                 num_systems_ * N_ * sizeof(*d_nblist_row_idxs_));
  cudaSafeMalloc(&d_nblist_col_idxs_,
                 num_systems_ * N_ * sizeof(*d_nblist_col_idxs_));

  // this needs to be large enough to be safe when resized
  const int mnkb = this->get_max_nonbonded_kernel_blocks();
  cudaSafeMalloc(&d_u_buffer_, num_systems_ * mnkb * sizeof(*d_u_buffer_));
  cudaSafeMalloc(&d_system_idxs_,
                 num_systems_ * mnkb * sizeof(*d_system_idxs_));
  // Initialize all of the system_idxs
  for (int i = 0; i < num_systems_; i++) {
    k_initialize_array<<<ceil_divide(mnkb, DEFAULT_THREADS_PER_BLOCK),
                         DEFAULT_THREADS_PER_BLOCK>>>(
        mnkb, d_system_idxs_ + i * mnkb, i);
    gpuErrchk(cudaPeekAtLastError());
  }

  cudaSafeMalloc(&d_perm_, num_systems_ * N_ * sizeof(*d_perm_));
  cudaSafeMalloc(&d_sorted_x_, num_systems_ * N_ * 3 * sizeof(*d_sorted_x_));
  cudaSafeMalloc(&d_sorted_p_,
                 num_systems_ * N_ * PARAMS_PER_ATOM * sizeof(*d_sorted_p_));

  cudaSafeMalloc(&d_nblist_x_, num_systems_ * N_ * 3 * sizeof(*d_nblist_x_));
  gpuErrchk(cudaMemset(d_nblist_x_, 0,
                       num_systems_ * N_ * 3 *
                           sizeof(*d_nblist_x_))); // set non-sensical positions
  cudaSafeMalloc(&d_nblist_box_, num_systems_ * 3 * 3 * sizeof(*d_nblist_box_));
  gpuErrchk(cudaMemset(d_nblist_box_, 0,
                       num_systems_ * 3 * 3 * sizeof(*d_nblist_box_)));
  gpuErrchk(cudaHostAlloc(&m_rebuild_nblist_, 1 * sizeof(*m_rebuild_nblist_),
                          cudaHostAllocMapped));
  m_rebuild_nblist_[0] = 0;
  gpuErrchk(cudaHostGetDevicePointer(&d_rebuild_nblist_, m_rebuild_nblist_, 0));

  if (!disable_hilbert_) {
    this->hilbert_sort_.reset(new HilbertSort<RealType>(N_));
  }

  this->set_atom_idxs(row_atom_idxs, col_atom_idxs);

  // Create event with timings disabled as timings slow down events
  gpuErrchk(cudaEventCreateWithFlags(&nblist_flag_sync_event_,
                                     cudaEventDisableTiming));
};

template <typename RealType>
NonbondedInteractionGroup<RealType>::~NonbondedInteractionGroup() {
  gpuErrchk(cudaFree(d_col_atom_idxs_));
  gpuErrchk(cudaFree(d_col_atom_idxs_counts_));
  gpuErrchk(cudaFree(d_row_atom_idxs_));
  gpuErrchk(cudaFree(d_row_atom_idxs_counts_));
  gpuErrchk(cudaFree(d_nblist_row_idxs_));
  gpuErrchk(cudaFree(d_nblist_col_idxs_));

  gpuErrchk(cudaFree(d_perm_));

  gpuErrchk(cudaFree(d_sorted_x_));
  gpuErrchk(cudaFree(d_u_buffer_));
  gpuErrchk(cudaFree(d_system_idxs_));

  gpuErrchk(cudaFree(d_sorted_p_));

  gpuErrchk(cudaFree(d_nblist_x_));
  gpuErrchk(cudaFree(d_nblist_box_));
  gpuErrchk(cudaFreeHost(m_rebuild_nblist_));

  gpuErrchk(cudaEventDestroy(nblist_flag_sync_event_));
};

template <typename RealType>
bool NonbondedInteractionGroup<RealType>::needs_sort() {
  return steps_since_last_sort_ % STEPS_PER_SORT == 0;
}

template <typename RealType>
int NonbondedInteractionGroup<RealType>::get_max_nonbonded_kernel_blocks()
    const {
  int max_nonbonded_kernel_blocks =
      static_cast<int>(ceil(N_ * NONBONDED_BLOCKS_TO_ROW_ATOMS_RATIO));
  return min(max_nonbonded_kernel_blocks, MAX_KERNEL_BLOCKS);
}

template <typename RealType>
int NonbondedInteractionGroup<RealType>::get_cur_nonbonded_kernel_blocks()
    const {
  const int NR = max_vector_int(this->row_idx_counts_);
  int cur_nonbonded_kernel_blocks =
      static_cast<int>(ceil(NR * NONBONDED_BLOCKS_TO_ROW_ATOMS_RATIO));
  int max_nonbonded_kernel_blocks = this->get_max_nonbonded_kernel_blocks();
  return min(cur_nonbonded_kernel_blocks, max_nonbonded_kernel_blocks);
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::sort(const RealType *d_coords,
                                               const RealType *d_box,
                                               cudaStream_t stream) {

  // We must rebuild the neighborlist after sorting, as the neighborlist is tied
  // to a particular sort order

  // To compute the d_perm_ layout, we:
  // 1) Compute the permutation moving the row_idxs first.
  // 2) Compute the permutation moving the non-overlapping column indices.

  // DISJOINT
  //        row_idxs   col_idxs
  // atom [b a e c d] [g i f h]
  //  idx  0 1 2 3 4   0 1 2 3
  // perm [a b c d e] [f g h i]

  // OVERLAPPING
  //        row_idxs        col_idxs
  // atom [b a e c d] [b a e c d g i f h]
  //  idx  0 1 2 3 4   0 1 2 3 4 5 6 7 8
  //          NR           NR   | NC-NR
  // perm [a b c d e] [f g h i]

  // Sort each system on its own.
  // TBD: Batch the hilbert curve sort across replicas
  for (int i = 0; i < num_systems_; i++) {
    int NC = column_idx_counts_[i];
    int NR = row_idx_counts_[i];
    if (!disable_hilbert_) {
      this->hilbert_sort_->sort_device(NR, d_row_atom_idxs_ + i * N_,
                                       d_coords + i * N_ * 3, d_box + i * 9,
                                       d_perm_ + i * N_, stream);
      if (this->interaction_type_ == NonbondedInteractionType::DISJOINT) {
        this->hilbert_sort_->sort_device(NC, d_col_atom_idxs_ + i * N_,
                                         d_coords + i * N_ * 3, d_box + i * 9,
                                         d_perm_ + i * N_ + NR, stream);
      } else if (this->interaction_type_ ==
                 NonbondedInteractionType::OVERLAPPING) {
        this->hilbert_sort_->sort_device(
            NC - NR, d_col_atom_idxs_ + i * N_ + NR, d_coords + i * N_ * 3,
            d_box + i * 9, d_perm_ + i * N_ + NR, stream);
      }

    } else {
      gpuErrchk(cudaMemcpyAsync(d_perm_ + i * N_, d_row_atom_idxs_ + i * N_,
                                NR * sizeof(*d_row_atom_idxs_),
                                cudaMemcpyDeviceToDevice, stream));
      if (this->interaction_type_ == NonbondedInteractionType::DISJOINT) {
        gpuErrchk(cudaMemcpyAsync(
            d_perm_ + i * N_ + NR, d_col_atom_idxs_ + i * N_,
            NC * sizeof(*d_col_atom_idxs_), cudaMemcpyDeviceToDevice, stream));
      } else if (this->interaction_type_ ==
                 NonbondedInteractionType::OVERLAPPING) {
        gpuErrchk(cudaMemcpyAsync(d_perm_ + i * N_ + NR,
                                  d_col_atom_idxs_ + i * N_ + NR,
                                  (NC - NR) * sizeof(*d_col_atom_idxs_),
                                  cudaMemcpyDeviceToDevice, stream));
      }
    }
  }

  // Set the mapped memory to indicate that we need to rebuild
  m_rebuild_nblist_[0] = 1;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::execute_device(
    const int num_systems, const int N, const int P,
    const RealType *d_x,   // [num_systems * N * 3]
    const RealType *d_p,   // [num_systems * N * PARAMS_PER_ATOM]
    const RealType *d_box, // [num_systems * 3 * 3]
    unsigned long long *d_du_dx, unsigned long long *d_du_dp, __int128 *d_u,
    cudaStream_t stream) {
  // (ytz) the nonbonded algorithm proceeds as follows:

  // (done in constructor), construct a hilbert curve mapping each of the
  // HILBERT_GRID_DIM x HILBERT_GRID_DIM x HILBERT_GRID_DIM cells into an index.
  // a. decide if we need to rebuild the neighborlist, if so:
  //     - look up which cell each particle belongs to, and its linear index
  //     along the hilbert curve.
  //     - use radix pair sort keyed on the hilbert index with values equal to
  //     the atomic index
  //     - resulting sorted values is the permutation array.
  //     - permute coords
  // b. else:
  //     - permute new coords
  // c. permute parameters
  // d. compute the nonbonded interactions using the neighborlist
  // e. inverse permute the forces, du/dps into the original index.
  // f. u is buffered into a per-particle array, and then reduced.

  if (num_systems != num_systems_) {
    throw std::runtime_error(
        "NonbondedInteractionGroup::execute_device():"
        "expected num_systems == num_systems_, got num_systems=" +
        std::to_string(num_systems) +
        ", num_systems_=" + std::to_string(num_systems_));
  }
  if (N != N_) {
    throw std::runtime_error("NonbondedInteractionGroup::execute_device(): "
                             "expected N == N_, got N=" +
                             std::to_string(N) + ", N_=" + std::to_string(N_));
  }

  if (P != num_systems_ * N_ * PARAMS_PER_ATOM) {
    throw std::runtime_error("NonbondedInteractionGroup::execute_device(): "
                             "expected P == num_systems_ * N_*" +
                             std::to_string(PARAMS_PER_ATOM) +
                             ", got P=" + std::to_string(P) +
                             ", num_systems_*" + std::to_string(num_systems_) +
                             ", N_*" + std::to_string(PARAMS_PER_ATOM) + "=" +
                             std::to_string(N_ * PARAMS_PER_ATOM));
  }

  // If the size of the row or cols is none, exit
  if (max_vector_int(row_idx_counts_) == 0 ||
      max_vector_int(column_idx_counts_) == 0) {
    return;
  }

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const int B = ceil_divide(N_, tpb);

  const int K = this->get_max_atoms();

  const dim3 dimGrid(ceil_divide(K, tpb), num_systems_, 1);

  if (this->needs_sort()) {
    // Sorting always triggers a neighborlist rebuild
    this->sort(d_x, d_box, stream);
  } else {
    // (ytz) see if we need to rebuild the neighborlist.
    // Reuse the d_perm_ here to avoid having to make two kernels calls.
    k_check_rebuild_coords_and_box_gather<RealType>
        <<<dimGrid, tpb, 0, stream>>>(num_systems_, N_, d_perm_, d_x,
                                      d_nblist_x_, d_box, d_nblist_box_,
                                      nblist_padding_, d_rebuild_nblist_);
    gpuErrchk(cudaPeekAtLastError());
    // we can optimize this away by doing the check on the GPU directly.
    gpuErrchk(cudaEventRecord(nblist_flag_sync_event_, stream));
  }

  k_gather_coords_and_params<RealType, 3, PARAMS_PER_ATOM>
      <<<dimGrid, tpb, 0, stream>>>(num_systems_, N, K, d_perm_, d_x, d_p,
                                    d_sorted_x_, d_sorted_p_);
  gpuErrchk(cudaPeekAtLastError());

  // look up which kernel we need for this computation
  int kernel_idx = 0;
  kernel_idx |= this->compute_col_grads_ ? 1 << 0 : 0;
  kernel_idx |= is_upper_triangular(this->interaction_type_) ? 1 << 1 : 0;
  kernel_idx |= d_du_dp ? 1 << 2 : 0;
  kernel_idx |= d_du_dx ? 1 << 3 : 0;
  kernel_idx |= d_u ? 1 << 4 : 0;

  const int mnkb = this->get_max_nonbonded_kernel_blocks();
  const int nkb = this->get_cur_nonbonded_kernel_blocks();

  // Zero out the energy buffer
  if (d_u) {
    // TBD: Test nkb instead of mnkb
    cudaMemsetAsync(d_u_buffer_, 0, num_systems_ * mnkb * sizeof(*d_u_buffer_),
                    stream);
  }

  // Syncing to an event allows kernels put into the queue after the event was
  // recorded to keep running during the sync Note that if no event is recorded,
  // this is effectively a no-op, such as in the case of sorting.
  gpuErrchk(cudaEventSynchronize(nblist_flag_sync_event_));
  if (m_rebuild_nblist_[0] > 0) {
    nblist_.build_nblist_device(K, d_sorted_x_, d_box, cutoff_, nblist_padding_,
                                stream);
    m_rebuild_nblist_[0] = 0;
    gpuErrchk(cudaMemcpyAsync(d_nblist_x_, d_x,
                              num_systems_ * N * 3 * sizeof(*d_x),
                              cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_nblist_box_, d_box,
                              num_systems_ * 3 * 3 * sizeof(*d_box),
                              cudaMemcpyDeviceToDevice, stream));

    // Useful diagnostic code (and doesn't seem to affect wall-clock time very
    // much), leave this here for easy access. unsigned int ixn_count;
    // cudaMemcpy(&ixn_count, nblist_.get_ixn_count(), sizeof(ixn_count),
    // cudaMemcpyDeviceToHost); std::cout << "ixn_count: " << ixn_count <<
    // std::endl;
  }

  kernel_ptrs_[kernel_idx]<<<dim3(nkb, num_systems_, 1),
                             NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, stream>>>(
      num_systems_, N_, K, mnkb, nblist_.max_ixn_count(),
      nblist_.get_num_row_idxs(), nblist_.get_ixn_count(), d_perm_, d_sorted_x_,
      d_sorted_p_, d_box, beta_, cutoff_, nblist_.get_ixn_tiles(),
      nblist_.get_ixn_atoms(), d_du_dx, d_du_dp,
      d_u == nullptr
          ? nullptr
          : d_u_buffer_ // switch to nullptr if we don't request energies
  );
  gpuErrchk(cudaPeekAtLastError());

  if (d_u) {
    // nullptr for the d_system_idxs if only simulating a single system
    nrg_accum_.sum_device(num_systems_ * mnkb, d_u_buffer_,
                          num_systems_ > 1 ? d_system_idxs_ : nullptr, d_u,
                          stream);
  }
  // Increment steps
  steps_since_last_sort_++;
}

// get_max_atoms returns the maximum number of atoms to consider for the
// interaction group this is the max of any replica, which may have different
// row/column atom indices
template <typename RealType>
int NonbondedInteractionGroup<RealType>::get_max_atoms() const {

  return max_atoms_from_row_and_columns(row_idx_counts_, column_idx_counts_,
                                        interaction_type_);
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_atom_idxs(
    const std::vector<std::vector<int>> &row_atom_idxs,
    const std::vector<std::vector<int>> &col_atom_idxs) {

  this->validate_idxs(N_, row_atom_idxs, col_atom_idxs, false);

  std::vector<int> row_counts(num_systems_);
  std::vector<int> col_counts(num_systems_);
  std::vector<unsigned int> row_atom_idxs_v;
  std::vector<unsigned int> col_atom_idxs_v;
  for (int i = 0; i < num_systems_; i++) {
    row_counts[i] = row_atom_idxs[i].size();
    col_counts[i] = col_atom_idxs[i].size();

    const size_t row_offset = row_atom_idxs_v.size();
    row_atom_idxs_v.resize(row_offset + N_, N_);
    std::memcpy(row_atom_idxs_v.data() + row_offset, row_atom_idxs[i].data(),
                row_atom_idxs[i].size() * sizeof(unsigned int));

    const size_t col_offset = col_atom_idxs_v.size();
    col_atom_idxs_v.resize(col_offset + N_, N_);
    std::memcpy(col_atom_idxs_v.data() + col_offset, col_atom_idxs[i].data(),
                col_atom_idxs[i].size() * sizeof(unsigned int));
  }

  cudaStream_t stream = static_cast<cudaStream_t>(0);
  DeviceBuffer<unsigned int> d_col(col_atom_idxs_v);
  DeviceBuffer<unsigned int> d_row(row_atom_idxs_v);
  this->set_atom_idxs_device(row_counts, col_counts, d_row.data, d_col.data,
                             stream);
  gpuErrchk(cudaStreamSynchronize(stream));
}

template <typename RealType>
std::vector<int> NonbondedInteractionGroup<RealType>::get_row_idxs() const {
  std::vector<int> h_row_idxs(num_systems_ * N_);

  gpuErrchk(cudaMemcpy(&h_row_idxs[0], d_row_atom_idxs_,
                       h_row_idxs.size() * sizeof(*d_row_atom_idxs_),
                       cudaMemcpyDeviceToHost));
  std::vector<int> row_out;
  for (size_t i = 0; i < h_row_idxs.size(); i++) {
    if (h_row_idxs[i] < N_) {
      row_out.push_back(h_row_idxs[i]);
    }
  }
  return row_out;
}

template <typename RealType>
std::vector<int> NonbondedInteractionGroup<RealType>::get_col_idxs() const {
  std::vector<int> h_col_idxs(num_systems_ * N_);

  gpuErrchk(cudaMemcpy(&h_col_idxs[0], d_col_atom_idxs_,
                       h_col_idxs.size() * sizeof(*d_col_atom_idxs_),
                       cudaMemcpyDeviceToHost));
  std::vector<int> col_out;
  for (size_t i = 0; i < h_col_idxs.size(); i++) {
    if (h_col_idxs[i] < N_) {
      col_out.push_back(h_col_idxs[i]);
    }
  }
  return col_out;
}

// set_atom_idxs_device is for use when idxs exist on the GPU already and are
// used as the new idxs to compute the neighborlist on.
template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_atom_idxs_device(
    const std::vector<int> &row_counts, const std::vector<int> &col_counts,
    const unsigned int *d_in_row_idxs, const unsigned int *d_in_column_idxs,
    const cudaStream_t stream) {

  if (row_counts.size() != num_systems_ || col_counts.size() != num_systems_) {
    throw std::runtime_error("row and column counts must match num_systems_");
  }
  const int K = max_atoms_from_row_and_columns(row_counts, col_counts,
                                               this->interaction_type_);
  if (this->interaction_type_ == NonbondedInteractionType::DISJOINT && K > N_) {
    throw std::runtime_error(
        "number of idxs must be less than or equal to N, got " +
        std::to_string(K));
  }

  constexpr int tpb = DEFAULT_THREADS_PER_BLOCK;
  // Set the permutation to all N_
  k_initialize_array<<<ceil_divide(num_systems_ * N_, tpb), tpb, 0, stream>>>(
      num_systems_ * N_, d_perm_, static_cast<unsigned int>(N_));
  gpuErrchk(cudaPeekAtLastError());
  if (K > 0) {
    // The indices must already be on the GPU and are copied into the
    // potential's buffers.
    gpuErrchk(cudaMemcpyAsync(d_row_atom_idxs_, d_in_row_idxs,
                              num_systems_ * N_ * sizeof(*d_row_atom_idxs_),
                              cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_col_atom_idxs_, d_in_column_idxs,
                              num_systems_ * N_ * sizeof(*d_col_atom_idxs_),
                              cudaMemcpyDeviceToDevice, stream));

    // TBD: Figure out a way to handle this more gracefully
    gpuErrchk(cudaMemcpyAsync(d_row_atom_idxs_counts_, &row_counts[0],
                              num_systems_ * sizeof(*d_row_atom_idxs_counts_),
                              cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_col_atom_idxs_counts_, &col_counts[0],
                              num_systems_ * sizeof(*d_col_atom_idxs_counts_),
                              cudaMemcpyHostToDevice, stream));

    // Resize the nblist
    nblist_.resize_device(K, stream);

    // The neighborlist only sees the permuted coordinates, so the row and atom
    // indices provided are sequential (e.g. [1, 2, 3, ...]) compared to the
    // interaction group which may have non-sequential row/col indices (e.g. [5,
    // 3, 10, ...])

    // -Example-
    // disjoint:
    // row_idxs = 012_____, col_idxs=34567___, n=8
    // nblist args:
    // row_idxs = 012, col_idxs=34567

    // overlapping:
    // row_idxs = 012_____, col_idxs=01234567, n=8
    // nblist args:
    // row_idxs = 012, col_idxs=34567
    k_setup_nblist_row_and_column_indices<<<
        dim3(ceil_divide(K, tpb), num_systems_), tpb, 0, stream>>>(
        num_systems_, K, d_row_atom_idxs_counts_, d_col_atom_idxs_counts_,
        interaction_type_ == NonbondedInteractionType::DISJOINT,
        d_nblist_row_idxs_, d_nblist_col_idxs_);
    gpuErrchk(cudaPeekAtLastError());

    nblist_.set_idxs_device(d_row_atom_idxs_counts_, d_col_atom_idxs_counts_,
                            d_nblist_row_idxs_, d_nblist_col_idxs_, stream);
  }

  // Update the row and column counts
  std::memcpy(column_idx_counts_.data(), col_counts.data(),
              num_systems_ * sizeof(int));
  std::memcpy(row_idx_counts_.data(), row_counts.data(),
              num_systems_ * sizeof(int));
  // Reset the steps so that we do a new sort, forcing a new nblist rebuild
  this->steps_since_last_sort_ = 0;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp,
    RealType *du_dp_float) {

  for (int i = 0; i < N; i++) {
    const int idx = i * PARAMS_PER_ATOM;
    const int idx_charge = idx + PARAM_OFFSET_CHARGE;
    const int idx_sig = idx + PARAM_OFFSET_SIG;
    const int idx_eps = idx + PARAM_OFFSET_EPS;
    const int idx_w = idx + PARAM_OFFSET_W;

    du_dp_float[idx_charge] =
        FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DCHARGE>(
            du_dp[idx_charge]);
    du_dp_float[idx_sig] =
        FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DSIG>(du_dp[idx_sig]);
    du_dp_float[idx_eps] =
        FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DEPS>(du_dp[idx_eps]);
    du_dp_float[idx_w] =
        FIXED_TO_FLOAT_DU_DP<RealType, FIXED_EXPONENT_DU_DW>(du_dp[idx_w]);
  }
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_nblist_padding(
    const RealType padding) {
  if (padding < static_cast<RealType>(0.0)) {
    throw std::runtime_error("nblist padding must be greater than 0.0");
  }
  this->nblist_padding_ = padding;
  // Reset the steps so that we do a new sort
  this->steps_since_last_sort_ = 0;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_compute_col_grads(
    const bool value) {
  // If compute_col_grads is true, we always compute gradients on the column
  // idxs. If we're in the disjoint case:
  //      compute_col_grads must always be true.
  // If we're in the overlapping case:
  //      compute_col_grads=False is guaranteed to compute correct gradients on
  //      the overlapping row_idxs that prefix col_idxs, and may or may not
  //      compute gradients on the suffix/remainder of col_idxs.
  if (!value && this->interaction_type_ == NonbondedInteractionType::DISJOINT) {
    throw std::runtime_error(
        "compute_col_grads must be true if interaction_type_ is DISJOINT");
  }
  compute_col_grads_ = value;
};

template <typename RealType>
void NonbondedInteractionGroup<RealType>::validate_idxs(
    const int N, const std::vector<std::vector<int>> &row_atom_idxs,
    const std::vector<std::vector<int>> &col_atom_idxs,
    const bool allow_empty) {

  if (row_atom_idxs.size() != num_systems_) {
    throw std::runtime_error(
        "row atom batches doesn't match expected number of batches");
  }
  if (row_atom_idxs.size() != col_atom_idxs.size()) {
    throw std::runtime_error(
        "row atom batches and column atom batches don't match");
  }
  NonbondedInteractionType new_ixn_type =
      get_nonbonded_interaction_type(row_atom_idxs, col_atom_idxs);
  if (new_ixn_type != this->interaction_type_) {
    throw std::runtime_error("switching interaction types is not supported");
  }
  for (int i = 0; i < row_atom_idxs.size(); i++) {
    if (!allow_empty) {
      if (row_atom_idxs[i].size() == 0) {
        throw std::runtime_error("row_atom_idxs must be nonempty");
      }
      if (col_atom_idxs[i].size() == 0) {
        throw std::runtime_error("col_atom_idxs must be nonempty");
      }
      if (row_atom_idxs[i].size() > static_cast<long unsigned int>(N)) {
        throw std::runtime_error("num row idxs must be <= N(" +
                                 std::to_string(N) + ")");
      }
      if (col_atom_idxs.size() > static_cast<long unsigned int>(N)) {
        throw std::runtime_error("num col idxs must be <= N(" +
                                 std::to_string(N) + ")");
      }
    }
    verify_atom_idxs(N, row_atom_idxs[i], allow_empty);
    verify_atom_idxs(N, col_atom_idxs[i], allow_empty);
  }

  return;
}

template <typename RealType>
int NonbondedInteractionGroup<RealType>::num_systems() const {
  return num_systems_;
}

template <typename RealType>
int NonbondedInteractionGroup<RealType>::get_num_col_idxs() const {
  return max_vector_int(column_idx_counts_);
};
template <typename RealType>
int NonbondedInteractionGroup<RealType>::get_num_row_idxs() const {
  return max_vector_int(row_idx_counts_);
};

template class NonbondedInteractionGroup<double>;
template class NonbondedInteractionGroup<float>;

} // namespace tmd
