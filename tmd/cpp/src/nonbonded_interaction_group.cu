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

NonbondedInteractionType
get_nonbonded_interaction_type(const std::vector<int> &row_atom_idxs,
                               const std::vector<int> &col_atom_idxs) {

  // row and col idxs must be either:
  // 1) disjoint: row_atom_idxs \intersection overlapping = empty set
  // 2) overlapping: row_atom_idxs == col_atom_idxs[:len(row_atom_idxs)]
  bool is_disjoint = true;
  std::set<int> unique_row_idxs(row_atom_idxs.begin(), row_atom_idxs.end());
  for (int col_atom_idx : col_atom_idxs) {
    if (unique_row_idxs.find(col_atom_idx) != unique_row_idxs.end()) {
      is_disjoint = false;
      break;
    }
  }
  if (is_disjoint) {
    return NonbondedInteractionType::DISJOINT;
  }

  if (row_atom_idxs.size() > col_atom_idxs.size()) {
    throw std::runtime_error(
        "num row atoms(" + std::to_string(row_atom_idxs.size()) +
        ") must be <= num col atoms(" + std::to_string(col_atom_idxs.size()) +
        ") if non-disjoint");
  }
  bool is_overlapping = true;
  for (int i = 0; i < row_atom_idxs.size(); i++) {
    if (row_atom_idxs[i] != col_atom_idxs[i]) {
      is_overlapping = false;
      break;
    }
  }
  if (is_overlapping) {
    return NonbondedInteractionType::OVERLAPPING;
  }

  throw std::runtime_error(
      "row and col indices are neither disjoint nor overlapping");
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

template <typename RealType>
NonbondedInteractionGroup<RealType>::NonbondedInteractionGroup(
    const int N, const std::vector<int> &row_atom_idxs,
    const std::vector<int> &col_atom_idxs, const RealType beta,
    const RealType cutoff, const bool disable_hilbert_sort,
    const RealType nblist_padding)
    : N_(N), NR_(row_atom_idxs.size()), NC_(col_atom_idxs.size()),
      interaction_type_(
          get_nonbonded_interaction_type(row_atom_idxs, col_atom_idxs)),
      compute_col_grads_(true), nrg_accum_(1, MAX_KERNEL_BLOCKS),
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

      beta_(beta), cutoff_(cutoff), steps_since_last_sort_(0),
      nblist_(N_, is_upper_triangular(interaction_type_)),
      nblist_padding_(nblist_padding), hilbert_sort_(nullptr),
      disable_hilbert_(disable_hilbert_sort) {

  this->validate_idxs(N_, row_atom_idxs, col_atom_idxs, false);

  cudaSafeMalloc(&d_col_atom_idxs_, N_ * sizeof(*d_col_atom_idxs_));
  cudaSafeMalloc(&d_row_atom_idxs_, N_ * sizeof(*d_row_atom_idxs_));

  cudaSafeMalloc(&d_arange_buffer_, N_ * sizeof(*d_arange_buffer_));

  k_arange<<<ceil_divide(N_, DEFAULT_THREADS_PER_BLOCK),
             DEFAULT_THREADS_PER_BLOCK, 0>>>(N_, d_arange_buffer_);
  gpuErrchk(cudaPeekAtLastError());

  // this needs to be large enough to be safe when resized
  const int mnkb = this->get_max_nonbonded_kernel_blocks();
  cudaSafeMalloc(&d_u_buffer_, mnkb * sizeof(*d_u_buffer_));

  cudaSafeMalloc(&d_perm_, N_ * sizeof(*d_perm_));
  cudaSafeMalloc(&d_sorted_x_, N_ * 3 * sizeof(*d_sorted_x_));
  cudaSafeMalloc(&d_sorted_p_, N_ * PARAMS_PER_ATOM * sizeof(*d_sorted_p_));

  cudaSafeMalloc(&d_nblist_x_, N_ * 3 * sizeof(*d_nblist_x_));
  gpuErrchk(
      cudaMemset(d_nblist_x_, 0,
                 N_ * 3 * sizeof(*d_nblist_x_))); // set non-sensical positions
  cudaSafeMalloc(&d_nblist_box_, 3 * 3 * sizeof(*d_nblist_box_));
  gpuErrchk(cudaMemset(d_nblist_box_, 0, 3 * 3 * sizeof(*d_nblist_box_)));
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
  gpuErrchk(cudaFree(d_row_atom_idxs_));
  gpuErrchk(cudaFree(d_arange_buffer_));

  gpuErrchk(cudaFree(d_perm_));

  gpuErrchk(cudaFree(d_sorted_x_));
  gpuErrchk(cudaFree(d_u_buffer_));

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
  int cur_nonbonded_kernel_blocks =
      static_cast<int>(ceil(NR_ * NONBONDED_BLOCKS_TO_ROW_ATOMS_RATIO));
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
  // 2) Compute the permtuation moving the non-overlapping column indices.

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

  if (!disable_hilbert_) {
    this->hilbert_sort_->sort_device(NR_, d_row_atom_idxs_, d_coords, d_box,
                                     d_perm_, stream);
    if (this->interaction_type_ == NonbondedInteractionType::DISJOINT) {
      this->hilbert_sort_->sort_device(NC_, d_col_atom_idxs_, d_coords, d_box,
                                       d_perm_ + NR_, stream);
    } else if (this->interaction_type_ ==
               NonbondedInteractionType::OVERLAPPING) {
      this->hilbert_sort_->sort_device(NC_ - NR_, d_col_atom_idxs_ + NR_,
                                       d_coords, d_box, d_perm_ + NR_, stream);
    }
  } else {
    gpuErrchk(cudaMemcpyAsync(d_perm_, d_row_atom_idxs_,
                              NR_ * sizeof(*d_row_atom_idxs_),
                              cudaMemcpyDeviceToDevice, stream));
    if (this->interaction_type_ == NonbondedInteractionType::DISJOINT) {
      gpuErrchk(cudaMemcpyAsync(d_perm_ + NR_, d_col_atom_idxs_,
                                NC_ * sizeof(*d_col_atom_idxs_),
                                cudaMemcpyDeviceToDevice, stream));
    } else if (this->interaction_type_ ==
               NonbondedInteractionType::OVERLAPPING) {
      gpuErrchk(cudaMemcpyAsync(d_perm_ + NR_, d_col_atom_idxs_ + NR_,
                                (NC_ - NR_) * sizeof(*d_col_atom_idxs_),
                                cudaMemcpyDeviceToDevice, stream));
    }
  }
  // Set the mapped memory to indicate that we need to rebuild
  m_rebuild_nblist_[0] = 1;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::execute_device(
    const int batches, const int N, const int P,
    const RealType *d_x,   // [batches * N * 3]
    const RealType *d_p,   // [batches * N * PARAMS_PER_ATOM]
    const RealType *d_box, // [batches * 3 * 3]
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

  assert(batches == 1);
  if (N != N_) {
    throw std::runtime_error("NonbondedInteractionGroup::execute_device(): "
                             "expected N == N_, got N=" +
                             std::to_string(N) + ", N_=" + std::to_string(N_));
  }

  if (P != N_ * PARAMS_PER_ATOM) {
    throw std::runtime_error(
        "NonbondedInteractionGroup::execute_device(): expected P == N_*" +
        std::to_string(PARAMS_PER_ATOM) + ", got P=" + std::to_string(P) +
        ", N_*" + std::to_string(PARAMS_PER_ATOM) + "=" +
        std::to_string(N_ * PARAMS_PER_ATOM));
  }

  // If the size of the row or cols is none, exit
  if (NR_ == 0 || NC_ == 0) {
    return;
  }

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const int B = ceil_divide(N_, tpb);

  int K; // number of atoms involved in the interaction group
  if (interaction_type_ == NonbondedInteractionType::DISJOINT) {
    K = NR_ + NC_;
  } else {
    // NC_ contains NR_ already, since they're overlapping
    K = NC_;
  }

  if (this->needs_sort()) {
    // Sorting always triggers a neighborlist rebuild
    this->sort(d_x, d_box, stream);
  } else {
    // (ytz) see if we need to rebuild the neighborlist.
    // Reuse the d_perm_ here to avoid having to make two kernels calls.
    k_check_rebuild_coords_and_box_gather<RealType>
        <<<ceil_divide(K, tpb), tpb, 0, stream>>>(
            K, d_perm_, d_x, d_nblist_x_, d_box, d_nblist_box_, nblist_padding_,
            d_rebuild_nblist_);
    gpuErrchk(cudaPeekAtLastError());
    // we can optimize this away by doing the check on the GPU directly.
    gpuErrchk(cudaEventRecord(nblist_flag_sync_event_, stream));
  }

  k_gather_coords_and_params<RealType, 3, PARAMS_PER_ATOM>
      <<<ceil_divide(K, tpb), tpb, 0, stream>>>(K, d_perm_, d_x, d_p,
                                                d_sorted_x_, d_sorted_p_);
  gpuErrchk(cudaPeekAtLastError());
  // Syncing to an event allows kernels put into the queue after the event was
  // recorded to keep running during the sync Note that if no event is recorded,
  // this is effectively a no-op, such as in the case of sorting.
  gpuErrchk(cudaEventSynchronize(nblist_flag_sync_event_));
  if (m_rebuild_nblist_[0] > 0) {
    nblist_.build_nblist_device(K, d_sorted_x_, d_box, cutoff_, nblist_padding_,
                                stream);
    m_rebuild_nblist_[0] = 0;
    gpuErrchk(cudaMemcpyAsync(d_nblist_x_, d_x, N * 3 * sizeof(*d_x),
                              cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_nblist_box_, d_box, 3 * 3 * sizeof(*d_box),
                              cudaMemcpyDeviceToDevice, stream));

    // Useful diagnostic code (and doesn't seem to affect wall-clock time very
    // much), leave this here for easy access. unsigned int ixn_count;
    // cudaMemcpy(&ixn_count, nblist_.get_ixn_count(), sizeof(ixn_count),
    // cudaMemcpyDeviceToHost); std::cout << "ixn_count: " << ixn_count <<
    // std::endl;
  }

  // look up which kernel we need for this computation
  int kernel_idx = 0;
  kernel_idx |= this->compute_col_grads_ ? 1 << 0 : 0;
  kernel_idx |= is_upper_triangular(this->interaction_type_) ? 1 << 1 : 0;
  kernel_idx |= d_du_dp ? 1 << 2 : 0;
  kernel_idx |= d_du_dx ? 1 << 3 : 0;
  kernel_idx |= d_u ? 1 << 4 : 0;

  int nkb = this->get_cur_nonbonded_kernel_blocks();
  kernel_ptrs_
      [kernel_idx]<<<nkb, NONBONDED_KERNEL_THREADS_PER_BLOCK, 0, stream>>>(
          K, nblist_.get_num_row_idxs(), nblist_.get_ixn_count(), d_perm_,
          d_sorted_x_, d_sorted_p_, d_box, beta_, cutoff_,
          nblist_.get_ixn_tiles(), nblist_.get_ixn_atoms(), d_du_dx, d_du_dp,
          d_u == nullptr
              ? nullptr
              : d_u_buffer_ // switch to nullptr if we don't request energies
      );
  gpuErrchk(cudaPeekAtLastError());

  if (d_u) {
    // nullptr for the d_system_idxs as batch size is fixed to 1
    nrg_accum_.sum_device(nkb, d_u_buffer_, nullptr, d_u, stream);
  }
  // Increment steps
  steps_since_last_sort_++;
}

template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_atom_idxs(
    const std::vector<int> &row_atom_idxs,
    const std::vector<int> &col_atom_idxs) {

  this->validate_idxs(N_, row_atom_idxs, col_atom_idxs, false);

  std::vector<unsigned int> row_atom_idxs_v(row_atom_idxs.begin(),
                                            row_atom_idxs.end());
  std::vector<unsigned int> col_atom_idxs_v(col_atom_idxs.begin(),
                                            col_atom_idxs.end());

  cudaStream_t stream = static_cast<cudaStream_t>(0);
  DeviceBuffer<unsigned int> d_col(col_atom_idxs_v);
  DeviceBuffer<unsigned int> d_row(row_atom_idxs_v);
  this->set_atom_idxs_device(row_atom_idxs_v.size(), col_atom_idxs_v.size(),
                             d_row.data, d_col.data, stream);
  gpuErrchk(cudaStreamSynchronize(stream));
}

template <typename RealType>
std::vector<int> NonbondedInteractionGroup<RealType>::get_row_idxs() const {
  std::vector<int> h_row_idxs(this->NR_);

  gpuErrchk(cudaMemcpy(&h_row_idxs[0], d_row_atom_idxs_,
                       this->NR_ * sizeof(*d_row_atom_idxs_),
                       cudaMemcpyDeviceToHost));
  return h_row_idxs;
}

template <typename RealType>
std::vector<int> NonbondedInteractionGroup<RealType>::get_col_idxs() const {
  std::vector<int> h_col_idxs(this->NC_);

  gpuErrchk(cudaMemcpy(&h_col_idxs[0], d_col_atom_idxs_,
                       this->NC_ * sizeof(*d_col_atom_idxs_),
                       cudaMemcpyDeviceToHost));
  return h_col_idxs;
}

// set_atom_idxs_device is for use when idxs exist on the GPU already and are
// used as the new idxs to compute the neighborlist on.
template <typename RealType>
void NonbondedInteractionGroup<RealType>::set_atom_idxs_device(
    const int NR, const int NC, unsigned int *d_in_row_idxs,
    unsigned int *d_in_column_idxs, const cudaStream_t stream) {

  if (this->interaction_type_ == NonbondedInteractionType::DISJOINT &&
      NC + NR > N_) {
    throw std::runtime_error("number of idxs must be less than or equal to N");
  }
  if (NR > 0 && NC > 0) {
    // The indices must already be on the GPU and are copied into the
    // potential's buffers.
    gpuErrchk(cudaMemcpyAsync(d_col_atom_idxs_, d_in_column_idxs,
                              NC * sizeof(*d_col_atom_idxs_),
                              cudaMemcpyDeviceToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(d_row_atom_idxs_, d_in_row_idxs,
                              NR * sizeof(*d_row_atom_idxs_),
                              cudaMemcpyDeviceToDevice, stream));

    // disjoint:
    // row_idxs = 012_____, col_idxs=34567___, n=8
    // nblist args:
    // row_idxs = 012, col_idxs=34567

    // overlapping:
    // row_idxs = 012_____, col_idxs=01234567, n=8
    // nblist args:
    // row_idxs = 012, col_idxs=34567
    // just do this once in constructor and be done with it.

    // Resize the nblist
    if (interaction_type_ == NonbondedInteractionType::DISJOINT) {
      nblist_.resize_device(NC + NR, stream);
    } else if (interaction_type_ == NonbondedInteractionType::OVERLAPPING) {
      nblist_.resize_device(NC, stream);
    }

    int col_offset =
        (interaction_type_ == NonbondedInteractionType::DISJOINT) ? NR : 0;
    // Offset into the ends of the arrays that now contain the row and column
    // indices for the nblist
    nblist_.set_idxs_device(NC, NR, d_arange_buffer_ + col_offset,
                            d_arange_buffer_, stream);
  }

  // Update the row and column counts
  this->NR_ = NR;
  this->NC_ = NC;
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
void NonbondedInteractionGroup<RealType>::set_compute_col_grads(bool value) {
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
    const int N, const std::vector<int> &row_atom_idxs,
    const std::vector<int> &col_atom_idxs, const bool allow_empty) {

  if (!allow_empty) {
    if (row_atom_idxs.size() == 0) {
      throw std::runtime_error("row_atom_idxs must be nonempty");
    }
    if (col_atom_idxs.size() == 0) {
      throw std::runtime_error("col_atom_idxs must be nonempty");
    }
    if (row_atom_idxs.size() > static_cast<long unsigned int>(N)) {
      throw std::runtime_error("num row idxs must be <= N(" +
                               std::to_string(N) + ")");
    }
    if (col_atom_idxs.size() > static_cast<long unsigned int>(N)) {
      throw std::runtime_error("num col idxs must be <= N(" +
                               std::to_string(N) + ")");
    }
  }
  verify_atom_idxs(N, row_atom_idxs, allow_empty);
  verify_atom_idxs(N, col_atom_idxs, allow_empty);

  NonbondedInteractionType new_ixn_type =
      get_nonbonded_interaction_type(row_atom_idxs, col_atom_idxs);

  if (new_ixn_type != this->interaction_type_) {
    throw std::runtime_error(
        "switching interaction types is probably not what you want");
  }

  return;
}

template class NonbondedInteractionGroup<double>;
template class NonbondedInteractionGroup<float>;

} // namespace tmd
