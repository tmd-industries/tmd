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

#include "assert.h"
#include "cub_utils.cuh"
#include "gpu_utils.cuh"
#include "harmonic_bond.hpp"
#include "k_harmonic_bond.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace tmd {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(const int num_batches, const int num_atoms,
                                     const std::vector<int> &bond_idxs,
                                     const std::vector<int> &system_idxs)
    : num_batches_(num_batches), num_atoms_(num_atoms),
      max_idxs_(bond_idxs.size() / IDXS_DIM), cur_num_idxs_(max_idxs_),
      sum_storage_bytes_(0),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP        U  X  P
                    &k_harmonic_bond<RealType, 0, 0, 0>,
                    &k_harmonic_bond<RealType, 0, 0, 1>,
                    &k_harmonic_bond<RealType, 0, 1, 0>,
                    &k_harmonic_bond<RealType, 0, 1, 1>,
                    &k_harmonic_bond<RealType, 1, 0, 0>,
                    &k_harmonic_bond<RealType, 1, 0, 1>,
                    &k_harmonic_bond<RealType, 1, 1, 0>,
                    &k_harmonic_bond<RealType, 1, 1, 1>}) {

  if (bond_idxs.size() % IDXS_DIM != 0) {
    throw std::runtime_error("bond_idxs.size() must be exactly " +
                             std::to_string(IDXS_DIM) + "*k!");
  }
  if (system_idxs.size() != max_idxs_) {
    throw std::runtime_error("system_idxs.size() != (bond_idxs.size() / " +
                             std::to_string(IDXS_DIM) + "), got " +
                             std::to_string(system_idxs.size()) + " and " +
                             std::to_string(max_idxs_));
  }

  static_assert(IDXS_DIM == 2);
  for (int b = 0; b < cur_num_idxs_; b++) {
    auto src = bond_idxs[b * IDXS_DIM + 0];
    auto dst = bond_idxs[b * IDXS_DIM + 1];
    if (src == dst) {
      throw std::runtime_error("src == dst");
    }
  }

  cudaSafeMalloc(&d_bond_idxs_,
                 cur_num_idxs_ * IDXS_DIM * sizeof(*d_bond_idxs_));

  cudaSafeMalloc(&d_bond_system_idxs_,
                 cur_num_idxs_ * sizeof(*d_bond_system_idxs_));

  // This could probably be moved to a utility, so nrg computations are cheaper
  // in the single case
  cudaSafeMalloc(&d_reductions_out_, sizeof(*d_reductions_out_));
  cudaSafeMalloc(&d_system_idxs_unique_,
                 cur_num_idxs_ * sizeof(d_system_idxs_unique_));

  gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0],
                       cur_num_idxs_ * IDXS_DIM * sizeof(*d_bond_idxs_),
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_bond_system_idxs_, &system_idxs[0],
                       cur_num_idxs_ * sizeof(*d_bond_system_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_u_buffer_, cur_num_idxs_ * sizeof(*d_u_buffer_));

  CUBSumOp reduction_op;
  gpuErrchk(cub::DeviceReduce::ReduceByKey(
      nullptr, sum_storage_bytes_, d_bond_system_idxs_, d_system_idxs_unique_,
      d_u_buffer_, d_u_buffer_, d_reductions_out_, reduction_op,
      cur_num_idxs_));

  gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

template <typename RealType> HarmonicBond<RealType>::~HarmonicBond() {
  gpuErrchk(cudaFree(d_bond_idxs_));
  gpuErrchk(cudaFree(d_bond_system_idxs_));
  gpuErrchk(cudaFree(d_reductions_out_));
  gpuErrchk(cudaFree(d_system_idxs_unique_));
  gpuErrchk(cudaFree(d_u_buffer_));
  gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType>
void HarmonicBond<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  // assert(batches == 1);
  if (cur_num_idxs_ > 0) {
    if (P != 2 * cur_num_idxs_) {
      throw std::runtime_error(
          "HarmonicBond::execute_device(): expected P == 2*B, got P=" +
          std::to_string(P) + ", 2*B=" + std::to_string(2 * cur_num_idxs_));
    }
    if (N != num_atoms_) {
      throw std::runtime_error(
          "HarmonicBond::execute_device(): Expected N == num_atoms, got N=" +
          std::to_string(N) + ", num_atoms=" + std::to_string(num_atoms_));
    }
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(cur_num_idxs_, tpb);

    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<blocks, tpb, 0, stream>>>(
        num_atoms_, cur_num_idxs_, d_x, d_box, d_p, d_bond_idxs_,
        d_bond_system_idxs_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);
    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      CUBSumOp reduction_op;
      // TBD Need to handle the fact that the d_reductions_out_ can be shuffled
      // as it is dependent on the ordering of d_bond_system_idxs_ Example of
      // different bond_sytem_idxs and the ordering of output nrgs [0, 1, 1, 2,
      // 2] -> [0, 1, 2] [1, 2, 2, 0, 1] -> [1, 2, 0]
      gpuErrchk(cub::DeviceReduce::ReduceByKey(
          d_sum_temp_storage_, sum_storage_bytes_, d_bond_system_idxs_,
          d_system_idxs_unique_, d_u_buffer_, d_u, d_reductions_out_,
          reduction_op, cur_num_idxs_, stream));
    }
  }
};

template <typename RealType>
void HarmonicBond<RealType>::set_idxs_device(const int num_idxs,
                                             const int *d_new_idxs,
                                             cudaStream_t stream) {
  if (max_idxs_ < num_idxs) {
    throw std::runtime_error(
        "set_idxs_device(): Max number of bonds " + std::to_string(max_idxs_) +
        " is less than new idxs " + std::to_string(num_idxs));
  }
  gpuErrchk(cudaMemcpyAsync(d_bond_idxs_, d_new_idxs,
                            num_idxs * IDXS_DIM * sizeof(*d_bond_idxs_),
                            cudaMemcpyDeviceToDevice, stream));
  cur_num_idxs_ = num_idxs;
}

template <typename RealType> int HarmonicBond<RealType>::get_num_idxs() const {
  return cur_num_idxs_;
}

template <typename RealType> int *HarmonicBond<RealType>::get_idxs_device() {
  return d_bond_idxs_;
}

template <typename RealType>
std::vector<int> HarmonicBond<RealType>::get_idxs_host() const {
  return device_array_to_vector<int>(cur_num_idxs_ * IDXS_DIM, d_bond_idxs_);
}

template <typename RealType> int HarmonicBond<RealType>::batch_size() const {
  return num_batches_;
}

template class HarmonicBond<double>;
template class HarmonicBond<float>;

} // namespace tmd
