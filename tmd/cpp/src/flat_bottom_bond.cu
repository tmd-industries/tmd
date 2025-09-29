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

#include "flat_bottom_bond.hpp"
#include "gpu_utils.cuh"
#include "k_flat_bottom_bond.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <vector>

namespace tmd {

template <typename RealType>
FlatBottomBond<RealType>::FlatBottomBond(const std::vector<int> &bond_idxs)
    : max_idxs_(bond_idxs.size() / IDXS_DIM), cur_num_idxs_(max_idxs_),
      nrg_accum_(1, cur_num_idxs_),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP           U  X  P
                    &k_flat_bottom_bond<RealType, 0, 0, 0>,
                    &k_flat_bottom_bond<RealType, 0, 0, 1>,
                    &k_flat_bottom_bond<RealType, 0, 1, 0>,
                    &k_flat_bottom_bond<RealType, 0, 1, 1>,
                    &k_flat_bottom_bond<RealType, 1, 0, 0>,
                    &k_flat_bottom_bond<RealType, 1, 0, 1>,
                    &k_flat_bottom_bond<RealType, 1, 1, 0>,
                    &k_flat_bottom_bond<RealType, 1, 1, 1>}) {

  // validate bond_idxs: even length, all idxs non-negative, and no self-edges
  if (bond_idxs.size() % IDXS_DIM != 0) {
    throw std::runtime_error("bond_idxs.size() must be exactly " +
                             std::to_string(IDXS_DIM) + "*k!");
  }
  static_assert(IDXS_DIM == 2);
  for (int b = 0; b < cur_num_idxs_; b++) {
    auto src = bond_idxs[b * IDXS_DIM + 0];
    auto dst = bond_idxs[b * IDXS_DIM + 1];
    if (src == dst) {
      throw std::runtime_error("src == dst");
    }

    if ((src < 0) or (dst < 0)) {
      throw std::runtime_error("idxs must be non-negative");
    }
  }

  // copy idxs to device
  cudaSafeMalloc(&d_bond_idxs_, cur_num_idxs_ * 2 * sizeof(*d_bond_idxs_));
  gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0],
                       cur_num_idxs_ * 2 * sizeof(*d_bond_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_u_buffer_, cur_num_idxs_ * sizeof(*d_u_buffer_));
};

template <typename RealType> FlatBottomBond<RealType>::~FlatBottomBond() {
  gpuErrchk(cudaFree(d_bond_idxs_));
  gpuErrchk(cudaFree(d_u_buffer_));
};

template <typename RealType>
void FlatBottomBond<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  const int num_params_per_bond = 3;
  int expected_P = num_params_per_bond * cur_num_idxs_;

  if (P != expected_P) {
    throw std::runtime_error(
        "FlatBottomBond::execute_device(): expected P == " +
        std::to_string(expected_P) + ", got P=" + std::to_string(P));
  }

  if (cur_num_idxs_ > 0) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(cur_num_idxs_, tpb);

    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<blocks, tpb, 0, stream>>>(
        cur_num_idxs_, d_x, d_box, d_p, d_bond_idxs_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);
    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      // nullptr for the d_system_idxs as batch size is fixed to 1
      nrg_accum_.sum_device(cur_num_idxs_, d_u_buffer_, nullptr, d_u, stream);
    }
  }
};

template <typename RealType>
void FlatBottomBond<RealType>::set_bonds_device(const int num_bonds,
                                                const int *d_bonds,
                                                const cudaStream_t stream) {
  if (max_idxs_ < num_bonds) {
    throw std::runtime_error(
        "set_bonds_device(): Max number of bonds " + std::to_string(max_idxs_) +
        " is less than new idxs " + std::to_string(num_bonds));
  }
  gpuErrchk(cudaMemcpyAsync(d_bond_idxs_, d_bonds,
                            num_bonds * IDXS_DIM * sizeof(*d_bond_idxs_),
                            cudaMemcpyDeviceToDevice, stream));
  cur_num_idxs_ = num_bonds;
}

template class FlatBottomBond<double>;
template class FlatBottomBond<float>;

} // namespace tmd
