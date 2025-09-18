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
#include "harmonic_bond.hpp"
#include "k_harmonic_bond.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace tmd {

template <typename RealType>
HarmonicBond<RealType>::HarmonicBond(const std::vector<int> &bond_idxs)
    : max_idxs_(bond_idxs.size() / IDXS_DIM), cur_num_idxs_(max_idxs_),
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
  gpuErrchk(cudaMemcpy(d_bond_idxs_, &bond_idxs[0],
                       cur_num_idxs_ * IDXS_DIM * sizeof(*d_bond_idxs_),
                       cudaMemcpyHostToDevice));
  cudaSafeMalloc(&d_u_buffer_, cur_num_idxs_ * sizeof(*d_u_buffer_));

  gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_,
                                   d_u_buffer_, cur_num_idxs_));

  gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

template <typename RealType> HarmonicBond<RealType>::~HarmonicBond() {
  gpuErrchk(cudaFree(d_bond_idxs_));
  gpuErrchk(cudaFree(d_u_buffer_));
  gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType>
void HarmonicBond<RealType>::execute_device(
    const int N, const int P, const RealType *d_x, const RealType *d_p,
    const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  if (cur_num_idxs_ > 0) {
    if (P != 2 * cur_num_idxs_) {
      throw std::runtime_error(
          "HarmonicBond::execute_device(): expected P == 2*B, got P=" +
          std::to_string(P) + ", 2*B=" + std::to_string(2 * cur_num_idxs_));
    }
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
      gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, sum_storage_bytes_,
                                       d_u_buffer_, d_u, cur_num_idxs_,
                                       stream));
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

template class HarmonicBond<double>;
template class HarmonicBond<float>;

} // namespace tmd
