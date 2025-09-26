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
#include "gpu_utils.cuh"
#include "harmonic_angle.hpp"
#include "k_harmonic_angle.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace tmd {

template <typename RealType>
HarmonicAngle<RealType>::HarmonicAngle(
    const std::vector<int> &angle_idxs // [A, 3]
    )
    : max_idxs_(angle_idxs.size() / IDXS_DIM), cur_num_idxs_(max_idxs_),
      sum_storage_bytes_(0),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP         U  X  P
                    &k_harmonic_angle<RealType, 0, 0, 0>,
                    &k_harmonic_angle<RealType, 0, 0, 1>,
                    &k_harmonic_angle<RealType, 0, 1, 0>,
                    &k_harmonic_angle<RealType, 0, 1, 1>,
                    &k_harmonic_angle<RealType, 1, 0, 0>,
                    &k_harmonic_angle<RealType, 1, 0, 1>,
                    &k_harmonic_angle<RealType, 1, 1, 0>,
                    &k_harmonic_angle<RealType, 1, 1, 1>}) {

  if (angle_idxs.size() % IDXS_DIM != 0) {
    throw std::runtime_error("angle_idxs.size() must be exactly " +
                             std::to_string(IDXS_DIM) + "*A");
  }
  static_assert(IDXS_DIM == 3);
  for (int a = 0; a < cur_num_idxs_; a++) {
    auto i = angle_idxs[a * IDXS_DIM + 0];
    auto j = angle_idxs[a * IDXS_DIM + 1];
    auto k = angle_idxs[a * IDXS_DIM + 2];
    if (i == j || j == k || i == k) {
      throw std::runtime_error("angle triplets must be unique");
    }
  }

  cudaSafeMalloc(&d_angle_idxs_,
                 cur_num_idxs_ * IDXS_DIM * sizeof(*d_angle_idxs_));
  gpuErrchk(cudaMemcpy(d_angle_idxs_, &angle_idxs[0],
                       cur_num_idxs_ * IDXS_DIM * sizeof(*d_angle_idxs_),
                       cudaMemcpyHostToDevice));
  cudaSafeMalloc(&d_u_buffer_, cur_num_idxs_ * sizeof(*d_u_buffer_));

  gpuErrchk(cub::DeviceReduce::Sum(nullptr, sum_storage_bytes_, d_u_buffer_,
                                   d_u_buffer_, cur_num_idxs_));

  gpuErrchk(cudaMalloc(&d_sum_temp_storage_, sum_storage_bytes_));
};

template <typename RealType> HarmonicAngle<RealType>::~HarmonicAngle() {
  gpuErrchk(cudaFree(d_angle_idxs_));
  gpuErrchk(cudaFree(d_u_buffer_));
  gpuErrchk(cudaFree(d_sum_temp_storage_));
};

template <typename RealType>
void HarmonicAngle<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  assert(batches == 1);
  if (cur_num_idxs_ > 0) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(cur_num_idxs_, tpb);

    if (P != cur_num_idxs_ * 3) {
      throw std::runtime_error("HarmonicAngle::execute_device(): expected P == "
                               "3*cur_num_idxs_, got P=" +
                               std::to_string(P) + "3*cur_num_idxs_=" +
                               std::to_string(3 * cur_num_idxs_));
    }
    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<blocks, tpb, 0, stream>>>(
        cur_num_idxs_, d_x, d_box, d_p, d_angle_idxs_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);
    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      gpuErrchk(cub::DeviceReduce::Sum(d_sum_temp_storage_, sum_storage_bytes_,
                                       d_u_buffer_, d_u, cur_num_idxs_,
                                       stream));
    }
  }
}

template <typename RealType>
void HarmonicAngle<RealType>::set_idxs_device(const int num_idxs,
                                              const int *d_new_idxs,
                                              cudaStream_t stream) {
  if (max_idxs_ < num_idxs) {
    throw std::runtime_error(
        "set_idxs_device(): Max number of angles " + std::to_string(max_idxs_) +
        " is less than new idxs " + std::to_string(num_idxs));
  }
  gpuErrchk(cudaMemcpyAsync(d_angle_idxs_, d_new_idxs,
                            num_idxs * IDXS_DIM * sizeof(*d_angle_idxs_),
                            cudaMemcpyDeviceToDevice, stream));
  cur_num_idxs_ = num_idxs;
}

template <typename RealType> int HarmonicAngle<RealType>::get_num_idxs() const {
  return cur_num_idxs_;
}

template <typename RealType> int *HarmonicAngle<RealType>::get_idxs_device() {
  return d_angle_idxs_;
}

template <typename RealType>
std::vector<int> HarmonicAngle<RealType>::get_idxs_host() const {
  return device_array_to_vector<int>(cur_num_idxs_ * IDXS_DIM, d_angle_idxs_);
}

template class HarmonicAngle<double>;
template class HarmonicAngle<float>;

} // namespace tmd
