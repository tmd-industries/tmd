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
#include "k_nonbonded_precomputed.cuh"
#include "kernel_utils.cuh"
#include "kernels/k_nonbonded_common.cuh"
#include "math_utils.cuh"
#include "nonbonded_precomputed.hpp"
#include <cub/cub.cuh>
#include <vector>

namespace tmd {

template <typename RealType>
NonbondedPairListPrecomputed<RealType>::NonbondedPairListPrecomputed(
    const int num_batches, const int num_atoms, const std::vector<int> &idxs,
    const std::vector<int> &system_idxs, const RealType beta,
    const RealType cutoff)
    : num_batches_(num_batches), num_atoms_(num_atoms), B_(idxs.size() / 2),
      beta_(beta), cutoff_(cutoff), nrg_accum_(num_batches_, B_),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP         U  X  P
                    &k_nonbonded_precomputed<RealType, 0, 0, 0>,
                    &k_nonbonded_precomputed<RealType, 0, 0, 1>,
                    &k_nonbonded_precomputed<RealType, 0, 1, 0>,
                    &k_nonbonded_precomputed<RealType, 0, 1, 1>,
                    &k_nonbonded_precomputed<RealType, 1, 0, 0>,
                    &k_nonbonded_precomputed<RealType, 1, 0, 1>,
                    &k_nonbonded_precomputed<RealType, 1, 1, 0>,
                    &k_nonbonded_precomputed<RealType, 1, 1, 1>}) {

  if (idxs.size() % 2 != 0) {
    throw std::runtime_error("idxs.size() must be exactly 2*B!");
  }

  if (system_idxs.size() != B_) {
    throw std::runtime_error(
        "system_idxs.size() != (pair_idxs.size() / 2), got " +
        std::to_string(system_idxs.size()) + " and " + std::to_string(B_));
  }

  for (int b = 0; b < B_; b++) {
    auto src = idxs[b * 2 + 0];
    auto dst = idxs[b * 2 + 1];
    if (src == dst) {
      throw std::runtime_error(
          "illegal pair with src == dst: " + std::to_string(src) + ", " +
          std::to_string(dst));
    }
  }

  cudaSafeMalloc(&d_idxs_, B_ * 2 * sizeof(*d_idxs_));
  gpuErrchk(cudaMemcpy(d_idxs_, &idxs[0], B_ * 2 * sizeof(*d_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_u_buffer_, B_ * sizeof(*d_u_buffer_));

  cudaSafeMalloc(&d_system_idxs_, B_ * sizeof(*d_system_idxs_));

  gpuErrchk(cudaMemcpy(d_system_idxs_, &system_idxs[0],
                       B_ * sizeof(*d_system_idxs_), cudaMemcpyHostToDevice));
};

template <typename RealType>
NonbondedPairListPrecomputed<RealType>::~NonbondedPairListPrecomputed() {
  gpuErrchk(cudaFree(d_idxs_));
  gpuErrchk(cudaFree(d_system_idxs_));
  gpuErrchk(cudaFree(d_u_buffer_));
};

template <typename RealType>
void NonbondedPairListPrecomputed<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  if (P != PARAMS_PER_PAIR * B_) {
    throw std::runtime_error(
        "NonbondedPairListPrecomputed::execute_device(): expected P == " +
        std::to_string(PARAMS_PER_PAIR) + "*B, got P=" + std::to_string(P) +
        ", " + std::to_string(PARAMS_PER_PAIR) +
        "*B=" + std::to_string(PARAMS_PER_PAIR * B_));
  }

  if (B_ > 0) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(B_, tpb);

    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<blocks, tpb, 0, stream>>>(
        num_atoms_, B_, d_x, d_p, d_box, d_idxs_, d_system_idxs_, beta_,
        cutoff_ * cutoff_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);
    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      nrg_accum_.sum_device(B_, d_u_buffer_, d_system_idxs_, d_u, stream);
    }
  }
};

template <typename RealType>
void NonbondedPairListPrecomputed<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp,
    RealType *du_dp_float) {

  if (P % PARAMS_PER_PAIR != 0) {
    throw std::runtime_error("NonbondedPairListPrecomputed::du_dp_fixed_to_"
                             "float(): expected P % B == 0, got P=" +
                             std::to_string(P) + ", " +
                             std::to_string(PARAMS_PER_PAIR) +
                             "*B=" + std::to_string(PARAMS_PER_PAIR * B_));
  }
  for (int i = 0; i < P / PARAMS_PER_PAIR; i++) {
    const int idx = i * PARAMS_PER_PAIR;
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
};

template <typename RealType>
int NonbondedPairListPrecomputed<RealType>::batch_size() const {
  return num_batches_;
}

template class NonbondedPairListPrecomputed<double>;
template class NonbondedPairListPrecomputed<float>;

} // namespace tmd
