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
#include "k_nonbonded_pair_list.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include "nonbonded_pair_list.hpp"
#include <stdexcept>
#include <vector>

namespace tmd {

template <typename RealType, bool Negated>
NonbondedPairList<RealType, Negated>::NonbondedPairList(
    const int num_batches, const int num_atoms,
    const std::vector<int> &pair_idxs,   // [M, 2]
    const std::vector<RealType> &scales, // [M, 2]
    const std::vector<int> &system_idxs, // [M]
    const RealType beta, const RealType cutoff)
    : num_batches_(num_batches), num_atoms_(num_atoms),
      max_idxs_(pair_idxs.size() / IDXS_DIM), cur_num_idxs_(max_idxs_),
      beta_(beta), cutoff_(cutoff), nrg_accum_(num_batches_, cur_num_idxs_),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP                       U  X  P
                    &k_nonbonded_pair_list<RealType, Negated, 0, 0, 0>,
                    &k_nonbonded_pair_list<RealType, Negated, 0, 0, 1>,
                    &k_nonbonded_pair_list<RealType, Negated, 0, 1, 0>,
                    &k_nonbonded_pair_list<RealType, Negated, 0, 1, 1>,
                    &k_nonbonded_pair_list<RealType, Negated, 1, 0, 0>,
                    &k_nonbonded_pair_list<RealType, Negated, 1, 0, 1>,
                    &k_nonbonded_pair_list<RealType, Negated, 1, 1, 0>,
                    &k_nonbonded_pair_list<RealType, Negated, 1, 1, 1>}) {

  if (pair_idxs.size() % IDXS_DIM != 0) {
    throw std::runtime_error("pair_idxs.size() must be even, but got " +
                             std::to_string(pair_idxs.size()));
  }

  if (system_idxs.size() != max_idxs_) {
    throw std::runtime_error("system_idxs.size() != (pair_idxs.size() / " +
                             std::to_string(IDXS_DIM) + "), got " +
                             std::to_string(system_idxs.size()) + " and " +
                             std::to_string(max_idxs_));
  }

  static_assert(IDXS_DIM == 2);
  for (int i = 0; i < cur_num_idxs_; i++) {
    auto src = pair_idxs[i * IDXS_DIM + 0];
    auto dst = pair_idxs[i * IDXS_DIM + 1];
    if (src == dst) {
      throw std::runtime_error(
          "illegal pair with src == dst: " + std::to_string(src) + ", " +
          std::to_string(dst));
    }
  }

  if (scales.size() / IDXS_DIM != cur_num_idxs_) {
    throw std::runtime_error(
        "expected same number of pairs and scale tuples, but got " +
        std::to_string(cur_num_idxs_) +
        " != " + std::to_string(scales.size() / IDXS_DIM));
  }

  cudaSafeMalloc(&d_u_buffer_, cur_num_idxs_ * sizeof(*d_u_buffer_));

  cudaSafeMalloc(&d_pair_idxs_,
                 cur_num_idxs_ * IDXS_DIM * sizeof(*d_pair_idxs_));
  gpuErrchk(cudaMemcpy(d_pair_idxs_, &pair_idxs[0],
                       cur_num_idxs_ * IDXS_DIM * sizeof(*d_pair_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_scales_, cur_num_idxs_ * IDXS_DIM * sizeof(*d_scales_));
  gpuErrchk(cudaMemcpy(d_scales_, &scales[0],
                       cur_num_idxs_ * IDXS_DIM * sizeof(*d_scales_),
                       cudaMemcpyHostToDevice));
  cudaSafeMalloc(&d_system_idxs_, cur_num_idxs_ * sizeof(*d_system_idxs_));

  gpuErrchk(cudaMemcpy(d_system_idxs_, &system_idxs[0],
                       cur_num_idxs_ * sizeof(*d_system_idxs_),
                       cudaMemcpyHostToDevice));
};

template <typename RealType, bool Negated>
NonbondedPairList<RealType, Negated>::~NonbondedPairList() {
  gpuErrchk(cudaFree(d_pair_idxs_));
  gpuErrchk(cudaFree(d_scales_));
  gpuErrchk(cudaFree(d_system_idxs_));
  gpuErrchk(cudaFree(d_u_buffer_));
};

template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  if (P != num_batches_ * num_atoms_ * PARAMS_PER_ATOM) {
    throw std::runtime_error(
        "NonbondedPairList::execute_device(): expected P == num_atoms_*" +
        std::to_string(PARAMS_PER_ATOM) +
        "*num_batces_, got P=" + std::to_string(P) + ", num_atoms_*" +
        std::to_string(PARAMS_PER_ATOM) + "*" + std::to_string(num_batches_) +
        "=" + std::to_string(num_atoms_ * PARAMS_PER_ATOM * num_batches_));
  }

  if (cur_num_idxs_ > 0) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int num_blocks_pairs = ceil_divide(cur_num_idxs_, tpb);

    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<num_blocks_pairs, tpb, 0, stream>>>(
        num_atoms_, cur_num_idxs_, d_x, d_p, d_box, d_pair_idxs_,
        d_system_idxs_, d_scales_, beta_, cutoff_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);

    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      nrg_accum_.sum_device(cur_num_idxs_, d_u_buffer_, d_system_idxs_, d_u,
                            stream);
    }
  }
}

// TODO: this implementation is duplicated from NonbondedInteractionGroup
template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::du_dp_fixed_to_float(
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

template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::set_idxs_device(
    const int num_idxs, const int *d_new_idxs, cudaStream_t stream) {
  if (max_idxs_ < num_idxs) {
    throw std::runtime_error(
        "set_idxs_device(): Max number of pairs " + std::to_string(max_idxs_) +
        " is less than new idxs " + std::to_string(num_idxs));
  }
  gpuErrchk(cudaMemcpyAsync(d_pair_idxs_, d_new_idxs,
                            num_idxs * IDXS_DIM * sizeof(*d_pair_idxs_),
                            cudaMemcpyDeviceToDevice, stream));
  cur_num_idxs_ = num_idxs;
}

template <typename RealType, bool Negated>
void NonbondedPairList<RealType, Negated>::set_scales_device(
    const int num_idxs, const RealType *d_new_scales, cudaStream_t stream) {
  if (max_idxs_ < num_idxs) {
    throw std::runtime_error("set_scales_device(): Max number of scales " +
                             std::to_string(max_idxs_) +
                             " is less than new idxs " +
                             std::to_string(num_idxs));
  }
  gpuErrchk(cudaMemcpyAsync(d_scales_, d_new_scales,
                            num_idxs * IDXS_DIM * sizeof(*d_scales_),
                            cudaMemcpyDeviceToDevice, stream));
}

template <typename RealType, bool Negated>
int NonbondedPairList<RealType, Negated>::get_num_idxs() const {
  return cur_num_idxs_;
}

template <typename RealType, bool Negated>
int *NonbondedPairList<RealType, Negated>::get_idxs_device() {
  return d_pair_idxs_;
}

template <typename RealType, bool Negated>
RealType *NonbondedPairList<RealType, Negated>::get_scales_device() {
  return d_scales_;
}

template <typename RealType, bool Negated>
std::vector<int> NonbondedPairList<RealType, Negated>::get_idxs_host() const {
  return device_array_to_vector<int>(cur_num_idxs_ * IDXS_DIM, d_pair_idxs_);
}

template <typename RealType, bool Negated>
std::vector<RealType>
NonbondedPairList<RealType, Negated>::get_scales_host() const {
  return device_array_to_vector<RealType>(cur_num_idxs_ * IDXS_DIM, d_scales_);
}

template <typename RealType, bool Negated>
int NonbondedPairList<RealType, Negated>::batch_size() const {
  return num_batches_;
}

template class NonbondedPairList<double, true>;
template class NonbondedPairList<float, true>;

template class NonbondedPairList<double, false>;
template class NonbondedPairList<float, false>;

} // namespace tmd
