// Copyright 2019-2025, Relay Therapeutics
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
#include "centroid_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_centroid_restraint.cuh"
#include "math_utils.cuh"
#include <vector>

namespace tmd {

template <typename RealType>
CentroidRestraint<RealType>::CentroidRestraint(
    const std::vector<int> &group_a_idxs, const std::vector<int> &group_b_idxs,
    const RealType kb, const RealType b0)
    : N_A_(group_a_idxs.size()), N_B_(group_b_idxs.size()), kb_(kb), b0_(b0) {

  cudaSafeMalloc(&d_group_a_idxs_, N_A_ * sizeof(*d_group_a_idxs_));
  gpuErrchk(cudaMemcpy(d_group_a_idxs_, &group_a_idxs[0],
                       N_A_ * sizeof(*d_group_a_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_group_b_idxs_, N_B_ * sizeof(*d_group_b_idxs_));
  gpuErrchk(cudaMemcpy(d_group_b_idxs_, &group_b_idxs[0],
                       N_B_ * sizeof(*d_group_b_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_centroid_a_, 3 * sizeof(*d_centroid_a_));
  cudaSafeMalloc(&d_centroid_b_, 3 * sizeof(*d_centroid_b_));
};

template <typename RealType> CentroidRestraint<RealType>::~CentroidRestraint() {
  gpuErrchk(cudaFree(d_group_a_idxs_));
  gpuErrchk(cudaFree(d_group_b_idxs_));
  gpuErrchk(cudaFree(d_centroid_a_));
  gpuErrchk(cudaFree(d_centroid_b_));
};

template <typename RealType>
void CentroidRestraint<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp,
    __int128 *d_u, // [1]
    cudaStream_t stream) {

  assert(batches == 1);
  if (N_B_ + N_A_ > 0) {
    int tpb = DEFAULT_THREADS_PER_BLOCK;

    int blocks = ceil_divide(N_B_ + N_A_, tpb);
    gpuErrchk(cudaMemsetAsync(d_centroid_a_, 0.0, 3 * sizeof(*d_centroid_a_),
                              stream));
    gpuErrchk(cudaMemsetAsync(d_centroid_b_, 0.0, 3 * sizeof(*d_centroid_b_),
                              stream));
    k_calc_centroid<RealType>
        <<<blocks, tpb, 0, stream>>>(d_x, d_group_a_idxs_, d_group_b_idxs_,
                                     N_A_, N_B_, d_centroid_a_, d_centroid_b_);
    gpuErrchk(cudaPeekAtLastError());

    k_centroid_restraint<RealType><<<blocks, tpb, 0, stream>>>(
        d_x, d_group_a_idxs_, d_group_b_idxs_, N_A_, N_B_, d_centroid_a_,
        d_centroid_b_, kb_, b0_, d_du_dx,
        d_u // Can write directly to the energy buffer for this potential.
    );
    gpuErrchk(cudaPeekAtLastError());
  }
};

template class CentroidRestraint<double>;
template class CentroidRestraint<float>;

} // namespace tmd
