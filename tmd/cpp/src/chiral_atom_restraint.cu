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
#include "chiral_atom_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_chiral_restraint.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <vector>

namespace tmd {

template <typename RealType>
ChiralAtomRestraint<RealType>::ChiralAtomRestraint(const std::vector<int> &idxs)
    : R_(idxs.size() / 4), nrg_accum_(1, R_),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP                U  X  P
                    &k_chiral_atom_restraint<RealType, 0, 0, 0>,
                    &k_chiral_atom_restraint<RealType, 0, 0, 1>,
                    &k_chiral_atom_restraint<RealType, 0, 1, 0>,
                    &k_chiral_atom_restraint<RealType, 0, 1, 1>,
                    &k_chiral_atom_restraint<RealType, 1, 0, 0>,
                    &k_chiral_atom_restraint<RealType, 1, 0, 1>,
                    &k_chiral_atom_restraint<RealType, 1, 1, 0>,
                    &k_chiral_atom_restraint<RealType, 1, 1, 1>}) {

  if (idxs.size() % 4 != 0) {
    throw std::runtime_error("idxs.size() must be exactly 4*k!");
  }

  cudaSafeMalloc(&d_idxs_, R_ * 4 * sizeof(*d_idxs_));
  gpuErrchk(cudaMemcpy(d_idxs_, &idxs[0], R_ * 4 * sizeof(*d_idxs_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_u_buffer_, R_ * sizeof(*d_u_buffer_));
};

template <typename RealType>
ChiralAtomRestraint<RealType>::~ChiralAtomRestraint() {
  gpuErrchk(cudaFree(d_idxs_));
  gpuErrchk(cudaFree(d_u_buffer_));
};

template <typename RealType>
void ChiralAtomRestraint<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  assert(batches == 1);
  if (P != R_) {
    throw std::runtime_error(
        "ChiralAtomRestraint::execute_device(): expected P == R, got P=" +
        std::to_string(P) + ", R=" + std::to_string(R_));
  }

  if (R_ > 0) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(R_, tpb);

    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<blocks, tpb, 0, stream>>>(
        R_, d_x, d_p, d_idxs_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);
    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      // nullptr for the d_system_idxs as batch size is fixed to 1
      nrg_accum_.sum_device(R_, d_u_buffer_, nullptr, d_u, stream);
    }
  }
};

template class ChiralAtomRestraint<double>;
template class ChiralAtomRestraint<float>;

} // namespace tmd
