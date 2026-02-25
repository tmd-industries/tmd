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
#include "fanout_summed_potential.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_indices.cuh"
#include "nonbonded_common.hpp"
#include "potential_utils.hpp"
#include <memory>

namespace tmd {

template <typename RealType>
FanoutSummedPotential<RealType>::FanoutSummedPotential(
    const std::vector<std::shared_ptr<Potential<RealType>>> &potentials,
    const bool parallel)
    : potentials_(potentials), parallel_(parallel),
      num_systems_(potentials.size() > 0 ? potentials[0]->num_systems() : 1),
      d_u_buffer_(num_systems_ * potentials_.size()),
      d_system_idxs_(num_systems_ * potentials_.size()),
      nrg_accum_(num_systems_, potentials_.size() * num_systems_) {

  verify_potentials_are_compatible(potentials_);
  k_segment_arange<<<dim3(ceil_divide(num_systems_, DEFAULT_THREADS_PER_BLOCK),
                          potentials_.size()),
                     DEFAULT_THREADS_PER_BLOCK>>>(
      potentials_.size(), num_systems_, d_system_idxs_.data);
  gpuErrchk(cudaPeekAtLastError());
};

template <typename RealType>
FanoutSummedPotential<RealType>::~FanoutSummedPotential(){};

template <typename RealType>
const std::vector<std::shared_ptr<Potential<RealType>>> &
FanoutSummedPotential<RealType>::get_potentials() {
  return potentials_;
}

template <typename RealType>
void FanoutSummedPotential<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  if (d_u) {
    gpuErrchk(cudaMemsetAsync(d_u_buffer_.data, 0, d_u_buffer_.size(), stream));
  }

  if (parallel_) {
    manager_.record_master_event(stream);
    for (auto i = 0; i < potentials_.size(); i++) {
      // Always sync the new streams with the incoming stream to ensure that the
      // state of the incoming buffers are valid
      manager_.wait_on_master(i, stream);
    }
  }
  cudaStream_t pot_stream = stream;
  for (auto i = 0; i < potentials_.size(); i++) {
    if (parallel_) {
      pot_stream = manager_.get_stream(i);
    }
    potentials_[i]->execute_device(
        batches, N, P, d_x, d_p, d_box, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_.data + num_systems_ * i,
        pot_stream);
  }

  if (parallel_) {
    for (auto i = 0; i < potentials_.size(); i++) {
      manager_.record_and_wait_on_child(i, stream);
    }
  }
  if (d_u) {
    // nullptr for the d_system_idxs if num_systems_ == 1
    nrg_accum_.sum_device(num_systems_ * potentials_.size(), d_u_buffer_.data,
                          num_systems_ > 1 ? d_system_idxs_.data : nullptr, d_u,
                          stream);
  }
};

template <typename RealType>
void FanoutSummedPotential<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp,
    RealType *du_dp_float) {

  if (!potentials_.empty()) {
    potentials_[0]->du_dp_fixed_to_float(N, P, du_dp, du_dp_float);
  }
}

template <typename RealType>
int FanoutSummedPotential<RealType>::num_systems() const {
  return num_systems_;
}

template class FanoutSummedPotential<double>;
template class FanoutSummedPotential<float>;

} // namespace tmd
