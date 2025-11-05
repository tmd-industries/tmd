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
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "nonbonded_common.hpp"
#include "summed_potential.hpp"
#include <memory>
#include <numeric>
#include <stdexcept>

namespace tmd {

template <typename RealType>
SummedPotential<RealType>::SummedPotential(
    const std::vector<std::shared_ptr<Potential<RealType>>> potentials,
    const std::vector<int> params_sizes, const bool parallel)
    : potentials_(potentials), params_sizes_(params_sizes),
      P_(std::accumulate(params_sizes.begin(), params_sizes.end(), 0)),
      parallel_(parallel), d_u_buffer_(potentials.size()),
      nrg_accum_(1, potentials_.size()) {
  if (potentials_.size() != params_sizes_.size()) {
    throw std::runtime_error(
        "number of potentials != number of parameter sizes");
  }
};

template <typename RealType> SummedPotential<RealType>::~SummedPotential(){};

template <typename RealType>
const std::vector<std::shared_ptr<Potential<RealType>>> &
SummedPotential<RealType>::get_potentials() {
  return potentials_;
}

template <typename RealType>
const std::vector<int> &SummedPotential<RealType>::get_parameter_sizes() {
  return params_sizes_;
}

template <typename RealType>
void SummedPotential<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  assert(batches == 1);
  if (P != P_) {
    throw std::runtime_error(
        "SummedPotential<RealType>::execute_device(): expected " +
        std::to_string(P_) + " parameters, got " + std::to_string(P));
  }
  if (d_u) {
    gpuErrchk(cudaMemsetAsync(d_u_buffer_.data, 0, d_u_buffer_.size(), stream));
  }

  int offset = 0;
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
        batches, N, params_sizes_[i], d_x, d_p + offset, d_box, d_du_dx,
        d_du_dp == nullptr ? nullptr : d_du_dp + offset,
        d_u == nullptr ? nullptr : d_u_buffer_.data + i, pot_stream);
    offset += params_sizes_[i];
  }
  if (parallel_) {
    for (auto i = 0; i < potentials_.size(); i++) {
      manager_.record_and_wait_on_child(i, stream);
    }
  }
  if (d_u) {
    // nullptr for the d_system_idxs as batch size is fixed to 1
    nrg_accum_.sum_device(potentials_.size(), d_u_buffer_.data, nullptr, d_u,
                          stream);
  }
};

template <typename RealType>
void SummedPotential<RealType>::du_dp_fixed_to_float(
    const int N, const int P, const unsigned long long *du_dp,
    RealType *du_dp_float) {

  int offset = 0;

  for (auto i = 0; i < potentials_.size(); i++) {
    potentials_[i]->du_dp_fixed_to_float(N, params_sizes_[i], du_dp + offset,
                                         du_dp_float + offset);
    offset += params_sizes_[i];
  }
}

template class SummedPotential<double>;
template class SummedPotential<float>;

} // namespace tmd
