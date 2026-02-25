// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025, Forrest York
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

#include "streamed_potential_runner.hpp"

namespace tmd {

template <typename RealType>
StreamedPotentialRunner<RealType>::StreamedPotentialRunner(){};

template <typename RealType>
StreamedPotentialRunner<RealType>::~StreamedPotentialRunner() {}

// wrap execute_device
template <typename RealType>
void StreamedPotentialRunner<RealType>::execute_potentials(
    const int num_systems,
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps, const int N,
    const RealType *d_x,   // [num_systems, N, 3]
    const RealType *d_box, // [num_systems, 3, 3]
    unsigned long long *d_du_dx, unsigned long long *d_du_dp,
    __int128 *d_u, // [num_systems, bps.size()]
    cudaStream_t stream) {

  manager_.record_master_event(stream);
  for (int i = 0; i < bps.size(); i++) {
    manager_.wait_on_master(i, stream);
  }

  for (int i = 0; i < bps.size(); i++) {
    bps[i]->execute_device(num_systems, N, d_x, d_box, d_du_dx, d_du_dp,
                           d_u == nullptr ? nullptr : d_u + num_systems * i,
                           manager_.get_stream(i));
  }

  for (int i = 0; i < bps.size(); i++) {
    manager_.record_and_wait_on_child(i, stream);
  }
};

template class StreamedPotentialRunner<double>;
template class StreamedPotentialRunner<float>;

} // namespace tmd
