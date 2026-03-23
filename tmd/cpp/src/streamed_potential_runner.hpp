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

// this implements a runner for running potentials in multiple streams and then
// syncing with the parent stream
#pragma once

#include "bound_potential.hpp"
#include "stream_manager.hpp"
#include <memory>
#include <vector>

namespace tmd {

template <typename RealType> class StreamedPotentialRunner {

public:
  StreamedPotentialRunner();

  ~StreamedPotentialRunner();

  // wrap execute_device
  void execute_potentials(
      const int num_systems,
      std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps, const int N,
      const RealType *d_x, const RealType *d_box, unsigned long long *d_du_dx,
      unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream);

private:
  StreamManager manager_;
};

} // namespace tmd
