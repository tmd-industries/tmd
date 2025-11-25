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

#pragma once

#include "bound_potential.hpp"
#include <vector>

namespace tmd {

template <typename RealType> class Integrator {

public:
  virtual ~Integrator() {};

  virtual int num_systems() const = 0;

  virtual void
  step_fwd(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
           RealType *d_x_t, RealType *d_v_t, RealType *d_box_t,
           unsigned int *d_idxs, cudaStream_t stream) = 0;

  virtual void
  initialize(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
             RealType *d_x_t, RealType *d_v_t, RealType *d_box_t,
             unsigned int *d_idxs, cudaStream_t stream) = 0;

  virtual void
  finalize(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
           RealType *d_x_t, RealType *d_v_t, RealType *d_box_t,
           unsigned int *d_idxs, cudaStream_t stream) = 0;
};

} // namespace tmd
