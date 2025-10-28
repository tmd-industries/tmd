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

// this file implements the MonteCarlo barostat from that of OpenMM
#pragma once

#include "barostat.hpp"
#include "bound_potential.hpp"
#include "curand_kernel.h"
#include "streamed_potential_runner.hpp"
#include <memory>
#include <random>
#include <vector>

namespace tmd {

template <typename RealType>
class AnisotropicMonteCarloBarostat : public MonteCarloBarostat<RealType> {

public:
  AnisotropicMonteCarloBarostat(
      const int N,
      const RealType pressure,    // in bar
      const RealType temperature, // in kelvin
      const std::vector<std::vector<int>> &group_idxs, const int interval,
      const std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
      const int seed, const bool adapt_volume_scale_factor,
      const RealType initial_volume_scale_factor, const bool scale_x,
      const bool scale_y, const bool scale_z);

  ~AnisotropicMonteCarloBarostat();

private:
  std::mt19937 generator_;
  std::uniform_real_distribution<RealType> distribution_;

  const bool scale_x_;
  const bool scale_y_;
  const bool scale_z_;

  virtual void propose_move(const int N, const RealType *d_x,
                            const RealType *d_box,
                            cudaStream_t stream) override;
};

} // namespace tmd
