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

#include <memory>
#include <vector>

#include "device_buffer.hpp"
#include "potential.hpp"

namespace tmd {

// a potential bounded to a set of parameters with some shape
template <typename RealType> struct BoundPotential {

  BoundPotential(std::shared_ptr<Potential<RealType>> potential,
                 const std::vector<RealType> &params, const int params_dim);

  int size;
  const int params_dim;
  DeviceBuffer<RealType> d_p;
  std::shared_ptr<Potential<RealType>> potential;

  std::vector<RealType> get_params() const;

  void set_params(const std::vector<RealType> &params);

  void set_params_device(const int size, const RealType *d_p,
                         const cudaStream_t stream);

  void execute_host(const int batches, const int N, const RealType *h_x,
                    const RealType *h_box, unsigned long long *h_du_dx,
                    __int128 *h_u);

  void execute_device(const int batches, const int N, const RealType *d_x,
                      const RealType *d_box, unsigned long long *d_du_dx,
                      unsigned long long *d_du_dp, __int128 *d_u,
                      cudaStream_t stream);

  void execute_batch_host(const int coord_batch_size, const int N,
                          const RealType *h_x, const RealType *h_box,
                          unsigned long long *h_du_dx, __int128 *h_u);
};

} // namespace tmd
