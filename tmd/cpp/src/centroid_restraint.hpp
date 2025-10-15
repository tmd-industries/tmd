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

#include "potential.hpp"
#include <vector>

namespace tmd {

template <typename RealType>
class CentroidRestraint : public Potential<RealType> {

private:
  int *d_group_a_idxs_;
  int *d_group_b_idxs_;

  unsigned long long *d_centroid_a_;
  unsigned long long *d_centroid_b_;

  __int128 *d_u_buffer_;

  int N_A_;
  int N_B_;

  RealType kb_;
  RealType b0_;

public:
  CentroidRestraint(const std::vector<int> &group_a_idxs,
                    const std::vector<int> &group_b_idxs, const RealType kb,
                    const RealType b0);

  ~CentroidRestraint();

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx,
                              unsigned long long *d_du_dp, __int128 *d_u,
                              cudaStream_t stream) override;
};

} // namespace tmd
