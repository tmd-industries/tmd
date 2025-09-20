// Copyright 2025, Forrest York
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

// implements a runner for running potentials from python
#pragma once

#include "potential.hpp"
#include "stream_manager.hpp"
#include <memory>
#include <vector>

namespace tmd {

template <typename RealType> class PotentialExecutor {

public:
  PotentialExecutor(const bool parallel = true);

  ~PotentialExecutor();

  static const int D = 3;

  void execute_potentials(
      const int N, const std::vector<int> &param_sizes,
      const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
      const RealType *h_x, const RealType *h_params, const RealType *h_box,
      unsigned long long *h_du_dx, unsigned long long *h_du_dp, __int128 *h_u);

  void execute_potentials_device(
      const int N, const std::vector<int> &param_sizes,
      const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
      const RealType *d_x, const RealType *d_params, const RealType *d_box,
      unsigned long long *d_du_dx, unsigned long long *d_du_dp, __int128 *d_u,
      cudaStream_t stream);

  void du_dp_fixed_to_float(
      const int N, const std::vector<int> &param_sizes,
      const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
      const unsigned long long *h_du_dp, RealType *h_du_dp_float);

  void execute_batch_potentials_sparse(
      const int N, const std::vector<int> &param_sizes, const int batch_size,
      const int coord_batches, const int param_batches,
      const unsigned int *coords_batch_idxs,
      const unsigned int *params_batch_idxs,
      const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
      const RealType *h_x, const RealType *h_params, const RealType *h_box,
      unsigned long long *h_du_dx, unsigned long long *h_du_dp, __int128 *h_u);

  int get_total_num_params(const std::vector<int> &param_sizes) const;

private:
  const bool parallel_;

  cudaStream_t stream_;

  StreamManager manager_;
};

} // namespace tmd
