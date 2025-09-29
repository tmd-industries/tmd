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

#pragma once

#include "energy_accum.hpp"
#include "potential.hpp"
#include <array>
#include <vector>

namespace tmd {

template <typename RealType>
class LogFlatBottomBond : public Potential<RealType> {

  typedef void (*k_log_flat_bond_fn)(
      const int N, const RealType *__restrict__ coords,
      const RealType *__restrict__ box, const RealType *__restrict__ params,
      const int *__restrict__ idxs, const RealType beta,
      unsigned long long *__restrict__ du_dx,
      unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u_buffer);

private:
  const int max_idxs_;
  int cur_num_idxs_;
  const RealType beta_;

  int *d_bond_idxs_;
  __int128 *d_u_buffer_;

  EnergyAccumulator nrg_accum_;

  std::array<k_log_flat_bond_fn, 8> kernel_ptrs_;

public:
  static const int IDXS_DIM = 2;

  LogFlatBottomBond(const int num_atoms, const std::vector<int> &bond_idxs,
                    RealType beta); // [B, 2]

  ~LogFlatBottomBond();

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx,
                              unsigned long long *d_du_dp, __int128 *d_u,
                              cudaStream_t stream) override;

  int num_bonds() const { return cur_num_idxs_; }

  void set_bonds_device(const int num_bonds, const int *d_bonds,
                        const cudaStream_t stream);
};

} // namespace tmd
