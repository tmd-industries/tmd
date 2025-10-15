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
#include <vector>

namespace tmd {

template <typename RealType>
class NonbondedPairListPrecomputed : public Potential<RealType> {

  typedef void (*k_nonbonded_precomputed_fn)(
      const int num_atoms, const int B, const RealType *__restrict__ coords,
      const RealType *__restrict__ params, const RealType *__restrict__ box,
      const int *__restrict__ pair_idxs, const int *__restrict__ system_idxs,
      const RealType beta, const RealType cutoff_squared,
      unsigned long long *__restrict__ du_dx,
      unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u_buffer);

private:
  const int num_batches_;
  const int num_atoms_;
  const int B_;

  const RealType beta_;
  const RealType cutoff_;

  int *d_idxs_;        // [B, 2]
  int *d_system_idxs_; // [B]
  __int128 *d_u_buffer_;

  EnergyAccumulator nrg_accum_;

  std::array<k_nonbonded_precomputed_fn, 8> kernel_ptrs_;

public:
  NonbondedPairListPrecomputed(const int num_batches, const int num_atoms,
                               const std::vector<int> &pair_idxs,   // [B, 2]
                               const std::vector<int> &system_idxs, // [B]
                               const RealType beta, const RealType cutoff);

  ~NonbondedPairListPrecomputed();

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx, // buffered
                              unsigned long long *d_du_dp,
                              __int128 *d_u, // buffered
                              cudaStream_t stream) override;

  void du_dp_fixed_to_float(const int N, const int P,
                            const unsigned long long *du_dp,
                            RealType *du_dp_float) override;

  virtual int batch_size() const override;
};

} // namespace tmd
