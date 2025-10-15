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

// Nonbonded Pair List that computes the interaction energies between pairs of
// atoms. The negated version of this potential should be used in conjunction
// with a NonbondedInteractionGroup as a way to compute the exclusions and
// cancel them out from the other potentials to ensure valid energies and du_dp
// values, combine the potentials using a FanoutSummedPotential
template <typename RealType, bool Negated>
class NonbondedPairList : public Potential<RealType> {

  typedef void (*k_nonbonded_pairlist_fn)(
      const int N, const int M, const RealType *__restrict__ coords,
      const RealType *__restrict__ params, const RealType *__restrict__ box,
      const int *__restrict__ idxs, const int *__restrict__ system_idxs,
      const RealType *__restrict__ scales, const RealType beta,
      const RealType cutoff, unsigned long long *__restrict__ du_dx,
      unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u_buffer);

private:
  const int num_batches_;
  const int num_atoms_;
  const int max_idxs_;
  int cur_num_idxs_; // number of pairs

  const RealType beta_;
  const RealType cutoff_;

  __int128 *d_u_buffer_; // [M]

  int *d_pair_idxs_;   // [M, 2]
  int *d_system_idxs_; // [M]
  RealType *d_scales_; // [M, 2]

  EnergyAccumulator nrg_accum_;

  std::array<k_nonbonded_pairlist_fn, 8> kernel_ptrs_;

public:
  static const int IDXS_DIM = 2;

  NonbondedPairList(const int num_batches, const int num_atoms,
                    const std::vector<int> &pair_idxs,   // [M, 2]
                    const std::vector<RealType> &scales, // [M, 2]
                    const std::vector<int> &system_idxs, // [M]
                    const RealType beta, const RealType cutoff);

  ~NonbondedPairList();

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx,
                              unsigned long long *d_du_dp, __int128 *d_u,
                              cudaStream_t stream) override;

  virtual int batch_size() const override;

  void du_dp_fixed_to_float(const int N, const int P,
                            const unsigned long long *du_dp,
                            RealType *du_dp_float) override;

  void set_idxs_device(const int num_idxs, const int *d_idxs,
                       cudaStream_t stream);

  void set_scales_device(const int num_idxs, const RealType *d_scales,
                         cudaStream_t stream);

  int get_num_idxs() const;

  int *get_idxs_device();

  RealType *get_scales_device();

  std::vector<int> get_idxs_host() const;

  std::vector<RealType> get_scales_host() const;
};

} // namespace tmd
