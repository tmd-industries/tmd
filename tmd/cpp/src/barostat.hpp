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

#include "bound_potential.hpp"
#include "curand_kernel.h"
#include "energy_accum.hpp"
#include "mover.hpp"
#include "streamed_potential_runner.hpp"
#include <memory>
#include <vector>

namespace tmd {

template <typename RealType> class MonteCarloBarostat : public Mover<RealType> {

public:
  MonteCarloBarostat(
      const int N,
      const RealType pressure,    // in bar
      const RealType temperature, // in kelvin
      const std::vector<std::vector<int>> &group_idxs, const int interval,
      const std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
      const int seed, const bool adapt_volume_scale_factor,
      const RealType initial_volume_scale_factor);

  ~MonteCarloBarostat();

  // move() may modify d_x and d_box
  virtual void move(const int num_systems, const int N, RealType *d_x,
                    RealType *d_box, cudaStream_t stream) override;

  std::vector<RealType> get_volume_scale_factor() const;

  void set_volume_scale_factor(const RealType volume_scale_factor);

  void set_pressure(const RealType pressure);

  void set_adaptive_scaling(const bool adaptive_scaling_enabled);

  bool get_adaptive_scaling();

protected:
  const int N_;
  const int num_mols_;

  bool adaptive_scaling_enabled_; // Whether or no to adapt d_volume_scale_

  void reset_counters();

  virtual void propose_move(const int N, const RealType *d_x,
                            const RealType *d_box, cudaStream_t stream);
  virtual void decide_move(const int N, RealType *d_x, RealType *d_box,
                           cudaStream_t stream);

  std::vector<std::shared_ptr<BoundPotential<RealType>>> bps_;

  RealType pressure_;
  const RealType temperature_;
  const int seed_;

  // device rng
  curandState_t *d_rand_state_;         // [1]
  RealType *d_metropolis_hasting_rand_; // [1]

  int num_grouped_atoms_;

  int *d_num_attempted_;
  int *d_num_accepted_;

  __int128 *d_u_buffer_;
  __int128 *d_u_proposed_buffer_;

  __int128 *d_init_u_;
  __int128 *d_final_u_;

  RealType *d_volume_;
  RealType *d_volume_delta_;
  RealType *d_length_scale_;
  RealType *d_volume_scale_;

  RealType *d_x_proposed_;
  RealType *d_box_proposed_;

  int *d_system_idxs_;

  int *d_atom_idxs_;   // grouped index to atom coords
  int *d_mol_idxs_;    // grouped index to molecule index
  int *d_mol_offsets_; // Offset of molecules to determine size of mols

  unsigned long long *d_centroids_; // Accumulate centroids in fixed point to
                                    // ensure deterministic behavior

  StreamedPotentialRunner<RealType> runner_;

  EnergyAccumulator nrg_accum_;
};

} // namespace tmd
