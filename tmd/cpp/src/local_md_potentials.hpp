// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025-2026, Forrest York
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
#include "device_buffer.hpp"
#include "pinned_host_buffer.hpp"
#include <memory>
#include <vector>

#include "bound_potential.hpp"
#include "curand.h"
#include "flat_bottom_bond.hpp"
#include "local_md_utils.hpp"
#include "log_flat_bottom_bond.hpp"
#include "potential.hpp"

namespace tmd {

template <typename RealType> class LocalMDPotentials {

public:
  constexpr static RealType DEFAULT_NBLIST_PADDING = static_cast<RealType>(0.2);

  LocalMDPotentials(
      const int num_systems, const int N,
      const std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
      const std::vector<std::shared_ptr<Potential<RealType>>> &nonbonded_pots,
      const bool freeze_reference = true,
      const RealType temperature = static_cast<RealType>(0.0),
      const RealType nblist_padding = DEFAULT_NBLIST_PADDING);

  ~LocalMDPotentials();

  unsigned int *get_free_idxs();

  void setup_from_idxs(RealType *d_x_t_, RealType *d_box_t,
                       const std::vector<int> &local_idxs, const int seed,
                       const RealType radius, const RealType k,
                       cudaStream_t stream);

  void setup_from_selection(const std::vector<int> &reference_idx,
                            const std::vector<int> &selection_idxs,
                            const RealType radius, const RealType k,
                            cudaStream_t stream);

  std::vector<std::shared_ptr<BoundPotential<RealType>>> get_potentials();

  void reset_potentials(cudaStream_t stream);

  const bool freeze_reference;
  const RealType temperature;
  const RealType nblist_padding; // Padding to set when configuring Local MD

private:
  const int num_systems_;
  const int N_;
  std::size_t temp_storage_bytes_;
  int num_allpairs_idxs_;

  // Bound potentials that were provided on construction. Parameters may change
  // and have to be copied to the true potentials All of the potentials in bps_
  // plus any additional potentials set up for local MD
  const std::vector<std::shared_ptr<BoundPotential<RealType>>> bps_;
  std::vector<std::shared_ptr<BoundPotential<RealType>>> local_md_potentials_;
  const std::shared_ptr<Potential<RealType>> nonbonded_pot_;

  const RealType
      initial_nblist_padding_; // Padding that is set by the incoming potential,
                               // used to reset to old value
  // Restraint for the free particles to the reference particle
  std::shared_ptr<FlatBottomBond<RealType>> free_restraint_;
  std::shared_ptr<BoundPotential<RealType>> bound_free_restraint_;

  // Restraint for the frozen particles to the reference particle
  std::shared_ptr<LogFlatBottomBond<RealType>> frozen_restraint_;
  std::shared_ptr<BoundPotential<RealType>> bound_frozen_restraint_;

  DeviceBuffer<int> d_restraint_pairs_;
  DeviceBuffer<RealType> d_bond_params_;
  DeviceBuffer<int> d_bond_system_idxs_;

  DeviceBuffer<RealType> d_probability_buffer_;

  // Buffers used to setup the flagged partition
  DeviceBuffer<unsigned int> d_arange_;
  DeviceBuffer<unsigned int> d_nonbonded_idxs_;
  DeviceBuffer<char>
      d_flags_; // 1 indicates a free atom, 0 indicates a frozen atom
  // The first partition will be the free indices, the rest the frozen.
  DeviceBuffer<unsigned int> d_partitioned_indices_;

  DeviceBuffer<int> d_reference_idxs_;
  DeviceBuffer<unsigned int> d_free_idxs_;
  DeviceBuffer<char> d_temp_storage_buffer_;

  // The size of the indices associated with each potential.
  // For potentials that don't have indices the size will be zero,
  // for all others it will be pot->get_num_idxs() * pot->IDXS_DIM
  // Neccessary for truncating the indices of potentials such that only
  // the free-free and free-frozen interactions are computed.
  const std::vector<int> idxs_sizes_;

  DeviceBuffer<RealType>
      d_scales_buffer_; // Temporary buffer for storing Exclusion scales

  DeviceBuffer<char> d_idxs_flags_; // Flag the idxs that overlap
  DeviceBuffer<int> d_idxs_buffer_; // Where to store the original idxs
  DeviceBuffer<int> d_idxs_temp_; // Temporary buffer for setting up the subset
                                  // of idxs for bonded terms
  DeviceBuffer<int>
      d_system_idxs_buffer_;             // Where to store original system idxs
  DeviceBuffer<int> d_system_idxs_temp_; // Temporary buffer for the system idxs
                                         // of bonded terms
  DeviceBuffer<RealType> d_params_temp_; // Temporary buffer for setting up the
                                         // subset of parameters for bonds

  int *m_counter_; // mapped, zero-copy memory
  int *d_counter_; // device version

  curandGenerator_t cr_rng_;
  cudaEvent_t sync_event_; // Event to synchronize with

  void _truncate_potentials(cudaStream_t stream);

  void _setup_free_idxs_given_parittions(const RealType radius,
                                         const RealType k, cudaStream_t stream);

  void _truncate_nonbonded_ixn_group(const int *d_num_free_idxs,
                                     unsigned int *d_permutation,
                                     cudaStream_t stream);

  void _truncate_bonded_potential_idxs(
      std::shared_ptr<BoundPotential<RealType>> bp,
      std::shared_ptr<Potential<RealType>> pot,
      int *d_idxs_buffer, // Where to store the original idxs
      unsigned int *d_permutation, int *d_system_idxs_buffer,
      cudaStream_t stream);

  void _truncate_nonbonded_exclusions_potential_idxs(
      std::shared_ptr<Potential<RealType>> pot,
      int *d_idxs_buffer, // Where to store the original idxs
      unsigned int *d_permutation, int *d_system_idxs_buffer,
      cudaStream_t stream);

  void _reset_nonbonded_ixn_group(std::shared_ptr<Potential<RealType>> pot,
                                  cudaStream_t stream);

  void _reset_bonded_potential_idxs(
      std::shared_ptr<BoundPotential<RealType>> bp,
      std::shared_ptr<Potential<RealType>> pot,
      const int total_idxs, // divide by potential->IDX_DIMS
      int *d_idxs_buffer,   // Contains original idxs
      unsigned int *d_permutation, int *d_src_system_idxs, cudaStream_t stream);

  void _reset_nonbonded_exclusions_potential_idxs(
      std::shared_ptr<Potential<RealType>> pot,
      const int total_idxs, // divide by potential->IDX_DIMS
      int *d_idxs_buffer,   // Contains original idxs
      unsigned int *d_permutation, int *d_src_system_idxs, cudaStream_t stream);
};

} // namespace tmd
