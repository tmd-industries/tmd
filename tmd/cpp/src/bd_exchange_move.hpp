// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025, Forrest York
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
#include "mover.hpp"
#include "nonbonded_mol_energy.hpp"
#include "pinned_host_buffer.hpp"
#include "segmented_sumexp.hpp"
#include "segmented_weighted_random_sampler.hpp"
#include <array>
#include <vector>

namespace tmd {

/* BDExchangeMove uses biased deletion to move waters randomly in a box. The
 * reference implementation is in
 * tmd/md/exchange/exchange_mover.py::BDExchangeMove
 *
 * Terminology:
 * - num_proposals_per_move: Number of states to evaluate each time move() is
 * called
 * - batch_size: The amount parallelism within each call to move()
 */
template <typename RealType> class BDExchangeMove : public Mover<RealType> {

protected:
  // Amount of random values to generate at a time
  static const int QUATERNIONS_PER_STEP = 4;
  const int N_;
  // Number of atom in all mols
  // All molecules are currently expected to have same number of atoms
  // (typically 3 for waters) Done to avoid having to determine the size of the
  // sample and allows us to test ion sampling by having two different
  // BDExchangemoves
  const int mol_size_;
  const int num_proposals_per_move_;
  // steps_per_move_ will likely be removed once we start batching due to need
  // for rewinding
  const int steps_per_move_; // num_proposals_per_move_ / batch_size
  const int num_target_mols_;
  const RealType nb_beta_;
  const RealType beta_; // 1 / kT
  const RealType cutoff_squared_;
  const int batch_size_;
  const int num_intermediates_per_reduce_; // Number of intermediate values to
                                           // reduce mol weights
  std::vector<size_t> num_attempted_;
  NonbondedMolEnergyPotential<RealType> mol_potential_;
  SegmentedWeightedRandomSampler<RealType> sampler_;
  SegmentedSumExp<RealType> logsumexp_;
  // Buffer for evaluating moves without touching the original coords
  DeviceBuffer<RealType> d_intermediate_coords_; // [batch_size_, mol_size_, 3]
  DeviceBuffer<RealType> d_params_;              // [N, PARAMS_PER_ATOM]
  DeviceBuffer<__int128> d_before_mol_energy_buffer_; // [num_target_mols_]
  DeviceBuffer<__int128>
      d_proposal_mol_energy_buffer_; // [batch_size, num_target_mols_]
  DeviceBuffer<RealType>
      d_sample_per_atom_energy_buffer_;         // [batch_size_, mol_size_ * N]
  DeviceBuffer<int> d_atom_idxs_;               // [num_target_mols_, mol_size_]
  DeviceBuffer<int> d_mol_offsets_;             // [num_target_mols_ + 1]
  DeviceBuffer<RealType> d_log_weights_before_; // [num_target_mols_]
  DeviceBuffer<RealType>
      d_log_weights_after_; // [batch_size_, num_target_mols_]

  // Arrays used for computing logsumexp, split into max component and the sum
  // component
  DeviceBuffer<RealType> d_lse_max_before_;     // [1]
  DeviceBuffer<RealType> d_lse_exp_sum_before_; // [1]
  DeviceBuffer<RealType> d_lse_max_after_;      // [batch_size_]
  DeviceBuffer<RealType> d_lse_exp_sum_after_;  // [batch_size_]

  DeviceBuffer<int> d_samples_; // [batch_size_] The indices of the molecules to
                                // make proposals for
  DeviceBuffer<int> d_selected_sample_; // [1] The mol selected from the batch
  DeviceBuffer<RealType>
      d_quaternions_; // Normal noise for uniform random rotations
  DeviceBuffer<RealType>
      d_mh_noise_; // Noise used in the Metropolis-Hastings check
  DeviceBuffer<size_t> d_num_accepted_;    // [num_systems]
  DeviceBuffer<int> d_target_mol_atoms_;   // [batch_size_, mol_size_]
  DeviceBuffer<int> d_target_mol_offsets_; // [num_target_mols + 1]
  DeviceBuffer<__int128>
      d_intermediate_sample_weights_; // [batch_size,
                                      // num_intermediates_per_reduce_]
  DeviceBuffer<RealType>
      d_sample_noise_; // Noise to use for selecting molecules
  DeviceBuffer<RealType>
      d_sampling_intermediate_; // [batch_size_, num_target_mols_] Intermediate
                                // buffer for weighted sampling
  DeviceBuffer<RealType>
      d_translations_; // Uniform noise for translation + the check
  DeviceBuffer<int> d_sample_segments_offsets_; // Segment offsets for the
                                                // sampler // [batch_size + 1]
  DeviceBuffer<int> d_noise_offset_;            // [1]  Offset into noise

  PinnedHostBuffer<int> p_noise_offset_; // [1]

  // If the RNGs are changed, make sure to modify the seeding of
  // TIBDExchangeMove translations RNG
  curandGenerator_t cr_rng_quat_;         // Generate noise for quaternions
  curandGenerator_t cr_rng_translations_; // Generate noise for translations
  curandGenerator_t cr_rng_samples_;      // Generate noise for selecting waters
  curandGenerator_t cr_rng_mh_; // Generate noise for Metropolis-Hastings

  void compute_initial_log_weights_device(const int N, const RealType *d_coords,
                                          const RealType *d_box,
                                          const RealType *d_params,
                                          cudaStream_t stream);

  BDExchangeMove(const int num_system, const int N,
                 const std::vector<std::vector<int>> &target_mols,
                 const std::vector<RealType> &params,
                 const RealType temperature, const RealType nb_beta,
                 const RealType cutoff, const int seed,
                 const int num_proposals_per_move, const int interval,
                 const int batch_size, const int translation_buffer_size);

public:
  BDExchangeMove(const int num_system, const int N,
                 const std::vector<std::vector<int>> &target_mols,
                 const std::vector<RealType> &params,
                 const RealType temperature, const RealType nb_beta,
                 const RealType cutoff, const int seed,
                 const int num_proposals_per_move, const int interval,
                 const int batch_size);

  void compute_incremental_log_weights_device(const int N, const bool scale,
                                              const RealType *d_box,
                                              const RealType *d_coords,
                                              const RealType *d_params,
                                              const RealType *d_quaternions,
                                              const RealType *d_translations,
                                              cudaStream_t stream);

  // compute_incremental_log_weights_host is used for testing the computation of
  // incremental weights with different batch sizes. Note that the translations
  // provided are used as is and are not scaled by the box extents.
  std::vector<std::vector<RealType>> compute_incremental_log_weights_host(
      const int N, const RealType *h_coords, const RealType *h_box,
      const int *mol_idxs, const RealType *h_quaternions,
      const RealType *h_translations);

  std::vector<RealType>
  compute_initial_log_weights_host(const int N, const RealType *h_coords,
                                   const RealType *h_box);

  // get_after_weights returns the per molecule weight before each proposal, may
  // come from either `compute_intial_weights_device` or from
  // `compute_incremental_log_weights_device`.
  std::vector<RealType> get_before_log_weights();

  // get_after_weights returns the per molecule weight after being computed
  // incrementally
  std::vector<RealType> get_after_log_weights();

  ~BDExchangeMove();

  virtual void move(const int num_systems, const int N,
                    RealType *d_coords, // [num_systems, N, 3]
                    RealType *d_box,    // [num_systems, 3, 3]
                    cudaStream_t stream) override;

  virtual RealType log_probability_host();
  virtual RealType raw_log_probability_host();

  size_t num_systems() const { return this->num_systems_; }

  size_t batch_size() const { return this->batch_size_; }

  std::vector<size_t> n_proposed() const { return num_attempted_; }

  std::vector<size_t> n_accepted() const;

  std::vector<RealType> get_params();

  void set_params(const std::vector<RealType> &params);

  void set_params_device(const int size, const RealType *d_p,
                         const cudaStream_t stream);

  std::vector<RealType> acceptance_fraction() const {
    std::vector<RealType> ratio(this->num_systems_);
    std::vector<size_t> accepted = this->n_accepted();
    std::vector<size_t> proposed = this->n_proposed();

    for (auto i = 0; i < this->num_systems_; i++) {
      ratio[i] = static_cast<RealType>(accepted[i]) /
                 static_cast<RealType>(proposed[i]);
    }
    return ratio;
  }
};

} // namespace tmd
