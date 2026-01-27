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

#include "constants.hpp"
#include "fanout_summed_potential.hpp"
#include "gpu_utils.cuh"
#include "harmonic_angle.hpp"
#include "harmonic_bond.hpp"
#include "kernel_utils.cuh"
#include "kernels/k_flat_bottom_bond.cuh"
#include "kernels/k_indices.cuh"
#include "kernels/k_local_md.cuh"
#include "local_md_potentials.hpp"
#include "math_utils.cuh"
#include "nonbonded_common.hpp"
#include "nonbonded_pair_list.hpp"
#include "periodic_torsion.hpp"
#include "potential.hpp"
#include "summed_potential.hpp"
#include <algorithm>
#include <cub/cub.cuh>
#include <random>
#include <vector>

#include <cub/config.cuh>
#include <cub/util_type.cuh>

namespace tmd {

template <typename RealType>
static std::shared_ptr<Potential<RealType>> get_and_verify_nonbonded_pots(
    const std::vector<std::shared_ptr<Potential<RealType>>> &nonbonded_pots,
    const int num_systems, const int N) {
  if (nonbonded_pots.size() != 1) {
    throw std::runtime_error(
        "must have exactly one NonbondedInteractionGroup potential");
  }
  auto nonbonded_pot = nonbonded_pots[0];
  verify_nonbonded_potential_for_local_md(nonbonded_pot, num_systems, N);
  return nonbonded_pot;
}

/**
 * # LocalMDPotentials
 *
 * This utility class modifies a set of input potentials (expected to define
 *interactions for an entire system) to run MD on a local region of a system.
 *
 * ## Expectations
 *
 * - A single `NonbondedInteractionGroup` potential with row and column indices
 *set to `np.arange(num_atoms)`.
 * - A `NonbondedPairList` potential that defines the exclusions.
 * - Optionally, a `NonbondedPrecomputedPairlist`.
 * - A set of `Bonded` potentials.
 *
 * ## Terminology
 *
 * - **Reference particle:** A particle chosen as the reference for local MD.
 *This particle typically resides at the center of the free region and will be
 *used to define restraints. When using `multiple_steps_local` the reference
 *particle will also be used to define the free region.
 * - **Free region:** Particles to be moved by an integrator. During local MD,
 *the `du_dx` values of free atoms are expected to be identical to those in
 *global MD. Free atoms are stored in the `d_free_idxs_` buffer, where an atom
 *is free if `d_free_idxs_[idx] == idx` and `d_free_idxs[idx] == num_atoms` when
 *frozen.
 * - **Frozen region:** Particles not moved by an integrator. Computation of
 *`du_dx` is unused and invalid during local MD.
 * - **Freeze reference:** If the reference particle is frozen, it's excluded
 *from `d_free_idxs_`; otherwise, it's in the free region and restrained.
 *
 * ## Input Potential Modification
 *
 * The following details how incoming potentials are modified when running for
 *Local MD.
 *
 * ### `NonbondedInteractionGroup`
 *
 * Once free or frozen atoms are determined, this class updates the row and
 *column indices of the interaction group. The row indices can be any superset
 *of the free atoms; a smaller superset generally performs better. In practice,
 ***the row indices are set to the free atoms plus the reference index**,
 *regardless of whether the reference is free. The indices determined be in the
 *row indices are stored in `d_flags_`, which may differ from `d_free_idxs_` for
 *the reference particle. This choice ensures consistency of row indices between
 *freezing and restraining the reference and reduces the number of interacting
 *nblist tiles in the common case (reference is typically at the center of the
 *free region). The reference particle may not be at the center of the free
 *region when the users provides their own selection via `setup_from_selection`,
 *the validity of the selection is left to the user as the radius and restraints
 * still apply.
 *
 * The column indices must be prefixed with the row indices followed by all
 *other atoms in the system. In practice we combine the row and column indices
 * into d_partitioned_indices_ with the first partition being the row indices
 *and the second partition the column indices. The function
 *cub::DevicePartition::Flagged is used to construct the partitions of row and
 *columns, placing the row atoms in ascending order while the column atoms are
 *in descending order.
 *
 * ### Bonded Terms (`HarmonicBond`, `HarmonicAngle`, etc.)
 *
 * Adjusted such that only the free-free and free-frozen interactions are
 *computed.
 *
 * ### `NonbondedPairList<RealType, true>`
 *
 * Adjusted such that only the free-free and free-frozen interactions are
 *computed.
 *
 * ### `NonbondedPrecomputedPairlist`
 *
 * Unadjusted. In practice, the potential is only over the ligand, so the
 *benefit of truncating off frozen atoms is minimal.
 *
 * ## Local MD Specific Potentials
 *
 * When running local MD, this class sets up one or two potentials depending on
 *the configuration.
 *
 * ### FlatBottomBond
 *
 * To ensure consistent density in the free region during local MD, a
 *flat-bottom restraint is constructed between the free region atoms and the
 *reference particle. The free particles have a zero potential energy when the
 *distance to the reference is below the defined radius, the restraint engages
 *outside of the radius to restrain the particles to within some radius of the
 *reference particle.
 *
 * ### LogFlatBottomBond
 *
 * When the reference particle is free to move, a restraint is applied between
 *all frozen particles and the reference. This ensures the reference particle
 *cannot move too far from its original location. In the future, consider
 *pruning the number of restraints.
 **/
template <typename RealType>
LocalMDPotentials<RealType>::LocalMDPotentials(
    const int num_systems, const int N,
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    const std::vector<std::shared_ptr<Potential<RealType>>> &nonbonded_pots,
    const bool freeze_reference, const RealType temperature,
    const RealType nblist_padding)
    : freeze_reference(freeze_reference), temperature(temperature),
      nblist_padding(nblist_padding), num_systems_(num_systems), N_(N),
      temp_storage_bytes_(0), bps_(bps), local_md_potentials_(bps_),
      nonbonded_pot_(
          get_and_verify_nonbonded_pots(nonbonded_pots, num_systems, N)),
      initial_nblist_padding_(
          get_nonbonded_ixn_potential_nblist_padding(nonbonded_pot_)),
      d_restraint_pairs_(num_systems_ * N_ * 2),
      d_bond_params_(num_systems_ * N_ * 3),
      d_bond_system_idxs_(num_systems_ * N_),
      d_probability_buffer_(num_systems_ * N_), d_arange_(0),
      d_nonbonded_idxs_(num_systems_ * N_), d_flags_(num_systems_ * N_),
      d_partitioned_indices_(0), d_reference_idxs_(num_systems_),
      d_free_idxs_(num_systems_ * N_), d_temp_storage_buffer_(0),
      idxs_sizes_(get_indices_buffer_sizes_from_bps(bps_)),
      d_scales_buffer_(get_scales_buffer_length_from_bps(bps_)),
      d_idxs_flags_(0), d_idxs_buffer_(0), d_idxs_temp_(0),
      d_system_idxs_buffer_(0), d_system_idxs_temp_(0), d_params_temp_(0) {

  if (temperature <= static_cast<RealType>(0.0)) {
    throw std::runtime_error("temperature must be greater than 0");
  }

  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  int max_required_idxs_temp_buffer =
      *std::max_element(idxs_sizes_.begin(), idxs_sizes_.end());
  size_t max_required_param_temp_buffer = 0;
  for (auto bp : bps_) {
    max_required_param_temp_buffer =
        max(max_required_param_temp_buffer, bp->d_p.length);
  }
  // This buffer is used both for d_scales and d_params, so must be large enough
  // to handle both cases This works because the size of the indices of the
  // nonbonded exclusions is the same size as the scales.
  d_params_temp_.realloc(
      max(max_required_param_temp_buffer,
          static_cast<size_t>(max_required_idxs_temp_buffer)));
  d_idxs_temp_.realloc(max_required_idxs_temp_buffer);

  d_idxs_flags_.realloc(max_required_idxs_temp_buffer);
  d_idxs_buffer_.realloc(
      std::accumulate(idxs_sizes_.begin(), idxs_sizes_.end(), 0));
  // TBD: Reduce the size of the system idxs buffers, but memory is cheap for
  // now Larger than they need to be because it includes idxs dims. There are
  // the same number of system idxs as there are idxs.
  d_system_idxs_temp_.realloc(max_required_idxs_temp_buffer);
  // d_arange is used both for idxs and for free, which is why it must be the
  // max of N_ or max number of idxs.
  d_arange_.realloc(max(max_required_idxs_temp_buffer, num_systems_ * N_));
  // Allow enough partition indices for us to store a permutation for each
  // potential
  d_partitioned_indices_.realloc(
      max(max_required_idxs_temp_buffer, num_systems_ * N_) *
      idxs_sizes_.size());
  d_system_idxs_buffer_.realloc(d_partitioned_indices_.length);

  // arange buffer that is constant throughout the lifetime of this class
  k_arange<<<ceil_divide(d_arange_.length, tpb), tpb>>>(d_arange_.length,
                                                        d_arange_.data);
  gpuErrchk(cudaPeekAtLastError());

  k_segment_arange<unsigned int>
      <<<dim3(ceil_divide(N_, tpb), num_systems_), tpb>>>(
          num_systems_, N_, d_nonbonded_idxs_.data);
  gpuErrchk(cudaPeekAtLastError());

  // Ensure that we allocate enough space for all potential bonds
  // default_bonds[i * 2 + 0] != default_bonds[i * 2 + 1], so set first value to
  // 0, second to i + 1
  std::vector<int> default_bonds(num_systems_ * N_ * 2);
  for (int i = 0; i < num_systems_ * N_; i++) {
    default_bonds[i * 2 + 0] = 0;
    default_bonds[i * 2 + 1] = i + 1;
  }
  std::vector<RealType> default_params(num_systems_ * N_ * 3);
  std::vector<int> system_idxs(num_systems_ * N_, 0);
  free_restraint_ =
      std::shared_ptr<FlatBottomBond<RealType>>(new FlatBottomBond<RealType>(
          num_systems_, N, default_bonds, system_idxs));
  // Construct a bound potential with 0 params
  bound_free_restraint_ = std::shared_ptr<BoundPotential<RealType>>(
      new BoundPotential<RealType>(free_restraint_, default_params, 3));

  if (!freeze_reference) {
    frozen_restraint_ = std::shared_ptr<LogFlatBottomBond<RealType>>(
        new LogFlatBottomBond<RealType>(
            num_systems_, N, default_bonds, system_idxs,
            1 / (temperature * static_cast<RealType>(BOLTZ))));
    bound_frozen_restraint_ = std::shared_ptr<BoundPotential<RealType>>(
        new BoundPotential<RealType>(frozen_restraint_, default_params, 3));
    // Push restraint to the front of the potentials for performance
    local_md_potentials_.insert(local_md_potentials_.begin(),
                                bound_frozen_restraint_);
  }
  // TBD: Investigate impact of potentials being pushed to the front when
  // NonbondedInteractionGroup has no sync Push restraint to the front of the
  // potentials for performance
  local_md_potentials_.insert(local_md_potentials_.begin(),
                              bound_free_restraint_);

  gpuErrchk(cudaHostAlloc(&m_counter_, num_systems_ * sizeof(*m_counter_),
                          cudaHostAllocMapped));
  for (int i = 0; i < num_systems_; i++) {
    m_counter_[i] = 0;
  }

  gpuErrchk(cudaHostGetDevicePointer(&d_counter_, m_counter_, 0));

  size_t max_free_flags_storage = 0;
  gpuErrchk(cub::DevicePartition::Flagged(
      nullptr, max_free_flags_storage, d_arange_.data, d_flags_.data,
      d_partitioned_indices_.data, d_counter_, N_));

  size_t max_idx_flags_storage = 0;
  gpuErrchk(cub::DevicePartition::Flagged(
      nullptr, max_idx_flags_storage, d_arange_.data, d_idxs_flags_.data,
      d_partitioned_indices_.data, d_counter_, d_idxs_flags_.length));

  temp_storage_bytes_ = max(max_free_flags_storage, max_idx_flags_storage);

  d_temp_storage_buffer_.realloc(temp_storage_bytes_);

  curandErrchk(curandCreateGenerator(&cr_rng_, CURAND_RNG_PSEUDO_DEFAULT));

  // Create event with timings disabled as timings slow down events
  gpuErrchk(cudaEventCreateWithFlags(&sync_event_, cudaEventDisableTiming));
};

template <typename RealType> LocalMDPotentials<RealType>::~LocalMDPotentials() {
  gpuErrchk(cudaFreeHost(m_counter_));

  gpuErrchk(cudaEventDestroy(sync_event_));
  curandErrchk(curandDestroyGenerator(cr_rng_));
}

// setup_from_idxs takes a set of idxs and a seed to determine the free
// particles. Fix the local_idxs to length one to ensure the same reference
// every time, though the seed also handles the probabilities of selecting
// particles, and it is suggested to provide a new seed at each step.
template <typename RealType>
void LocalMDPotentials<RealType>::setup_from_idxs(
    RealType *d_x_t, RealType *d_box_t, const std::vector<int> &local_idxs,
    const int seed, const RealType radius, const RealType k,
    cudaStream_t stream) {
  std::mt19937 rng;
  rng.seed(seed);
  std::uniform_int_distribution<unsigned int> random_dist(0, local_idxs.size() -
                                                                 1);

  // TBD: Avoid implied sync
  std::vector<int> reference_idxs(num_systems_);
  for (int i = 0; i < num_systems_; i++) {
    reference_idxs[i] = local_idxs[random_dist(rng)];
  }
  d_reference_idxs_.copy_from(&reference_idxs[0]);

  curandErrchk(curandSetStream(cr_rng_, stream));
  curandErrchk(curandSetPseudoRandomGeneratorSeed(cr_rng_, seed));
  // Reset the generator offset to ensure same values for the same seed are
  // produced Simply reseeding does NOT produce identical results
  curandErrchk(curandSetGeneratorOffset(cr_rng_, 0));

  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  // TBD: Try using local RNGs instead of batching like this
  // Generate values between (0, 1.0]
  curandErrchk(templateCurandUniform(cr_rng_, d_probability_buffer_.data,
                                     d_probability_buffer_.length));

  const RealType kBT = static_cast<RealType>(BOLTZ) * temperature;
  // flag all of the particles that will be in the row atoms, includes the
  // reference.
  k_log_probability_flag<RealType>
      <<<dim3(ceil_divide(N_, tpb), num_systems_), tpb, 0, stream>>>(
          num_systems_, N_, kBT, radius, k, d_reference_idxs_.data, d_x_t,
          d_box_t, d_probability_buffer_.data, d_flags_.data);
  gpuErrchk(cudaPeekAtLastError());

  this->_setup_free_idxs_given_parittions(radius, k, stream);
}

// setup_from_selection takes a set of idxs, flat-bottom restraint parameters
// (radius, k) assumes selection_idxs are sampled based on exp(-beta
// U_flat_bottom(distance_to_reference, radius, k)) (or that the user is
// otherwise accounting for selection probabilities) The selection indices
// shouldn't contain the reference idx, as upstream python verifies that
// beforehand, though nothing should break if the reference particle is
// included.
template <typename RealType>
void LocalMDPotentials<RealType>::setup_from_selection(
    const std::vector<int> &reference_idxs,
    const std::vector<int> &selection_idxs, const RealType radius,
    const RealType k, const cudaStream_t stream) {

  if (reference_idxs.size() != 1) {
    throw std::runtime_error(
        "Setup from selection only supported for single system simulations");
  }
  if (reference_idxs.size() != num_systems_) {
    throw std::runtime_error(
        "Number of references must match number of systems");
  }
  for (int i = 0; i < num_systems_; i++) {
    if (reference_idxs[i] < 0 || reference_idxs[i] >= N_) {
      throw std::runtime_error("Reference indices must be [0, N), got " +
                               std::to_string(reference_idxs[i]) +
                               " at index " + std::to_string(i));
    }
  }
  // TBD: Need to handle this more carefully. Since this is rarely used,
  // inclined to say the selection is constant across hosts for simplicity
  gpuErrchk(cudaMemcpyAsync(d_free_idxs_.data, &selection_idxs[0],
                            selection_idxs.size() * sizeof(*d_free_idxs_.data),
                            cudaMemcpyHostToDevice, stream));

  // Zero out all of the flags, indicating all particles are frozen
  gpuErrchk(cudaMemsetAsync(d_flags_.data, 0, d_flags_.size(), stream));
  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  d_reference_idxs_.copy_from(&reference_idxs[0]);

  // For each provided atom index, set d_flags_[idx] = 1 to indicate the
  // particles should be free and thus in the row indices.
  k_flag_free<<<ceil_divide(selection_idxs.size(), tpb), tpb, 0, stream>>>(
      N_, selection_idxs.size(), d_free_idxs_.data, d_flags_.data);
  gpuErrchk(cudaPeekAtLastError());
  // Push the reference into the flags, sets it up in the row indices of the ixn
  // group for performance
  k_update_index<char>
      <<<1, 1, 0, stream>>>(d_flags_.data, reference_idxs[0], 1);
  gpuErrchk(cudaPeekAtLastError());

  this->_setup_free_idxs_given_parittions(radius, k, stream);
}

template <typename RealType>
void LocalMDPotentials<RealType>::_setup_free_idxs_given_parittions(
    const RealType radius, const RealType k, cudaStream_t stream) {
  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  for (int i = 0; i < num_systems_; i++) {
    // Partition the flagged indices, separating the free and frozen indices.
    // Each system will be partitioned separately
    gpuErrchk(cub::DevicePartition::Flagged(
        d_temp_storage_buffer_.data, temp_storage_bytes_, d_arange_.data,
        d_flags_.data + i * N_, d_partitioned_indices_.data + i * N_,
        d_counter_ + i, N_, stream));
  }
  gpuErrchk(cudaEventRecord(sync_event_, stream));

  k_initialize_array<unsigned int>
      <<<ceil_divide(num_systems_ * N_, tpb), tpb, 0, stream>>>(
          num_systems_ * N_, d_free_idxs_.data, N_);
  gpuErrchk(cudaPeekAtLastError());

  // TBD: Try reducing the size of N.
  if (freeze_reference) {
    k_setup_free_indices_from_partitions<true>
        <<<dim3(ceil_divide(N_, tpb), num_systems_), tpb, 0, stream>>>(
            num_systems_, N_, d_counter_, d_reference_idxs_.data,
            d_partitioned_indices_.data, d_free_idxs_.data);
  } else {
    k_setup_free_indices_from_partitions<false>
        <<<dim3(ceil_divide(N_, tpb), num_systems_), tpb, 0, stream>>>(
            num_systems_, N_, d_counter_, d_reference_idxs_.data,
            d_partitioned_indices_.data, d_free_idxs_.data);
  }
  gpuErrchk(cudaPeekAtLastError());

  k_construct_bonded_params_and_system_idxs<RealType, false>
      <<<dim3(ceil_divide(N_, tpb), num_systems_), tpb, 0, stream>>>(
          num_systems_, N_, d_counter_, d_reference_idxs_.data, k, 0.0, radius,
          d_partitioned_indices_.data, d_restraint_pairs_.data,
          d_bond_params_.data, d_bond_system_idxs_.data);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaEventSynchronize(sync_event_));

  int total_free = 0;
  int total_frozen = 0;

  for (int i = 0; i < num_systems_; i++) {
    const int num_row_idxs = m_counter_[i];
    // If only the reference particle is in the free region, we haven't selected
    // anything. Put it at the end, so we can correctly reset the potentials
    if (num_row_idxs == N_) {
      fprintf(
          stderr,
          "LocalMDPotentials setup has entire system selected in system \n");
    } else if (num_row_idxs == 1) {
      throw std::runtime_error(
          "LocalMDPotentials setup has no free particles selected in system " +
          std::to_string(i));
    }
    total_free += num_row_idxs;
    total_frozen += N_ - num_row_idxs;
  }

  // Setup the flat bottom restraints
  bound_free_restraint_->set_params_device(bound_free_restraint_->params_dim *
                                               total_free,
                                           d_bond_params_.data, stream);
  free_restraint_->set_bonds_device(total_free, d_restraint_pairs_.data,
                                    stream);
  free_restraint_->set_system_idxs_device(total_free, d_bond_system_idxs_.data,
                                          stream);

  if (!freeze_reference) {
    // Only update the log flat bottom's restraint parameters if there are
    // frozen atoms
    if (total_frozen > 0) {
      k_construct_bonded_params_and_system_idxs<RealType, true>
          <<<dim3(ceil_divide(N_, tpb), num_systems_), tpb, 0, stream>>>(
              num_systems_, N_, d_counter_, d_reference_idxs_.data, k, 0.0,
              radius, d_partitioned_indices_.data, d_restraint_pairs_.data,
              d_bond_params_.data, d_bond_system_idxs_.data);
      gpuErrchk(cudaPeekAtLastError());
    }

    bound_frozen_restraint_->set_params_device(
        bound_frozen_restraint_->params_dim * total_frozen, d_bond_params_.data,
        stream);
    frozen_restraint_->set_bonds_device(total_frozen, d_restraint_pairs_.data,
                                        stream);
    frozen_restraint_->set_system_idxs_device(total_frozen,
                                              d_bond_system_idxs_.data, stream);
  }
  this->_truncate_potentials(stream);
}

template <typename RealType>
void LocalMDPotentials<RealType>::_truncate_potentials(cudaStream_t stream) {

  this->_truncate_nonbonded_ixn_group(m_counter_, d_partitioned_indices_.data,
                                      stream);

  int i = 0;
  int idxs_offset = 0;
  int partition_offset = 0;
  for (auto bp : bps_) {
    if (std::shared_ptr<FanoutSummedPotential<RealType>> fanned_potential =
            std::dynamic_pointer_cast<FanoutSummedPotential<RealType>>(
                bp->potential);
        fanned_potential != nullptr) {
      for (auto pot : fanned_potential->get_potentials()) {
        if (is_exclusions_nonbonded_all_pairs_potential(pot)) {
          this->_truncate_nonbonded_exclusions_potential_idxs(
              pot, d_idxs_buffer_.data + idxs_offset,
              d_partitioned_indices_.data + partition_offset,
              d_system_idxs_buffer_.data + idxs_offset, stream);
        } else {
          this->_truncate_bonded_potential_idxs(
              bp, pot, d_idxs_buffer_.data + idxs_offset,
              d_partitioned_indices_.data + partition_offset,
              d_system_idxs_buffer_.data + idxs_offset, stream);
        }
        partition_offset += d_arange_.length;
        idxs_offset += idxs_sizes_[i];
        i++;
      }
    } else {
      if (is_exclusions_nonbonded_all_pairs_potential(bp->potential)) {
        this->_truncate_nonbonded_exclusions_potential_idxs(
            bp->potential, d_idxs_buffer_.data + idxs_offset,
            d_partitioned_indices_.data + partition_offset,
            d_system_idxs_buffer_.data + idxs_offset, stream);
      } else {
        this->_truncate_bonded_potential_idxs(
            bp, bp->potential, d_idxs_buffer_.data + idxs_offset,
            d_partitioned_indices_.data + partition_offset,
            d_system_idxs_buffer_.data + idxs_offset, stream);
      }
      partition_offset += d_arange_.length;
      idxs_offset += idxs_sizes_[i];
      i++;
    }
  }
}

template <typename RealType>
std::vector<std::shared_ptr<BoundPotential<RealType>>>
LocalMDPotentials<RealType>::get_potentials() {
  return local_md_potentials_;
}

template <typename RealType>
unsigned int *LocalMDPotentials<RealType>::get_free_idxs() {
  return d_free_idxs_.data;
}

/** _truncate_bonded_potential_idxs modifies the bonded potentials so that when
 * calling `execute_device` only the free-free and free-frozen interactions are
 * computed. This is done in the following steps::
 * - Determine the number of indices and the associated dimensions of the
 * indices (e.g. HarmonicBond will have K indices and each set of indices will
 * have a dimension of 2)
 * - Flag the indices that compute interactions that contain a free atom
 * (d_free_idxs_.data)
 * - Create a permutation of the indices that orders the flagged indices such
 * that the computed indices come first
 * - Create a copy of the starting indices so that the potential idxs can be
 * reset later
 * - Permute the starting indices and parameters into temporary buffers, the
 * parameters and indices need to be similarly ordered.
 * - Update the potential's bound potential in its entirety, will later
 * unpermute the parameters to reset the bound potential. Set number of
 * parameters to number of interactions to compute.
 * - Set the potential's indices to only compute the free-free and free-frozen
 * interactions.
 */
template <typename RealType>
void LocalMDPotentials<RealType>::_truncate_bonded_potential_idxs(
    std::shared_ptr<BoundPotential<RealType>> bp,
    std::shared_ptr<Potential<RealType>> pot,
    int *d_idxs_buffer, // Where to store the original idxs
    unsigned int *d_permutation, int *d_system_idxs_buffer,
    cudaStream_t stream) {
  if (!is_truncatable_potential(pot)) {
    return;
  }

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const int params_dim = bp->params_dim;
  int num_idxs = 0;
  int idxs_dim = 0;
  int *d_src_idxs = nullptr;
  int *d_system_idxs = nullptr;
  if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
          std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
      trunc_pot != nullptr) {
    num_idxs = trunc_pot->get_num_idxs();
    idxs_dim = trunc_pot->IDXS_DIM;
    d_src_idxs = trunc_pot->get_idxs_device();
    d_system_idxs = trunc_pot->get_system_idxs_device();
  } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
             trunc_pot != nullptr) {
    num_idxs = trunc_pot->get_num_idxs();
    idxs_dim = trunc_pot->IDXS_DIM;
    d_src_idxs = trunc_pot->get_idxs_device();
    d_system_idxs = trunc_pot->get_system_idxs_device();
  } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(pot);
             trunc_pot != nullptr) {
    num_idxs = trunc_pot->get_num_idxs();
    idxs_dim = trunc_pot->IDXS_DIM;
    d_src_idxs = trunc_pot->get_idxs_device();
    d_system_idxs = trunc_pot->get_system_idxs_device();
  } else {
    throw std::runtime_error(
        "_truncate_bonded_potential_idxs::Unexpected truncatable type");
  }
  // If the potential is empty, do nothing
  if (num_idxs == 0) {
    return;
  }
  assert(idxs_dim > 0 && d_src_idxs != nullptr);

  const int blocks = ceil_divide(num_idxs, tpb);
  // Assume that the full bound potential params are being used
  assert(bp->d_p.length / params_dim == num_idxs);
  assert(bp->size == bp->d_p.length);

  k_flag_indices_to_keep<<<blocks, tpb, 0, stream>>>(
      num_idxs, idxs_dim, N_, d_src_idxs, d_free_idxs_.data, d_system_idxs,
      d_idxs_flags_.data);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cub::DevicePartition::Flagged(
      d_temp_storage_buffer_.data, temp_storage_bytes_, d_arange_.data,
      d_idxs_flags_.data, d_permutation, d_counter_, num_idxs, stream));
  // Record so we know when we can check d_counter_ for the number of
  // overlapping indices
  gpuErrchk(cudaEventRecord(sync_event_, stream));

  gpuErrchk(cudaMemcpyAsync(d_idxs_buffer, d_src_idxs,
                            num_idxs * idxs_dim * sizeof(*d_idxs_buffer),
                            cudaMemcpyDeviceToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_system_idxs_buffer, d_system_idxs,
                            num_idxs * sizeof(*d_system_idxs_buffer),
                            cudaMemcpyDeviceToDevice, stream));

  // Permute indices such that the indices still computed are stored at the
  // start of d_idxs_temp_.data
  k_permute_chunks<int, true><<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
      num_idxs, idxs_dim, d_permutation, d_idxs_buffer, d_idxs_temp_.data);
  gpuErrchk(cudaPeekAtLastError());

  k_permute_chunks<int, true><<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
      num_idxs, 1, d_permutation, d_system_idxs_buffer,
      d_system_idxs_temp_.data);
  gpuErrchk(cudaPeekAtLastError());

  // Permute parameters
  k_permute_chunks<RealType, true>
      <<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
          num_idxs, params_dim, d_permutation, bp->d_p.data,
          d_params_temp_.data);
  gpuErrchk(cudaPeekAtLastError());

  // Copy the permuted parameters to the bound potential
  gpuErrchk(cudaMemcpyAsync(bp->d_p.data, d_params_temp_.data, bp->d_p.size(),
                            cudaMemcpyDeviceToDevice, stream));

  // Synchronize to ensure that the number of indices left to compute is stored
  // in *m_counter_
  gpuErrchk(cudaEventSynchronize(sync_event_));
  bp->size = *m_counter_ * params_dim;

  if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
          std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
      trunc_pot != nullptr) {
    trunc_pot->set_idxs_device(*m_counter_, d_idxs_temp_.data, stream);
    trunc_pot->set_system_idxs_device(*m_counter_, d_system_idxs_temp_.data,
                                      stream);
  } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
             trunc_pot != nullptr) {
    trunc_pot->set_idxs_device(*m_counter_, d_idxs_temp_.data, stream);
    trunc_pot->set_system_idxs_device(*m_counter_, d_system_idxs_temp_.data,
                                      stream);
  } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(pot);
             trunc_pot != nullptr) {
    trunc_pot->set_idxs_device(*m_counter_, d_idxs_temp_.data, stream);
    trunc_pot->set_system_idxs_device(*m_counter_, d_system_idxs_temp_.data,
                                      stream);
  } else {
    throw std::runtime_error(
        "_truncate_bonded_potential_idxs::Unexpected truncatable type");
  }
}

/** _truncate_nonbonded_ixn_group modifies the NonbondedInteractionGroup
 * potential so that when calling `execute_device` only the free-free and
 * free-frozen interactions are computed. This is done by setting the row
 * indices to the free atoms (which always includes the reference) and the
 * column indices to the free atoms followed by the frozen. Note that this
 * function is reliant on the permutation set up in
 * _setup_free_idxs_given_parittions. TBD: Clean so the code is more
 * flexible and not tied to _setup_free_idxs_given_parittions
 */
template <typename RealType>
void LocalMDPotentials<RealType>::_truncate_nonbonded_ixn_group(
    const int *num_free_idxs,
    unsigned int *d_partitioned_idxs, // Partitioned in num_free_idxs at the
                                      // front and the frozen afterwards
    cudaStream_t stream) {
  set_nonbonded_ixn_potential_nblist_padding(nonbonded_pot_, nblist_padding);
  // Permutation will have the free indices (plus reference) at the front and
  // frozen indices at the back
  std::vector<int> free_vect(num_free_idxs, num_free_idxs + num_systems_);
  set_nonbonded_ixn_potential_idxs(
      nonbonded_pot_, free_vect, std::vector<int>(num_systems_, N_),
      d_partitioned_idxs, d_partitioned_idxs, stream);
}

template <typename RealType>
void LocalMDPotentials<RealType>::_reset_nonbonded_ixn_group(
    std::shared_ptr<Potential<RealType>> pot, cudaStream_t stream) {
  auto idxs_counts = std::vector<int>(num_systems_, N_);
  set_nonbonded_ixn_potential_idxs(pot, idxs_counts, idxs_counts,
                                   d_nonbonded_idxs_.data,
                                   d_nonbonded_idxs_.data, stream);
  set_nonbonded_ixn_potential_nblist_padding(pot, initial_nblist_padding_);
}

/** _truncate_nonbonded_exclusions_potential_idxs modifies the
 * NonbondedPairList<RealType, true> potential so that when calling
 * `execute_device` only the free-free and free-frozen interactions are
 * computed. This is done in the following steps::
 * - Determine the number of indices and the associated dimensions of the
 * indices
 * - Flag the indices that compute interactions that contain a free atom
 * (d_free_idxs_.data)
 * - Create a permutation of the indices that orders the flagged indices such
 * that the computed indices come first
 * - Create a copy of the starting indices and scales so that the potential idxs
 * can be reset later
 * - Permute the starting indices and scales into temporary buffers, the scales
 * and indices need to be similarly ordered.
 * - Set the potential's indices and scales to only compute the free-free and
 * free-frozen interactions.
 */
template <typename RealType>
void LocalMDPotentials<RealType>::_truncate_nonbonded_exclusions_potential_idxs(
    std::shared_ptr<Potential<RealType>> pot, int *d_idxs_buffer,
    unsigned int *d_permutation, int *d_system_idxs_buffer,
    cudaStream_t stream) {
  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  std::shared_ptr<NonbondedPairList<RealType, true>> nb_pot =
      std::dynamic_pointer_cast<NonbondedPairList<RealType, true>>(pot);
  if (!nb_pot) {
    throw std::runtime_error("_truncate_nonbonded_exclusions_potential_idxs:: "
                             "Unexpectedly couldn't convert to "
                             "Negated NonbondedPairList");
  }
  RealType *d_scales_ptr = nb_pot->get_scales_device();
  int *src_idxs_ptr = nb_pot->get_idxs_device();
  int *d_system_idxs = nb_pot->get_system_idxs_device();
  const int num_idxs = nb_pot->get_num_idxs();
  const int idxs_dim = nb_pot->IDXS_DIM;

  const int blocks = ceil_divide(num_idxs, tpb);

  k_flag_indices_to_keep<<<blocks, tpb, 0, stream>>>(
      num_idxs, idxs_dim, N_, src_idxs_ptr, d_free_idxs_.data, d_system_idxs,
      d_idxs_flags_.data);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cub::DevicePartition::Flagged(
      d_temp_storage_buffer_.data, temp_storage_bytes_, d_arange_.data,
      d_idxs_flags_.data, d_permutation, d_counter_, num_idxs, stream));
  // Record so we know when we can check d_counter_ for the number of
  // overlapping indices
  gpuErrchk(cudaEventRecord(sync_event_, stream));

  // Store the original values in buffers
  gpuErrchk(cudaMemcpyAsync(d_idxs_buffer, src_idxs_ptr,
                            num_idxs * idxs_dim * sizeof(*d_idxs_buffer),
                            cudaMemcpyDeviceToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_system_idxs_buffer, d_system_idxs,
                            num_idxs * sizeof(*d_system_idxs_buffer),
                            cudaMemcpyDeviceToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_scales_buffer_.data, d_scales_ptr,
                            num_idxs * idxs_dim * sizeof(*d_scales_ptr),
                            cudaMemcpyDeviceToDevice, stream));

  k_permute_chunks<int, true><<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
      num_idxs, idxs_dim, d_permutation, d_idxs_buffer, d_idxs_temp_.data);
  gpuErrchk(cudaPeekAtLastError());
  k_permute_chunks<int, true><<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
      num_idxs, 1, d_permutation, d_system_idxs_buffer,
      d_system_idxs_temp_.data);
  gpuErrchk(cudaPeekAtLastError());

  // Permute scales
  k_permute_chunks<RealType, true>
      <<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
          num_idxs, idxs_dim, d_permutation, d_scales_ptr, d_params_temp_.data);

  gpuErrchk(cudaEventSynchronize(sync_event_));

  // Don't change the parameter sizes here, as parameters are constant in the
  // case of nonbonded exclusions
  nb_pot->set_idxs_device(*m_counter_, d_idxs_temp_.data, stream);
  nb_pot->set_scales_device(*m_counter_, d_params_temp_.data, stream);
  nb_pot->set_system_idxs_device(*m_counter_, d_system_idxs_temp_.data, stream);
}

template <typename RealType>
void LocalMDPotentials<RealType>::_reset_bonded_potential_idxs(
    std::shared_ptr<BoundPotential<RealType>> bp,
    std::shared_ptr<Potential<RealType>> pot, const int total_idxs,
    int *d_src_idxs, unsigned int *d_permutation, int *d_src_system_idxs,
    cudaStream_t stream) {
  if (!is_truncatable_potential(pot)) {
    return;
  }
  int idxs_dim = 0;
  if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
          std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
      trunc_pot != nullptr) {
    idxs_dim = trunc_pot->IDXS_DIM;
  } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
             trunc_pot != nullptr) {
    idxs_dim = trunc_pot->IDXS_DIM;
  } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(pot);
             trunc_pot != nullptr) {
    idxs_dim = trunc_pot->IDXS_DIM;
  } else {
    throw std::runtime_error(
        "_reset_bonded_potential_idxs::Unexpected truncatable type");
  }
  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const int params_dim = bp->params_dim;
  const int num_idxs = total_idxs / idxs_dim;
  // If the potential is empty, return
  if (num_idxs == 0) {
    return;
  }

  k_permute_chunks<RealType, false>
      <<<ceil_divide(num_idxs, tpb), tpb, 0, stream>>>(
          num_idxs, params_dim, d_permutation, bp->d_p.data,
          d_params_temp_.data);
  gpuErrchk(cudaPeekAtLastError());

  if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
          std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
      trunc_pot != nullptr) {
    trunc_pot->set_idxs_device(num_idxs, d_src_idxs, stream);
    trunc_pot->set_system_idxs_device(num_idxs, d_src_system_idxs, stream);
  } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
             trunc_pot != nullptr) {
    trunc_pot->set_idxs_device(num_idxs, d_src_idxs, stream);
    trunc_pot->set_system_idxs_device(num_idxs, d_src_system_idxs, stream);
  } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(pot);
             trunc_pot != nullptr) {
    trunc_pot->set_idxs_device(num_idxs, d_src_idxs, stream);
    trunc_pot->set_system_idxs_device(num_idxs, d_src_system_idxs, stream);
  } else {
    throw std::runtime_error(
        "_reset_bonded_potential_idxs::Unexpected truncatable type");
  }
  bp->size = bp->d_p.length;
  // Copy the unpermuted parameters to the bound potential, returning it to the
  // original state
  gpuErrchk(cudaMemcpyAsync(bp->d_p.data, d_params_temp_.data,
                            num_idxs * params_dim * sizeof(*bp->d_p.data),
                            cudaMemcpyDeviceToDevice, stream));
}

template <typename RealType>
void LocalMDPotentials<RealType>::_reset_nonbonded_exclusions_potential_idxs(
    std::shared_ptr<Potential<RealType>> pot, const int total_idxs,
    int *d_src_idxs, unsigned int *d_permutation, int *d_src_system_idxs,
    cudaStream_t stream) {

  std::shared_ptr<NonbondedPairList<RealType, true>> nb_pot =
      std::dynamic_pointer_cast<NonbondedPairList<RealType, true>>(pot);

  if (!nb_pot) {
    throw std::runtime_error(
        "_reset_nonbonded_exclusions_potential_idxs::Unexpectedly couldn't "
        "convert to Negated NonbondedPairList");
  }

  const int num_idxs = total_idxs / nb_pot->IDXS_DIM;

  nb_pot->set_idxs_device(num_idxs, d_src_idxs, stream);
  nb_pot->set_scales_device(num_idxs, d_scales_buffer_.data, stream);
  nb_pot->set_system_idxs_device(num_idxs, d_src_system_idxs, stream);
}

template <typename RealType>
void LocalMDPotentials<RealType>::reset_potentials(cudaStream_t stream) {
  this->_reset_nonbonded_ixn_group(nonbonded_pot_, stream);
  int i = 0;
  int idxs_offset = 0;
  int partition_offset = 0;
  for (auto bp : bps_) {
    if (std::shared_ptr<FanoutSummedPotential<RealType>> fanned_potential =
            std::dynamic_pointer_cast<FanoutSummedPotential<RealType>>(
                bp->potential);
        fanned_potential != nullptr) {
      for (auto pot : fanned_potential->get_potentials()) {
        if (is_exclusions_nonbonded_all_pairs_potential(pot)) {
          this->_reset_nonbonded_exclusions_potential_idxs(
              pot, idxs_sizes_[i], d_idxs_buffer_.data + idxs_offset,
              d_partitioned_indices_.data + partition_offset,
              d_system_idxs_buffer_.data + idxs_offset, stream);
        } else {
          this->_reset_bonded_potential_idxs(
              bp, pot, idxs_sizes_[i], d_idxs_buffer_.data + idxs_offset,
              d_partitioned_indices_.data + partition_offset,
              d_system_idxs_buffer_.data + idxs_offset, stream);
        }

        partition_offset += d_arange_.length;
        idxs_offset += idxs_sizes_[i];
        i++;
      }
    } else {
      if (is_exclusions_nonbonded_all_pairs_potential(bp->potential)) {
        this->_reset_nonbonded_exclusions_potential_idxs(
            bp->potential, idxs_sizes_[i], d_idxs_buffer_.data + idxs_offset,
            d_partitioned_indices_.data + partition_offset,
            d_system_idxs_buffer_.data + idxs_offset, stream);
      } else {
        this->_reset_bonded_potential_idxs(
            bp, bp->potential, idxs_sizes_[i],
            d_idxs_buffer_.data + idxs_offset,
            d_partitioned_indices_.data + partition_offset,
            d_system_idxs_buffer_.data + idxs_offset, stream);
      }
      partition_offset += d_arange_.length;
      idxs_offset += idxs_sizes_[i];
      i++;
    }
  }
}

template class LocalMDPotentials<double>;
template class LocalMDPotentials<float>;

} // namespace tmd
