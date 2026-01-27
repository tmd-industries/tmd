// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025-2026 Forrest York
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

#include "local_md_utils.hpp"
#include "nonbonded_interaction_group.hpp"

#include "fanout_summed_potential.hpp"
#include "harmonic_angle.hpp"
#include "harmonic_bond.hpp"
#include "nonbonded_common.hpp"
#include "nonbonded_pair_list.hpp"
#include "periodic_torsion.hpp"
#include "summed_potential.hpp"

#include "gpu_utils.cuh"
#include <string>
#include <type_traits>

namespace tmd {

template <typename RealType>
void verify_nonbonded_potential_for_local_md(
    const std::shared_ptr<Potential<RealType>> pot,
    const int expected_system_count, const int expected_idx_count) {
  if (std::shared_ptr<NonbondedInteractionGroup<RealType>> nb_pot =
          std::dynamic_pointer_cast<NonbondedInteractionGroup<RealType>>(pot);
      nb_pot) {
    if (nb_pot->get_num_row_idxs() != expected_idx_count ||
        nb_pot->get_num_col_idxs() != expected_idx_count) {
      throw std::runtime_error("unable to run local MD with nonbonded "
                               "potential on subset of the system");
    } else if (nb_pot->num_systems() != expected_system_count) {
      throw std::runtime_error(
          "local MD expected " + std::to_string(expected_system_count) +
          " systems, got " + std::to_string(nb_pot->num_systems()) + "systems");
    }
  } else {
    throw std::runtime_error(
        "unable to cast potential to NonbondedInteractionGroup");
  }
}

template void verify_nonbonded_potential_for_local_md<float>(
    const std::shared_ptr<Potential<float>> pot,
    const int expected_system_count, const int expected_idx_count);
template void verify_nonbonded_potential_for_local_md<double>(
    const std::shared_ptr<Potential<double>> pot,
    const int expected_system_count, const int expected_idx_count);

template <typename RealType>
void set_nonbonded_ixn_potential_idxs(std::shared_ptr<Potential<RealType>> pot,
                                      const std::vector<int> &num_row_idxs,
                                      const std::vector<int> &num_col_idxs,
                                      const unsigned int *d_row_idxs,
                                      const unsigned int *d_col_idxs,
                                      const cudaStream_t stream) {

  if (num_row_idxs.size() != num_col_idxs.size()) {
    throw std::runtime_error(
        "Number of row counts and number of column counts must match");
  }
  bool set_compute_col_grads = false;
  for (auto i = 0; i < num_row_idxs.size(); i++) {
    if (num_row_idxs[i] == num_col_idxs[i]) {
      set_compute_col_grads = true;
      break;
    }
  }
  if (std::shared_ptr<NonbondedInteractionGroup<RealType>> nb_pot =
          std::dynamic_pointer_cast<NonbondedInteractionGroup<RealType>>(pot);
      nb_pot) {
    nb_pot->set_atom_idxs_device(num_row_idxs, num_col_idxs, d_row_idxs,
                                 d_col_idxs, stream);
    nb_pot->set_compute_col_grads(set_compute_col_grads);
  } else {
    throw std::runtime_error(
        "Unable to cast potential to NonbondedInteractionGroup");
  }
}

template void set_nonbonded_ixn_potential_idxs<float>(
    std::shared_ptr<Potential<float>> pot, const std::vector<int> &num_row_idxs,
    const std::vector<int> &num_col_idxs, const unsigned int *d_row_idxs,
    const unsigned int *d_col_idxs, const cudaStream_t stream);
template void set_nonbonded_ixn_potential_idxs<double>(
    std::shared_ptr<Potential<double>> pot,
    const std::vector<int> &num_row_idxs, const std::vector<int> &num_col_idxs,
    const unsigned int *d_row_idxs, const unsigned int *d_col_idxs,
    const cudaStream_t stream);

template <typename RealType>
RealType get_nonbonded_ixn_potential_nblist_padding(
    const std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<NonbondedInteractionGroup<RealType>> nb_pot =
          std::dynamic_pointer_cast<NonbondedInteractionGroup<RealType>>(pot);
      nb_pot) {
    return nb_pot->get_nblist_padding();
  } else {
    throw std::runtime_error(
        "Unable to cast potential to NonbondedInteractionGroup");
  }
}

template float get_nonbonded_ixn_potential_nblist_padding<float>(
    const std::shared_ptr<Potential<float>> pot);
template double get_nonbonded_ixn_potential_nblist_padding<double>(
    const std::shared_ptr<Potential<double>> pot);

template <typename RealType>
void set_nonbonded_ixn_potential_nblist_padding(
    std::shared_ptr<Potential<RealType>> pot, const RealType nblist_padding) {
  if (std::shared_ptr<NonbondedInteractionGroup<RealType>> nb_pot =
          std::dynamic_pointer_cast<NonbondedInteractionGroup<RealType>>(pot);
      nb_pot) {
    nb_pot->set_nblist_padding(nblist_padding);
  } else {
    throw std::runtime_error(
        "Unable to cast potential to NonbondedInteractionGroup");
  }
}

template void set_nonbonded_ixn_potential_nblist_padding<float>(
    std::shared_ptr<Potential<float>> pot, const float nblist_padding);
template void set_nonbonded_ixn_potential_nblist_padding<double>(
    std::shared_ptr<Potential<double>> pot, const double nblist_padding);

template <typename RealType>
void verify_local_md_parameters(const RealType radius, const RealType k) {
  // Lower bound on radius selected to be 1 Angstrom, to avoid case where no
  // particles are moved. TBD whether or not this is a good lower bound
  const RealType min_radius = static_cast<RealType>(0.1);
  if (radius < min_radius) {
    throw std::runtime_error("radius must be greater or equal to " +
                             std::to_string(min_radius));
  }
  if (k < static_cast<RealType>(1.0)) {
    throw std::runtime_error("k must be at least one");
  }
  // TBD determine a more precise threshold, currently 10x what has been tested
  const RealType max_k = 1e6;
  if (k > max_k) {
    throw std::runtime_error("k must be less than than " +
                             std::to_string(max_k));
  }
}

template void verify_local_md_parameters<float>(const float radius,
                                                const float k);
template void verify_local_md_parameters<double>(const double radius,
                                                 const double k);

template <typename RealType>
int get_scales_buffer_length(
    const std::vector<std::shared_ptr<Potential<RealType>>> pots) {
  for (auto pot : pots) {
    if (std::shared_ptr<FanoutSummedPotential<RealType>> fanned_potential =
            std::dynamic_pointer_cast<FanoutSummedPotential<RealType>>(pot);
        fanned_potential != nullptr) {

      int scale_buffer_length =
          get_scales_buffer_length(fanned_potential->get_potentials());
      if (scale_buffer_length > 0) {
        return scale_buffer_length;
      }
    } else if (is_exclusions_nonbonded_all_pairs_potential(pot)) {
      if (std::shared_ptr<NonbondedPairList<RealType, true>> trunc_pot =
              std::dynamic_pointer_cast<NonbondedPairList<RealType, true>>(pot);
          trunc_pot != nullptr) {
        return trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM;
      } else {
        throw std::runtime_error(
            "get_scales_buffer_length()::something went wrong");
      }
    }
  }
  return 0;
}

template int get_scales_buffer_length(
    const std::vector<std::shared_ptr<Potential<float>>> pots);
template int get_scales_buffer_length(
    const std::vector<std::shared_ptr<Potential<double>>> pots);

template <typename RealType>
int get_scales_buffer_length_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> bps) {
  for (auto bp : bps) {
    if (std::shared_ptr<FanoutSummedPotential<RealType>> fanned_potential =
            std::dynamic_pointer_cast<FanoutSummedPotential<RealType>>(
                bp->potential);
        fanned_potential != nullptr) {

      int scale_buffer_length =
          get_scales_buffer_length(fanned_potential->get_potentials());
      if (scale_buffer_length > 0) {
        return scale_buffer_length;
      }
    } else if (is_exclusions_nonbonded_all_pairs_potential(bp->potential)) {
      if (std::shared_ptr<NonbondedPairList<RealType, true>> trunc_pot =
              std::dynamic_pointer_cast<NonbondedPairList<RealType, true>>(
                  bp->potential);
          trunc_pot != nullptr) {
        return trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM;
      } else {
        throw std::runtime_error(
            "get_scales_buffer_length_from_bps()::something went wrong");
      }
    }
  }
  return 0;
}

template int get_scales_buffer_length_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<float>>> pots);
template int get_scales_buffer_length_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<double>>> pots);

template <typename RealType>
std::vector<int> get_indices_buffer_sizes_from_pots(
    const std::vector<std::shared_ptr<Potential<RealType>>> pots) {
  std::vector<int> idx_sizes;
  for (auto pot : pots) {
    if (std::shared_ptr<FanoutSummedPotential<RealType>> fanned_potential =
            std::dynamic_pointer_cast<FanoutSummedPotential<RealType>>(pot);
        fanned_potential != nullptr) {

      std::vector<int> summed_idxs = get_indices_buffer_sizes_from_pots(
          fanned_potential->get_potentials());
      idx_sizes.insert(idx_sizes.end(), summed_idxs.begin(), summed_idxs.end());
    } else if (is_truncatable_potential(pot)) {
      if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
              std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
          trunc_pot != nullptr) {
        idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
      } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                     std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
                 trunc_pot != nullptr) {
        idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
      } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                     std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(pot);
                 trunc_pot != nullptr) {
        idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
      } else if (std::shared_ptr<NonbondedPairList<RealType, true>> trunc_pot =
                     std::dynamic_pointer_cast<
                         NonbondedPairList<RealType, true>>(pot);
                 trunc_pot != nullptr) {
        idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
      } else {
        throw std::runtime_error("get_indices_buffer_sizes_from_pots()::Got a "
                                 "truncatable potential, but not implemented");
      }
    } else {
      idx_sizes.push_back(0);
    }
  }
  return idx_sizes;
}

template std::vector<int> get_indices_buffer_sizes_from_pots(
    const std::vector<std::shared_ptr<Potential<float>>> pots);
template std::vector<int> get_indices_buffer_sizes_from_pots(
    const std::vector<std::shared_ptr<Potential<double>>> pots);

template <typename RealType>
std::vector<int> get_indices_buffer_sizes_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> bps) {
  std::vector<int> idx_sizes;
  for (auto bp : bps) {
    if (is_summed_potential(bp->potential)) {
      throw std::runtime_error("get_indices_buffer_sizes_from_bps()::Unable to "
                               "handle SummedPotential");
    } else if (std::shared_ptr<FanoutSummedPotential<RealType>>
                   fanned_potential = std::dynamic_pointer_cast<
                       FanoutSummedPotential<RealType>>(bp->potential);
               fanned_potential != nullptr) {
      std::vector<int> summed_idxs = get_indices_buffer_sizes_from_pots(
          fanned_potential->get_potentials());
      idx_sizes.insert(idx_sizes.end(), summed_idxs.begin(), summed_idxs.end());
    } else {
      auto pot = bp->potential;
      if (is_truncatable_potential(pot)) {
        if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
                std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
            trunc_pot != nullptr) {
          idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
        } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                       std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
                   trunc_pot != nullptr) {
          idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
        } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                       std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(
                           pot);
                   trunc_pot != nullptr) {
          idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
        } else if (std::shared_ptr<NonbondedPairList<RealType, true>>
                       trunc_pot = std::dynamic_pointer_cast<
                           NonbondedPairList<RealType, true>>(pot);
                   trunc_pot != nullptr) {
          idx_sizes.push_back(trunc_pot->get_num_idxs() * trunc_pot->IDXS_DIM);
        } else {
          throw std::runtime_error(
              "get_indices_buffer_sizes_from_bps()::Got a truncatable "
              "potential, but not implemented");
        }
      } else {
        idx_sizes.push_back(0);
      }
    }
  }
  return idx_sizes;
}

template std::vector<int> get_indices_buffer_sizes_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<float>>> bps);
template std::vector<int> get_indices_buffer_sizes_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<double>>> bps);

template <typename RealType>
bool is_truncatable_potential(const std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<HarmonicBond<RealType>> trunc_pot =
          std::dynamic_pointer_cast<HarmonicBond<RealType>>(pot);
      trunc_pot != nullptr) {
    return true;
  } else if (std::shared_ptr<HarmonicAngle<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<HarmonicAngle<RealType>>(pot);
             trunc_pot != nullptr) {
    return true;
  } else if (std::shared_ptr<PeriodicTorsion<RealType>> trunc_pot =
                 std::dynamic_pointer_cast<PeriodicTorsion<RealType>>(pot);
             trunc_pot != nullptr) {
    return true;
  } else if (std::shared_ptr<NonbondedPairList<RealType, true>> trunc_pot =
                 std::dynamic_pointer_cast<NonbondedPairList<RealType, true>>(
                     pot);
             trunc_pot != nullptr) {
    return true;
  } else {
    return false;
  }
}

template bool
is_truncatable_potential(const std::shared_ptr<Potential<float>> pot);
template bool
is_truncatable_potential(const std::shared_ptr<Potential<double>> pot);

} // namespace tmd
