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

#include <set>
#include <stdexcept>
#include <string>

#include "harmonic_bond.hpp"
#include "nonbonded_common.hpp"
#include "nonbonded_interaction_group.hpp"
#include "nonbonded_pair_list.hpp"

#include "cuda_runtime.h"

#include "fanout_summed_potential.hpp"
#include "summed_potential.hpp"

#include "set_utils.hpp"

namespace tmd {

template <typename RealType>
bool is_fanout_summed_potential(std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<FanoutSummedPotential<RealType>> fanned_potential =
          std::dynamic_pointer_cast<FanoutSummedPotential<RealType>>(pot);
      fanned_potential != nullptr) {
    return true;
  }
  return false;
}

// Return true if the potential is a summed potential.
template <typename RealType>
bool is_summed_potential(std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<SummedPotential<RealType>> summed_potential =
          std::dynamic_pointer_cast<SummedPotential<RealType>>(pot);
      summed_potential != nullptr) {
    return true;
  }
  return false;
}

template <typename RealType>
bool is_nonbonded_ixn_group_potential(
    std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<NonbondedInteractionGroup<RealType>> nb_pot =
          std::dynamic_pointer_cast<NonbondedInteractionGroup<RealType>>(pot);
      nb_pot) {
    return true;
  }
  return false;
}

template <typename RealType>
bool is_exclusions_nonbonded_all_pairs_potential(
    std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<NonbondedPairList<RealType, true>> nb_pot =
          std::dynamic_pointer_cast<NonbondedPairList<RealType, true>>(pot);
      nb_pot) {
    return true;
  }
  return false;
}

void verify_atom_idxs(const int N, const std::vector<int> &atom_idxs,
                      const bool allow_empty) {
  if (atom_idxs.size() == 0) {
    if (allow_empty) {
      // No further checks if we allow the indices to be empty
      return;
    }
    throw std::runtime_error("indices can't be empty");
  }
  std::set<int> unique_idxs(atom_idxs.begin(), atom_idxs.end());
  if (unique_idxs.size() != atom_idxs.size()) {
    throw std::runtime_error("atom indices must be unique");
  }
  if (*std::max_element(atom_idxs.begin(), atom_idxs.end()) >= N) {
    throw std::runtime_error("index values must be less than N(" +
                             std::to_string(N) + ")");
  }
  if (*std::min_element(atom_idxs.begin(), atom_idxs.end()) < 0) {
    throw std::runtime_error("index values must be greater or equal to zero");
  }
}

// get_nonbonded_ixn_group_cutoff_with_padding returns the cutoff plus padding.
// Using these value can be used to validate the box dimensions
template <typename RealType>
RealType get_nonbonded_ixn_group_cutoff_with_padding(
    std::shared_ptr<Potential<RealType>> pot) {
  if (std::shared_ptr<NonbondedInteractionGroup<RealType>> nb_pot =
          std::dynamic_pointer_cast<NonbondedInteractionGroup<RealType>>(pot);
      nb_pot) {
    return nb_pot->get_cutoff() + nb_pot->get_nblist_padding();
  } else {
    throw std::runtime_error(
        "unable to cast potential to NonbondedInteractionGroup");
  }
}

// Recursively populate nb_pots potentials with the NonbondedInteractionGroup
// potentials
template <typename RealType>
void get_nonbonded_ixn_group_potentials(
    std::vector<std::shared_ptr<Potential<RealType>>> input,
    std::vector<std::shared_ptr<Potential<RealType>>> &nb_pots) {
  for (auto pot : input) {
    if (is_summed_potential(pot)) {
      throw std::runtime_error("Not allowed to pass a SummedPotential to C++");
    } else if (std::shared_ptr<FanoutSummedPotential<RealType>>
                   fanned_potential = std::dynamic_pointer_cast<
                       FanoutSummedPotential<RealType>>(pot);
               fanned_potential != nullptr) {
      for (auto summed_pot : fanned_potential->get_potentials()) {
        if (is_summed_potential(summed_pot) ||
            is_fanout_summed_potential(summed_pot)) {
          throw std::runtime_error(
              "Not allowed to wrap a summed pot in a fanout summed pot");
        } else if (is_nonbonded_ixn_group_potential(summed_pot)) {
          nb_pots.push_back(summed_pot);
        }
      }
    } else if (is_nonbonded_ixn_group_potential(pot)) {
      nb_pots.push_back(pot);
    }
  }
}

template bool
is_fanout_summed_potential<float>(std::shared_ptr<Potential<float>> pot);
template bool
is_fanout_summed_potential<double>(std::shared_ptr<Potential<double>> pot);

template bool is_summed_potential<float>(std::shared_ptr<Potential<float>> pot);
template bool
is_summed_potential<double>(std::shared_ptr<Potential<double>> pot);

template bool
is_nonbonded_ixn_group_potential<float>(std::shared_ptr<Potential<float>> pot);
template bool is_nonbonded_ixn_group_potential<double>(
    std::shared_ptr<Potential<double>> pot);

template bool is_exclusions_nonbonded_all_pairs_potential<float>(
    std::shared_ptr<Potential<float>> pot);
template bool is_exclusions_nonbonded_all_pairs_potential<double>(
    std::shared_ptr<Potential<double>> pot);

template void get_nonbonded_ixn_group_potentials<float>(
    std::vector<std::shared_ptr<Potential<float>>> input,
    std::vector<std::shared_ptr<Potential<float>>> &nb_pots);
template void get_nonbonded_ixn_group_potentials<double>(
    std::vector<std::shared_ptr<Potential<double>>> input,
    std::vector<std::shared_ptr<Potential<double>>> &nb_pots);

template float get_nonbonded_ixn_group_cutoff_with_padding<float>(
    std::shared_ptr<Potential<float>> pot);
template double get_nonbonded_ixn_group_cutoff_with_padding<double>(
    std::shared_ptr<Potential<double>> pot);

} // namespace tmd
