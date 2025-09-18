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

#include <memory>
#include <vector>

#include "bound_potential.hpp"
#include "potential.hpp"

namespace tmd {

// each atom parameterized by a 4-tuple: charge, lj sigma, lj epsilon, 4D
// coordinate w
enum {
  PARAM_OFFSET_CHARGE = 0,
  PARAM_OFFSET_SIG,
  PARAM_OFFSET_EPS,
  PARAM_OFFSET_W,
  PARAMS_PER_ATOM
};

void verify_atom_idxs(int N, const std::vector<int> &atom_idxs,
                      const bool allow_empty = false);

template <typename RealType>
bool is_fanout_summed_potential(std::shared_ptr<Potential<RealType>> pot);
template <typename RealType>
bool is_summed_potential(std::shared_ptr<Potential<RealType>> pot);
template <typename RealType>
bool is_nonbonded_ixn_group_potential(std::shared_ptr<Potential<RealType>> pot);
template <typename RealType>
bool is_exclusions_nonbonded_all_pairs_potential(
    std::shared_ptr<Potential<RealType>> pot);

// Recursively populate nb_pots potentials with the NonbondedInteractionGroup
// potentials
template <typename RealType>
void get_nonbonded_ixn_group_potentials(
    std::vector<std::shared_ptr<Potential<RealType>>> input,
    std::vector<std::shared_ptr<Potential<RealType>>> &nb_pots);

template <typename RealType>
RealType get_nonbonded_ixn_group_cutoff_with_padding(
    std::shared_ptr<Potential<RealType>> pot);

} // namespace tmd
