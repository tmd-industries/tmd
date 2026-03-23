// Copyright 2025 Forrest York
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
#include "potential_utils.hpp"
#include <iostream>

namespace tmd {

template <typename RealType>
void verify_potentials_are_compatible(
    const std::vector<std::shared_ptr<Potential<RealType>>> &potentials) {
  auto num_systems = -1;
  for (auto pot : potentials) {
    if (num_systems < 0) {
      num_systems = pot->num_systems();
      if (num_systems < 1) {
        throw std::runtime_error(
            "Potentials must have at least a system size of 1, got " +
            std::to_string(num_systems));
      }
    } else if (pot->num_systems() != num_systems) {
      throw std::runtime_error("Potentials must all have the same system size" +
                               std::to_string(pot->num_systems()) + ", " +
                               std::to_string(num_systems));
    }
  }
}

template <typename RealType>
void verify_potentials_are_compatible(
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> &potentials) {
  auto num_systems = -1;
  for (auto pot : potentials) {
    if (num_systems < 0) {
      num_systems = pot->num_systems();
      if (num_systems < 1) {
        throw std::runtime_error(
            "Bound potentials must have at least a system size of 1, got " +
            std::to_string(num_systems));
      }
    } else if (pot->num_systems() != num_systems) {
      throw std::runtime_error(
          "Bound potentials must all have the same system size" +
          std::to_string(pot->num_systems()) + ", " +
          std::to_string(num_systems));
    }
  }
}

template void verify_potentials_are_compatible<float>(
    const std::vector<std::shared_ptr<Potential<float>>> &potentials);
template void verify_potentials_are_compatible<double>(
    const std::vector<std::shared_ptr<Potential<double>>> &potentials);

template void verify_potentials_are_compatible<float>(
    const std::vector<std::shared_ptr<BoundPotential<float>>> &potentials);
template void verify_potentials_are_compatible<double>(
    const std::vector<std::shared_ptr<BoundPotential<double>>> &potentials);

} // namespace tmd
