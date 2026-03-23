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
#pragma once

#include "bound_potential.hpp"
#include "potential.hpp"
#include <memory>
#include <vector>

namespace tmd {

template <typename RealType>
void verify_potentials_are_compatible(
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> &potentials);

template <typename RealType>
void verify_potentials_are_compatible(
    const std::vector<std::shared_ptr<Potential<RealType>>> &potentials);

} // namespace tmd
