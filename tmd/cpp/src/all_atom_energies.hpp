// Copyright 2019-2025, Relay Therapeutics
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

#include <vector>

namespace tmd {

template <typename RealType>
std::vector<RealType> compute_atom_by_atom_energies(
    const int N, const std::vector<int> &target_atoms,
    const std::vector<RealType> &coords, const std::vector<RealType> &params,
    std::vector<RealType> &box, const RealType nb_beta, const RealType cutoff);

} // namespace tmd
