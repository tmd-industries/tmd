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
#include <array>
#include <vector>

namespace tmd {

void verify_group_idxs(const int N,
                       const std::vector<std::vector<int>> &group_idxs);

// verify_mols_contiguous verifies that all of the atoms in the molecules are
// sequential. IE mol 0 is [0, 1, ..., K] and mol 1 is [K + 1, ....] and so on.
// This is used by water sampling and is an acceptable approach as long as the
// water molecules are all at the start of the system
void verify_mols_contiguous(const std::vector<std::vector<int>> &group_idxs);

// prepare_group_idxs_for_gpu takes a set of group indices and flattens it into
// three vectors. The first is the atom indices, the second is the mol indices
// and the last is the mol offsets. The first two arrays are both the length of
// the total number of atoms in the group idxs and the offsets are of the number
// of groups + 1.
std::array<std::vector<int>, 3>
prepare_group_idxs_for_gpu(const std::vector<std::vector<int>> &group_idxs);

std::vector<int>
get_mol_offsets(const std::vector<std::vector<int>> &group_idxs);

std::vector<int>
get_atom_indices(const std::vector<std::vector<int>> &group_idxs);

} // namespace tmd
