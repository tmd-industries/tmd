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

namespace tmd {

template <typename RealType>
void rotate_coordinates_host(const int N, const int n_rotations,
                             const RealType *coords,
                             const RealType *quaternions, RealType *output);

template <typename RealType>
void rotate_coordinates_and_translate_mol_host(
    const int N, const int batch_size, const RealType *mol_coords,
    const RealType *box, const RealType *quaternion,
    const RealType *translation, RealType *output);

} // namespace tmd
