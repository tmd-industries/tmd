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
#include <typeinfo>

#include "bound_potential.hpp"
#include "potential.hpp"

namespace tmd {

template <typename RealType>
void verify_nonbonded_potential_for_local_md(
    const std::shared_ptr<Potential<RealType>> pot,
    const int expected_system_count, const int expected_idx_count);

template <typename RealType>
void set_nonbonded_ixn_potential_idxs(std::shared_ptr<Potential<RealType>> pot,
                                      const std::vector<int> &num_col_idxs,
                                      const std::vector<int> &num_row_idxs,
                                      const unsigned int *d_col_idxs,
                                      const unsigned int *d_row_idxs,
                                      const cudaStream_t stream);

template <typename RealType>
RealType get_nonbonded_ixn_potential_nblist_padding(
    const std::shared_ptr<Potential<RealType>> pot);

template <typename RealType>
void set_nonbonded_ixn_potential_nblist_padding(
    std::shared_ptr<Potential<RealType>> pot, const RealType nblist_padding);

template <typename RealType>
void verify_local_md_parameters(const RealType radius, const RealType k);

template <typename RealType>
int get_scales_buffer_length(
    const std::vector<std::shared_ptr<Potential<RealType>>> pots);

template <typename RealType>
int get_scales_buffer_length_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> bps);

template <typename RealType>
std::vector<int> get_indices_buffer_sizes_from_pots(
    const std::vector<std::shared_ptr<Potential<RealType>>> pots);

template <typename RealType>
std::vector<int> get_indices_buffer_sizes_from_bps(
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> bps);

template <typename RealType>
bool is_truncatable_potential(const std::shared_ptr<Potential<RealType>> pot);

} // namespace tmd
