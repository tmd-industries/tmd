// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025, Forrest York
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

#include "bd_exchange_move.hpp"
#include "curand_kernel.h"
#include "device_buffer.hpp"
#include "pinned_host_buffer.hpp"
#include <array>
#include <vector>

namespace tmd {

// TIBDExchangeMove uses targeted insertion and biased deletion to move into a
// sphere around a set of ligand atoms. The reference implementation is in
// tmd/md/exchange/exchange_mover.py::TIBDExchangeMove Refer to
// tmd/cpp/src/bd_exchange_move.hpp::BDExchangeMove for the definitions of the
// terminology
template <typename RealType>
class TIBDExchangeMove : public BDExchangeMove<RealType> {

protected:
  const RealType radius_;
  const RealType inner_volume_;

  DeviceBuffer<curandState_t> d_rand_states_;

  DeviceBuffer<int> d_inner_mols_count_;    // [1]
  DeviceBuffer<int> d_identify_indices_;    // [this->num_target_mols_]
  DeviceBuffer<int> d_partitioned_indices_; // [this->num_target_mols_]
  DeviceBuffer<char> d_temp_storage_buffer_;
  size_t temp_storage_bytes_;

  DeviceBuffer<RealType> d_center_; // [3]
  // Uniform noise with the first element used for deciding directionality of
  // insertion and the second element is used for comparison against the
  // acceptance rate in the Metropolis-Hastings check
  DeviceBuffer<RealType> d_uniform_noise_buffer_; // [2 * this->batch_size_ *
                                                  // this->steps_per_move_]
  DeviceBuffer<int> d_targeting_inner_vol_;       // [1]

  DeviceBuffer<int> d_ligand_idxs_;
  DeviceBuffer<RealType>
      d_src_log_weights_; // [this->num_target_mols_ * this->batch_size_]
  DeviceBuffer<RealType>
      d_dest_log_weights_; // [this->num_target_mols_ * this->batch_size_]
  DeviceBuffer<int> d_inner_flags_;     // [this->num_target_mols_]
  DeviceBuffer<RealType> d_box_volume_; // [1]

private:
  DeviceBuffer<RealType>
      d_selected_translations_; // [this->batch_size_, 3] The translation
                                // selected to run
  DeviceBuffer<int> d_sample_after_segment_offsets_; // [this->batch_size_ + 1]
  DeviceBuffer<int> d_weights_before_counts_;        // [this->batch_size_]
  DeviceBuffer<int> d_weights_after_counts_;         // [this->batch_size_]

  DeviceBuffer<RealType>
      d_lse_max_src_; // [this->batch_size, this->num_target_mols_]
  DeviceBuffer<RealType>
      d_lse_exp_sum_src_; // [this->batch_size, this->num_target_mols_]

public:
  TIBDExchangeMove(const int num_systems, const int N,
                   const std::vector<int> ligand_idxs,
                   const std::vector<std::vector<int>> &target_mols,
                   const std::vector<RealType> &params,
                   const RealType temperature, const RealType nb_beta,
                   const RealType cutoff, const RealType radius, const int seed,
                   const int proposals_per_move, const int interval,
                   const int batch_size);

  ~TIBDExchangeMove();

  void move(const int num_systems, const int N,
            RealType *d_coords, // [num_systems, N, 3]
            RealType *d_box,    // [num_systems, 3, 3]
            cudaStream_t stream) override;

  std::array<std::vector<RealType>, 2>
  move_host(const int N, const RealType *h_coords,
            const RealType *h_box) override;

  RealType log_probability_host() override;
  RealType raw_log_probability_host() override;
};

} // namespace tmd
