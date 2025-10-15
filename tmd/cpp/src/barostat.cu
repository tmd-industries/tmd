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

#include "barostat.hpp"
#include "constants.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"
#include <stdio.h>
#include <variant>

#include "kernels/k_barostat.cuh"

namespace tmd {

template <typename RealType>
MonteCarloBarostat<RealType>::MonteCarloBarostat(
    const int N,
    const RealType pressure,    // Expected in Bar
    const RealType temperature, // Kelvin
    const std::vector<std::vector<int>> group_idxs, const int interval,
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> bps,
    const int seed, const bool adaptive_scaling_enabled,
    const RealType initial_volume_scale_factor)
    : Mover<RealType>(interval), N_(N),
      adaptive_scaling_enabled_(adaptive_scaling_enabled), bps_(bps),
      pressure_(pressure), temperature_(temperature), seed_(seed),
      group_idxs_(group_idxs), num_grouped_atoms_(0), runner_(),
      nrg_accum_(1, bps_.size()) {

  // Trigger check that interval is valid
  this->set_interval(this->interval_);

  // lets not have another facepalm moment again...
  if (temperature < static_cast<RealType>(100.0)) {
    std::cout << "warning temperature less than 100K" << std::endl;
  }

  if (pressure > static_cast<RealType>(10.0)) {
    std::cout << "warning pressure more than 10bar" << std::endl;
  }

  const int num_mols = group_idxs_.size();

  verify_group_idxs(N, group_idxs);
  // Array of flattened atom indices, mol indices and mol offsets
  std::array<std::vector<int>, 3> flattened_groups =
      prepare_group_idxs_for_gpu(group_idxs_);

  num_grouped_atoms_ = flattened_groups[0].size();

  cudaSafeMalloc(&d_x_proposed_, N_ * 3 * sizeof(*d_x_proposed_));
  cudaSafeMalloc(&d_box_proposed_, 3 * 3 * sizeof(*d_box_proposed_));
  cudaSafeMalloc(&d_u_buffer_, bps_.size() * sizeof(*d_u_buffer_));
  cudaSafeMalloc(&d_u_proposed_buffer_,
                 bps_.size() * sizeof(*d_u_proposed_buffer_));

  cudaSafeMalloc(&d_init_u_, 1 * sizeof(*d_init_u_));
  cudaSafeMalloc(&d_final_u_, 1 * sizeof(*d_final_u_));

  cudaSafeMalloc(&d_num_accepted_, 1 * sizeof(*d_num_accepted_));
  cudaSafeMalloc(&d_num_attempted_, 1 * sizeof(*d_num_attempted_));

  cudaSafeMalloc(&d_volume_, 1 * sizeof(*d_volume_));
  cudaSafeMalloc(&d_length_scale_, 1 * sizeof(*d_length_scale_));
  cudaSafeMalloc(&d_volume_scale_, 1 * sizeof(*d_volume_scale_));
  cudaSafeMalloc(&d_volume_delta_, 1 * sizeof(*d_volume_delta_));
  cudaSafeMalloc(&d_rand_state_, 1 * sizeof(*d_rand_state_));
  cudaSafeMalloc(&d_metropolis_hasting_rand_,
                 1 * sizeof(*d_metropolis_hasting_rand_));

  gpuErrchk(cudaMemcpy(d_volume_scale_, &initial_volume_scale_factor,
                       1 * sizeof(*d_volume_scale_), cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_centroids_, num_mols * 3 * sizeof(*d_centroids_));
  cudaSafeMalloc(&d_mol_offsets_,
                 flattened_groups[2].size() * sizeof(*d_mol_offsets_));

  cudaSafeMalloc(&d_atom_idxs_, num_grouped_atoms_ * sizeof(*d_atom_idxs_));
  cudaSafeMalloc(&d_mol_idxs_, num_grouped_atoms_ * sizeof(*d_mol_idxs_));

  gpuErrchk(cudaMemcpy(d_atom_idxs_, &flattened_groups[0][0],
                       flattened_groups[1].size() * sizeof(*d_atom_idxs_),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_mol_idxs_, &flattened_groups[1][0],
                       flattened_groups[0].size() * sizeof(*d_mol_idxs_),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_mol_offsets_, &flattened_groups[2][0],
                       flattened_groups[2].size() * sizeof(*d_mol_offsets_),
                       cudaMemcpyHostToDevice));

  k_initialize_curand_states<<<1, 1, 0>>>(1, seed_, d_rand_state_);
  gpuErrchk(cudaPeekAtLastError());

  this->reset_counters();
};

template <typename RealType>
MonteCarloBarostat<RealType>::~MonteCarloBarostat() {
  gpuErrchk(cudaFree(d_x_proposed_));
  gpuErrchk(cudaFree(d_centroids_));
  gpuErrchk(cudaFree(d_atom_idxs_));
  gpuErrchk(cudaFree(d_mol_idxs_));
  gpuErrchk(cudaFree(d_mol_offsets_));
  gpuErrchk(cudaFree(d_box_proposed_));
  gpuErrchk(cudaFree(d_u_proposed_buffer_));
  gpuErrchk(cudaFree(d_u_buffer_));
  gpuErrchk(cudaFree(d_init_u_));
  gpuErrchk(cudaFree(d_final_u_));
  gpuErrchk(cudaFree(d_length_scale_));
  gpuErrchk(cudaFree(d_volume_));
  gpuErrchk(cudaFree(d_volume_scale_));
  gpuErrchk(cudaFree(d_volume_delta_));
  gpuErrchk(cudaFree(d_rand_state_));
  gpuErrchk(cudaFree(d_metropolis_hasting_rand_));
  gpuErrchk(cudaFree(d_num_accepted_));
  gpuErrchk(cudaFree(d_num_attempted_));
};

template <typename RealType>
void MonteCarloBarostat<RealType>::reset_counters() {
  gpuErrchk(cudaMemset(d_num_accepted_, 0, sizeof(*d_num_accepted_)));
  gpuErrchk(cudaMemset(d_num_attempted_, 0, sizeof(*d_num_attempted_)));
}

template <typename RealType>
RealType MonteCarloBarostat<RealType>::get_volume_scale_factor() {
  RealType h_scaling;
  gpuErrchk(cudaMemcpy(&h_scaling, d_volume_scale_,
                       1 * sizeof(*d_volume_scale_), cudaMemcpyDeviceToHost));
  return h_scaling;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::set_volume_scale_factor(
    const RealType volume_scale_factor) {
  gpuErrchk(cudaMemcpy(d_volume_scale_, &volume_scale_factor,
                       1 * sizeof(*d_volume_scale_), cudaMemcpyHostToDevice));
  this->reset_counters();
}

template <typename RealType>
bool MonteCarloBarostat<RealType>::get_adaptive_scaling() {
  return this->adaptive_scaling_enabled_;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::set_adaptive_scaling(
    const bool adaptive_scaling_enabled) {
  this->adaptive_scaling_enabled_ = adaptive_scaling_enabled;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::move(const int N,
                                        RealType *d_x,   // [N*3]
                                        RealType *d_box, // [3*3]
                                        cudaStream_t stream) {
  if (N != N_) {
    throw std::runtime_error("N != N_");
  }
  this->step_++;
  if (this->step_ % this->interval_ != 0) {
    return;
  }

  const int num_molecules = group_idxs_.size();
  gpuErrchk(cudaMemsetAsync(d_centroids_, 0,
                            num_molecules * 3 * sizeof(*d_centroids_), stream));

  k_setup_barostat_move<RealType><<<1, 1, 0, stream>>>(
      adaptive_scaling_enabled_, d_rand_state_, d_box, d_volume_delta_,
      d_volume_scale_, d_length_scale_, d_volume_, d_metropolis_hasting_rand_);
  gpuErrchk(cudaPeekAtLastError());

  // Create duplicates of the coords/box that we can modify
  gpuErrchk(cudaMemcpyAsync(d_x_proposed_, d_x, N_ * 3 * sizeof(*d_x),
                            cudaMemcpyDeviceToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_box_proposed_, d_box,
                            3 * 3 * sizeof(*d_box_proposed_),
                            cudaMemcpyDeviceToDevice, stream));

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  // TBD: For larger systems (20k >) may be better to reduce the number of
  // blocks, rather than matching the number of blocks to be
  // ceil_divide(units_of_work, tpb). The kernels already support this, but at
  // the moment we match the blocks * tpb to equal units_of_work
  const int blocks = ceil_divide(num_grouped_atoms_, tpb);

  k_find_group_centroids<RealType>
      <<<blocks, tpb, 0, stream>>>(num_grouped_atoms_, d_x_proposed_,
                                   d_atom_idxs_, d_mol_idxs_, d_centroids_);
  gpuErrchk(cudaPeekAtLastError());

  // Scale centroids
  k_rescale_positions<RealType><<<blocks, tpb, 0, stream>>>(
      num_grouped_atoms_, d_x_proposed_, d_length_scale_, d_box,
      d_box_proposed_, // Box will be rescaled by length_scale
      d_atom_idxs_, d_mol_idxs_, d_mol_offsets_, d_centroids_);
  gpuErrchk(cudaPeekAtLastError());

  runner_.execute_potentials(bps_, N_, d_x, d_box, nullptr, nullptr,
                             d_u_buffer_, stream);
  // nullptr for the d_system_idxs as batch size is fixed to 1
  nrg_accum_.sum_device(bps_.size(), d_u_buffer_, nullptr, d_init_u_, stream);

  runner_.execute_potentials(bps_, N_, d_x_proposed_, d_box_proposed_, nullptr,
                             nullptr, d_u_proposed_buffer_, stream);
  // nullptr for the d_system_idxs as batch size is fixed to 1
  nrg_accum_.sum_device(bps_.size(), d_u_proposed_buffer_, nullptr, d_final_u_,
                        stream);

  RealType pressure = pressure_ * static_cast<RealType>(AVOGADRO * 1e-25);
  const RealType kT = static_cast<RealType>(BOLTZ) * temperature_;

  k_decide_move<RealType><<<ceil_divide(N_, tpb), tpb, 0, stream>>>(
      N_, adaptive_scaling_enabled_, num_molecules, kT, pressure,
      d_metropolis_hasting_rand_, d_volume_, d_volume_delta_, d_volume_scale_,
      d_init_u_, d_final_u_, d_box, d_box_proposed_, d_x, d_x_proposed_,
      d_num_accepted_, d_num_attempted_);
  gpuErrchk(cudaPeekAtLastError());
};

template <typename RealType>
void MonteCarloBarostat<RealType>::set_pressure(const RealType pressure) {
  pressure_ = static_cast<RealType>(pressure);
  // Could have equilibrated and be a large number of steps from shifting volume
  // adjustment, ie num attempted = 300 and num accepted = 150
  this->reset_counters();
}

template class MonteCarloBarostat<float>;
template class MonteCarloBarostat<double>;

} // namespace tmd
