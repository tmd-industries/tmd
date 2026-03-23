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
#include "potential_utils.hpp"
#include <stdio.h>
#include <variant>

#include "kernels/k_barostat.cuh"
#include "kernels/k_indices.cuh"

namespace tmd {

template <typename RealType>
MonteCarloBarostat<RealType>::MonteCarloBarostat(
    const int N,
    const RealType pressure,    // Expected in Bar
    const RealType temperature, // Kelvin
    const std::vector<std::vector<int>> &group_idxs, const int interval,
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    const int seed, const bool adaptive_scaling_enabled,
    const RealType initial_volume_scale_factor)
    : Mover<RealType>(bps.size() > 0 ? bps[0]->num_systems() : 1, interval),
      N_(N), num_mols_(group_idxs.size()),
      adaptive_scaling_enabled_(adaptive_scaling_enabled), bps_(bps),
      pressure_(pressure), temperature_(temperature), seed_(seed),
      num_grouped_atoms_(0), runner_(),
      nrg_accum_(this->num_systems_, bps_.size() * this->num_systems_) {

  verify_potentials_are_compatible(bps_);
  // Trigger check that interval is valid
  this->set_interval(this->interval_);

  // lets not have another facepalm moment again...
  if (temperature < static_cast<RealType>(100.0)) {
    std::cout << "warning temperature less than 100K" << std::endl;
  }

  if (pressure > static_cast<RealType>(10.0)) {
    std::cout << "warning pressure more than 10bar" << std::endl;
  }

  verify_group_idxs(N, group_idxs);
  // Array of flattened atom indices, mol indices and mol offsets
  std::array<std::vector<int>, 3> flattened_groups =
      prepare_group_idxs_for_gpu(group_idxs);

  num_grouped_atoms_ = flattened_groups[0].size();

  cudaSafeMalloc(&d_x_proposed_,
                 this->num_systems_ * N_ * 3 * sizeof(*d_x_proposed_));
  cudaSafeMalloc(&d_box_proposed_,
                 this->num_systems_ * 3 * 3 * sizeof(*d_box_proposed_));
  cudaSafeMalloc(&d_u_buffer_,
                 this->num_systems_ * bps_.size() * sizeof(*d_u_buffer_));
  cudaSafeMalloc(&d_u_proposed_buffer_, this->num_systems_ * bps_.size() *
                                            sizeof(*d_u_proposed_buffer_));

  cudaSafeMalloc(&d_init_u_, this->num_systems_ * sizeof(*d_init_u_));
  cudaSafeMalloc(&d_final_u_, this->num_systems_ * sizeof(*d_final_u_));

  cudaSafeMalloc(&d_num_accepted_,
                 this->num_systems_ * sizeof(*d_num_accepted_));
  cudaSafeMalloc(&d_num_attempted_,
                 this->num_systems_ * sizeof(*d_num_attempted_));

  cudaSafeMalloc(&d_volume_, this->num_systems_ * sizeof(*d_volume_));
  cudaSafeMalloc(&d_length_scale_,
                 this->num_systems_ * sizeof(*d_length_scale_));
  cudaSafeMalloc(&d_volume_scale_,
                 this->num_systems_ * sizeof(*d_volume_scale_));
  cudaSafeMalloc(&d_volume_delta_,
                 this->num_systems_ * sizeof(*d_volume_delta_));
  cudaSafeMalloc(&d_rand_state_, this->num_systems_ * sizeof(*d_rand_state_));
  cudaSafeMalloc(&d_metropolis_hasting_rand_,
                 this->num_systems_ * sizeof(*d_metropolis_hasting_rand_));

  k_fill<RealType><<<ceil_divide(this->num_systems_, DEFAULT_THREADS_PER_BLOCK),
                     DEFAULT_THREADS_PER_BLOCK, 0>>>(
      this->num_systems_, d_volume_scale_, initial_volume_scale_factor);
  gpuErrchk(cudaPeekAtLastError());

  cudaSafeMalloc(&d_centroids_,
                 this->num_systems_ * num_mols_ * 3 * sizeof(*d_centroids_));
  cudaSafeMalloc(&d_mol_offsets_,
                 flattened_groups[2].size() * sizeof(*d_mol_offsets_));

  cudaSafeMalloc(&d_atom_idxs_, num_grouped_atoms_ * sizeof(*d_atom_idxs_));
  cudaSafeMalloc(&d_mol_idxs_, num_grouped_atoms_ * sizeof(*d_mol_idxs_));

  cudaSafeMalloc(&d_system_idxs_,
                 this->num_systems_ * bps_.size() * sizeof(*d_system_idxs_));

  gpuErrchk(cudaMemcpy(d_atom_idxs_, &flattened_groups[0][0],
                       flattened_groups[1].size() * sizeof(*d_atom_idxs_),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_mol_idxs_, &flattened_groups[1][0],
                       flattened_groups[0].size() * sizeof(*d_mol_idxs_),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_mol_offsets_, &flattened_groups[2][0],
                       flattened_groups[2].size() * sizeof(*d_mol_offsets_),
                       cudaMemcpyHostToDevice));

  k_initialize_curand_states<<<ceil_divide(this->num_systems_,
                                           DEFAULT_THREADS_PER_BLOCK),
                               DEFAULT_THREADS_PER_BLOCK, 0>>>(
      this->num_systems_, seed_, d_rand_state_);
  gpuErrchk(cudaPeekAtLastError());

  // Zero out the box proposals, as kernels currently only touch diagonal
  gpuErrchk(cudaMemset(d_box_proposed_, 0,
                       this->num_systems_ * 3 * 3 * sizeof(*d_box_proposed_)));

  k_segment_arange<<<dim3(ceil_divide(this->num_systems_,
                                      DEFAULT_THREADS_PER_BLOCK),
                          bps_.size()),
                     DEFAULT_THREADS_PER_BLOCK>>>(
      bps_.size(), this->num_systems_, d_system_idxs_);
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
  gpuErrchk(cudaFree(d_system_idxs_));
};

template <typename RealType>
void MonteCarloBarostat<RealType>::reset_counters() {
  gpuErrchk(cudaMemset(d_num_accepted_, 0,
                       this->num_systems_ * sizeof(*d_num_accepted_)));
  gpuErrchk(cudaMemset(d_num_attempted_, 0,
                       this->num_systems_ * sizeof(*d_num_attempted_)));
}

template <typename RealType>
std::vector<RealType>
MonteCarloBarostat<RealType>::get_volume_scale_factor() const {
  std::vector<RealType> h_scaling(this->num_systems_);
  gpuErrchk(cudaMemcpy(&h_scaling[0], d_volume_scale_,
                       this->num_systems_ * sizeof(*d_volume_scale_),
                       cudaMemcpyDeviceToHost));
  return h_scaling;
}

template <typename RealType>
void MonteCarloBarostat<RealType>::set_volume_scale_factor(
    const RealType volume_scale_factor) {
  k_fill<RealType><<<ceil_divide(this->num_systems_, DEFAULT_THREADS_PER_BLOCK),
                     DEFAULT_THREADS_PER_BLOCK, 0>>>(
      this->num_systems_, d_volume_scale_, volume_scale_factor);
  gpuErrchk(cudaPeekAtLastError());
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
void MonteCarloBarostat<RealType>::move(const int num_systems, const int N,
                                        RealType *d_x,   // [num_systems, N, 3]
                                        RealType *d_box, // [num_systems, 3, 3]
                                        cudaStream_t stream) {
  if (num_systems != this->num_systems_) {
    throw std::runtime_error("num_systems != num_systems_");
  }
  if (N != N_) {
    throw std::runtime_error("N != N_");
  }
  this->step_++;
  if (this->step_ % this->interval_ != 0) {
    return;
  }

  if (this->num_systems_ > 1) {
    gpuErrchk(cudaMemsetAsync(d_init_u_, 0,
                              this->num_systems_ * sizeof(*d_init_u_), stream));
    gpuErrchk(cudaMemsetAsync(
        d_final_u_, 0, this->num_systems_ * sizeof(*d_final_u_), stream));

    gpuErrchk(cudaMemsetAsync(
        d_u_buffer_, 0, this->num_systems_ * bps_.size() * sizeof(*d_u_buffer_),
        stream));
    gpuErrchk(cudaMemsetAsync(d_u_proposed_buffer_, 0,
                              this->num_systems_ * bps_.size() *
                                  sizeof(*d_u_proposed_buffer_),
                              stream));
  }

  gpuErrchk(cudaMemsetAsync(
      d_centroids_, 0,
      this->num_systems_ * num_mols_ * 3 * sizeof(*d_centroids_), stream));

  runner_.execute_potentials(this->num_systems_, bps_, N_, d_x, d_box, nullptr,
                             nullptr, d_u_buffer_, stream);

  nrg_accum_.sum_device(this->num_systems_ * bps_.size(), d_u_buffer_,
                        d_system_idxs_, d_init_u_, stream);

  this->propose_move(N, d_x, d_box, stream);

  runner_.execute_potentials(this->num_systems_, bps_, N_, d_x_proposed_,
                             d_box_proposed_, nullptr, nullptr,
                             d_u_proposed_buffer_, stream);

  nrg_accum_.sum_device(this->num_systems_ * bps_.size(), d_u_proposed_buffer_,
                        d_system_idxs_, d_final_u_, stream);

  this->decide_move(N, d_x, d_box, stream);
};

template <typename RealType>
void MonteCarloBarostat<RealType>::propose_move(const int N,
                                                const RealType *d_x,
                                                const RealType *d_box,
                                                cudaStream_t stream) {
  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  // TBD: For larger systems (20k >) may be better to reduce the number of
  // blocks, rather than matching the number of blocks to be
  // ceil_divide(units_of_work, tpb). The kernels already support this, but at
  // the moment we match the blocks * tpb to equal units_of_work
  const dim3 blocks(ceil_divide(num_grouped_atoms_, tpb), this->num_systems_);

  k_setup_barostat_move<RealType, true, true, true>
      <<<dim3(1, this->num_systems_), 1, 0, stream>>>(
          this->num_systems_, adaptive_scaling_enabled_, d_rand_state_, d_box,
          d_volume_delta_, d_volume_scale_, d_length_scale_, d_volume_,
          d_metropolis_hasting_rand_);
  gpuErrchk(cudaPeekAtLastError());

  // Create duplicates of the coords that we can modify
  gpuErrchk(cudaMemcpyAsync(d_x_proposed_, d_x,
                            this->num_systems_ * N_ * 3 * sizeof(*d_x),
                            cudaMemcpyDeviceToDevice, stream));

  k_find_group_centroids<RealType><<<blocks, tpb, 0, stream>>>(
      this->num_systems_, N_, this->num_mols_, num_grouped_atoms_,
      d_x_proposed_, d_atom_idxs_, d_mol_idxs_, d_centroids_);
  gpuErrchk(cudaPeekAtLastError());

  // Scale centroids
  k_rescale_positions<RealType, true, true, true><<<blocks, tpb, 0, stream>>>(
      this->num_systems_, N_, this->num_mols_, num_grouped_atoms_,
      d_x_proposed_, d_length_scale_, d_box,
      d_box_proposed_, // Proposed box will be d_box rescaled by length_scale
      d_atom_idxs_, d_mol_idxs_, d_mol_offsets_, d_centroids_);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void MonteCarloBarostat<RealType>::decide_move(const int N, RealType *d_x,
                                               RealType *d_box,
                                               cudaStream_t stream) {
  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 blocks(ceil_divide(N_, tpb), this->num_systems_);

  const RealType pressure = pressure_ * static_cast<RealType>(AVOGADRO * 1e-25);
  const RealType kT = static_cast<RealType>(BOLTZ) * temperature_;

  k_decide_move<RealType><<<blocks, tpb, 0, stream>>>(
      this->num_systems_, N_, adaptive_scaling_enabled_, num_mols_, kT,
      pressure, d_metropolis_hasting_rand_, d_volume_, d_volume_delta_,
      d_volume_scale_, d_init_u_, d_final_u_, d_box, d_box_proposed_, d_x,
      d_x_proposed_, d_num_accepted_, d_num_attempted_);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void MonteCarloBarostat<RealType>::set_pressure(const RealType pressure) {
  pressure_ = pressure;
  // Could have equilibrated and be a large number of steps from shifting volume
  // adjustment, ie num attempted = 300 and num accepted = 150
  this->reset_counters();
}

template class MonteCarloBarostat<float>;
template class MonteCarloBarostat<double>;

} // namespace tmd
