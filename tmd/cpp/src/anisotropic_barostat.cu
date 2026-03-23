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

#include "anisotropic_barostat.hpp"
#include "assert.h"
#include "constants.hpp"
#include "fixed_point.hpp"
#include "gpu_utils.cuh"
#include "math_utils.cuh"
#include "mol_utils.hpp"
#include <cub/cub.cuh>
#include <stdio.h>
#include <variant>

#include "kernels/k_barostat.cuh"

namespace tmd {

template <typename RealType>
AnisotropicMonteCarloBarostat<RealType>::AnisotropicMonteCarloBarostat(
    const int N,
    const RealType pressure,    // Expected in Bar
    const RealType temperature, // Kelvin
    const std::vector<std::vector<int>> &group_idxs, const int interval,
    const std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    const int seed, const bool adaptive_scaling_enabled,
    const RealType initial_volume_scale_factor, const bool scale_x,
    const bool scale_y, const bool scale_z)
    : MonteCarloBarostat<RealType>(
          N, pressure, temperature, group_idxs, interval, bps, seed,
          adaptive_scaling_enabled, initial_volume_scale_factor),
      generator_(seed), distribution_(0.0, 3.0), scale_x_(scale_x),
      scale_y_(scale_y), scale_z_(scale_z) {

  if (!(scale_x_ | scale_y_ | scale_z_)) {
    throw std::runtime_error("must scale at least one dimension");
  }
};

template <typename RealType>
AnisotropicMonteCarloBarostat<RealType>::~AnisotropicMonteCarloBarostat(){};

template <typename RealType>
void AnisotropicMonteCarloBarostat<RealType>::propose_move(
    const int N, const RealType *d_x, const RealType *d_box,
    cudaStream_t stream) {
  // Create duplicates of the coords that we can modify
  gpuErrchk(cudaMemcpyAsync(this->d_x_proposed_, d_x,
                            this->num_systems_ * this->N_ * 3 * sizeof(*d_x),
                            cudaMemcpyDeviceToDevice, stream));

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  // TBD: For larger systems (20k >) may be better to reduce the number of
  // blocks, rather than matching the number of blocks to be
  // ceil_divide(units_of_work, tpb). The kernels already support this, but at
  // the moment we match the blocks * tpb to equal units_of_work
  const dim3 blocks(ceil_divide(this->num_grouped_atoms_, tpb),
                    this->num_systems_);

  k_find_group_centroids<RealType><<<blocks, tpb, 0, stream>>>(
      this->num_systems_, this->N_, this->num_mols_, this->num_grouped_atoms_,
      this->d_x_proposed_, this->d_atom_idxs_, this->d_mol_idxs_,
      this->d_centroids_);
  gpuErrchk(cudaPeekAtLastError());

  RealType rand = distribution_(generator_);
  int scaling_dim = -1;
  while (true) {
    if (scale_x_ && rand < 1.0) {
      scaling_dim = 0;
      break;
    } else if (scale_y_ && rand < 2.0) {
      scaling_dim = 1;
      break;
    } else if (scale_z_ && rand < 3.0) {
      scaling_dim = 2;
      break;
    }

    rand = distribution_(generator_);
  }
  assert(0 <= scaling_dim < 3);

  if (scaling_dim == 0) {
    k_setup_barostat_move<RealType, true, false, false>
        <<<dim3(1, this->num_systems_), 1, 0, stream>>>(
            this->num_systems_, this->adaptive_scaling_enabled_,
            this->d_rand_state_, d_box, this->d_volume_delta_,
            this->d_volume_scale_, this->d_length_scale_, this->d_volume_,
            this->d_metropolis_hasting_rand_);
    gpuErrchk(cudaPeekAtLastError());
    // Scale centroids
    k_rescale_positions<RealType, true, false, false>
        <<<blocks, tpb, 0, stream>>>(
            this->num_systems_, this->N_, this->num_mols_,
            this->num_grouped_atoms_, this->d_x_proposed_,
            this->d_length_scale_, d_box,
            this->d_box_proposed_, // Proposed box will be d_box rescaled by
                                   // length_scale
            this->d_atom_idxs_, this->d_mol_idxs_, this->d_mol_offsets_,
            this->d_centroids_);
    gpuErrchk(cudaPeekAtLastError());
  } else if (scaling_dim == 1) {
    k_setup_barostat_move<RealType, false, true, false>
        <<<dim3(1, this->num_systems_), 1, 0, stream>>>(
            this->num_systems_, this->adaptive_scaling_enabled_,
            this->d_rand_state_, d_box, this->d_volume_delta_,
            this->d_volume_scale_, this->d_length_scale_, this->d_volume_,
            this->d_metropolis_hasting_rand_);
    gpuErrchk(cudaPeekAtLastError());
    // Scale centroids
    k_rescale_positions<RealType, false, true, false>
        <<<blocks, tpb, 0, stream>>>(
            this->num_systems_, this->N_, this->num_mols_,
            this->num_grouped_atoms_, this->d_x_proposed_,
            this->d_length_scale_, d_box,
            this->d_box_proposed_, // Proposed box will be d_box rescaled by
                                   // length_scale
            this->d_atom_idxs_, this->d_mol_idxs_, this->d_mol_offsets_,
            this->d_centroids_);
    gpuErrchk(cudaPeekAtLastError());
  } else {
    k_setup_barostat_move<RealType, false, false, true>
        <<<dim3(1, this->num_systems_), 1, 0, stream>>>(
            this->num_systems_, this->adaptive_scaling_enabled_,
            this->d_rand_state_, d_box, this->d_volume_delta_,
            this->d_volume_scale_, this->d_length_scale_, this->d_volume_,
            this->d_metropolis_hasting_rand_);
    gpuErrchk(cudaPeekAtLastError());
    // Scale centroids
    k_rescale_positions<RealType, false, false, true>
        <<<blocks, tpb, 0, stream>>>(
            this->num_systems_, this->N_, this->num_mols_,
            this->num_grouped_atoms_, this->d_x_proposed_,
            this->d_length_scale_, d_box,
            this->d_box_proposed_, // Proposed box will be d_box rescaled by
                                   // length_scale
            this->d_atom_idxs_, this->d_mol_idxs_, this->d_mol_offsets_,
            this->d_centroids_);
    gpuErrchk(cudaPeekAtLastError());
  }
}

template class AnisotropicMonteCarloBarostat<float>;
template class AnisotropicMonteCarloBarostat<double>;

} // namespace tmd
