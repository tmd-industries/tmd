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

#include "barostat.hpp"
#include "bound_potential.hpp"
#include "constants.hpp"
#include "context.hpp"

#include "fixed_point.hpp"
#include "flat_bottom_bond.hpp"
#include "gpu_utils.cuh"
#include "integrator.hpp"
#include "kernels/kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "local_md_potentials.hpp"
#include "math_utils.cuh"
#include "nonbonded_common.hpp"
#include "pinned_host_buffer.hpp"
#include "set_utils.hpp"

namespace tmd {

template <typename RealType>
static bool is_barostat(std::shared_ptr<Mover<RealType>> &mover) {
  if (std::shared_ptr<MonteCarloBarostat<RealType>> baro =
          std::dynamic_pointer_cast<MonteCarloBarostat<RealType>>(mover);
      baro) {
    return true;
  }
  return false;
}

template <typename RealType>
Context<RealType>::Context(
    int N, const RealType *x_0, const RealType *v_0, const RealType *box_0,
    std::shared_ptr<Integrator<RealType>> intg,
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    std::vector<std::shared_ptr<Mover<RealType>>> &movers)
    : N_(N), movers_(movers), step_(0), intg_(intg), bps_(bps),
      nonbonded_pots_(0) {

  std::vector<std::shared_ptr<Potential<RealType>>> pots;
  for (auto bp : bps_) {
    // Defensive, should be removed later in favor of ensuring all batch sizes
    // are identical
    if (bp->potential->batch_size() != 1) {
      throw std::runtime_error("Batch sizes of all potentials must be 1");
    }
    pots.push_back(bp->potential);
  }
  // A no-op if running in vacuum or there are no NonbondedInteractionGroup
  // potentials
  get_nonbonded_ixn_group_potentials(pots, nonbonded_pots_);

  d_x_t_ = gpuErrchkCudaMallocAndCopy(x_0, N * 3);
  d_v_t_ = gpuErrchkCudaMallocAndCopy(v_0, N * 3);
  d_box_t_ = gpuErrchkCudaMallocAndCopy(box_0, 3 * 3);

  gpuErrchk(cudaStreamCreate(&stream_));
};

template <typename RealType> Context<RealType>::~Context() {
  gpuErrchk(cudaFree(d_x_t_));
  gpuErrchk(cudaFree(d_v_t_));
  gpuErrchk(cudaFree(d_box_t_));

  gpuErrchk(cudaStreamDestroy(stream_));
};

template <typename RealType>
void Context<RealType>::_verify_coords_and_box(const RealType *coords_buffer,
                                               const RealType *box_buffer,
                                               cudaStream_t stream) {
  // If there are no nonbonded potentials (ie Vacuum), nothing to check.
  if (nonbonded_pots_.size() == 0) {
    return;
  }
  gpuErrchk(cudaStreamSynchronize(stream));
  for (auto pot : nonbonded_pots_) {
    RealType cutoff = get_nonbonded_ixn_group_cutoff_with_padding(pot);
    RealType db_cutoff = 2 * cutoff;
    for (int i = 0; i < 3; i++) {
      if (box_buffer[i * 3 + i] < db_cutoff) {
        throw std::runtime_error(
            "cutoff with padding is more than half of the box width, "
            "neighborlist is no longer reliable");
      }
    }
  }

  const double max_box_dim = max(
      box_buffer[0 * 3 + 0], max(box_buffer[1 * 3 + 1], box_buffer[2 * 3 + 2]));
  const auto [min_coord, max_coord] =
      std::minmax_element(coords_buffer, coords_buffer + N_ * 3);
  // Look at the largest difference in all dimensions, since coordinates are not
  // imaged into the home box per se, rather into the nearest periodic box
  const double max_coord_delta = *max_coord - *min_coord;
  if (max_box_dim * 100.0 < max_coord_delta) {
    throw std::runtime_error(
        "simulation unstable: dimensions of coordinates two orders of "
        "magnitude larger than max box dimension");
  }
}

template <typename RealType> RealType Context<RealType>::_get_temperature() {
  if (std::shared_ptr<LangevinIntegrator<RealType>> langevin =
          std::dynamic_pointer_cast<LangevinIntegrator<RealType>>(intg_);
      langevin != nullptr) {
    return langevin->get_temperature();
  } else {
    throw std::runtime_error("integrator must be LangevinIntegrator.");
  }
}

template <typename RealType>
void Context<RealType>::setup_local_md(RealType temperature,
                                       bool freeze_reference,
                                       RealType nblist_padding) {
  if (this->local_md_pots_ != nullptr) {
    if (this->local_md_pots_->temperature != temperature ||
        this->local_md_pots_->freeze_reference != freeze_reference ||
        this->local_md_pots_->nblist_padding != nblist_padding) {
      throw std::runtime_error(
          "local md configured with different parameters, current parameters: "
          "Temperature " +
          std::to_string(this->local_md_pots_->temperature) +
          " Freeze Reference " +
          std::to_string(this->local_md_pots_->freeze_reference) +
          " Nblist padding " +
          std::to_string(this->local_md_pots_->nblist_padding));
    }
    return;
  }
  this->local_md_pots_.reset(
      new LocalMDPotentials(N_, bps_, nonbonded_pots_, freeze_reference,
                            temperature, nblist_padding));
}

template <typename RealType>
void Context<RealType>::_ensure_local_md_intialized() {
  if (this->local_md_pots_ == nullptr) {
    RealType temperature = this->_get_temperature();
    this->setup_local_md(temperature, true);
  }
}

template <typename RealType>
void Context<RealType>::multiple_steps_local(const int n_steps,
                                             const std::vector<int> &local_idxs,
                                             const int n_samples,
                                             const RealType radius,
                                             const RealType k, const int seed,
                                             RealType *h_x, RealType *h_box) {
  const int store_x_interval =
      n_samples > 0 ? n_steps / n_samples : n_steps + 1;
  if (n_samples < 0) {
    throw std::runtime_error("n_samples < 0");
  }
  if (n_steps % store_x_interval != 0) {
    std::cout << "warning:: n_steps modulo store_x_interval does not equal zero"
              << std::endl;
  }
  this->_ensure_local_md_intialized();
  try {

    local_md_pots_->setup_from_idxs(d_x_t_, d_box_t_, local_idxs, seed, radius,
                                    k, stream_);

    unsigned int *d_free_idxs = local_md_pots_->get_free_idxs();

    std::vector<std::shared_ptr<BoundPotential<RealType>>> local_pots =
        local_md_pots_->get_potentials();

    intg_->initialize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs,
                      stream_);
    for (int i = 1; i <= n_steps; i++) {
      this->_step(local_pots, d_free_idxs, stream_);
      if (i % store_x_interval == 0) {
        RealType *box_ptr = h_box + ((i / store_x_interval) - 1) * 3 * 3;
        RealType *coord_ptr = h_x + ((i / store_x_interval) - 1) * N_ * 3;
        gpuErrchk(cudaMemcpyAsync(coord_ptr, d_x_t_, N_ * 3 * sizeof(*d_x_t_),
                                  cudaMemcpyDeviceToHost, stream_));
        gpuErrchk(cudaMemcpyAsync(box_ptr, d_box_t_, 3 * 3 * sizeof(*d_box_t_),
                                  cudaMemcpyDeviceToHost, stream_));
        this->_verify_coords_and_box(coord_ptr, box_ptr, stream_);
      }
    }
    intg_->finalize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs, stream_);
    local_md_pots_->reset_potentials(stream_);
  } catch (...) {
    // Always reset the potentials. _verify_coords_and_box may trigger an
    // exception and without resetting the potentials are in an invalid state
    local_md_pots_->reset_potentials(stream_);
    gpuErrchk(cudaStreamSynchronize(stream_));
    throw;
  }
  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType>
std::vector<std::shared_ptr<BoundPotential<RealType>>>
Context<RealType>::truncate_potentials_local_selection(
    const int reference_idx, const std::vector<int> &selection_idxs,
    const RealType radius, const RealType k) {

  this->_ensure_local_md_intialized();

  local_md_pots_->setup_from_selection(reference_idx, selection_idxs, radius, k,
                                       stream_);

  gpuErrchk(cudaStreamSynchronize(stream_));

  return local_md_pots_->get_potentials();
}

template <typename RealType>
void Context<RealType>::multiple_steps_local_selection(
    const int n_steps, const int reference_idx,
    const std::vector<int> &selection_idxs, const int n_samples,
    const RealType radius, const RealType k, RealType *h_x, RealType *h_box) {
  const int store_x_interval =
      n_samples > 0 ? n_steps / n_samples : n_steps + 1;
  if (n_samples < 0) {
    throw std::runtime_error("n_samples < 0");
  }
  if (n_steps % store_x_interval != 0) {
    std::cout << "warning:: n_steps modulo store_x_interval does not equal zero"
              << std::endl;
  }

  this->_ensure_local_md_intialized();

  try {

    local_md_pots_->setup_from_selection(reference_idx, selection_idxs, radius,
                                         k, stream_);

    unsigned int *d_free_idxs = local_md_pots_->get_free_idxs();

    std::vector<std::shared_ptr<BoundPotential<RealType>>> local_pots =
        local_md_pots_->get_potentials();

    intg_->initialize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs,
                      stream_);
    for (int i = 1; i <= n_steps; i++) {
      this->_step(local_pots, d_free_idxs, stream_);
      if (i % store_x_interval == 0) {
        RealType *box_ptr = h_box + ((i / store_x_interval) - 1) * 3 * 3;
        RealType *coord_ptr = h_x + ((i / store_x_interval) - 1) * N_ * 3;
        gpuErrchk(cudaMemcpyAsync(coord_ptr, d_x_t_, N_ * 3 * sizeof(*d_x_t_),
                                  cudaMemcpyDeviceToHost, stream_));
        gpuErrchk(cudaMemcpyAsync(box_ptr, d_box_t_, 3 * 3 * sizeof(*d_box_t_),
                                  cudaMemcpyDeviceToHost, stream_));
        this->_verify_coords_and_box(coord_ptr, box_ptr, stream_);
      }
    }
    intg_->finalize(local_pots, d_x_t_, d_v_t_, d_box_t_, d_free_idxs, stream_);
    local_md_pots_->reset_potentials(stream_);
  } catch (...) {
    // Always reset the potentials. _verify_coords_and_box may trigger an
    // exception and without resetting the potentials are in an invalid state
    local_md_pots_->reset_potentials(stream_);
    gpuErrchk(cudaStreamSynchronize(stream_));
    throw;
  }
  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType>
void Context<RealType>::multiple_steps(const int n_steps, const int n_samples,
                                       RealType *h_x, RealType *h_box) {
  const int store_x_interval =
      n_samples > 0 ? n_steps / n_samples : n_steps + 1;
  if (n_samples < 0) {
    throw std::runtime_error("n_samples < 0");
  }
  if (n_steps % store_x_interval != 0) {
    std::cout << "warning:: n_steps modulo store_x_interval does not equal zero"
              << std::endl;
  }

  intg_->initialize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream_);
  for (int i = 1; i <= n_steps; i++) {
    this->_step(bps_, nullptr, stream_);

    if (i % store_x_interval == 0) {
      RealType *box_ptr = h_box + ((i / store_x_interval) - 1) * 3 * 3;
      RealType *coord_ptr = h_x + ((i / store_x_interval) - 1) * N_ * 3;
      gpuErrchk(cudaMemcpyAsync(coord_ptr, d_x_t_, N_ * 3 * sizeof(*d_x_t_),
                                cudaMemcpyDeviceToHost, stream_));
      gpuErrchk(cudaMemcpyAsync(box_ptr, d_box_t_, 3 * 3 * sizeof(*d_box_t_),
                                cudaMemcpyDeviceToHost, stream_));
      this->_verify_coords_and_box(coord_ptr, box_ptr, stream_);
    }
  }
  intg_->finalize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream_);

  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType> void Context<RealType>::step() {
  this->_step(bps_, nullptr, stream_);
  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType> void Context<RealType>::finalize() {
  intg_->finalize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream_);
  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType> void Context<RealType>::initialize() {
  intg_->initialize(bps_, d_x_t_, d_v_t_, d_box_t_, nullptr, stream_);
  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType>
void Context<RealType>::_step(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    unsigned int *d_atom_idxs, const cudaStream_t stream) {
  intg_->step_fwd(bps, d_x_t_, d_v_t_, d_box_t_, d_atom_idxs, stream);

  // If atom idxs are passed, indicates that only a subset of the system should
  // move. Don't run any additional movers in this situation. TBD: Handle movers
  // in the local MD case.
  if (d_atom_idxs == nullptr) {
    for (auto mover : movers_) {
      // May modify coords and box size
      mover->move(N_, d_x_t_, d_box_t_, stream);
    }
  }

  step_ += 1;
};

template <typename RealType> int Context<RealType>::num_atoms() const {
  return N_;
}

template <typename RealType>
void Context<RealType>::set_x_t(const RealType *in_buffer) {
  gpuErrchk(cudaMemcpy(d_x_t_, in_buffer, N_ * 3 * sizeof(*in_buffer),
                       cudaMemcpyHostToDevice));
}

template <typename RealType>
void Context<RealType>::set_v_t(const RealType *in_buffer) {
  gpuErrchk(cudaMemcpy(d_v_t_, in_buffer, N_ * 3 * sizeof(*in_buffer),
                       cudaMemcpyHostToDevice));
}

template <typename RealType>
void Context<RealType>::set_box(const RealType *in_buffer) {
  gpuErrchk(cudaMemcpy(d_box_t_, in_buffer, 3 * 3 * sizeof(*in_buffer),
                       cudaMemcpyHostToDevice));
}

template <typename RealType>
void Context<RealType>::get_x_t(RealType *out_buffer) const {
  gpuErrchk(cudaMemcpy(out_buffer, d_x_t_, N_ * 3 * sizeof(*out_buffer),
                       cudaMemcpyDeviceToHost));
}

template <typename RealType>
void Context<RealType>::get_v_t(RealType *out_buffer) const {
  gpuErrchk(cudaMemcpy(out_buffer, d_v_t_, N_ * 3 * sizeof(*out_buffer),
                       cudaMemcpyDeviceToHost));
}

template <typename RealType>
void Context<RealType>::get_box(RealType *out_buffer) const {
  gpuErrchk(cudaMemcpy(out_buffer, d_box_t_, 3 * 3 * sizeof(*out_buffer),
                       cudaMemcpyDeviceToHost));
}

template <typename RealType>
std::shared_ptr<Integrator<RealType>>
Context<RealType>::get_integrator() const {
  return intg_;
}

template <typename RealType>
std::vector<std::shared_ptr<BoundPotential<RealType>>>
Context<RealType>::get_potentials() const {
  return bps_;
}

template <typename RealType>
std::vector<std::shared_ptr<BoundPotential<RealType>>>
Context<RealType>::get_local_md_potentials() const {
  if (this->local_md_pots_ == nullptr) {
    throw std::runtime_error(
        "Local MD has not been configured, call `setup_local_md` or "
        "`multiple_steps_local` to configure.");
  }
  return this->local_md_pots_->get_potentials();
}

template <typename RealType>
std::vector<std::shared_ptr<Mover<RealType>>>
Context<RealType>::get_movers() const {
  return movers_;
}

template <typename RealType>
std::shared_ptr<MonteCarloBarostat<RealType>>
Context<RealType>::get_barostat() const {
  for (auto mover : movers_) {
    if (is_barostat<RealType>(mover)) {
      std::shared_ptr<MonteCarloBarostat<RealType>> baro =
          std::dynamic_pointer_cast<MonteCarloBarostat<RealType>>(mover);
      return baro;
    }
  }
  return nullptr;
}

template class Context<double>;
template class Context<float>;

} // namespace tmd
