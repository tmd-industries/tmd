// Copyright 2026 Justin Gullingsrud
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

#include <cmath>
#include <stdexcept>

#include "constants.hpp"
#include "constrained_langevin_integrator.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_constraints.cuh"

namespace tmd {

template <typename RealType>
ConstrainedLangevinIntegrator<RealType>::ConstrainedLangevinIntegrator(
    const int batch_size, const int N, const RealType *masses,
    const RealType temperature, const RealType dt, const RealType friction,
    const int seed, std::shared_ptr<Constraints<RealType>> constraints)
    : batch_size_(batch_size), N_(N), temperature_(temperature), dt_(dt),
      friction_(friction), ca_(exp(-friction * dt)),
      d_rand_states_(batch_size_ * N_), constraints_(constraints), runner_() {

  if (!constraints_) {
    throw std::runtime_error(
        "ConstrainedLangevinIntegrator requires a Constraints object");
  }

  const RealType kT = static_cast<RealType>(BOLTZ) * temperature;
  const RealType ccs_adjustment = sqrt(1 - exp(-2 * friction * dt));

  std::vector<RealType> h_ccs(batch_size_ * N_);
  std::vector<RealType> h_cbs(batch_size_ * N_);
  std::vector<RealType> h_inv_mass(batch_size_ * N_);
  for (int i = 0; i < batch_size_ * N_; i++) {
    h_cbs[i] = static_cast<RealType>(dt_ / masses[i]);
    h_ccs[i] = static_cast<RealType>(ccs_adjustment * sqrt(kT / masses[i]));
    h_inv_mass[i] = std::isinf(masses[i])
                        ? static_cast<RealType>(0.0)
                        : static_cast<RealType>(1.0) / masses[i];
  }

  d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs.data(), batch_size_ * N_);
  d_ccs_ = gpuErrchkCudaMallocAndCopy(h_ccs.data(), batch_size_ * N_);
  d_inv_mass_ = gpuErrchkCudaMallocAndCopy(h_inv_mass.data(), batch_size_ * N_);

  cudaSafeMalloc(&d_du_dx_, batch_size_ * N_ * 3 * sizeof(*d_du_dx_));
  gpuErrchk(cudaMemset(d_du_dx_, 0, batch_size_ * N_ * 3 * sizeof(*d_du_dx_)));

  cudaSafeMalloc(&d_x_ref_, batch_size_ * N_ * 3 * sizeof(*d_x_ref_));

  k_initialize_curand_states<<<ceil_divide(d_rand_states_.length,
                                           DEFAULT_THREADS_PER_BLOCK),
                               DEFAULT_THREADS_PER_BLOCK, 0>>>(
      static_cast<int>(d_rand_states_.length), seed, d_rand_states_.data);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
ConstrainedLangevinIntegrator<RealType>::~ConstrainedLangevinIntegrator() {
  gpuErrchk(cudaFree(d_cbs_));
  gpuErrchk(cudaFree(d_ccs_));
  gpuErrchk(cudaFree(d_inv_mass_));
  gpuErrchk(cudaFree(d_x_ref_));
  gpuErrchk(cudaFree(d_du_dx_));
}

template <typename RealType>
RealType ConstrainedLangevinIntegrator<RealType>::get_temperature() const {
  return this->temperature_;
}

template <typename RealType>
void ConstrainedLangevinIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {
  const int D = 3;

  // d_idxs (non-null under local MD) is the free-index buffer: atoms whose
  // entry equals N are frozen and skipped by the sub-step kernels and treated
  // as infinite-mass anchors by SHAKE/RATTLE.

  constexpr size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 atom_grid(ceil_divide(N_, tpb), batch_size_);
  const RealType half_dt = static_cast<RealType>(0.5) * dt_;
  const size_t coords_size = batch_size_ * N_ * D * sizeof(*d_x_t);

  // Force evaluation at the current positions.
  runner_.execute_potentials(batch_size_, bps, N_, d_x_t, d_box_t, d_du_dx_,
                             nullptr, nullptr, stream);

  // B: full velocity kick, then project velocities onto the constraint
  // tangent space.
  k_constrained_kick<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, d_cbs_, d_idxs, d_v_t, d_du_dx_);
  gpuErrchk(cudaPeekAtLastError());
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);

  // A (first half drift): save pre-drift positions, drift, then SHAKE and apply
  // the constraint-consistent velocity correction. A subsequent RATTLE removes
  // any residual velocity component normal to the constraints.
  gpuErrchk(cudaMemcpyAsync(d_x_ref_, d_x_t, coords_size,
                            cudaMemcpyDeviceToDevice, stream));
  k_constrained_drift<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, d_v_t, d_idxs, d_x_t, half_dt);
  gpuErrchk(cudaPeekAtLastError());
  constraints_->apply_position_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_ref_, d_x_t, d_v_t, half_dt,
                                           /*update_velocity=*/true, stream);
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);

  // O: Ornstein-Uhlenbeck velocity update, then RATTLE.
  k_constrained_ornstein<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, ca_, d_ccs_, d_idxs, d_rand_states_.data, d_v_t);
  gpuErrchk(cudaPeekAtLastError());
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);

  // A (second half drift): save, drift, SHAKE, velocity correction, RATTLE.
  gpuErrchk(cudaMemcpyAsync(d_x_ref_, d_x_t, coords_size,
                            cudaMemcpyDeviceToDevice, stream));
  k_constrained_drift<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, d_v_t, d_idxs, d_x_t, half_dt);
  gpuErrchk(cudaPeekAtLastError());
  constraints_->apply_position_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_ref_, d_x_t, d_v_t, half_dt,
                                           /*update_velocity=*/true, stream);
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);
}

template <typename RealType>
void ConstrainedLangevinIntegrator<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {
  const int D = 3;
  const size_t coords_size = batch_size_ * N_ * D * sizeof(*d_x_t);

  // Ensure the initial state satisfies the constraints: project positions onto
  // the manifold (without modifying velocities) and then remove any velocity
  // component along the constraints. d_idxs (non-null under local MD) freezes
  // atoms whose entry equals N.
  gpuErrchk(cudaMemcpyAsync(d_x_ref_, d_x_t, coords_size,
                            cudaMemcpyDeviceToDevice, stream));
  constraints_->apply_position_constraints(
      batch_size_, N_, d_inv_mass_, d_idxs, d_x_ref_, d_x_t, d_v_t,
      static_cast<RealType>(1.0), /*update_velocity=*/false, stream);
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);
}

template <typename RealType>
void ConstrainedLangevinIntegrator<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {}

template class ConstrainedLangevinIntegrator<float>;
template class ConstrainedLangevinIntegrator<double>;

} // namespace tmd
