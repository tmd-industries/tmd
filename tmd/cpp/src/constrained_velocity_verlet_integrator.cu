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

#include "constrained_velocity_verlet_integrator.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_constraints.cuh"

namespace tmd {

template <typename RealType>
ConstrainedVelocityVerletIntegrator<RealType>::ConstrainedVelocityVerletIntegrator(
    const int batch_size, const int N, const RealType *masses,
    const RealType dt, std::shared_ptr<Constraints<RealType>> constraints)
    : batch_size_(batch_size), N_(N), dt_(dt), constraints_(constraints),
      runner_() {

  if (!constraints_) {
    throw std::runtime_error(
        "ConstrainedVelocityVerletIntegrator requires a Constraints object");
  }

  std::vector<RealType> h_half_cbs(batch_size_ * N_);
  std::vector<RealType> h_inv_mass(batch_size_ * N_);
  for (int i = 0; i < batch_size_ * N_; i++) {
    h_half_cbs[i] = static_cast<RealType>(static_cast<RealType>(0.5) * dt_ / masses[i]);
    h_inv_mass[i] = std::isinf(masses[i])
                        ? static_cast<RealType>(0.0)
                        : static_cast<RealType>(1.0) / masses[i];
  }

  d_half_cbs_ = gpuErrchkCudaMallocAndCopy(h_half_cbs.data(), batch_size_ * N_);
  d_inv_mass_ = gpuErrchkCudaMallocAndCopy(h_inv_mass.data(), batch_size_ * N_);

  cudaSafeMalloc(&d_du_dx_, batch_size_ * N_ * 3 * sizeof(*d_du_dx_));
  gpuErrchk(cudaMemset(d_du_dx_, 0, batch_size_ * N_ * 3 * sizeof(*d_du_dx_)));

  cudaSafeMalloc(&d_du_dx_cached_, batch_size_ * N_ * 3 * sizeof(*d_du_dx_cached_));
  gpuErrchk(
      cudaMemset(d_du_dx_cached_, 0, batch_size_ * N_ * 3 * sizeof(*d_du_dx_cached_)));

  cudaSafeMalloc(&d_x_ref_, batch_size_ * N_ * 3 * sizeof(*d_x_ref_));
}

template <typename RealType>
ConstrainedVelocityVerletIntegrator<RealType>::~ConstrainedVelocityVerletIntegrator() {
  gpuErrchk(cudaFree(d_half_cbs_));
  gpuErrchk(cudaFree(d_inv_mass_));
  gpuErrchk(cudaFree(d_x_ref_));
  gpuErrchk(cudaFree(d_du_dx_));
  gpuErrchk(cudaFree(d_du_dx_cached_));
}

template <typename RealType>
void ConstrainedVelocityVerletIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {
  const int D = 3;

  constexpr size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 atom_grid(ceil_divide(N_, tpb), batch_size_);
  const size_t coords_size = batch_size_ * N_ * D * sizeof(*d_x_t);
  const size_t force_size = batch_size_ * N_ * D * sizeof(*d_du_dx_);

  // First half kick using the force cached from the previous step (or from
  // initialize). k_constrained_kick negates the stored du/dx and zeroes the
  // cache after use.
  k_constrained_kick<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, d_half_cbs_, d_idxs, d_v_t, d_du_dx_cached_);
  gpuErrchk(cudaPeekAtLastError());

  // Full drift, then SHAKE. The position projection's constraint-consistent
  // velocity update (x_constrained - x_ref) / dt is exactly the RATTLE
  // position-stage velocity v(t + dt/2).
  gpuErrchk(cudaMemcpyAsync(d_x_ref_, d_x_t, coords_size,
                            cudaMemcpyDeviceToDevice, stream));
  k_constrained_drift<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, d_v_t, d_idxs, d_x_t, dt_);
  gpuErrchk(cudaPeekAtLastError());
  constraints_->apply_position_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_ref_, d_x_t, d_v_t, dt_,
                                           /*update_velocity=*/true, stream);

  // Single force evaluation at the new positions.
  gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, force_size, stream));
  runner_.execute_potentials(batch_size_, bps, N_, d_x_t, d_box_t, d_du_dx_,
                             nullptr, nullptr, stream);

  // Cache f(x_{t+dt}) for the next step's first half kick before the second
  // kick consumes (and zeroes) d_du_dx_.
  gpuErrchk(cudaMemcpyAsync(d_du_dx_cached_, d_du_dx_, force_size,
                            cudaMemcpyDeviceToDevice, stream));

  // Second half kick, then RATTLE velocity projection.
  k_constrained_kick<RealType, D><<<atom_grid, tpb, 0, stream>>>(
      batch_size_, N_, d_half_cbs_, d_idxs, d_v_t, d_du_dx_);
  gpuErrchk(cudaPeekAtLastError());
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);
}

template <typename RealType>
void ConstrainedVelocityVerletIntegrator<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {
  const int D = 3;
  const size_t coords_size = batch_size_ * N_ * D * sizeof(*d_x_t);
  const size_t force_size = batch_size_ * N_ * D * sizeof(*d_du_dx_);

  // Project the initial state onto the constraint manifold (positions without
  // altering velocities, then remove any velocity component along the
  // constraints).
  gpuErrchk(cudaMemcpyAsync(d_x_ref_, d_x_t, coords_size,
                            cudaMemcpyDeviceToDevice, stream));
  constraints_->apply_position_constraints(
      batch_size_, N_, d_inv_mass_, d_idxs, d_x_ref_, d_x_t, d_v_t,
      static_cast<RealType>(1.0), /*update_velocity=*/false, stream);
  constraints_->apply_velocity_constraints(batch_size_, N_, d_inv_mass_, d_idxs,
                                           d_x_t, d_v_t, stream);

  // Seed the force cache with f(x_0) for the first step's first half kick.
  gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, force_size, stream));
  runner_.execute_potentials(batch_size_, bps, N_, d_x_t, d_box_t, d_du_dx_,
                             nullptr, nullptr, stream);
  gpuErrchk(cudaMemcpyAsync(d_du_dx_cached_, d_du_dx_, force_size,
                            cudaMemcpyDeviceToDevice, stream));
}

template <typename RealType>
void ConstrainedVelocityVerletIntegrator<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {}

template class ConstrainedVelocityVerletIntegrator<float>;
template class ConstrainedVelocityVerletIntegrator<double>;

} // namespace tmd
