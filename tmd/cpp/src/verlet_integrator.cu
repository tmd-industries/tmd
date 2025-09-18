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

#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include "verlet_integrator.hpp"

#include "kernels/k_integrator.cuh"

namespace tmd {

template <typename RealType>
VelocityVerletIntegrator<RealType>::VelocityVerletIntegrator(
    int N, RealType dt, const RealType *h_cbs)
    : N_(N), dt_(dt), initialized_(false), runner_() {

  d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs, N);
  cudaSafeMalloc(&d_du_dx_, N * 3 * sizeof(*d_du_dx_));
}

template <typename RealType>
VelocityVerletIntegrator<RealType>::~VelocityVerletIntegrator() {
  gpuErrchk(cudaFree(d_cbs_));
  gpuErrchk(cudaFree(d_du_dx_));
}

template <typename RealType>
void VelocityVerletIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {

  gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

  const int D = 3;
  size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  size_t n_blocks = ceil_divide(N_, tpb);
  dim3 dimGrid_dx(n_blocks, D);
  for (int i = 0; i < bps.size(); i++) {
    bps[i]->execute_device(N_, d_x_t, d_box_t, d_du_dx_, nullptr, nullptr,
                           stream);
  }
  update_forward_velocity_verlet<RealType><<<dimGrid_dx, tpb, 0, stream>>>(
      N_, D, d_idxs, d_cbs_, d_x_t, d_v_t, d_du_dx_, dt_);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void VelocityVerletIntegrator<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {

  if (initialized_) {
    throw std::runtime_error("initialized twice");
  }
  gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

  const int D = 3;
  size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  size_t n_blocks = ceil_divide(N_, tpb);
  dim3 dimGrid_dx(n_blocks, D);

  runner_.execute_potentials(bps, N_, d_x_t, d_box_t,
                             d_du_dx_, // we only need the forces
                             nullptr, nullptr, stream);

  half_step_velocity_verlet<RealType, true><<<dimGrid_dx, tpb, 0, stream>>>(
      N_, D, d_idxs, d_cbs_, d_x_t, d_v_t, d_du_dx_, dt_);
  gpuErrchk(cudaPeekAtLastError());
  initialized_ = true;
};

template <typename RealType>
void VelocityVerletIntegrator<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {

  if (!initialized_) {
    throw std::runtime_error("not initialized");
  }
  gpuErrchk(cudaMemsetAsync(d_du_dx_, 0, N_ * 3 * sizeof(*d_du_dx_), stream));

  const int D = 3;
  size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  size_t n_blocks = ceil_divide(N_, tpb);
  dim3 dimGrid_dx(n_blocks, D);

  for (int i = 0; i < bps.size(); i++) {
    bps[i]->execute_device(N_, d_x_t, d_box_t,
                           d_du_dx_, // we only need the forces
                           nullptr, nullptr, stream);
  }
  half_step_velocity_verlet<RealType, false><<<dimGrid_dx, tpb, 0, stream>>>(
      N_, D, d_idxs, d_cbs_, d_x_t, d_v_t, d_du_dx_, dt_);
  gpuErrchk(cudaPeekAtLastError());
  initialized_ = false;
};

template class VelocityVerletIntegrator<double>;
template class VelocityVerletIntegrator<float>;

} // namespace tmd
