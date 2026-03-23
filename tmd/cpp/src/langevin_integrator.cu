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

#include "constants.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "math_utils.cuh"

#include "kernels/k_integrator.cuh"

namespace tmd {
template <typename RealType>
LangevinIntegrator<RealType>::LangevinIntegrator(
    const int batch_size, const int N, const RealType *masses,
    const RealType temperature, const RealType dt, const RealType friction,
    const int seed)
    : batch_size_(batch_size), N_(N), temperature_(temperature), dt_(dt),
      friction_(friction), ca_(exp(-friction * dt)),
      d_rand_states_(batch_size_ * N_), runner_() {

  const RealType kT = static_cast<RealType>(BOLTZ) * temperature;
  const RealType ccs_adjustment = sqrt(1 - exp(-2 * friction * dt));

  std::vector<RealType> h_ccs(batch_size_ * N_);
  std::vector<RealType> h_cbs(batch_size_ * N_);
  for (int i = 0; i < batch_size_ * N_; i++) {
    h_cbs[i] = static_cast<RealType>(dt_ / masses[i]);
    h_ccs[i] = static_cast<RealType>(ccs_adjustment * sqrt(kT / masses[i]));
  }

  d_cbs_ = gpuErrchkCudaMallocAndCopy(h_cbs.data(), batch_size_ * N_);
  d_ccs_ = gpuErrchkCudaMallocAndCopy(h_ccs.data(), batch_size_ * N_);

  cudaSafeMalloc(&d_du_dx_, batch_size_ * N_ * 3 * sizeof(*d_du_dx_));

  // Only need to memset the forces to zero once at initialization;
  // k_update_forward_baoab will zero forces after every step
  gpuErrchk(cudaMemset(d_du_dx_, 0, batch_size_ * N_ * 3 * sizeof(*d_du_dx_)));

  k_initialize_curand_states<<<ceil_divide(d_rand_states_.length,
                                           DEFAULT_THREADS_PER_BLOCK),
                               DEFAULT_THREADS_PER_BLOCK, 0>>>(
      static_cast<int>(d_rand_states_.length), seed, d_rand_states_.data);
  gpuErrchk(cudaPeekAtLastError());
}
template <typename RealType>
LangevinIntegrator<RealType>::~LangevinIntegrator() {
  gpuErrchk(cudaFree(d_cbs_));
  gpuErrchk(cudaFree(d_ccs_));
  gpuErrchk(cudaFree(d_du_dx_));
}

template <typename RealType>
RealType LangevinIntegrator<RealType>::get_temperature() const {
  return this->temperature_;
}

template <typename RealType>
void LangevinIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {
  const int D = 3;

  runner_.execute_potentials(batch_size_, bps, N_, d_x_t, d_box_t,
                             d_du_dx_, // we only need the forces
                             nullptr, nullptr, stream);

  constexpr size_t tpb = DEFAULT_THREADS_PER_BLOCK;

  k_update_forward_baoab<RealType, D>
      <<<dim3(ceil_divide(N_, tpb), batch_size_), tpb, 0, stream>>>(
          batch_size_, N_, ca_, d_idxs, d_cbs_, d_ccs_, d_rand_states_.data,
          d_x_t, d_v_t, d_du_dx_, dt_);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void LangevinIntegrator<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {};

template <typename RealType>
void LangevinIntegrator<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {};

template class LangevinIntegrator<float>;
template class LangevinIntegrator<double>;

} // namespace tmd
