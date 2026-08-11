#include "constrained_langevin_integrator.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "math_utils.cuh"

#include "kernels/k_constraints.cuh"
#include "kernels/k_integrator.cuh"
#include "kernels/k_math.cuh"

namespace tmd {
template <typename RealType>
ConstrainedLangevinIntegrator<RealType>::ConstrainedLangevinIntegrator(
    const int batch_size, const int N, const RealType *masses,
    const RealType temperature, const RealType dt, const RealType friction,
    const int seed, std::shared_ptr<ConstraintGroups<RealType>> constraints)
    : LangevinIntegrator<RealType>(batch_size, N, masses, temperature, dt,
                                   friction, seed),
      constraints_(constraints) {}

template <typename RealType>
ConstrainedLangevinIntegrator<RealType>::~ConstrainedLangevinIntegrator() {}

template <typename RealType>
void ConstrainedLangevinIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {

  this->runner_.execute_potentials(this->batch_size_, bps, this->N_, d_x_t,
                                   d_box_t,
                                   this->d_du_dx_, // we only need the forces
                                   nullptr, nullptr, stream);
  constexpr int D = 3;
  constexpr int tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 intg_dim(ceil_divide(this->N_, tpb), this->batch_size_);

  // Perform the initial Kick that adjusts velocity based on forces
  k_update_forward_kick<RealType, D><<<intg_dim, tpb, 0, stream>>>(
      this->batch_size_, this->N_, d_idxs, this->d_cbs_, d_v_t, this->d_du_dx_);
  gpuErrchk(cudaPeekAtLastError());

  constraints_->constrain_velocities(this->batch_size_, this->N_, d_x_t, d_v_t,
                                     d_idxs, stream);

  k_update_forward_half_step<RealType, D><<<intg_dim, tpb, 0, stream>>>(
      this->batch_size_, this->N_, this->dt_, this->ca_, d_idxs, this->d_ccs_,
      this->d_rand_states_.data, d_x_t, d_v_t);
  gpuErrchk(cudaPeekAtLastError());

  constraints_->constrain_positions(this->batch_size_, this->N_, d_x_t, d_idxs,
                                    true, stream);

  const int num_groups = constraints_->n_groups();

  // Correct the final velocities, only if constrains are being applied
  if (num_groups > 0) {
    const dim3 constraint_dim(ceil_divide(num_groups, tpb), this->batch_size_);
    k_apply_velocity_correction<RealType, D>
        <<<constraint_dim, tpb, 0, stream>>>(
            this->batch_size_, this->N_, num_groups, this->dt_, d_idxs,
            constraints_->get_group_offsets(),
            constraints_->get_group_indices(), d_x_t,
            constraints_->get_previous_group_coords(), d_v_t);
    gpuErrchk(cudaPeekAtLastError());
  }
}

template class ConstrainedLangevinIntegrator<float>;
template class ConstrainedLangevinIntegrator<double>;

} // namespace tmd
