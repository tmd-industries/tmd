#include "constants.hpp"
#include "fire_minimizer.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"

#include <cub/cub.cuh>

#include "k_fire_minimizer.cuh"

namespace tmd {
template <typename RealType>
FireMinimizer<RealType>::FireMinimizer(
    const int N,
    const int n_min,
    const RealType dt_start,
    const RealType dt_max,
    const RealType f_inc,
    const RealType f_dec,
    const RealType alpha_start,
    const RealType f_alpha)
    : N_(N), n_min_(n_min), thread_blocks_(ceil_divide(N, DEFAULT_THREADS_PER_BLOCK)), dt_max_(dt_max),
      f_alpha_(f_alpha), f_inc_(f_inc), f_dec_(f_dec), alpha_start_(alpha_start), d_dt_(1), d_alpha_(1), d_v_norm_(1),
      d_f_norm_(1), n_positive_(1), d_forces_(N_ * 3), d_force_velo_dot_(1), d_du_dx_old_(N * 3), d_du_dx_(N * 3),
      runner_(), initialized_(false) {

    d_dt_.copy_from(&dt_start);
    d_alpha_.copy_from(&alpha_start_);

    gpuErrchk(cudaMemset(n_positive_.data, 0, sizeof(*n_positive_.data)));

    cublasErrchk(cublasCreate(&cublas_handle_));
    // Need to tell cublas that the output pointers are device pointers
    cublasErrchk(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
}
template <typename RealType> FireMinimizer<RealType>::~FireMinimizer() { cublasErrchk(cublasDestroy(cublas_handle_)); }

template <typename RealType>
void FireMinimizer<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t,
    RealType *d_v_t,
    RealType *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    const size_t tpb = DEFAULT_THREADS_PER_BLOCK;

    cublasErrchk(cublasSetStream(cublas_handle_, stream));

    // Copy the d_du_dx values to old and reset d_du_dx to zero
    k_fire_shift<RealType>
        <<<thread_blocks_, tpb, 0, stream>>>(N_, d_idxs, d_dt_.data, d_x_t, d_v_t, d_du_dx_.data, d_du_dx_old_.data);
    gpuErrchk(cudaPeekAtLastError());

    runner_.execute_potentials(
        bps,
        N_,
        d_x_t,
        d_box_t,
        d_du_dx_.data, // we only need the forces
        nullptr,
        nullptr,
        stream);

    // Update velocities
    k_fire_update_velocity_and_store_fp_force<RealType>
        <<<thread_blocks_, tpb, 0, stream>>>(N_, d_dt_.data, d_du_dx_.data, d_du_dx_old_.data, d_forces_.data, d_v_t);
    gpuErrchk(cudaPeekAtLastError());

    // Compute the force norm and the velo norm
    cublasErrchk(templateCublasNorm2(cublas_handle_, N_ * 3, d_v_t, 1, d_v_norm_.data));
    cublasErrchk(templateCublasNorm2(cublas_handle_, N_ * 3, d_forces_.data, 1, d_f_norm_.data));
    // Compute the dot product of forces/velocities
    cublasErrchk(templateCublasDot(cublas_handle_, N_ * 3, d_forces_.data, 1, d_v_t, 1, d_force_velo_dot_.data));

    // Update the velocities from norms/dot
    k_fire_final_velocity_update<RealType><<<thread_blocks_, tpb, 0, stream>>>(
        N_, d_force_velo_dot_.data, d_f_norm_.data, d_v_norm_.data, d_alpha_.data, d_du_dx_.data, d_v_t);
    gpuErrchk(cudaPeekAtLastError());

    // Update the params
    k_fire_update_params<<<1, 1, 0, stream>>>(
        n_min_,
        dt_max_,
        f_inc_,
        f_dec_,
        f_alpha_,
        alpha_start_,
        d_force_velo_dot_.data,
        d_dt_.data,
        d_alpha_.data,
        n_positive_.data);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void FireMinimizer<RealType>::initialize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t,
    RealType *d_v_t,
    RealType *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {

    if (initialized_) {
        throw std::runtime_error("initialized twice");
    }

    gpuErrchk(cudaMemsetAsync(d_du_dx_.data, 0, d_du_dx_.size(), stream));

    runner_.execute_potentials(
        bps,
        N_,
        d_x_t,
        d_box_t,
        d_du_dx_.data, // we only need the forces
        nullptr,
        nullptr,
        stream);
    initialized_ = true;
};

template <typename RealType>
void FireMinimizer<RealType>::finalize(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t,
    RealType *d_v_t,
    RealType *d_box_t,
    unsigned int *d_idxs,
    cudaStream_t stream) {
    if (!initialized_) {
        throw std::runtime_error("not initialized");
    }
    initialized_ = false;
};

} // namespace tmd
