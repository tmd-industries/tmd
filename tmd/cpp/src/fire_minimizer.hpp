#pragma once

#include "cublas_v2.h"
#include <vector>

#include "bound_potential.hpp"
#include "device_buffer.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace tmd {

template <typename RealType> class FireMinimizer : public Integrator<RealType> {

private:
    const int N_;
    const int n_min_;
    const int thread_blocks_;
    const RealType dt_max_;
    const RealType f_alpha_;
    const RealType f_inc_;
    const RealType f_dec_;
    const RealType alpha_start_;

    DeviceBuffer<RealType> d_dt_;
    DeviceBuffer<RealType> d_alpha_;
    DeviceBuffer<RealType> d_v_norm_;
    DeviceBuffer<RealType> d_f_norm_;
    DeviceBuffer<int> n_positive_;            // Number of steps taken in the right direction

    DeviceBuffer<RealType> d_forces_;         // RealType version of the forces
    DeviceBuffer<RealType> d_force_velo_dot_; // The actual P value
    DeviceBuffer<unsigned long long> d_du_dx_old_;
    DeviceBuffer<unsigned long long> d_du_dx_;
    StreamedPotentialRunner<RealType> runner_;

    bool initialized_;

    cublasHandle_t cublas_handle_;

public:
    FireMinimizer(
        const int N,
        const int n_min,
        const RealType dt_start,
        const RealType dt_max,
        const RealType f_inc,
        const RealType f_dec,
        const RealType alpha_start,
        const RealType f_alpha);

    virtual ~FireMinimizer();

    virtual void step_fwd(
        std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
        RealType *d_x_t,
        RealType *d_v_t,
        RealType *d_box_t_,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void initialize(
        std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
        RealType *d_x_t,
        RealType *d_v_t,
        RealType *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;

    virtual void finalize(
        std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
        RealType *d_x_t,
        RealType *d_v_t,
        RealType *d_box_t,
        unsigned int *d_idxs,
        cudaStream_t stream) override;
};

template class FireMinimizer<double>;
template class FireMinimizer<float>;

} // namespace tmd
