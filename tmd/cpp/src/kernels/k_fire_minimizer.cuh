#include "k_fixed_point.cuh"

namespace tmd {

template <typename RealType>
__global__ void k_fire_shift(
    const int N,
    const unsigned int *atom_idxs,             // [N]
    const RealType *__restrict__ current_dt,   // [1]
    RealType *__restrict__ x_t,                // [N, 3]
    const RealType *__restrict__ v_t,          // [N, 3]
    unsigned long long *__restrict__ du_dx,    // [N, 3]
    unsigned long long *__restrict__ du_dx_old // [N, 3]
) {

    const RealType dt = *current_dt;
    const RealType dt_sq = dt * dt;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {
        int atom_idx = (atom_idxs == nullptr ? idx : atom_idxs[idx]);

        if (atom_idx < N) {
            unsigned long long force_x = du_dx[atom_idx * 3 + 0];
            unsigned long long force_y = du_dx[atom_idx * 3 + 1];
            unsigned long long force_z = du_dx[atom_idx * 3 + 2];

            // Copy over the current du_dx to the old
            du_dx_old[atom_idx * 3 + 0] = force_x;
            du_dx_old[atom_idx * 3 + 1] = force_y;
            du_dx_old[atom_idx * 3 + 2] = force_z;

            du_dx[atom_idx * 3 + 0] = 0;
            du_dx[atom_idx * 3 + 1] = 0;
            du_dx[atom_idx * 3 + 2] = 0;

            RealType v_x = v_t[atom_idx * 3 + 0];
            RealType v_y = v_t[atom_idx * 3 + 1];
            RealType v_z = v_t[atom_idx * 3 + 2];

            x_t[atom_idx * 3 + 0] += (dt * v_x) + (dt_sq * -FIXED_TO_FLOAT<RealType>(force_x));
            x_t[atom_idx * 3 + 1] += (dt * v_y) + (dt_sq * -FIXED_TO_FLOAT<RealType>(force_y));
            x_t[atom_idx * 3 + 2] += (dt * v_z) + (dt_sq * -FIXED_TO_FLOAT<RealType>(force_z));
        } else {
            // Still zero out the values, to avoid contributing to the force/velo norm
            du_dx[idx * 3 + 0] = 0;
            du_dx[idx * 3 + 1] = 0;
            du_dx[idx * 3 + 2] = 0;

            du_dx_old[idx * 3 + 0] = 0;
            du_dx_old[idx * 3 + 1] = 0;
            du_dx_old[idx * 3 + 2] = 0;
        }

        idx += gridDim.x * blockDim.x;
    }
};

template <typename RealType>
__global__ void k_fire_update_velocity_and_store_fp_force(
    const int N,
    const RealType *__restrict__ current_dt,
    const unsigned long long *__restrict__ du_dx,     // [N, 3]
    const unsigned long long *__restrict__ du_dx_old, // [N, 3]
    RealType *__restrict__ fp_force,                  // [N, 3]
    RealType *__restrict__ v_t                        // [N, 3]
) {
    const RealType half_dt = *current_dt * static_cast<RealType>(0.5);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {

        unsigned long long force_x = du_dx[idx * 3 + 0];
        unsigned long long force_y = du_dx[idx * 3 + 1];
        unsigned long long force_z = du_dx[idx * 3 + 2];

        v_t[idx * 3 + 0] += (half_dt * -FIXED_TO_FLOAT<RealType>(force_x + du_dx_old[idx * 3 + 0]));
        v_t[idx * 3 + 1] += (half_dt * -FIXED_TO_FLOAT<RealType>(force_y + du_dx_old[idx * 3 + 1]));
        v_t[idx * 3 + 2] += (half_dt * -FIXED_TO_FLOAT<RealType>(force_z + du_dx_old[idx * 3 + 2]));

        fp_force[idx * 3 + 0] = -FIXED_TO_FLOAT<RealType>(force_x);
        fp_force[idx * 3 + 1] = -FIXED_TO_FLOAT<RealType>(force_y);
        fp_force[idx * 3 + 2] = -FIXED_TO_FLOAT<RealType>(force_z);

        idx += gridDim.x * blockDim.x;
    }
};

template <typename RealType>
__global__ void k_fire_final_velocity_update(
    const int N,
    const RealType *__restrict__ force_velo_dot,  // [1]
    const RealType *__restrict__ force_norm,      // [1]
    const RealType *__restrict__ velo_norm,       // [1]
    const RealType *__restrict__ alpha,           // [1] - Can't update in kernel as would be a race condition
    const unsigned long long *__restrict__ du_dx, // [N, 3]
    RealType *__restrict__ v_t                    // [N, 3]
) {

    const RealType P = *force_velo_dot;

    const RealType local_alpha = *alpha;

    // Add 1e-6 to be consistent with tmd/_vendored/fire.py
    const RealType local_f_norm = *force_norm + static_cast<RealType>(1e-6);

    const RealType local_v_norm = *velo_norm;

    const RealType norm_ratio = local_v_norm / local_f_norm;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {

        // If the force_velo_dot product is less than zero, zero out the velocities
        if (P < 0) {
            v_t[idx * 3 + 0] = 0;
            v_t[idx * 3 + 1] = 0;
            v_t[idx * 3 + 2] = 0;
        } else {
            RealType v_x = v_t[idx * 3 + 0];
            RealType v_y = v_t[idx * 3 + 1];
            RealType v_z = v_t[idx * 3 + 2];

            v_x += local_alpha * (-FIXED_TO_FLOAT<RealType>(du_dx[idx * 3 + 0]) * norm_ratio - v_x);
            v_y += local_alpha * (-FIXED_TO_FLOAT<RealType>(du_dx[idx * 3 + 1]) * norm_ratio - v_y);
            v_z += local_alpha * (-FIXED_TO_FLOAT<RealType>(du_dx[idx * 3 + 2]) * norm_ratio - v_z);

            v_t[idx * 3 + 0] = v_x;
            v_t[idx * 3 + 1] = v_y;
            v_t[idx * 3 + 2] = v_z;
        }

        idx += gridDim.x * blockDim.x;
    }
};

template <typename RealType>
__global__ void k_fire_update_params(
    const int n_min,
    const RealType dt_max,
    const RealType f_inc,
    const RealType f_dec,
    const RealType f_alpha,
    const RealType alpha_start,
    const RealType *__restrict__ force_velo_dot, // [1]
    RealType *__restrict__ dt,                   // [1]
    RealType *__restrict__ alpha,                // [1] - Can't update in kernel as would be a race condition
    int *__restrict__ n_positive                 // [1]
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    assert(idx == 0);                            // Only runs a single thread

    const RealType P = *force_velo_dot;
    const RealType current_dt = *dt;
    if (P < 0) {
        *n_positive = 0;          // Reset the number of positive moves to zero
        *dt = current_dt * f_dec; // Decrease the timestep
        *alpha = alpha_start;     // Reset the alpha
    } else {
        const int positive_moves_so_far = *n_positive;
        *n_positive++;            // Increment in global memory
        if (positive_moves_so_far + 1 > n_min) {
            // Differs from tmd/_vendored/fire.py in that it doesn't handle P == 0 differently
            // Follows implementation detailed in https://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
            *dt = min(current_dt * f_inc, dt_max); // Increase the timestep
            *alpha *= f_alpha;                     // Scale down the alpha
        }
    }
}

} // namespace tmd
