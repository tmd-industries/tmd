#include "../fixed_point.hpp"
#include "k_fixed_point.cuh"
#include "k_flat_bottom_bond.cuh"

namespace tmd {

template <typename RealType, bool COMPUTE_U, bool COMPUTE_DU_DX,
          bool COMPUTE_DU_DP>
void __global__ k_flat_bottom_restraint(
    const int N, // number of atoms
    const int R, // number of restraints
    const RealType *__restrict__ coords, const RealType *__restrict__ box,
    const RealType *__restrict__ params,           // [R, 3]
    const int *__restrict__ atom_idxs,             // [R]
    const int *__restrict__ system_idxs,           // [R]
    const RealType *__restrict__ restraint_coords, // [R, 3]
    unsigned long long *__restrict__ du_dx,
    unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u) {

  // which restraint
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < R) {
    const int system_idx = system_idxs[idx];
    const int coord_offset = system_idx * N;
    const int box_offset = system_idx * 9;

    const int atom_idx = atom_idxs[idx] + coord_offset;

    // look up params
    constexpr int num_params = 3;
    int params_idx = idx * num_params;
    int k_idx = params_idx + 0;
    int rmin_idx = params_idx + 1;
    int rmax_idx = params_idx + 2;

    RealType k = params[k_idx];
    RealType rmin = params[rmin_idx];
    RealType rmax = params[rmax_idx];

    // compute common subexpressions involving distance, displacements
    RealType dx[3];
    RealType r2 = 0;
#pragma unroll
    for (int d = 0; d < 3; d++) {
      RealType delta = coords[atom_idx * 3 + d] - restraint_coords[idx * 3 + d];
      delta -= box[box_offset + d * 3 + d] *
               nearbyint(delta / box[box_offset + d * 3 + d]);
      dx[d] = delta;
      r2 += delta * delta;
    }
    RealType r = sqrt(r2);

    // branches -> masks
    RealType r_gt_rmax = static_cast<RealType>(r > rmax);
    RealType r_lt_rmin = static_cast<RealType>(r < rmin);

    if (COMPUTE_U) {
      RealType u_real = compute_flat_bottom_energy<RealType>(k, r, rmin, rmax);

      // Always set the energy buffer value to ensure buffer is initialized
      u[idx] = FLOAT_TO_FIXED_ENERGY<RealType>(u_real);
    }
    if (COMPUTE_DU_DP || COMPUTE_DU_DX) {
      if (r_gt_rmax != 0 || r_lt_rmin != 0) {
        RealType d_r_min = r - rmin;
        RealType d_r_max = r - rmax;
        RealType d_rmin_3 = d_r_min * d_r_min * d_r_min;
        RealType d_rmax_3 = d_r_max * d_r_max * d_r_max;
        if (COMPUTE_DU_DP) {
          // compute parameter derivatives
          RealType du_dk_real =
              (r_gt_rmax *
                   ((d_rmax_3 * d_r_max) * static_cast<RealType>(0.25)) +
               (r_lt_rmin *
                ((d_rmin_3 * d_r_min) * static_cast<RealType>(0.25))));
          RealType du_drmin_real = r_lt_rmin * (-k * d_rmin_3);
          RealType du_drmax_real = r_gt_rmax * (-k * d_rmax_3);

          // cast float -> fixed
          unsigned long long du_dk =
              FLOAT_TO_FIXED_BONDED<RealType>(du_dk_real);
          unsigned long long du_drmin =
              FLOAT_TO_FIXED_BONDED<RealType>(du_drmin_real);
          unsigned long long du_drmax =
              FLOAT_TO_FIXED_BONDED<RealType>(du_drmax_real);

          // increment du_dp array
          atomicAdd(du_dp + k_idx, du_dk);
          atomicAdd(du_dp + rmin_idx, du_drmin);
          atomicAdd(du_dp + rmax_idx, du_drmax);
        }

        if (COMPUTE_DU_DX) {
          RealType du_dr =
              k * ((r_gt_rmax * d_rmax_3) + (r_lt_rmin * d_rmin_3));
          RealType inv_r = 1 / r;
#pragma unroll
          for (int d = 0; d < 3; d++) {
            // compute du/dcoords
            RealType du_dsrc_real = du_dr * dx[d] * inv_r;

            // cast float -> fixed
            unsigned long long du_dsrc =
                FLOAT_TO_FIXED_BONDED<RealType>(du_dsrc_real);

            // increment du_dx array
            atomicAdd(du_dx + atom_idx * 3 + d, du_dsrc);
          }
        }
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
