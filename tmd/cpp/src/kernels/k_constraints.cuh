#pragma once
#include <cassert>
#include <cmath>

namespace tmd {

/**
 * CUDA kernel applying the SHAKE algorithm to constrain bond lengths
 * within atom groups. Each group has one anchor atom and up to
 * MAX_GROUP_SIZE hydrogen atoms.
 *
 * @tparam RealType Floating-point type (e.g. float, double).
 * @tparam D Number of dimensions
 * @tparam MAX_GROUP_SIZE Max number of non-anchor atoms per group.
 * @param num_systems Number of independent systems processed in parallel.
 * @param N Number of atoms per system.
 * @param iterations Maximum Newton-Raphson iterations per group.
 * @param n_groups Total number of groups across all systems.
 * @param tolerance Tolerance for distance check.
 * @param idxs Optional [num_systems x N] array; if non-null, entries < N
 *             mark frozen atoms.
 * @param group_offsets [n_groups + 1] start/end indices into group_indices.
 * @param group_indices Flattened list of atom indices per group; first entry
 *                      is the anchor.
 * @param distances Target bond lengths [total atoms in all groups].
 * @param inv_masses [num_systems x N] inverse atomic masses.
 * @param x_t [num_systems x N x 3] in/out position array (modified in-place).
 * @param x_t_copy [n_groups, 3] a copy of the input positions. Useful for
 * correcting velocities down stream. May be null
 */
template <typename RealType, int D, int MAX_GROUP_SIZE>
__global__ void
k_apply_shake(const int num_systems, const int N, const int iterations,
              const int n_groups, const RealType tolerance,
              const unsigned int *__restrict__ idxs, // [num_systems, N]
              const int *__restrict__ group_offsets, // [n_groups + 1]
              const int *__restrict__ group_indices,
              const int *__restrict__ distance_offsets, // [n_groups + 1]
              const RealType *__restrict__ distances,
              const RealType *__restrict__ inv_masses, // [num_systems, N]
              RealType *__restrict__ x_t,              // [num_systems, N, D]
              RealType *__restrict__ x_t_copy          // [atoms_in_group, 3]
) {

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  // Will be zero if aren't copying out initial coordinates
  const int atoms_in_constraints =
      x_t_copy != nullptr ? group_offsets[n_groups] : 0;

  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < n_groups) {
    const int dist_start = distance_offsets[kernel_idx];
    const int offset_start = group_offsets[kernel_idx];
    const int offset_end = group_offsets[kernel_idx + 1];
    const int anchor_atom = group_indices[offset_start];

    const int n_hydrogens = (offset_end - offset_start) - 1;
    assert(n_hydrogens <= MAX_GROUP_SIZE);

    RealType anchor_x = x_t[system_idx * N * D + anchor_atom * D + 0];
    RealType anchor_y = x_t[system_idx * N * D + anchor_atom * D + 1];
    RealType anchor_z = x_t[system_idx * N * D + anchor_atom * D + 2];
    if (atoms_in_constraints > 0) {
      x_t_copy[system_idx * atoms_in_constraints * D + offset_start * D + 0] =
          anchor_x;
      x_t_copy[system_idx * atoms_in_constraints * D + offset_start * D + 1] =
          anchor_y;
      x_t_copy[system_idx * atoms_in_constraints * D + offset_start * D + 2] =
          anchor_z;
    }

    const bool frozen_anchor =
        idxs == nullptr ? false : idxs[system_idx * N + anchor_atom] >= N;

    const RealType inv_anchor_mass =
        !frozen_anchor ? inv_masses[system_idx * N + anchor_atom]
                       : static_cast<RealType>(0.0);

    // Setup the reference distances using the initial coordinates
    RealType ref_deltas[MAX_GROUP_SIZE][D];
    for (int j = 0; j < n_hydrogens; j++) {
      int atom_idx = group_indices[offset_start + j + 1];
      const RealType atom_x = x_t[system_idx * N * D + atom_idx * D + 0];
      const RealType atom_y = x_t[system_idx * N * D + atom_idx * D + 1];
      const RealType atom_z = x_t[system_idx * N * D + atom_idx * D + 2];
      ref_deltas[j][0] = anchor_x - atom_x;
      ref_deltas[j][1] = anchor_y - atom_y;
      ref_deltas[j][2] = anchor_z - atom_z;

      if (atoms_in_constraints > 0) {
        x_t_copy[system_idx * atoms_in_constraints * D +
                 (offset_start + j + 1) * D + 0] = atom_x;
        x_t_copy[system_idx * atoms_in_constraints * D +
                 (offset_start + j + 1) * D + 1] = atom_y;
        x_t_copy[system_idx * atoms_in_constraints * D +
                 (offset_start + j + 1) * D + 2] = atom_z;
      }
    }

    for (int i = 0; i < iterations; i++) {
      bool converged = true;
      for (int j = 0; j < n_hydrogens; j++) {
        int atom_idx = group_indices[offset_start + j + 1];
        const bool frozen_atom =
            idxs == nullptr ? false : idxs[system_idx * N + atom_idx] >= N;
        if (frozen_atom && frozen_anchor) {
          continue;
        }

        const RealType inv_atom_mass =
            !frozen_atom ? inv_masses[system_idx * N + atom_idx] : 0.0;

        const RealType target_dist2 =
            distances[dist_start + j] * distances[dist_start + j];

        RealType delta_x =
            anchor_x - x_t[system_idx * N * D + atom_idx * D + 0];
        RealType delta_y =
            anchor_y - x_t[system_idx * N * D + atom_idx * D + 1];
        RealType delta_z =
            anchor_z - x_t[system_idx * N * D + atom_idx * D + 2];

        RealType dist2 =
            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;

        RealType diff = dist2 - target_dist2;

        if (fabs(diff) < tolerance * target_dist2) {
          continue;
        }

        RealType delta_dot_ref = (delta_x * ref_deltas[j][0]) +
                                 (delta_y * ref_deltas[j][1]) +
                                 (delta_z * ref_deltas[j][2]);
        RealType denom = static_cast<RealType>(2.0) *
                         (inv_anchor_mass + inv_atom_mass) * delta_dot_ref;
        // If the denominator is very small, skip
        if (fabs(denom) < static_cast<RealType>(1e-8)) {
          continue;
        }
        converged = false;
        RealType grad = diff / denom;

        anchor_x -= inv_anchor_mass * grad * ref_deltas[j][0];
        anchor_y -= inv_anchor_mass * grad * ref_deltas[j][1];
        anchor_z -= inv_anchor_mass * grad * ref_deltas[j][2];

        x_t[system_idx * N * D + atom_idx * D + 0] +=
            inv_atom_mass * grad * ref_deltas[j][0];
        x_t[system_idx * N * D + atom_idx * D + 1] +=
            inv_atom_mass * grad * ref_deltas[j][1];
        x_t[system_idx * N * D + atom_idx * D + 2] +=
            inv_atom_mass * grad * ref_deltas[j][2];
      }
      if (converged) {
        break;
      }
    }
    // Write out the anchor coords
    x_t[system_idx * N * D + anchor_atom * D + 0] = anchor_x;
    x_t[system_idx * N * D + anchor_atom * D + 1] = anchor_y;
    x_t[system_idx * N * D + anchor_atom * D + 2] = anchor_z;

    kernel_idx += gridDim.x * blockDim.x;
  }
}

/**
 * CUDA kernel applying the RATTLE algorithm to constrain velocities
 * so that bond-length constraints are maintained. Each group has one
 * anchor atom and up to MAX_GROUP_SIZE hydrogen atoms.
 *
 * @tparam RealType Floating-point type (e.g. float, double).
 * @tparam D Number of dimensions
 * @tparam MAX_GROUP_SIZE Max number of non-anchor atoms per group.
 * @param num_systems Number of independent systems processed in parallel.
 * @param N Number of atoms per system.
 * @param iterations Maximum Newton-Raphson iterations per group.
 * @param n_groups Total number of groups across all systems.
 * @param tolerance Tolerance for velocity constraint check.
 * @param idxs Optional [num_systems x N] array; if non-null, entries < N
 *             mark frozen atoms.
 * @param group_offsets [n_groups + 1] start/end indices into group_indices.
 * @param group_indices Flattened list of atom indices per group; first entry
 *                      is the anchor.
 * @param inv_masses [num_systems x N] inverse atomic masses.
 * @param x_t [num_systems x N x D] position array (read-only).
 * @param v_t [num_systems x N x D] in/out velocity array (modified in-place).
 */
template <typename RealType, int D, int MAX_GROUP_SIZE>
__global__ void
k_apply_rattle(const int num_systems, const int N, const int iterations,
               const int n_groups, const RealType tolerance,
               const unsigned int *__restrict__ idxs, // [num_systems, N]
               const int *__restrict__ group_offsets, // [n_groups + 1]
               const int *__restrict__ group_indices,
               const RealType *__restrict__ inv_masses, // [num_systems, N]
               const RealType *__restrict__ x_t,        // [num_systems, N, D]
               RealType *__restrict__ v_t               // [num_systems, N, D]
) {

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < n_groups) {
    const int offset_start = group_offsets[kernel_idx];
    const int offset_end = group_offsets[kernel_idx + 1];
    const int anchor_atom = group_indices[offset_start];

    const int n_hydrogens = (offset_end - offset_start) - 1;

    const RealType anchor_x = x_t[system_idx * N * D + anchor_atom * D + 0];
    const RealType anchor_y = x_t[system_idx * N * D + anchor_atom * D + 1];
    const RealType anchor_z = x_t[system_idx * N * D + anchor_atom * D + 2];

    RealType anchor_vx = v_t[system_idx * N * D + anchor_atom * D + 0];
    RealType anchor_vy = v_t[system_idx * N * D + anchor_atom * D + 1];
    RealType anchor_vz = v_t[system_idx * N * D + anchor_atom * D + 2];

    const bool frozen_anchor =
        idxs == nullptr ? false : idxs[system_idx * N + anchor_atom] >= N;

    const RealType inv_anchor_mass =
        !frozen_anchor ? inv_masses[system_idx * N + anchor_atom]
                       : static_cast<RealType>(0.0);

    for (int i = 0; i < iterations; i++) {
      bool converged = true;
      for (int j = 0; j < n_hydrogens; j++) {
        int atom_idx = group_indices[offset_start + j + 1];
        const bool frozen_atom =
            idxs == nullptr ? false : idxs[system_idx * N + atom_idx] >= N;
        if (frozen_atom && frozen_anchor) {
          continue;
        }
        const RealType inv_atom_mass =
            !frozen_atom ? inv_masses[system_idx * N + atom_idx] : 0.0;

        RealType delta_x =
            anchor_x - x_t[system_idx * N * D + atom_idx * D + 0];
        RealType delta_y =
            anchor_y - x_t[system_idx * N * D + atom_idx * D + 1];
        RealType delta_z =
            anchor_z - x_t[system_idx * N * D + atom_idx * D + 2];

        RealType dist2 =
            (delta_x * delta_x) + (delta_y * delta_y) + (delta_z * delta_z);

        RealType denom = dist2 * (inv_anchor_mass + inv_atom_mass);

        if (fabs(denom) < static_cast<RealType>(1e-8)) {
          continue;
        }

        RealType delta_vx =
            anchor_vx - v_t[system_idx * N * D + atom_idx * D + 0];
        RealType delta_vy =
            anchor_vy - v_t[system_idx * N * D + atom_idx * D + 1];
        RealType delta_vz =
            anchor_vz - v_t[system_idx * N * D + atom_idx * D + 2];

        // Velocity constraint: (v_anchor - v_atom) . delta_r = 0
        RealType rv =
            (delta_x * delta_vx) + (delta_y * delta_vy) + (delta_z * delta_vz);

        if (fabs(rv) < tolerance) {
          continue;
        }

        converged = false;
        RealType grad = rv / denom;

        anchor_vx -= inv_anchor_mass * grad * delta_x;
        anchor_vy -= inv_anchor_mass * grad * delta_y;
        anchor_vz -= inv_anchor_mass * grad * delta_z;

        v_t[system_idx * N * D + atom_idx * D + 0] +=
            inv_atom_mass * grad * delta_x;
        v_t[system_idx * N * D + atom_idx * D + 1] +=
            inv_atom_mass * grad * delta_y;
        v_t[system_idx * N * D + atom_idx * D + 2] +=
            inv_atom_mass * grad * delta_z;
      }
      if (converged) {
        break;
      }
    }
    // Write out the anchor velocities
    v_t[system_idx * N * D + anchor_atom * D + 0] = anchor_vx;
    v_t[system_idx * N * D + anchor_atom * D + 1] = anchor_vy;
    v_t[system_idx * N * D + anchor_atom * D + 2] = anchor_vz;

    kernel_idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
