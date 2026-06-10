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

#pragma once

#include <cstdint>

#include "../gpu_utils.cuh"
#include "k_fixed_point.cuh"

namespace tmd {

// Maximum number of atoms in a single constraint cluster. A cluster is a heavy
// atom together with the hydrogens whose bonds are constrained (plus, for rigid
// water, the two hydrogens connected by the H-H angle constraint). The largest
// chemically reasonable cluster is a heavy atom bonded to four hydrogens (e.g.
// methane / ammonium), giving five atoms. The Python builder validates that no
// cluster exceeds this bound.
static const int CONSTRAINT_MAX_CLUSTER_ATOMS = 5;
// Maximum number of constraints in a single cluster: up to four heavy-hydrogen
// bonds, plus rigid-water angle constraints.
static const int CONSTRAINT_MAX_CLUSTER_CONSTRAINTS = 6;

// k_constrain_positions implements SHAKE (Ryckaert, Ciccotti, Berendsen 1977).
//
// Each thread handles a single (cluster, system) pair. The cluster atoms form a
// disjoint star around one heavy atom, so no two threads touch the same atom
// and no atomics are required. The unconstrained ("drifted") positions in x are
// projected back onto the constraint manifold using the reference positions in
// x_ref (the positions at the start of the drift) to build the constraint
// directions. When update_velocity is true the constraint-consistent velocity
//   v = (x_constrained - x_ref) / dt_drift
// is written for each cluster atom; this is exactly the RATTLE position-stage
// velocity update for a geodesic (BAOAB) drift of length dt_drift.
template <typename RealType>
__global__ void k_constrain_positions(
    const int num_systems, const int N, const int num_clusters,
    const int *__restrict__ cluster_atom_offsets,       // [num_clusters + 1]
    const int *__restrict__ cluster_atoms,              // [total_cluster_atoms]
    const int *__restrict__ cluster_constraint_offsets, // [num_clusters + 1]
    const int *__restrict__ constraint_local_i,         // [total_constraints]
    const int *__restrict__ constraint_local_j,         // [total_constraints]
    const RealType *__restrict__ constraint_r0,         // [total_constraints]
    const RealType *__restrict__ inv_mass,              // [num_systems * N]
    const unsigned int *__restrict__ idxs,             // [num_systems * N] or null
    const RealType *__restrict__ x_ref,                 // [num_systems * N * 3]
    RealType *__restrict__ x,                           // [num_systems * N * 3]
    RealType *__restrict__ v,                           // [num_systems * N * 3]
    const RealType dt_drift, const RealType tol, const int max_iters,
    const bool update_velocity) {

  const int D = 3;
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (cluster_idx < num_clusters) {
    const int atom_begin = cluster_atom_offsets[cluster_idx];
    const int atom_end = cluster_atom_offsets[cluster_idx + 1];
    const int n_atoms = atom_end - atom_begin;

    const int con_begin = cluster_constraint_offsets[cluster_idx];
    const int con_end = cluster_constraint_offsets[cluster_idx + 1];

    // Load cluster state into registers.
    int global_atom[CONSTRAINT_MAX_CLUSTER_ATOMS];
    bool frozen[CONSTRAINT_MAX_CLUSTER_ATOMS];
    RealType pos[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType ref[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType winv[CONSTRAINT_MAX_CLUSTER_ATOMS];

    for (int a = 0; a < n_atoms; a++) {
      const int atom = cluster_atoms[atom_begin + a];
      global_atom[a] = atom;
      const int base = system_idx * N * D + atom * D;
      pos[a][0] = x[base + 0];
      pos[a][1] = x[base + 1];
      pos[a][2] = x[base + 2];
      ref[a][0] = x_ref[base + 0];
      ref[a][1] = x_ref[base + 1];
      ref[a][2] = x_ref[base + 2];
      // A frozen atom (local MD) acts as an infinite-mass anchor: it does not
      // move during the drift, so its position equals its reference and its
      // inverse mass is treated as zero.
      const bool is_frozen =
          idxs != nullptr &&
          idxs[system_idx * N + atom] >= static_cast<unsigned int>(N);
      frozen[a] = is_frozen;
      winv[a] = is_frozen ? static_cast<RealType>(0.0)
                          : inv_mass[system_idx * N + atom];
    }

    // Iterative SHAKE (Gauss-Seidel sweeps over constraints).
    for (int iter = 0; iter < max_iters; iter++) {
      bool done = true;
      for (int c = con_begin; c < con_end; c++) {
        const int li = constraint_local_i[c];
        const int lj = constraint_local_j[c];
        const RealType r0 = constraint_r0[c];
        const RealType target = r0 * r0;

        const RealType dx = pos[li][0] - pos[lj][0];
        const RealType dy = pos[li][1] - pos[lj][1];
        const RealType dz = pos[li][2] - pos[lj][2];
        const RealType r2 = dx * dx + dy * dy + dz * dz;

        const RealType diff = r2 - target;
        if (fabs(diff) > tol * target) {
          done = false;
        }

        // Reference (start-of-drift) separation provides the constraint
        // direction, as in the original SHAKE algorithm.
        const RealType rx = ref[li][0] - ref[lj][0];
        const RealType ry = ref[li][1] - ref[lj][1];
        const RealType rz = ref[li][2] - ref[lj][2];

        const RealType rdotd = rx * dx + ry * dy + rz * dz;
        const RealType reduced = winv[li] + winv[lj];
        const RealType denom = static_cast<RealType>(2.0) * reduced * rdotd;
        // Guard against a degenerate (near-orthogonal) configuration.
        if (denom == static_cast<RealType>(0.0)) {
          continue;
        }
        const RealType g = diff / denom;

        pos[li][0] -= winv[li] * g * rx;
        pos[li][1] -= winv[li] * g * ry;
        pos[li][2] -= winv[li] * g * rz;
        pos[lj][0] += winv[lj] * g * rx;
        pos[lj][1] += winv[lj] * g * ry;
        pos[lj][2] += winv[lj] * g * rz;
      }
      if (done) {
        break;
      }
    }

    const RealType inv_dt =
        update_velocity ? static_cast<RealType>(1.0) / dt_drift
                        : static_cast<RealType>(0.0);

    for (int a = 0; a < n_atoms; a++) {
      // Frozen atoms are anchors: their position is unchanged and their
      // velocity must not be overwritten with the constraint-consistent value.
      if (frozen[a]) {
        continue;
      }
      const int base = system_idx * N * D + global_atom[a] * D;
      x[base + 0] = pos[a][0];
      x[base + 1] = pos[a][1];
      x[base + 2] = pos[a][2];
      if (update_velocity) {
        v[base + 0] = (pos[a][0] - ref[a][0]) * inv_dt;
        v[base + 1] = (pos[a][1] - ref[a][1]) * inv_dt;
        v[base + 2] = (pos[a][2] - ref[a][2]) * inv_dt;
      }
    }

    cluster_idx += gridDim.x * blockDim.x;
  }
}

// k_constrain_velocities implements the RATTLE velocity projection: it removes
// the component of relative velocity along each constraint so that
//   d/dt |x_i - x_j|^2 = 2 (x_i - x_j) . (v_i - v_j) = 0
// holds for every constraint. One thread per (cluster, system).
template <typename RealType>
__global__ void k_constrain_velocities(
    const int num_systems, const int N, const int num_clusters,
    const int *__restrict__ cluster_atom_offsets,
    const int *__restrict__ cluster_atoms,
    const int *__restrict__ cluster_constraint_offsets,
    const int *__restrict__ constraint_local_i,
    const int *__restrict__ constraint_local_j,
    const RealType *__restrict__ inv_mass,
    const unsigned int *__restrict__ idxs, // [num_systems * N] or null
    const RealType *__restrict__ x, // [num_systems * N * 3]
    RealType *__restrict__ v,       // [num_systems * N * 3]
    const RealType tol, const int max_iters) {

  const int D = 3;
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (cluster_idx < num_clusters) {
    const int atom_begin = cluster_atom_offsets[cluster_idx];
    const int atom_end = cluster_atom_offsets[cluster_idx + 1];
    const int n_atoms = atom_end - atom_begin;

    const int con_begin = cluster_constraint_offsets[cluster_idx];
    const int con_end = cluster_constraint_offsets[cluster_idx + 1];

    int global_atom[CONSTRAINT_MAX_CLUSTER_ATOMS];
    bool frozen[CONSTRAINT_MAX_CLUSTER_ATOMS];
    RealType pos[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType vel[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType winv[CONSTRAINT_MAX_CLUSTER_ATOMS];

    for (int a = 0; a < n_atoms; a++) {
      const int atom = cluster_atoms[atom_begin + a];
      global_atom[a] = atom;
      const int base = system_idx * N * D + atom * D;
      pos[a][0] = x[base + 0];
      pos[a][1] = x[base + 1];
      pos[a][2] = x[base + 2];
      // A frozen atom (local MD) is physically stationary: treat its velocity
      // as zero and its inverse mass as zero so it anchors the constraint
      // without absorbing momentum. Its stored velocity is left untouched.
      const bool is_frozen =
          idxs != nullptr &&
          idxs[system_idx * N + atom] >= static_cast<unsigned int>(N);
      frozen[a] = is_frozen;
      if (is_frozen) {
        vel[a][0] = static_cast<RealType>(0.0);
        vel[a][1] = static_cast<RealType>(0.0);
        vel[a][2] = static_cast<RealType>(0.0);
        winv[a] = static_cast<RealType>(0.0);
      } else {
        vel[a][0] = v[base + 0];
        vel[a][1] = v[base + 1];
        vel[a][2] = v[base + 2];
        winv[a] = inv_mass[system_idx * N + atom];
      }
    }

    for (int iter = 0; iter < max_iters; iter++) {
      bool done = true;
      for (int c = con_begin; c < con_end; c++) {
        const int li = constraint_local_i[c];
        const int lj = constraint_local_j[c];

        const RealType rx = pos[li][0] - pos[lj][0];
        const RealType ry = pos[li][1] - pos[lj][1];
        const RealType rz = pos[li][2] - pos[lj][2];

        const RealType vx = vel[li][0] - vel[lj][0];
        const RealType vy = vel[li][1] - vel[lj][1];
        const RealType vz = vel[li][2] - vel[lj][2];

        const RealType rv = rx * vx + ry * vy + rz * vz;
        const RealType r2 = rx * rx + ry * ry + rz * rz;
        const RealType reduced = winv[li] + winv[lj];
        const RealType denom = reduced * r2;
        if (denom == static_cast<RealType>(0.0)) {
          continue;
        }
        // Tolerance is on the rate-of-change of the (squared) bond length.
        if (fabs(rv) > tol * r2) {
          done = false;
        }
        const RealType k = rv / denom;

        vel[li][0] -= winv[li] * k * rx;
        vel[li][1] -= winv[li] * k * ry;
        vel[li][2] -= winv[li] * k * rz;
        vel[lj][0] += winv[lj] * k * rx;
        vel[lj][1] += winv[lj] * k * ry;
        vel[lj][2] += winv[lj] * k * rz;
      }
      if (done) {
        break;
      }
    }

    for (int a = 0; a < n_atoms; a++) {
      if (frozen[a]) {
        continue;
      }
      const int base = system_idx * N * D + global_atom[a] * D;
      v[base + 0] = vel[a][0];
      v[base + 1] = vel[a][1];
      v[base + 2] = vel[a][2];
    }

    cluster_idx += gridDim.x * blockDim.x;
  }
}

// k_constrained_kick performs the BAOAB "B" sub-step: v += (dt / m) * force.
// Forces are stored as negated fixed-point du/dx and are zeroed after use, in
// keeping with the rest of the integrator kernels.
template <typename RealType, int D>
__global__ void
k_constrained_kick(const int num_systems, const int N,
                   const RealType *__restrict__ cbs, // [num_systems * N], dt/m
                   const unsigned int *__restrict__ idxs, // [num_systems*N] or null
                   RealType *__restrict__ v,
                   unsigned long long *__restrict__ du_dx) {
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < N) {
    const int atom_idx =
        (idxs == nullptr ? kernel_idx : idxs[system_idx * N + kernel_idx]);
    if (atom_idx < N) {
      const RealType cb = cbs[system_idx * N + atom_idx];
      for (int d = 0; d < D; d++) {
        const int idx = system_idx * N * D + atom_idx * D + d;
        const RealType force = -FIXED_TO_FLOAT<RealType>(du_dx[idx]);
        v[idx] += cb * force;
        du_dx[idx] = 0;
      }
    } else if (idxs != nullptr) {
      // Frozen atom under local MD. The potentials still accumulate a force on
      // this atom (e.g. free-frozen nonbonded interactions), but it is never
      // integrated, so that force must be zeroed here. Otherwise it would
      // accumulate across local-MD steps and be injected as a spurious velocity
      // kick the next time this atom is integrated -- a subsequent global step,
      // or a later local burst in which it becomes free -- destabilizing the
      // simulation. Mirrors the frozen-atom branch in k_update_forward_baoab;
      // relies on idxs[kernel_idx] == kernel_idx for free atoms, so kernel_idx
      // is this frozen atom's storage slot.
      for (int d = 0; d < D; d++) {
        du_dx[system_idx * N * D + kernel_idx * D + d] = 0;
      }
    }
    kernel_idx += gridDim.x * blockDim.x;
  }
}

// k_constrained_drift performs a BAOAB "A" sub-step: x += frac * dt * v.
template <typename RealType, int D>
__global__ void k_constrained_drift(const int num_systems, const int N,
                                    const RealType *__restrict__ v,
                                    const unsigned int *__restrict__ idxs,
                                    RealType *__restrict__ x,
                                    const RealType frac_dt) {
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < N) {
    const int atom_idx =
        (idxs == nullptr ? kernel_idx : idxs[system_idx * N + kernel_idx]);
    if (atom_idx < N) {
      for (int d = 0; d < D; d++) {
        const int idx = system_idx * N * D + atom_idx * D + d;
        x[idx] += frac_dt * v[idx];
      }
    }
    kernel_idx += gridDim.x * blockDim.x;
  }
}

// k_constrained_ornstein performs the BAOAB "O" sub-step (Ornstein-Uhlenbeck):
//   v <- ca * v + ccs * gaussian_noise
template <typename RealType, int D>
__global__ void k_constrained_ornstein(
    const int num_systems, const int N, const RealType ca,
    const RealType *__restrict__ ccs, // [num_systems * N]
    const unsigned int *__restrict__ idxs, // [num_systems * N] or null
    curandState_t *__restrict__ rand_states, RealType *__restrict__ v) {
  static_assert(D == 3);
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < N) {
    const int atom_idx =
        (idxs == nullptr ? kernel_idx : idxs[system_idx * N + kernel_idx]);
    if (atom_idx < N) {
      const RealType cc = ccs[system_idx * N + atom_idx];
      curandState_t local_state = rand_states[system_idx * N + atom_idx];

      const int base = system_idx * N * D + atom_idx * D;
      RealType noise_x;
      RealType noise_y;
      template_curand_normal2(noise_x, noise_y, &local_state);
      const RealType noise_z = template_curand_normal<RealType>(&local_state);

      v[base + 0] = ca * v[base + 0] + cc * noise_x;
      v[base + 1] = ca * v[base + 1] + cc * noise_y;
      v[base + 2] = ca * v[base + 2] + cc * noise_z;

      rand_states[system_idx * N + atom_idx] = local_state;
    }
    kernel_idx += gridDim.x * blockDim.x;
  }
}

// ---------------------------------------------------------------------------
// Fused constrained BAOAB step.
//
// The reference implementation issues one small kernel per integrator sub-step
// (kick, drift, SHAKE, RATTLE, ...), each chained on the previous one's output.
// For the small/medium systems typical of hydration free energies, the per-step
// runtime is dominated by the fixed launch+scheduling latency of this serial
// chain of ~12 tiny kernels rather than by the constraint arithmetic itself.
//
// Because the entire post-force B-A-O-A sequence depends only on the (already
// computed) forces and per-atom random numbers, it can be evaluated for a whole
// cluster inside a single kernel, keeping the cluster's positions/velocities in
// registers across all sub-steps. This collapses the chain to one kernel for
// the constrained (cluster) atoms plus one kernel for any unconstrained atoms,
// dramatically reducing the number of dependent launches. The arithmetic and
// its order are identical to the per-kernel path.
// ---------------------------------------------------------------------------

// SHAKE position projection (Gauss-Seidel sweeps) on a cluster held in local
// arrays. Constraint directions use the reference (start-of-drift) separations.
template <typename RealType, int D>
__device__ __forceinline__ void
cluster_shake_sweeps(const int con_begin, const int con_end,
                     const int *__restrict__ constraint_local_i,
                     const int *__restrict__ constraint_local_j,
                     const RealType *__restrict__ constraint_r0,
                     RealType (&pos)[CONSTRAINT_MAX_CLUSTER_ATOMS][D],
                     const RealType (&ref)[CONSTRAINT_MAX_CLUSTER_ATOMS][D],
                     const RealType (&winv)[CONSTRAINT_MAX_CLUSTER_ATOMS],
                     const RealType tol, const int max_iters) {
  for (int iter = 0; iter < max_iters; iter++) {
    bool done = true;
    for (int c = con_begin; c < con_end; c++) {
      const int li = constraint_local_i[c];
      const int lj = constraint_local_j[c];
      const RealType r0 = constraint_r0[c];
      const RealType target = r0 * r0;

      const RealType dx = pos[li][0] - pos[lj][0];
      const RealType dy = pos[li][1] - pos[lj][1];
      const RealType dz = pos[li][2] - pos[lj][2];
      const RealType r2 = dx * dx + dy * dy + dz * dz;

      const RealType diff = r2 - target;
      if (fabs(diff) > tol * target) {
        done = false;
      }

      const RealType rx = ref[li][0] - ref[lj][0];
      const RealType ry = ref[li][1] - ref[lj][1];
      const RealType rz = ref[li][2] - ref[lj][2];

      const RealType rdotd = rx * dx + ry * dy + rz * dz;
      const RealType reduced = winv[li] + winv[lj];
      const RealType denom = static_cast<RealType>(2.0) * reduced * rdotd;
      if (denom == static_cast<RealType>(0.0)) {
        continue;
      }
      const RealType g = diff / denom;

      pos[li][0] -= winv[li] * g * rx;
      pos[li][1] -= winv[li] * g * ry;
      pos[li][2] -= winv[li] * g * rz;
      pos[lj][0] += winv[lj] * g * rx;
      pos[lj][1] += winv[lj] * g * ry;
      pos[lj][2] += winv[lj] * g * rz;
    }
    if (done) {
      break;
    }
  }
}

// RATTLE velocity projection (Gauss-Seidel sweeps) on a cluster held in local
// arrays, using the current positions for the constraint directions.
template <typename RealType, int D>
__device__ __forceinline__ void
cluster_rattle_sweeps(const int con_begin, const int con_end,
                      const int *__restrict__ constraint_local_i,
                      const int *__restrict__ constraint_local_j,
                      const RealType (&pos)[CONSTRAINT_MAX_CLUSTER_ATOMS][D],
                      RealType (&vel)[CONSTRAINT_MAX_CLUSTER_ATOMS][D],
                      const RealType (&winv)[CONSTRAINT_MAX_CLUSTER_ATOMS],
                      const RealType tol, const int max_iters) {
  for (int iter = 0; iter < max_iters; iter++) {
    bool done = true;
    for (int c = con_begin; c < con_end; c++) {
      const int li = constraint_local_i[c];
      const int lj = constraint_local_j[c];

      const RealType rx = pos[li][0] - pos[lj][0];
      const RealType ry = pos[li][1] - pos[lj][1];
      const RealType rz = pos[li][2] - pos[lj][2];

      const RealType vx = vel[li][0] - vel[lj][0];
      const RealType vy = vel[li][1] - vel[lj][1];
      const RealType vz = vel[li][2] - vel[lj][2];

      const RealType rv = rx * vx + ry * vy + rz * vz;
      const RealType r2 = rx * rx + ry * ry + rz * rz;
      const RealType reduced = winv[li] + winv[lj];
      const RealType denom = reduced * r2;
      if (denom == static_cast<RealType>(0.0)) {
        continue;
      }
      if (fabs(rv) > tol * r2) {
        done = false;
      }
      const RealType k = rv / denom;

      vel[li][0] -= winv[li] * k * rx;
      vel[li][1] -= winv[li] * k * ry;
      vel[li][2] -= winv[li] * k * rz;
      vel[lj][0] += winv[lj] * k * rx;
      vel[lj][1] += winv[lj] * k * ry;
      vel[lj][2] += winv[lj] * k * rz;
    }
    if (done) {
      break;
    }
  }
}

// k_constrained_baoab_cluster performs the full post-force constrained BAOAB
// step for every cluster atom in a single kernel (one thread per cluster).
// The sub-step sequence -- B, RATTLE, A(half)+SHAKE+velocity-correction, RATTLE,
// O, RATTLE, A(half)+SHAKE+velocity-correction, RATTLE -- matches the per-kernel
// integrator exactly. Forces (negated fixed-point du/dx) are consumed and zeroed
// here, mirroring the per-kernel kick.
template <typename RealType, int D>
__global__ void k_constrained_baoab_cluster(
    const int num_systems, const int N, const int num_clusters,
    const int *__restrict__ cluster_atom_offsets,
    const int *__restrict__ cluster_atoms,
    const int *__restrict__ cluster_constraint_offsets,
    const int *__restrict__ constraint_local_i,
    const int *__restrict__ constraint_local_j,
    const RealType *__restrict__ constraint_r0,
    const RealType *__restrict__ cbs,      // [num_systems * N] dt / m
    const RealType *__restrict__ ccs,      // [num_systems * N] noise scale
    const RealType *__restrict__ inv_mass, // [num_systems * N]
    const unsigned int *__restrict__ idxs, // [num_systems * N] or null
    curandState_t *__restrict__ rand_states, // [num_systems * N]
    unsigned long long *__restrict__ du_dx,  // [num_systems * N * 3]
    RealType *__restrict__ x,                // [num_systems * N * 3]
    RealType *__restrict__ v,                // [num_systems * N * 3]
    const RealType ca, const RealType half_dt, const RealType pos_tol,
    const RealType vel_tol, const int max_iters) {
  static_assert(D == 3);
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (cluster_idx < num_clusters) {
    const int atom_begin = cluster_atom_offsets[cluster_idx];
    const int atom_end = cluster_atom_offsets[cluster_idx + 1];
    const int n_atoms = atom_end - atom_begin;
    const int con_begin = cluster_constraint_offsets[cluster_idx];
    const int con_end = cluster_constraint_offsets[cluster_idx + 1];

    int global_atom[CONSTRAINT_MAX_CLUSTER_ATOMS];
    bool frozen[CONSTRAINT_MAX_CLUSTER_ATOMS];
    RealType pos[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType vel[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType ref[CONSTRAINT_MAX_CLUSTER_ATOMS][D];
    RealType winv[CONSTRAINT_MAX_CLUSTER_ATOMS];
    RealType cb[CONSTRAINT_MAX_CLUSTER_ATOMS];
    RealType cc[CONSTRAINT_MAX_CLUSTER_ATOMS];
    curandState_t rng[CONSTRAINT_MAX_CLUSTER_ATOMS];

    // Load cluster state. Frozen atoms (local MD) are infinite-mass anchors:
    // zero inverse mass, zero velocity, no kick/noise, no writeback.
    for (int a = 0; a < n_atoms; a++) {
      const int atom = cluster_atoms[atom_begin + a];
      global_atom[a] = atom;
      const int sN = system_idx * N + atom;
      const int base = sN * D;
      const bool is_frozen =
          idxs != nullptr && idxs[sN] >= static_cast<unsigned int>(N);
      frozen[a] = is_frozen;
      pos[a][0] = x[base + 0];
      pos[a][1] = x[base + 1];
      pos[a][2] = x[base + 2];
      if (is_frozen) {
        vel[a][0] = static_cast<RealType>(0.0);
        vel[a][1] = static_cast<RealType>(0.0);
        vel[a][2] = static_cast<RealType>(0.0);
        winv[a] = static_cast<RealType>(0.0);
        cb[a] = static_cast<RealType>(0.0);
        cc[a] = static_cast<RealType>(0.0);
      } else {
        vel[a][0] = v[base + 0];
        vel[a][1] = v[base + 1];
        vel[a][2] = v[base + 2];
        winv[a] = inv_mass[sN];
        cb[a] = cbs[sN];
        cc[a] = ccs[sN];
        rng[a] = rand_states[sN];
      }
    }

    // B: velocity kick. Consume and zero the force on every cluster atom,
    // including frozen ones (whose force must not accumulate across steps).
    for (int a = 0; a < n_atoms; a++) {
      const int base = (system_idx * N + global_atom[a]) * D;
      for (int d = 0; d < D; d++) {
        const RealType force = -FIXED_TO_FLOAT<RealType>(du_dx[base + d]);
        if (!frozen[a]) {
          vel[a][d] += cb[a] * force;
        }
        du_dx[base + d] = 0;
      }
    }
    cluster_rattle_sweeps<RealType, D>(con_begin, con_end, constraint_local_i,
                                       constraint_local_j, pos, vel, winv,
                                       vel_tol, max_iters);

    const RealType inv_half_dt = static_cast<RealType>(1.0) / half_dt;

    // A (first half drift): save reference, drift, SHAKE, constraint-consistent
    // velocity correction, RATTLE.
    for (int a = 0; a < n_atoms; a++) {
      ref[a][0] = pos[a][0];
      ref[a][1] = pos[a][1];
      ref[a][2] = pos[a][2];
      if (!frozen[a]) {
        pos[a][0] += half_dt * vel[a][0];
        pos[a][1] += half_dt * vel[a][1];
        pos[a][2] += half_dt * vel[a][2];
      }
    }
    cluster_shake_sweeps<RealType, D>(con_begin, con_end, constraint_local_i,
                                      constraint_local_j, constraint_r0, pos,
                                      ref, winv, pos_tol, max_iters);
    for (int a = 0; a < n_atoms; a++) {
      if (!frozen[a]) {
        vel[a][0] = (pos[a][0] - ref[a][0]) * inv_half_dt;
        vel[a][1] = (pos[a][1] - ref[a][1]) * inv_half_dt;
        vel[a][2] = (pos[a][2] - ref[a][2]) * inv_half_dt;
      }
    }
    cluster_rattle_sweeps<RealType, D>(con_begin, con_end, constraint_local_i,
                                       constraint_local_j, pos, vel, winv,
                                       vel_tol, max_iters);

    // O: Ornstein-Uhlenbeck velocity update, then RATTLE.
    for (int a = 0; a < n_atoms; a++) {
      if (frozen[a]) {
        continue;
      }
      RealType noise_x;
      RealType noise_y;
      template_curand_normal2(noise_x, noise_y, &rng[a]);
      const RealType noise_z = template_curand_normal<RealType>(&rng[a]);
      vel[a][0] = ca * vel[a][0] + cc[a] * noise_x;
      vel[a][1] = ca * vel[a][1] + cc[a] * noise_y;
      vel[a][2] = ca * vel[a][2] + cc[a] * noise_z;
    }
    cluster_rattle_sweeps<RealType, D>(con_begin, con_end, constraint_local_i,
                                       constraint_local_j, pos, vel, winv,
                                       vel_tol, max_iters);

    // A (second half drift): save, drift, SHAKE, velocity correction, RATTLE.
    for (int a = 0; a < n_atoms; a++) {
      ref[a][0] = pos[a][0];
      ref[a][1] = pos[a][1];
      ref[a][2] = pos[a][2];
      if (!frozen[a]) {
        pos[a][0] += half_dt * vel[a][0];
        pos[a][1] += half_dt * vel[a][1];
        pos[a][2] += half_dt * vel[a][2];
      }
    }
    cluster_shake_sweeps<RealType, D>(con_begin, con_end, constraint_local_i,
                                      constraint_local_j, constraint_r0, pos,
                                      ref, winv, pos_tol, max_iters);
    for (int a = 0; a < n_atoms; a++) {
      if (!frozen[a]) {
        vel[a][0] = (pos[a][0] - ref[a][0]) * inv_half_dt;
        vel[a][1] = (pos[a][1] - ref[a][1]) * inv_half_dt;
        vel[a][2] = (pos[a][2] - ref[a][2]) * inv_half_dt;
      }
    }
    cluster_rattle_sweeps<RealType, D>(con_begin, con_end, constraint_local_i,
                                       constraint_local_j, pos, vel, winv,
                                       vel_tol, max_iters);

    // Write back updated positions, velocities and RNG state for free atoms.
    for (int a = 0; a < n_atoms; a++) {
      if (frozen[a]) {
        continue;
      }
      const int sN = system_idx * N + global_atom[a];
      const int base = sN * D;
      x[base + 0] = pos[a][0];
      x[base + 1] = pos[a][1];
      x[base + 2] = pos[a][2];
      v[base + 0] = vel[a][0];
      v[base + 1] = vel[a][1];
      v[base + 2] = vel[a][2];
      rand_states[sN] = rng[a];
    }

    cluster_idx += gridDim.x * blockDim.x;
  }
}

// k_baoab_noncluster performs the full post-force BAOAB step (B, A, O, A) for
// atoms that do not participate in any constraint cluster. These atoms are
// unconstrained, so no SHAKE/RATTLE projection is applied; the sub-step order
// and arithmetic match the per-kernel path for non-cluster atoms exactly. The
// is_cluster_atom mask (indexed by per-system atom index) selects the atoms
// handled by k_constrained_baoab_cluster, which this kernel must skip.
template <typename RealType, int D>
__global__ void k_baoab_noncluster(
    const int num_systems, const int N,
    const RealType *__restrict__ cbs, // [num_systems * N] dt / m
    const RealType *__restrict__ ccs, // [num_systems * N] noise scale
    const unsigned int *__restrict__ idxs,       // [num_systems * N] or null
    const uint8_t *__restrict__ is_cluster_atom, // [N]
    curandState_t *__restrict__ rand_states,     // [num_systems * N]
    unsigned long long *__restrict__ du_dx,      // [num_systems * N * 3]
    RealType *__restrict__ x, RealType *__restrict__ v, const RealType ca,
    const RealType half_dt) {
  static_assert(D == 3);
  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (kernel_idx < N) {
    const int atom_idx =
        (idxs == nullptr ? kernel_idx : idxs[system_idx * N + kernel_idx]);
    if (atom_idx < N) {
      // Cluster atoms are integrated by k_constrained_baoab_cluster (which also
      // zeroes their forces); skip them here.
      if (is_cluster_atom[atom_idx]) {
        kernel_idx += gridDim.x * blockDim.x;
        continue;
      }
      const RealType cb = cbs[system_idx * N + atom_idx];
      const RealType cc = ccs[system_idx * N + atom_idx];
      const int base = system_idx * N * D + atom_idx * D;
      curandState_t local_state = rand_states[system_idx * N + atom_idx];

      // B: kick (consume and zero force).
      RealType vel[D];
      for (int d = 0; d < D; d++) {
        const RealType force = -FIXED_TO_FLOAT<RealType>(du_dx[base + d]);
        vel[d] = v[base + d] + cb * force;
        du_dx[base + d] = 0;
      }
      // A (first half drift).
      RealType pos[D];
      for (int d = 0; d < D; d++) {
        pos[d] = x[base + d] + half_dt * vel[d];
      }
      // O: Ornstein-Uhlenbeck.
      RealType noise_x;
      RealType noise_y;
      template_curand_normal2(noise_x, noise_y, &local_state);
      const RealType noise_z = template_curand_normal<RealType>(&local_state);
      vel[0] = ca * vel[0] + cc * noise_x;
      vel[1] = ca * vel[1] + cc * noise_y;
      vel[2] = ca * vel[2] + cc * noise_z;
      // A (second half drift).
      for (int d = 0; d < D; d++) {
        pos[d] += half_dt * vel[d];
        x[base + d] = pos[d];
        v[base + d] = vel[d];
      }
      rand_states[system_idx * N + atom_idx] = local_state;
    } else if (idxs != nullptr) {
      // Frozen non-cluster atom under local MD: zero its (un-integrated) force
      // so it does not accumulate. Frozen cluster atoms are handled by the
      // cluster kernel. kernel_idx is the storage slot (idxs[slot] == slot for
      // free atoms), so it indexes the mask for this atom.
      if (!is_cluster_atom[kernel_idx]) {
        for (int d = 0; d < D; d++) {
          du_dx[system_idx * N * D + kernel_idx * D + d] = 0;
        }
      }
    }
    kernel_idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
