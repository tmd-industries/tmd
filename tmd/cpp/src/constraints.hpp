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

#include "device_buffer.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

namespace tmd {

// Constraints holds the GPU-resident description of a set of distance
// constraints partitioned into disjoint clusters, and applies SHAKE (position)
// and RATTLE (velocity) projections.
//
// A cluster is a group of atoms whose constraints only involve atoms within the
// group. For hydrogen-bond constraints under the assumption that no hydrogen is
// shared between alchemical end states, every hydrogen is bonded to exactly one
// heavy atom, so each cluster is a star around a single heavy atom (optionally
// with a rigid-water H-H constraint). Clusters are mutually disjoint, which lets
// each be solved independently by a single GPU thread without atomics.
//
// The constraint topology (atoms, indices, target distances) is fixed for the
// lifetime of the object; only positions/velocities change between steps.
template <typename RealType> class Constraints {

private:
  const int num_clusters_;
  const int num_constraints_;
  const int num_water_clusters_;
  const int num_general_clusters_;
  const RealType pos_tol_;
  const RealType vel_tol_;
  const int max_iters_;

  // CSR layout over clusters.
  DeviceBuffer<int> d_cluster_atom_offsets_;       // [num_clusters + 1]
  DeviceBuffer<int> d_cluster_atoms_;              // [total cluster atoms]
  DeviceBuffer<int> d_cluster_constraint_offsets_; // [num_clusters + 1]
  DeviceBuffer<int> d_constraint_local_i_;         // [num_constraints]
  DeviceBuffer<int> d_constraint_local_j_;         // [num_constraints]
  DeviceBuffer<RealType> d_constraint_r0_;         // [num_constraints]

  // Cluster ids of the rigid 3-point water clusters (handled by the analytic
  // SETTLE kernel) and the complementary set of all other clusters (handled by
  // the iterative SHAKE/RATTLE kernel). Together they partition [0,
  // num_clusters).
  DeviceBuffer<int> d_water_cluster_ids_;   // [num_water_clusters_]
  DeviceBuffer<int> d_general_cluster_ids_; // [num_general_clusters_]

  // Host copy of the cluster member atom indices (one system), used to build
  // the cluster-membership mask consumed by the fused integrator.
  const std::vector<int> h_cluster_atoms_;

  // Representative rigid-water model parameters. All waters share the same
  // geometry and masses, so the SETTLE kernel uses these scalars instead of
  // per-atom array lookups. The geometry (constraint lengths and the per-system
  // global atom indices of a representative oxygen/hydrogen) is derived from the
  // cluster layout at construction; the dynamics coefficients (inverse masses
  // and the BAOAB dt/m and noise-scale factors) are supplied by the integrator
  // via set_water_baoab_scalars, since they depend on dt/friction/temperature.
  int water_o_index_ = -1;
  int water_h_index_ = -1;
  RealType water_dOH_ = 0;
  RealType water_dHH_ = 0;
  RealType water_inv_mO_ = 0;
  RealType water_inv_mH_ = 0;
  RealType water_cb_O_ = 0;
  RealType water_cb_H_ = 0;
  RealType water_cc_O_ = 0;
  RealType water_cc_H_ = 0;
  bool water_scalars_set_ = false;

public:
  // cluster_atom_offsets / cluster_constraint_offsets are CSR offset arrays of
  // length num_clusters + 1. cluster_atoms holds the global atom index of each
  // cluster member. constraint_local_{i,j} index into a cluster's own atom list
  // (i.e. in [0, n_atoms_in_cluster)). constraint_r0 holds the target bond
  // length for each constraint. water_cluster_ids lists the clusters that are
  // rigid 3-point water (canonical layout: local atom 0 oxygen, 1/2 hydrogens;
  // constraints O-H1, O-H2, H1-H2) and are eligible for the analytic SETTLE
  // kernel; every such cluster must have exactly 3 atoms and 3 constraints.
  Constraints(const std::vector<int> &cluster_atom_offsets,
              const std::vector<int> &cluster_atoms,
              const std::vector<int> &cluster_constraint_offsets,
              const std::vector<int> &constraint_local_i,
              const std::vector<int> &constraint_local_j,
              const std::vector<RealType> &constraint_r0,
              const std::vector<int> &water_cluster_ids, const RealType pos_tol,
              const RealType vel_tol, const int max_iters);

  ~Constraints();

  int num_clusters() const { return num_clusters_; }
  int num_constraints() const { return num_constraints_; }
  int num_water_clusters() const { return num_water_clusters_; }

  // Per-system global atom indices (in [0, N)) of a representative rigid-water
  // oxygen and hydrogen. Valid only when num_water_clusters() > 0. The
  // integrator uses these to look up the water atoms' BAOAB coefficients.
  int water_oxygen_index() const { return water_o_index_; }
  int water_hydrogen_index() const { return water_h_index_; }

  // Supply the BAOAB coefficients (inverse mass, cb = dt/m, cc = noise scale)
  // for the representative water oxygen and hydrogen. Because all waters share
  // mass, these scalars apply to every water and let the SETTLE kernel avoid
  // per-atom array lookups. Must be called before apply_constrained_baoab when
  // num_water_clusters() > 0.
  void set_water_baoab_scalars(RealType inv_mO, RealType inv_mH, RealType cb_O,
                               RealType cb_H, RealType cc_O, RealType cc_H);

  // Per-system global atom indices (in [0, N)) of every cluster member. The
  // clusters are disjoint, so these indices are distinct; an atom not in this
  // list participates in no constraint. Used to build the cluster-membership
  // mask for the fused integrator.
  const std::vector<int> &cluster_atoms_host() const {
    return h_cluster_atoms_;
  }

  // Project drifted positions x back onto the constraint manifold using x_ref
  // (positions at the start of the drift) for the constraint directions. When
  // update_velocity is true, velocities of constrained atoms are reset to the
  // constraint-consistent value (x - x_ref) / dt_drift. When d_idxs is non-null
  // (local MD) it is the free-index buffer: atoms whose entry equals N are
  // frozen and treated as infinite-mass anchors.
  void apply_position_constraints(const int num_systems, const int N,
                                  const RealType *d_inv_mass,
                                  const unsigned int *d_idxs,
                                  const RealType *d_x_ref, RealType *d_x,
                                  RealType *d_v, const RealType dt_drift,
                                  const bool update_velocity,
                                  cudaStream_t stream);

  // Project velocities so that every constrained pair satisfies r . v = 0. When
  // d_idxs is non-null (local MD) frozen atoms (entry == N) are treated as
  // stationary infinite-mass anchors.
  void apply_velocity_constraints(const int num_systems, const int N,
                                  const RealType *d_inv_mass,
                                  const unsigned int *d_idxs,
                                  const RealType *d_x, RealType *d_v,
                                  cudaStream_t stream);

  // Perform the entire post-force constrained BAOAB step for all cluster atoms
  // in a single fused kernel (one thread per cluster), consuming and zeroing
  // the forces in d_du_dx. This is equivalent to the sequence of per-sub-step
  // kernels plus their interleaved SHAKE/RATTLE projections, but issues a single
  // launch. Non-cluster atoms are not touched and must be integrated separately.
  void apply_constrained_baoab(
      const int num_systems, const int N, const RealType *d_cbs,
      const RealType *d_ccs, const RealType *d_inv_mass,
      const unsigned int *d_idxs, curandState_t *d_rand_states,
      unsigned long long *d_du_dx, RealType *d_x, RealType *d_v,
      const RealType ca, const RealType half_dt, cudaStream_t stream);
};

} // namespace tmd
