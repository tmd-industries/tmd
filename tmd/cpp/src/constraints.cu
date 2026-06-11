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

#include "constraints.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_constraints.cuh"
#include <cstdlib>
#include <stdexcept>

namespace tmd {

template <typename RealType>
Constraints<RealType>::Constraints(
    const std::vector<int> &cluster_atom_offsets,
    const std::vector<int> &cluster_atoms,
    const std::vector<int> &cluster_constraint_offsets,
    const std::vector<int> &constraint_local_i,
    const std::vector<int> &constraint_local_j,
    const std::vector<RealType> &constraint_r0,
    const std::vector<int> &water_cluster_ids, const RealType pos_tol,
    const RealType vel_tol, const int max_iters)
    : num_clusters_(static_cast<int>(cluster_atom_offsets.size()) - 1),
      num_constraints_(static_cast<int>(constraint_r0.size())),
      num_water_clusters_(static_cast<int>(water_cluster_ids.size())),
      num_general_clusters_(static_cast<int>(cluster_atom_offsets.size()) - 1 -
                            static_cast<int>(water_cluster_ids.size())),
      pos_tol_(pos_tol), vel_tol_(vel_tol), max_iters_(max_iters),
      d_cluster_atom_offsets_(cluster_atom_offsets),
      d_cluster_atoms_(cluster_atoms),
      d_cluster_constraint_offsets_(cluster_constraint_offsets),
      d_constraint_local_i_(constraint_local_i),
      d_constraint_local_j_(constraint_local_j),
      d_constraint_r0_(constraint_r0),
      d_water_cluster_ids_(water_cluster_ids.size()),
      d_general_cluster_ids_(num_general_clusters_ < 0
                                 ? 0
                                 : static_cast<size_t>(num_general_clusters_)),
      h_cluster_atoms_(cluster_atoms) {

  if (cluster_atom_offsets.size() < 1) {
    throw std::runtime_error("cluster_atom_offsets must be non-empty");
  }
  if (cluster_constraint_offsets.size() != cluster_atom_offsets.size()) {
    throw std::runtime_error(
        "cluster_constraint_offsets and cluster_atom_offsets must have the "
        "same length");
  }
  if (constraint_local_i.size() != constraint_local_j.size() ||
      constraint_local_i.size() != constraint_r0.size()) {
    throw std::runtime_error(
        "constraint_local_i, constraint_local_j and constraint_r0 must have "
        "the same length");
  }
  if (max_iters <= 0) {
    throw std::runtime_error("max_iters must be positive");
  }

  // Validate cluster sizes against the register-array bounds used by the
  // kernels, and validate the CSR offsets.
  for (int c = 0; c < num_clusters_; c++) {
    const int n_atoms =
        cluster_atom_offsets[c + 1] - cluster_atom_offsets[c];
    const int n_cons =
        cluster_constraint_offsets[c + 1] - cluster_constraint_offsets[c];
    if (n_atoms < 0 || n_cons < 0) {
      throw std::runtime_error("CSR offsets must be non-decreasing");
    }
    if (n_atoms > CONSTRAINT_MAX_CLUSTER_ATOMS) {
      throw std::runtime_error(
          "cluster has more atoms than CONSTRAINT_MAX_CLUSTER_ATOMS");
    }
    if (n_cons > CONSTRAINT_MAX_CLUSTER_CONSTRAINTS) {
      throw std::runtime_error(
          "cluster has more constraints than "
          "CONSTRAINT_MAX_CLUSTER_CONSTRAINTS");
    }
  }

  // Validate the water-cluster ids and build the complementary (general)
  // cluster id list. The SETTLE kernel assumes the canonical water layout, so
  // each water cluster must have exactly 3 atoms and 3 constraints; ids must be
  // distinct and in range.
  std::vector<char> is_water(num_clusters_, 0);
  for (const int id : water_cluster_ids) {
    if (id < 0 || id >= num_clusters_) {
      throw std::runtime_error("water cluster id out of range");
    }
    if (is_water[id]) {
      throw std::runtime_error("duplicate water cluster id");
    }
    is_water[id] = 1;
    const int n_atoms =
        cluster_atom_offsets[id + 1] - cluster_atom_offsets[id];
    const int n_cons =
        cluster_constraint_offsets[id + 1] - cluster_constraint_offsets[id];
    if (n_atoms != 3 || n_cons != 3) {
      throw std::runtime_error(
          "water cluster must have exactly 3 atoms and 3 constraints");
    }
  }
  if (num_water_clusters_ > 0) {
    d_water_cluster_ids_.copy_from(water_cluster_ids.data());
  }
  std::vector<int> general_cluster_ids;
  general_cluster_ids.reserve(static_cast<size_t>(num_general_clusters_));
  for (int c = 0; c < num_clusters_; c++) {
    if (!is_water[c]) {
      general_cluster_ids.push_back(c);
    }
  }
  if (num_general_clusters_ > 0) {
    d_general_cluster_ids_.copy_from(general_cluster_ids.data());
  }
}

template <typename RealType> Constraints<RealType>::~Constraints() {}

template <typename RealType>
void Constraints<RealType>::apply_position_constraints(
    const int num_systems, const int N, const RealType *d_inv_mass,
    const unsigned int *d_idxs, const RealType *d_x_ref, RealType *d_x,
    RealType *d_v, const RealType dt_drift, const bool update_velocity,
    cudaStream_t stream) {
  if (num_clusters_ == 0) {
    return;
  }
  const size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  dim3 grid(ceil_divide(num_clusters_, tpb), num_systems);
  k_constrain_positions<RealType><<<grid, tpb, 0, stream>>>(
      num_systems, N, num_clusters_, d_cluster_atom_offsets_.data,
      d_cluster_atoms_.data, d_cluster_constraint_offsets_.data,
      d_constraint_local_i_.data, d_constraint_local_j_.data,
      d_constraint_r0_.data, d_inv_mass, d_idxs, d_x_ref, d_x, d_v, dt_drift,
      pos_tol_, max_iters_, update_velocity);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Constraints<RealType>::apply_velocity_constraints(
    const int num_systems, const int N, const RealType *d_inv_mass,
    const unsigned int *d_idxs, const RealType *d_x, RealType *d_v,
    cudaStream_t stream) {
  if (num_clusters_ == 0) {
    return;
  }
  const size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  dim3 grid(ceil_divide(num_clusters_, tpb), num_systems);
  k_constrain_velocities<RealType><<<grid, tpb, 0, stream>>>(
      num_systems, N, num_clusters_, d_cluster_atom_offsets_.data,
      d_cluster_atoms_.data, d_cluster_constraint_offsets_.data,
      d_constraint_local_i_.data, d_constraint_local_j_.data, d_inv_mass,
      d_idxs, d_x, d_v, vel_tol_, max_iters_);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Constraints<RealType>::apply_constrained_baoab(
    const int num_systems, const int N, const RealType *d_cbs,
    const RealType *d_ccs, const RealType *d_inv_mass,
    const unsigned int *d_idxs, curandState_t *d_rand_states,
    unsigned long long *d_du_dx, RealType *d_x, RealType *d_v,
    const RealType ca, const RealType half_dt, cudaStream_t stream) {
  if (num_clusters_ == 0) {
    return;
  }
  // Read once: TMD_NO_SETTLE forces every cluster through the iterative
  // SHAKE/RATTLE path (A/B baseline for the analytic SETTLE water kernel).
  static const bool no_settle = std::getenv("TMD_NO_SETTLE") != nullptr;

  const size_t tpb = DEFAULT_THREADS_PER_BLOCK;

  // SETTLE assumes all three water atoms are mobile, so it is only used in
  // global MD (d_idxs == null). Under local MD some water atoms may be frozen
  // anchors, so water is routed through the iterative path with the rest.
  const bool use_settle =
      !no_settle && d_idxs == nullptr && num_water_clusters_ > 0;

  if (use_settle) {
    dim3 wgrid(ceil_divide(num_water_clusters_, tpb), num_systems);
    k_settle_baoab_water<RealType, 3><<<wgrid, tpb, 0, stream>>>(
        num_systems, N, num_water_clusters_, d_water_cluster_ids_.data,
        d_cluster_atom_offsets_.data, d_cluster_atoms_.data,
        d_cluster_constraint_offsets_.data, d_constraint_r0_.data, d_cbs, d_ccs,
        d_inv_mass, d_rand_states, d_du_dx, d_x, d_v, ca, half_dt);
    gpuErrchk(cudaPeekAtLastError());

    if (num_general_clusters_ > 0) {
      dim3 ggrid(ceil_divide(num_general_clusters_, tpb), num_systems);
      k_constrained_baoab_cluster<RealType, 3><<<ggrid, tpb, 0, stream>>>(
          num_systems, N, num_clusters_, d_general_cluster_ids_.data,
          num_general_clusters_, d_cluster_atom_offsets_.data,
          d_cluster_atoms_.data, d_cluster_constraint_offsets_.data,
          d_constraint_local_i_.data, d_constraint_local_j_.data,
          d_constraint_r0_.data, d_cbs, d_ccs, d_inv_mass, d_idxs,
          d_rand_states, d_du_dx, d_x, d_v, ca, half_dt, pos_tol_, vel_tol_,
          max_iters_);
      gpuErrchk(cudaPeekAtLastError());
    }
  } else {
    dim3 grid(ceil_divide(num_clusters_, tpb), num_systems);
    k_constrained_baoab_cluster<RealType, 3><<<grid, tpb, 0, stream>>>(
        num_systems, N, num_clusters_, nullptr, 0, d_cluster_atom_offsets_.data,
        d_cluster_atoms_.data, d_cluster_constraint_offsets_.data,
        d_constraint_local_i_.data, d_constraint_local_j_.data,
        d_constraint_r0_.data, d_cbs, d_ccs, d_inv_mass, d_idxs, d_rand_states,
        d_du_dx, d_x, d_v, ca, half_dt, pos_tol_, vel_tol_, max_iters_);
    gpuErrchk(cudaPeekAtLastError());
  }
}

template class Constraints<float>;
template class Constraints<double>;

} // namespace tmd
