// Copyright 2025 Forrest York
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
#include <stdexcept>

namespace tmd {

template <typename RealType>
Constraints<RealType>::Constraints(
    const std::vector<int> &cluster_atom_offsets,
    const std::vector<int> &cluster_atoms,
    const std::vector<int> &cluster_constraint_offsets,
    const std::vector<int> &constraint_local_i,
    const std::vector<int> &constraint_local_j,
    const std::vector<RealType> &constraint_r0, const RealType pos_tol,
    const RealType vel_tol, const int max_iters)
    : num_clusters_(static_cast<int>(cluster_atom_offsets.size()) - 1),
      num_constraints_(static_cast<int>(constraint_r0.size())),
      pos_tol_(pos_tol), vel_tol_(vel_tol), max_iters_(max_iters),
      d_cluster_atom_offsets_(cluster_atom_offsets),
      d_cluster_atoms_(cluster_atoms),
      d_cluster_constraint_offsets_(cluster_constraint_offsets),
      d_constraint_local_i_(constraint_local_i),
      d_constraint_local_j_(constraint_local_j),
      d_constraint_r0_(constraint_r0) {

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
}

template <typename RealType> Constraints<RealType>::~Constraints() {}

template <typename RealType>
void Constraints<RealType>::apply_position_constraints(
    const int num_systems, const int N, const RealType *d_inv_mass,
    const RealType *d_x_ref, RealType *d_x, RealType *d_v,
    const RealType dt_drift, const bool update_velocity, cudaStream_t stream) {
  if (num_clusters_ == 0) {
    return;
  }
  const size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  dim3 grid(ceil_divide(num_clusters_, tpb), num_systems);
  k_constrain_positions<RealType><<<grid, tpb, 0, stream>>>(
      num_systems, N, num_clusters_, d_cluster_atom_offsets_.data,
      d_cluster_atoms_.data, d_cluster_constraint_offsets_.data,
      d_constraint_local_i_.data, d_constraint_local_j_.data,
      d_constraint_r0_.data, d_inv_mass, d_x_ref, d_x, d_v, dt_drift, pos_tol_,
      max_iters_, update_velocity);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void Constraints<RealType>::apply_velocity_constraints(
    const int num_systems, const int N, const RealType *d_inv_mass,
    const RealType *d_x, RealType *d_v, cudaStream_t stream) {
  if (num_clusters_ == 0) {
    return;
  }
  const size_t tpb = DEFAULT_THREADS_PER_BLOCK;
  dim3 grid(ceil_divide(num_clusters_, tpb), num_systems);
  k_constrain_velocities<RealType><<<grid, tpb, 0, stream>>>(
      num_systems, N, num_clusters_, d_cluster_atom_offsets_.data,
      d_cluster_atoms_.data, d_cluster_constraint_offsets_.data,
      d_constraint_local_i_.data, d_constraint_local_j_.data, d_inv_mass, d_x,
      d_v, vel_tol_, max_iters_);
  gpuErrchk(cudaPeekAtLastError());
}

template class Constraints<float>;
template class Constraints<double>;

} // namespace tmd
