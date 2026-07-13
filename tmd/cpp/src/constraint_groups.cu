#include "constraint_groups.hpp"
#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"

#include "kernels/k_constraints.cuh"
#include "kernels/k_math.cuh"

namespace tmd {
template <typename RealType>
ConstraintGroups<RealType>::ConstraintGroups(
    const int batch_size, const int N, const RealType *masses,
    const std::vector<std::vector<int>> groups,
    const std::vector<std::vector<RealType>> distances, const int iterations,
    const RealType tolerance)
    : batch_size_(batch_size), N_(N), n_groups_(groups.size()),
      max_group_size_(0), iterations_(iterations), tolerance_(tolerance) {

  if (iterations <= 0) {
    throw std::runtime_error("iterations must be at least one");
  }
  if (tolerance_ <= 0.0) {
    throw std::runtime_error("tolerance must be greater than 0.0");
  }
  if (n_groups_ != distances.size()) {
    throw std::runtime_error(
        "number of groups must match number of distance arrays");
  }
  std::vector<int> group_offsets;
  std::vector<int> flat_group_idxs;
  int offset = 0;
  total_atoms_in_constraints_ = 0;
  for (auto group : groups) {
    group_offsets.push_back(offset);
    const int group_size = group.size();
    if (group_size <= 1) {
      throw std::runtime_error("must provide groups with at least 2 atoms");
    }
    total_atoms_in_constraints_ += group_size;
    offset += group_size;
    max_group_size_ = max(max_group_size_, group_size);
    flat_group_idxs.reserve(flat_group_idxs.size() +
                            distance(group.begin(), group.end()));
    flat_group_idxs.insert(flat_group_idxs.end(), group.begin(), group.end());
  }
  if (max_group_size_ > 7) {
    throw std::runtime_error("Does not support groups with more than 7 atoms");
  }

  group_offsets.push_back(offset);

  std::vector<int> distance_offsets;
  std::vector<RealType> flat_distances;
  int dist_offset = 0;
  for (auto dists : distances) {
    distance_offsets.push_back(dist_offset);
    dist_offset += dists.size();

    flat_distances.reserve(flat_distances.size() +
                           distance(dists.begin(), dists.end()));
    flat_distances.insert(flat_distances.end(), dists.begin(), dists.end());
  }
  distance_offsets.push_back(dist_offset);

  d_inv_masses_ =
      gpuErrchkCudaMallocAndCopy(masses, this->batch_size_ * this->N_);
  d_group_offsets_ =
      gpuErrchkCudaMallocAndCopy(group_offsets.data(), group_offsets.size());
  d_group_indices_ = gpuErrchkCudaMallocAndCopy(flat_group_idxs.data(),
                                                flat_group_idxs.size());

  d_distance_offsets_ = gpuErrchkCudaMallocAndCopy(distance_offsets.data(),
                                                   distance_offsets.size());
  d_distances_ =
      gpuErrchkCudaMallocAndCopy(flat_distances.data(), flat_distances.size());

  // Buffer to store the pre-SHAKE positions to correct the velocities
  if (total_atoms_in_constraints_ > 0) {
    cudaSafeMalloc(&d_unadjusted_group_coords_,
                   this->batch_size_ * total_atoms_in_constraints_ * 3 *
                       sizeof(*d_unadjusted_group_coords_));
  } else {
    d_unadjusted_group_coords_ = nullptr;
  }

  k_invert_array<RealType>
      <<<ceil_divide(this->batch_size_ * this->N_, DEFAULT_THREADS_PER_BLOCK),
         DEFAULT_THREADS_PER_BLOCK>>>(this->batch_size_ * this->N_,
                                      d_inv_masses_);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType> ConstraintGroups<RealType>::~ConstraintGroups() {
  gpuErrchk(cudaFree(d_inv_masses_));

  gpuErrchk(cudaFree(d_group_indices_));
  gpuErrchk(cudaFree(d_group_offsets_));

  gpuErrchk(cudaFree(d_distances_));
  gpuErrchk(cudaFree(d_distance_offsets_));

  if (d_unadjusted_group_coords_ != nullptr) {
    gpuErrchk(cudaFree(d_unadjusted_group_coords_));
  }
}

template <typename RealType>
void ConstraintGroups<RealType>::constrain_positions(
    const int num_systems, const int N, RealType *d_x_t,
    const unsigned int *idxs, const bool store_current_x,
    cudaStream_t stream) const {

  this->run_shake(num_systems, N, d_x_t, idxs, store_current_x, stream);
}

template <typename RealType>
void ConstraintGroups<RealType>::run_rattle(const int num_systems, const int N,
                                            const RealType *d_x_t,
                                            RealType *d_v_t,
                                            const unsigned int *idxs,
                                            cudaStream_t stream) const {
  constexpr int D = 3;
  constexpr int tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 constraint_dim(ceil_divide(max(1, n_groups_), tpb), num_systems);

  // Apply velocity constraints with RATTLE
  switch (max_group_size_) {
  case 2:
    k_apply_rattle<RealType, D, 2><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
    break;
  case 3:
    k_apply_rattle<RealType, D, 3><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
    break;
  case 4:
    k_apply_rattle<RealType, D, 4><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
    break;
  case 5:
    k_apply_rattle<RealType, D, 5><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
    break;
  case 6:
    k_apply_rattle<RealType, D, 6><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
    break;
  case 7:
    k_apply_rattle<RealType, D, 7><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
    break;
  default:
    throw std::runtime_error("Unexpected group size " +
                             std::to_string(max_group_size_));
  }
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void ConstraintGroups<RealType>::run_shake(const int num_systems, const int N,
                                           RealType *d_x_t,
                                           const unsigned int *idxs,
                                           const bool store_current_x,
                                           cudaStream_t stream) const {
  constexpr int D = 3;
  constexpr int tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 constraint_dim(ceil_divide(max(1, n_groups_), tpb), num_systems);

  // Apply positional constraints with SHAKE
  switch (max_group_size_) {
  case 2:
    k_apply_shake<RealType, D, 2><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_distance_offsets_, d_distances_,
        d_inv_masses_, d_x_t,
        store_current_x ? d_unadjusted_group_coords_ : nullptr);
    break;
  case 3:
    k_apply_shake<RealType, D, 3><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_distance_offsets_, d_distances_,
        d_inv_masses_, d_x_t,
        store_current_x ? d_unadjusted_group_coords_ : nullptr);
    break;
  case 4:
    k_apply_shake<RealType, D, 4><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_distance_offsets_, d_distances_,
        d_inv_masses_, d_x_t,
        store_current_x ? d_unadjusted_group_coords_ : nullptr);
    break;
  case 5:
    k_apply_shake<RealType, D, 5><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_distance_offsets_, d_distances_,
        d_inv_masses_, d_x_t,
        store_current_x ? d_unadjusted_group_coords_ : nullptr);
    break;
  case 6:
    k_apply_shake<RealType, D, 6><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_distance_offsets_, d_distances_,
        d_inv_masses_, d_x_t,
        store_current_x ? d_unadjusted_group_coords_ : nullptr);
    break;
  case 7:
    k_apply_shake<RealType, D, 7><<<constraint_dim, tpb, 0, stream>>>(
        num_systems, N, iterations_, n_groups_, tolerance_, idxs,
        d_group_offsets_, d_group_indices_, d_distance_offsets_, d_distances_,
        d_inv_masses_, d_x_t,
        store_current_x ? d_unadjusted_group_coords_ : nullptr);
    break;
  default:
    throw std::runtime_error("Unexpected group size " +
                             std::to_string(max_group_size_));
  }
  gpuErrchk(cudaPeekAtLastError());
}

template <typename RealType>
void ConstraintGroups<RealType>::constrain_velocities(
    const int num_systems, const int N, const RealType *d_x_t, RealType *d_v_t,
    const unsigned int *idxs, cudaStream_t stream) const {
  this->run_rattle(num_systems, N, d_x_t, d_v_t, idxs, stream);
}

template <typename RealType>
RealType *ConstraintGroups<RealType>::constrain_positions_host(
    const int num_systems, const int N, const RealType *h_coords,
    const unsigned int *h_idxs) const {
  constexpr int D = 3;
  const int total_elements = num_systems * N * D;

  DeviceBuffer<RealType> d_coords(total_elements);
  d_coords.copy_from(h_coords);

  DeviceBuffer<unsigned int> d_idxs_buffer;
  if (h_idxs) {
    d_idxs_buffer.realloc(num_systems * N);
    d_idxs_buffer.copy_from(h_idxs);
  }

  constrain_positions(num_systems, N, d_coords.data,
                      h_idxs ? d_idxs_buffer.data : nullptr, false,
                      static_cast<cudaStream_t>(0));

  gpuErrchk(cudaStreamSynchronize(static_cast<cudaStream_t>(0)));

  RealType *h_result = nullptr;
  cudaMallocHost(&h_result, total_elements * sizeof(RealType));
  gpuErrchk(cudaMemcpy(h_result, d_coords.data,
                       total_elements * sizeof(RealType),
                       cudaMemcpyDeviceToHost));

  return h_result;
}

template <typename RealType>
RealType *ConstraintGroups<RealType>::constrain_velocities_host(
    const int num_systems, const int N, const RealType *h_coords,
    const RealType *h_velocities, const unsigned int *h_idxs) const {
  constexpr int D = 3;
  const int total_elements = num_systems * N * D;

  DeviceBuffer<RealType> d_coords(total_elements);
  DeviceBuffer<RealType> d_velocities(total_elements);
  d_coords.copy_from(h_coords);
  d_velocities.copy_from(h_velocities);

  DeviceBuffer<unsigned int> d_idxs_buffer;
  if (h_idxs) {
    d_idxs_buffer.realloc(num_systems * N);
    d_idxs_buffer.copy_from(h_idxs);
  }

  constrain_velocities(num_systems, N, d_coords.data, d_velocities.data,
                       h_idxs ? d_idxs_buffer.data : nullptr,
                       static_cast<cudaStream_t>(0));

  gpuErrchk(cudaStreamSynchronize(static_cast<cudaStream_t>(0)));

  RealType *h_result = nullptr;
  cudaMallocHost(&h_result, total_elements * sizeof(RealType));
  gpuErrchk(cudaMemcpy(h_result, d_velocities.data,
                       total_elements * sizeof(RealType),
                       cudaMemcpyDeviceToHost));

  return h_result;
}

template class ConstraintGroups<float>;
template class ConstraintGroups<double>;

} // namespace tmd
