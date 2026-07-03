#include "constrained_langevin_integrator.hpp"
#include "gpu_utils.cuh"
#include "kernel_utils.cuh"
#include "langevin_integrator.hpp"
#include "math_utils.cuh"

#include "kernels/k_constraints.cuh"
#include "kernels/k_integrator.cuh"
#include "kernels/k_math.cuh"

namespace tmd {
template <typename RealType>
ConstrainedLangevinIntegrator<RealType>::ConstrainedLangevinIntegrator(
    const int batch_size, const int N, const RealType *masses,
    const RealType temperature, const RealType dt, const RealType friction,
    const int seed, const std::vector<std::vector<int>> groups,
    const std::vector<std::vector<RealType>> distances,
    const RealType tolerance, const int iterations)
    : LangevinIntegrator<RealType>(batch_size, N, masses, temperature, dt,
                                   friction, seed),
      n_groups_(groups.size()), iterations_(iterations), tolerance_(tolerance),
      max_group_size_(0) {

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
  for (auto group : groups) {
    group_offsets.push_back(offset);
    const int group_size = group.size();
    if (group_size <= 1) {
      throw std::runtime_error("must provide groups with at least 2 atoms");
    }
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

  d_distances_offsets_ = gpuErrchkCudaMallocAndCopy(distance_offsets.data(),
                                                    distance_offsets.size());
  d_distances_ =
      gpuErrchkCudaMallocAndCopy(flat_distances.data(), flat_distances.size());

  // Buffer to store the adjusted constraint coordinates
  cudaSafeMalloc(&d_unadjusted_group_coords_,
                 this->batch_size_ * flat_group_idxs.size() * 3 *
                     sizeof(*d_unadjusted_group_coords_));
  k_invert_array<RealType>
      <<<this->batch_size_ *this->N_, DEFAULT_THREADS_PER_BLOCK>>>(
          this->batch_size_ * this->N_, d_inv_masses_);
  gpuErrchk(cudaPeekAtLastError());
}
template <typename RealType>
ConstrainedLangevinIntegrator<RealType>::~ConstrainedLangevinIntegrator() {
  gpuErrchk(cudaFree(d_inv_masses_));

  gpuErrchk(cudaFree(d_group_indices_));
  gpuErrchk(cudaFree(d_group_offsets_));

  gpuErrchk(cudaFree(d_distances_));
  gpuErrchk(cudaFree(d_distances_offsets_));

  gpuErrchk(cudaFree(d_unadjusted_group_coords_));
}

template <typename RealType>
void ConstrainedLangevinIntegrator<RealType>::step_fwd(
    std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
    RealType *d_x_t, RealType *d_v_t, RealType *d_box_t, unsigned int *d_idxs,
    cudaStream_t stream) {

  this->runner_.execute_potentials(this->batch_size_, bps, this->N_, d_x_t,
                                   d_box_t,
                                   this->d_du_dx_, // we only need the forces
                                   nullptr, nullptr, stream);
  constexpr int D = 3;
  constexpr int tpb = DEFAULT_THREADS_PER_BLOCK;
  const dim3 intg_dim(ceil_divide(this->N_, tpb), this->batch_size_);
  // Even if n_groups == 0, run the constraints. Left in for testing purposes
  // to be sure the constrain kernels don't adjust anything unexpectedly.
  const dim3 constraint_dim(ceil_divide(max(1, n_groups_), tpb),
                            this->batch_size_);

  // Perform the initial Kick that adjusts velocity based on forces
  k_update_forward_kick<RealType, D><<<intg_dim, tpb, 0, stream>>>(
      this->batch_size_, this->N_, d_idxs, this->d_cbs_, d_v_t, this->d_du_dx_);
  gpuErrchk(cudaPeekAtLastError());

  // Apply velocity constraints with RATTLE
  k_apply_rattle<RealType, D, 7><<<constraint_dim, tpb, 0, stream>>>(
      this->batch_size_, this->N_, iterations_, n_groups_, tolerance_, d_idxs,
      d_group_offsets_, d_group_indices_, d_inv_masses_, d_x_t, d_v_t);
  gpuErrchk(cudaPeekAtLastError());

  k_update_forward_half_step<RealType, D><<<intg_dim, tpb, 0, stream>>>(
      this->batch_size_, this->N_, this->dt_, this->ca_, d_idxs, this->d_ccs_,
      this->d_rand_states_.data, d_x_t, d_v_t);
  gpuErrchk(cudaPeekAtLastError());

  k_apply_shake<RealType, D, 7><<<constraint_dim, tpb, 0, stream>>>(
      this->batch_size_, this->N_, iterations_, n_groups_, tolerance_, d_idxs,
      d_group_offsets_, d_group_indices_, d_distances_offsets_, d_distances_,
      d_inv_masses_, d_x_t, d_unadjusted_group_coords_);
  gpuErrchk(cudaPeekAtLastError());

  // Correct the final velocities, only if constrains are being applied
  if (n_groups_ > 0) {
    k_apply_velocity_correction<RealType, D, 7>
        <<<constraint_dim, tpb, 0, stream>>>(
            this->batch_size_, this->N_, n_groups_, this->dt_, d_group_offsets_,
            d_group_indices_, d_x_t, d_unadjusted_group_coords_, d_v_t);
    gpuErrchk(cudaPeekAtLastError());
  }
}

template class ConstrainedLangevinIntegrator<float>;
template class ConstrainedLangevinIntegrator<double>;

} // namespace tmd
