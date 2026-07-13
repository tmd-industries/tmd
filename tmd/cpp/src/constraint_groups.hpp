#pragma once

#include <vector>

namespace tmd {

template <typename RealType> class ConstraintGroups {

private:
  const int batch_size_;
  const int N_;
  const int n_groups_;
  int max_group_size_;
  const int iterations_;
  const RealType tolerance_;
  int total_atoms_in_constraints_;

  RealType *d_inv_masses_;
  int *d_group_offsets_;
  int *d_group_indices_;
  int *d_distance_offsets_;
  RealType *d_distances_;
  // Optionally store positions of constrained atoms before applying positional
  // restraints Useful for some integrators (BAOAB notably)
  RealType *d_unadjusted_group_coords_;

public:
  ConstraintGroups(const int batch_size, const int N, const RealType *masses,
                   const std::vector<std::vector<int>> groups,
                   const std::vector<std::vector<RealType>> distances,
                   const int iterations = 15, const RealType tolerance = 1e-5);

  virtual ~ConstraintGroups();

  int num_constrained_atoms() const { return total_atoms_in_constraints_; }

  int num_systems() const { return batch_size_; }

  int num_atoms() const { return N_; }

  int n_groups() const { return n_groups_; }

  int max_group_size() const { return max_group_size_; }

  int iterations() const { return iterations_; }

  RealType tolerance() const { return tolerance_; }

  int *get_group_offsets() const { return d_group_offsets_; }

  int *get_group_indices() const { return d_group_indices_; }

  RealType *get_previous_group_coords() const {
    return d_unadjusted_group_coords_;
  }

  void constrain_positions(const int num_systems, const int N, RealType *d_x_t,
                           const unsigned int *idxs, const bool store_current_x,
                           cudaStream_t stream) const;

  void constrain_velocities(const int num_systems, const int N,
                            const RealType *d_x_t, RealType *d_v_t,
                            const unsigned int *idxs,
                            cudaStream_t stream) const;

  RealType *constrain_positions_host(const int num_systems, const int N,
                                     const RealType *h_coords,
                                     const unsigned int *h_idxs) const;

  RealType *constrain_velocities_host(const int num_systems, const int N,
                                      const RealType *h_coords,
                                      const RealType *h_velocities,
                                      const unsigned int *h_idxs) const;

private:
  void run_rattle(const int num_systems, const int N, const RealType *d_x_t,
                  RealType *d_v_t, const unsigned int *idxs,
                  cudaStream_t stream) const;

  void run_shake(const int num_systems, const int N, RealType *d_x_t,
                 const unsigned int *idxs, const bool store_current_x,
                 cudaStream_t stream) const;
};

} // namespace tmd
