#pragma once

#include <memory>
#include <vector>

#include "langevin_integrator.hpp"

namespace tmd {

template <typename RealType>
class ConstrainedLangevinIntegrator : public LangevinIntegrator<RealType> {

private:
  const int n_groups_;
  const int iterations_;
  const RealType tolerance_;
  int max_group_size_; // Largest group size

  RealType *d_inv_masses_;
  int *d_group_offsets_; // [n_groups + 1]
  int *d_group_indices_;
  RealType *d_distances_;
  int *d_distances_offsets_;
  // Store the pre-SHAKE positions to correct the velocities
  RealType *d_unadjusted_group_coords_;

public:
  ConstrainedLangevinIntegrator(
      const int batch_size, const int N, const RealType *masses,
      const RealType temperature, const RealType dt, const RealType friction,
      const int seed, const std::vector<std::vector<int>> groups,
      const std::vector<std::vector<RealType>> distances,
      const RealType tolerance, const int iterations);

  virtual ~ConstrainedLangevinIntegrator();

  virtual void
  step_fwd(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
           RealType *d_x_t, RealType *d_v_t, RealType *d_box_t_,
           unsigned int *d_idxs, cudaStream_t stream) override;
};

} // namespace tmd
