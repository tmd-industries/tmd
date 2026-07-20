#pragma once

#include <memory>
#include <vector>

#include "constraint_groups.hpp"
#include "langevin_integrator.hpp"

namespace tmd {

template <typename RealType>
class ConstrainedLangevinIntegrator : public LangevinIntegrator<RealType> {

private:
  std::shared_ptr<ConstraintGroups<RealType>> constraints_;

public:
  ConstrainedLangevinIntegrator(
      const int batch_size, const int N, const RealType *masses,
      const RealType temperature, const RealType dt, const RealType friction,
      const int seed, std::shared_ptr<ConstraintGroups<RealType>> constraints);

  virtual ~ConstrainedLangevinIntegrator();

  std::shared_ptr<ConstraintGroups<RealType>> get_constraints() const {
    return constraints_;
  };

  virtual void
  step_fwd(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
           RealType *d_x_t, RealType *d_v_t, RealType *d_box_t_,
           unsigned int *d_idxs, cudaStream_t stream) override;
};

} // namespace tmd
