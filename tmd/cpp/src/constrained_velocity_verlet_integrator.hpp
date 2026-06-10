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

#include <memory>
#include <vector>

#include "constraints.hpp"
#include "device_buffer.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace tmd {

// ConstrainedVelocityVerletIntegrator implements a RATTLE velocity-Verlet (NVE)
// integrator augmented with SHAKE/RATTLE distance constraints, used to freeze
// bonds involving hydrogens. Unlike ConstrainedLangevinIntegrator there is no
// thermostat, so the total energy is conserved up to integration error: this
// makes the integrator a clean diagnostic for measuring energy drift as a
// function of timestep.
//
// Velocities are stored synchronized with positions (at integer steps). A step
// consists of: a half kick using the cached force f(x_t); a full drift followed
// by a SHAKE position projection (whose constraint-consistent velocity update
// yields the half-step velocity); a single force evaluation at the new
// positions f(x_{t+dt}); a second half kick; and a RATTLE velocity projection.
// The force at the new positions is cached for the next step's first half kick,
// so exactly one force evaluation is performed per step.
template <typename RealType>
class ConstrainedVelocityVerletIntegrator : public Integrator<RealType> {

private:
  const int batch_size_;
  const int N_;
  const RealType dt_;

  RealType *d_half_cbs_;  // [batch_size * N] (dt / 2) / mass
  RealType *d_inv_mass_;  // [batch_size * N] 1 / mass (0 for frozen atoms)
  RealType *d_x_ref_;     // [batch_size * N * 3] scratch for pre-drift positions
  unsigned long long *d_du_dx_;       // [batch_size * N * 3] force accumulator
  unsigned long long *d_du_dx_cached_; // [batch_size * N * 3] cached f(x_t)

  std::shared_ptr<Constraints<RealType>> constraints_;

  StreamedPotentialRunner<RealType> runner_;

public:
  ConstrainedVelocityVerletIntegrator(
      const int batch_size, const int N, const RealType *masses,
      const RealType dt,
      std::shared_ptr<Constraints<RealType>> constraints);

  virtual ~ConstrainedVelocityVerletIntegrator();

  virtual int num_systems() const override { return batch_size_; };

  virtual void
  step_fwd(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
           RealType *d_x_t, RealType *d_v_t, RealType *d_box_t,
           unsigned int *d_idxs, cudaStream_t stream) override;

  virtual void
  initialize(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
             RealType *d_x_t, RealType *d_v_t, RealType *d_box_t,
             unsigned int *d_idxs, cudaStream_t stream) override;

  virtual void
  finalize(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
           RealType *d_x_t, RealType *d_v_t, RealType *d_box_t,
           unsigned int *d_idxs, cudaStream_t stream) override;
};

} // namespace tmd
