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
#include <memory>
#include <vector>

#include "constraints.hpp"
#include "curand_kernel.h"
#include "device_buffer.hpp"
#include "integrator.hpp"
#include "streamed_potential_runner.hpp"

namespace tmd {

// ConstrainedLangevinIntegrator implements a BAOAB Langevin integrator (as in
// LangevinIntegrator) augmented with SHAKE/RATTLE distance constraints, used to
// freeze bonds involving hydrogens. The integrator uses the geodesic /
// "middle" constrained-BAOAB scheme: the velocity (B) and Ornstein-Uhlenbeck
// (O) sub-steps are followed by a RATTLE velocity projection, and each drift
// (A) sub-step is followed by a SHAKE position projection together with the
// associated constraint-consistent velocity correction. Exactly one force
// evaluation is performed per step, matching the unconstrained integrator.
template <typename RealType>
class ConstrainedLangevinIntegrator : public Integrator<RealType> {

private:
  const int batch_size_;
  const int N_;
  const RealType temperature_;
  const RealType dt_;
  const RealType friction_;
  const RealType ca_;
  DeviceBuffer<curandState_t> d_rand_states_;

  RealType *d_cbs_;      // [batch_size * N] dt / mass
  RealType *d_ccs_;      // [batch_size * N] noise scale
  RealType *d_inv_mass_; // [batch_size * N] 1 / mass (0 for frozen atoms)
  RealType *d_x_ref_;    // [batch_size * N * 3] scratch for pre-drift positions
  unsigned long long *d_du_dx_;

  std::shared_ptr<Constraints<RealType>> constraints_;

  StreamedPotentialRunner<RealType> runner_;

  // Per-atom (per system) mask: 1 if the atom belongs to a constraint cluster.
  // Cluster atoms are integrated by the fused per-cluster kernel; the remaining
  // atoms are integrated by a separate per-atom kernel.
  DeviceBuffer<uint8_t> d_is_cluster_atom_;
  // False when every atom belongs to a cluster (e.g. pure water): the per-atom
  // non-cluster kernel can then be skipped entirely.
  bool has_noncluster_atoms_;
  // Fuse the entire post-force sub-step sequence into one per-cluster kernel
  // (plus one non-cluster kernel) instead of issuing a kernel per sub-step.
  // Disabled by setting TMD_NO_CONSTRAINT_FUSION (e.g. for A/B comparison).
  bool use_fusion_;

  // Issue the fixed post-force sequence of integrator sub-step kernels.
  // Dispatches to the fused or per-sub-step implementation depending on
  // use_fusion_.
  void step_post_force(RealType *d_x_t, RealType *d_v_t, unsigned int *d_idxs,
                       cudaStream_t stream);
  // Fused implementation: one per-cluster kernel (plus one non-cluster kernel).
  void step_post_force_fused(RealType *d_x_t, RealType *d_v_t,
                             unsigned int *d_idxs, cudaStream_t stream);
  // Reference implementation: one kernel per integrator sub-step.
  void step_post_force_unfused(RealType *d_x_t, RealType *d_v_t,
                               unsigned int *d_idxs, cudaStream_t stream);

public:
  ConstrainedLangevinIntegrator(
      const int batch_size, const int N, const RealType *masses,
      const RealType temperature, const RealType dt, const RealType friction,
      const int seed, std::shared_ptr<Constraints<RealType>> constraints);

  virtual ~ConstrainedLangevinIntegrator();

  RealType get_temperature() const;

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
