// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025 Forrest York
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

#include "barostat.hpp"
#include "bound_potential.hpp"
#include "integrator.hpp"
#include "local_md_potentials.hpp"
#include "mover.hpp"
#include "potential.hpp"
#include <array>
#include <vector>

namespace tmd {

template <typename RealType> class Context {

public:
  Context(const int num_systems, const int N, const RealType *x_0,
          const RealType *v_0, const RealType *box_0,
          std::shared_ptr<Integrator<RealType>> intg,
          std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
          std::vector<std::shared_ptr<Mover<RealType>>> &movers);

  ~Context();

  void step();
  void initialize();
  void finalize();

  void multiple_steps(const int n_steps, const int n_samples, RealType *h_x,
                      RealType *h_box);

  void multiple_steps_local(const int n_steps,
                            const std::vector<int> &local_idxs,
                            const int n_samples, const RealType radius,
                            const RealType k, const int seed, RealType *h_x,
                            RealType *h_box);

  void multiple_steps_local_selection(const int n_steps,
                                      const int reference_idx,
                                      const std::vector<int> &selection_idxs,
                                      const int n_samples,
                                      const RealType radius, const RealType k,
                                      RealType *h_x, RealType *h_box);

  int num_atoms() const;

  int num_systems() const { return num_systems_; };

  void set_x_t(const RealType *in_buffer);

  void get_x_t(RealType *out_buffer) const;

  void set_v_t(const RealType *in_buffer);

  void get_v_t(RealType *out_buffer) const;

  void set_box(const RealType *in_buffer);

  void get_box(RealType *out_buffer) const;

  void setup_local_md(RealType temperature, bool freeze_reference,
                      RealType nblist_padding =
                          LocalMDPotentials<RealType>::DEFAULT_NBLIST_PADDING);

  std::vector<std::shared_ptr<BoundPotential<RealType>>>
  truncate_potentials_local_selection(const int reference_idx,
                                      const std::vector<int> &selection_idxs,
                                      const RealType radius, const RealType k);

  std::shared_ptr<Integrator<RealType>> get_integrator() const;

  std::vector<std::shared_ptr<BoundPotential<RealType>>> get_potentials() const;

  std::vector<std::shared_ptr<BoundPotential<RealType>>>
  get_local_md_potentials() const;

  std::vector<std::shared_ptr<Mover<RealType>>> get_movers() const;

  std::shared_ptr<MonteCarloBarostat<RealType>> get_barostat() const;

private:
  const int num_systems_;
  int N_; // number of particles

  cudaStream_t stream_;

  std::vector<std::shared_ptr<Mover<RealType>>> movers_;

  void _step(std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
             unsigned int *d_atom_idxs, const cudaStream_t stream);

  RealType _get_temperature();

  void _ensure_local_md_intialized();

  void _verify_coords_and_box(const RealType *coords_buffer,
                              const RealType *box_buffer, cudaStream_t stream);

  int step_;

  RealType *d_x_t_;   // coordinates
  RealType *d_v_t_;   // velocities
  RealType *d_box_t_; // box vectors

  std::shared_ptr<Integrator<RealType>> intg_;
  std::vector<std::shared_ptr<BoundPotential<RealType>>> bps_;
  std::vector<std::shared_ptr<Potential<RealType>>>
      nonbonded_pots_; // Potentials used to verify the system hasn't blown up
  std::unique_ptr<LocalMDPotentials<RealType>> local_md_pots_;
};

} // namespace tmd
