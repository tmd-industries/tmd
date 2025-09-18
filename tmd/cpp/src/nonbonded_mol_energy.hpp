// Copyright 2019-2025, Relay Therapeutics
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

#include <vector>

#include "device_buffer.hpp"

namespace tmd {

// NonbondedMolEnergyPotential computes the energies of one set of molecules
// against another
template <typename RealType> class NonbondedMolEnergyPotential {

private:
  const int N_;
  const int num_target_mols_;
  const RealType beta_;
  const RealType cutoff_squared_;

  DeviceBuffer<int> d_target_atom_idxs_;
  DeviceBuffer<int> d_target_mol_idxs_;
  DeviceBuffer<int> d_target_mol_offsets_;

  // Intermediate buffer for storing the per atom energies
  DeviceBuffer<__int128> d_atom_energy_buffer_;

public:
  NonbondedMolEnergyPotential(const int N,
                              const std::vector<std::vector<int>> &target_mols,
                              const RealType beta, const RealType cutoff);

  ~NonbondedMolEnergyPotential() {};

  void mol_energies_device(const int N, const int target_mols,
                           const RealType *d_coords,    // [N, 3]
                           const RealType *d_params,    // [N, PARAMS_PER_ATOM]
                           const RealType *d_box,       // [3, 3]
                           __int128 *d_output_energies, // [target_mols]
                           cudaStream_t stream);

  std::vector<__int128> mol_energies_host(const int N, const int P,
                                          const RealType *h_coords,
                                          const RealType *h_params,
                                          const RealType *h_box);
};

} // namespace tmd
