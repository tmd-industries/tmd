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

#include "all_atom_energies.hpp"
#include "device_buffer.hpp"
#include "kernels/k_nonbonded.cuh"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"

namespace tmd {

template <typename RealType>
std::vector<RealType> compute_atom_by_atom_energies(
    const int N, const std::vector<int> &target_atoms,
    const std::vector<RealType> &coords, const std::vector<RealType> &params,
    std::vector<RealType> &box, const RealType nb_beta, const RealType cutoff) {
  const DeviceBuffer<int> d_target_atoms(target_atoms);
  const DeviceBuffer<RealType> d_coords(coords);
  const DeviceBuffer<RealType> d_params(params);
  const DeviceBuffer<RealType> d_box(box);
  DeviceBuffer<RealType> d_energy_output(N * target_atoms.size());
  RealType cutoff_squared = cutoff * cutoff;

  cudaStream_t stream = static_cast<cudaStream_t>(0);

  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  dim3 dimGrid(ceil_divide(N, tpb), d_target_atoms.length, 1);

  k_atom_by_atom_energies<RealType><<<dimGrid, tpb, 0, stream>>>(
      N, static_cast<int>(d_target_atoms.length), d_target_atoms.data,
      nullptr, // Use the provided coords to compute the energies
      d_coords.data, d_params.data, d_box.data, nb_beta, cutoff_squared,
      d_energy_output.data);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaStreamSynchronize(stream));

  std::vector<RealType> energy_out(d_energy_output.length);
  d_energy_output.copy_to(&energy_out[0]);

  return energy_out;
}

template std::vector<float> compute_atom_by_atom_energies<float>(
    const int N, const std::vector<int> &target_atoms,
    const std::vector<float> &coords, const std::vector<float> &params,
    std::vector<float> &box, float nb_beta, float cutoff);

template std::vector<double> compute_atom_by_atom_energies<double>(
    const int N, const std::vector<int> &target_atoms,
    const std::vector<double> &coords, const std::vector<double> &params,
    std::vector<double> &box, double nb_beta, double cutoff);

} // namespace tmd
