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

#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include "kernels/k_translations.cuh"
#include "math_utils.cuh"
#include "translations.hpp"

namespace tmd {

template <typename RealType>
std::vector<RealType> translations_inside_and_outside_sphere_host(
    const int n_translations, const std::vector<RealType> &box,
    const std::vector<RealType> &center, const RealType radius,
    const int seed) {

  const int tpb = DEFAULT_THREADS_PER_BLOCK;

  DeviceBuffer<RealType> d_box(box);
  DeviceBuffer<RealType> d_center(center);
  DeviceBuffer<curandState_t> d_states(tpb);
  DeviceBuffer<RealType> d_output(n_translations * 3 * 2);

  cudaStream_t stream = static_cast<cudaStream_t>(0);

  k_initialize_curand_states<<<1, tpb, 0, stream>>>(
      static_cast<int>(d_states.length), seed, d_states.data);
  gpuErrchk(cudaPeekAtLastError());

  // Generate two translations per translation requested, one inner, one outer
  k_generate_translations_inside_and_outside_sphere<RealType>
      <<<1, tpb, 0, stream>>>(n_translations, d_box.data, d_center.data, radius,
                              d_states.data, d_output.data);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaStreamSynchronize(stream));

  std::vector<RealType> h_out(d_output.length);
  d_output.copy_to(&h_out[0]);
  return h_out;
}

template std::vector<float> translations_inside_and_outside_sphere_host<float>(
    const int, const std::vector<float> &box, const std::vector<float> &center,
    const float, const int);
template std::vector<double>
translations_inside_and_outside_sphere_host<double>(
    const int, const std::vector<double> &box,
    const std::vector<double> &center, const double, const int);

} // namespace tmd
