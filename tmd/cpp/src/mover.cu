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
#include "mover.hpp"

namespace tmd {

template <typename RealType>
std::array<std::vector<RealType>, 2>
Mover<RealType>::move_host(const int N, const RealType *h_x,
                           const RealType *h_box) {

  DeviceBuffer<RealType> d_x(N * 3);
  DeviceBuffer<RealType> d_box(3 * 3);
  d_x.copy_from(h_x);
  d_box.copy_from(h_box);

  cudaStream_t stream = static_cast<cudaStream_t>(0);

  this->move(1, N, d_x.data, d_box.data, stream);
  gpuErrchk(cudaStreamSynchronize(stream));
  std::vector<RealType> out_coords(d_x.length);
  std::vector<RealType> out_box(d_box.length);
  d_x.copy_to(&out_coords[0]);
  d_box.copy_to(&out_box[0]);
  return std::array<std::vector<RealType>, 2>({out_coords, out_box});
}

template class Mover<double>;
template class Mover<float>;

} // namespace tmd
