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

#include "gpu_utils.cuh"
#include "pinned_host_buffer.hpp"

namespace tmd {

template <typename T>
static T *allocate_pinned_host_memory(const std::size_t length) {
  if (length < 1) {
    throw std::runtime_error("device buffer length must at least be 1");
  }
  T *buffer;
  gpuErrchk(cudaMallocHost(&buffer, length * sizeof(T)));
  return buffer;
}

template <typename T>
PinnedHostBuffer<T>::PinnedHostBuffer(const std::size_t length)
    : size(length * sizeof(T)), data(allocate_pinned_host_memory<T>(length)) {}

template <typename T> PinnedHostBuffer<T>::~PinnedHostBuffer() {
  // TODO: the file/line context reported by gpuErrchk on failure is
  // not very useful when it's called from here. Is there a way to
  // report a stack trace?
  gpuErrchk(cudaFreeHost(data));
}

template <typename T>
void PinnedHostBuffer<T>::copy_from(const T *host_buffer) const {
  memcpy(data, host_buffer, size);
}

template <typename T> void PinnedHostBuffer<T>::copy_to(T *host_buffer) const {
  memcpy(host_buffer, data, size);
}

template class PinnedHostBuffer<double>;
template class PinnedHostBuffer<float>;
template class PinnedHostBuffer<int>;
template class PinnedHostBuffer<char>;
template class PinnedHostBuffer<unsigned int>;
template class PinnedHostBuffer<unsigned long long>;
} // namespace tmd
