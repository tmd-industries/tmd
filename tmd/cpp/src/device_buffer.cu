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

#include "curand_kernel.h"
#include "device_buffer.hpp"
#include "gpu_utils.cuh"
#include <cstddef>
#include <cub/util_type.cuh>

namespace tmd {

template <typename T> static T *allocate_gpu_memory(const std::size_t length) {
  T *buffer;
  cudaSafeMalloc(&buffer, length * sizeof(T));
  return buffer;
}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::vector<T> &vec)
    : DeviceBuffer(vec.size()) {
  this->copy_from(&vec[0]);
}

// Create a new DeviceBuffer that copies the data in an existing device array.
template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t length, const T *d_arr)
    : DeviceBuffer(length) {
  gpuErrchk(cudaMemcpy(data, d_arr, this->size(), cudaMemcpyDeviceToDevice));
}

template <typename T> DeviceBuffer<T>::DeviceBuffer() : DeviceBuffer(0) {}

template <typename T>
DeviceBuffer<T>::DeviceBuffer(const std::size_t length)
    : length(length), data(allocate_gpu_memory<T>(length)) {}

template <typename T> void DeviceBuffer<T>::realloc(const size_t new_length) {
  // Print a warning if buffers were non-zero when resized, this can have real
  // performance impacts
  if (this->length > 0) {
    std::cout << "warning:: resizing device buffer that is non-zero"
              << std::endl;
  }
  // Free the existing data
  gpuErrchk(cudaFree(data));
  this->length = new_length;
  this->data = allocate_gpu_memory<T>(new_length);
}

template <typename T> DeviceBuffer<T>::~DeviceBuffer() {
  // TODO: the file/line context reported by gpuErrchk on failure is
  // not very useful when it's called from here. Is there a way to
  // report a stack trace?
  gpuErrchk(cudaFree(data));
}

template <typename T> size_t DeviceBuffer<T>::size() const {
  return this->length * sizeof(T);
}

template <typename T>
void DeviceBuffer<T>::copy_from(const T *host_buffer) const {
  gpuErrchk(
      cudaMemcpy(data, host_buffer, this->size(), cudaMemcpyHostToDevice));
}

template <typename T> void DeviceBuffer<T>::copy_to(T *host_buffer) const {
  gpuErrchk(
      cudaMemcpy(host_buffer, data, this->size(), cudaMemcpyDeviceToHost));
}

template class DeviceBuffer<double>;
template class DeviceBuffer<float>;
template class DeviceBuffer<int>;
template class DeviceBuffer<size_t>;
template class DeviceBuffer<char>;
template class DeviceBuffer<unsigned int>;
template class DeviceBuffer<unsigned long long>;
template class DeviceBuffer<__int128>;
template class DeviceBuffer<cub::KeyValuePair<int, double>>;
template class DeviceBuffer<cub::KeyValuePair<int, float>>;
template class DeviceBuffer<curandState_t>;
} // namespace tmd
