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
#include <cstddef>
#include <vector>

namespace tmd {

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer();
  DeviceBuffer(const size_t length);
  DeviceBuffer(const std::vector<T> &vec);
  DeviceBuffer(const size_t length, const T *d_arr);

  ~DeviceBuffer();

  size_t length;

  T *data;

  void realloc(const size_t length);

  // Size returns the number of bytes that make up the buffer unlike the
  // std::container which returns the number of elements. For the number of
  // elements use the `length` property.
  size_t size() const;

  void copy_from(const T *host_buffer) const;

  void copy_to(T *host_buffer) const;
};

} // namespace tmd
