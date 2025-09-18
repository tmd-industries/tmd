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

namespace tmd {

template <typename T> class PinnedHostBuffer {
public:
  PinnedHostBuffer(const size_t length);

  ~PinnedHostBuffer();

  const size_t size;

  T *const data;

  void copy_from(const T *host_buffer) const;

  void copy_to(T *host_buffer) const;
};

} // namespace tmd
