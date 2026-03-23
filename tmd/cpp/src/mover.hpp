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

#include <array>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace tmd {

// Base class for generic moves that can be accepted by Context
template <typename RealType> class Mover {

protected:
  Mover<RealType>(const int num_systems, const int interval)
      : num_systems_(num_systems), interval_(interval), step_(0){};
  const int num_systems_;
  int interval_;
  int step_;

public:
  virtual ~Mover<RealType>(){};

  // set_step is to deal with HREX where a mover may not be called during the
  // singular frame being generated. TBD come up with an explicit, chainable API
  // that works around this jank
  void set_step(const int step) {
    if (step < 0) {
      throw std::runtime_error("step must be at least 0");
    }
    this->step_ = step;
  }

  void set_interval(const int interval) {
    if (interval <= 0) {
      throw std::runtime_error("interval must be greater than 0");
    }
    this->interval_ = interval;
    // Clear the step, to ensure user can expect that in interval steps the
    // barostat will trigger
    this->step_ = 0;
  }

  int get_interval() const { return this->interval_; };

  int num_systems() const { return this->num_systems_; };

  virtual void move(const int num_systems, const int N, RealType *d_x,
                    RealType *d_box, cudaStream_t stream) = 0;

  virtual std::array<std::vector<RealType>, 2>
  move_host(const int N, const RealType *h_x, const RealType *h_box);
};

} // namespace tmd
