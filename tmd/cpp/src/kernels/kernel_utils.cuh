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

namespace tmd {

// DEFAULT_THREADS_PER_BLOCK should be at least 128 to ensure maximum occupancy
// for Cuda Arch 8.6 with 48 SMs given that there aren't too many registers in
// the kernel. Refer to the occupancy calculator in Nsight Compute for more
// details
static const int DEFAULT_THREADS_PER_BLOCK = 128;
static const int WARP_SIZE = 32;
// DEFAULT_THREADS_PER_BLOCK should be multiple of WARP_SIZE, else it is
// wasteful
static_assert(DEFAULT_THREADS_PER_BLOCK % WARP_SIZE == 0);

} // namespace tmd
