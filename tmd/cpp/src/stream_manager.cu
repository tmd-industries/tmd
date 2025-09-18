// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025 Forrest York
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
#include "kernel_utils.cuh"
#include "stream_manager.hpp"

namespace tmd {

StreamManager::StreamManager() {};

StreamManager::~StreamManager() {
  for (const auto &[key, value] : streams_) {
    gpuErrchk(cudaStreamDestroy(value));
  }
  for (const auto &[key, value] : events_) {
    gpuErrchk(cudaEventDestroy(value));
  }
}

cudaEvent_t StreamManager::get_master_event() {
  int master_key = -1;
  return this->get_stream_event(master_key);
}

cudaStream_t StreamManager::get_stream(int key) {
  if (streams_.count(key) == 1) {
    return streams_[key];
  }
  cudaStream_t new_stream;
  // Create stream that doesn't block with the null stream to avoid
  // unintentional blocking.
  gpuErrchk(cudaStreamCreateWithFlags(&new_stream, cudaStreamNonBlocking));

  streams_[key] = new_stream;
  return new_stream;
};

cudaEvent_t StreamManager::get_stream_event(int key) {
  if (events_.count(key) == 1) {
    return events_[key];
  }
  cudaEvent_t new_event;
  // Create event with timings disabled as timings slow down events
  gpuErrchk(cudaEventCreateWithFlags(&new_event, cudaEventDisableTiming));

  events_[key] = new_event;
  return new_event;
};

void StreamManager::record_master_event(cudaStream_t parent_stream) {
  cudaEvent_t master_event = this->get_master_event();
  gpuErrchk(cudaEventRecord(master_event, parent_stream));
}

// indicate that child streams should wait on master
void StreamManager::wait_on_master(int key, cudaStream_t parent_stream) {
  cudaEvent_t master_event = this->get_master_event();
  cudaStream_t child_stream = this->get_stream(key);
  gpuErrchk(cudaStreamWaitEvent(child_stream, master_event));
};

// indicate that master should wait on children
void StreamManager::record_and_wait_on_child(int key,
                                             cudaStream_t parent_stream) {
  cudaStream_t child_stream = this->get_stream(key);
  cudaEvent_t event = this->get_stream_event(key);
  gpuErrchk(cudaEventRecord(event, child_stream));
  gpuErrchk(cudaStreamWaitEvent(parent_stream, event));
};

} // namespace tmd
