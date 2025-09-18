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

// this implements a stream manager that allows creation of new streams as well
// as syncing two streams. Handles the creation and destruction of streams and
// events, of which stream destruction is blocking while event destruction is
// not. Streams are all created to not sync with the NULL stream and events have
// timings disabled.
#pragma once

#include <map>

namespace tmd {

class StreamManager {

public:
  StreamManager();

  ~StreamManager();

  // get_stream handles the creation and retrieval of cuda streams. Streams are
  // configured to not sync implicitly with the NULL stream.
  cudaStream_t get_stream(int key);

  // get_event handles the creation and retrieval of cuda events. The events
  // have timings disabled for performance.
  cudaEvent_t get_stream_event(int key);

  cudaEvent_t get_master_event();

  void record_master_event(cudaStream_t parent_stream);

  // sync_to will sync the to_stream stream with the stream associated with the
  // key. This is done on the GPU and does not block.
  void wait_on_master(int key, cudaStream_t parent_stream);

  // sync_from will sync the stream associated with the key to the from_stream.
  // This is done on the GPU and does not block.
  void record_and_wait_on_child(int key, cudaStream_t parent_stream);

private:
  std::map<int, cudaStream_t> streams_;
  std::map<int, cudaEvent_t> events_;
};

} // namespace tmd
