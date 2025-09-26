// Copyright 2025, Forrest York
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
#include "potential_executor.hpp"

namespace tmd {

template <typename RealType>
PotentialExecutor<RealType>::PotentialExecutor(bool parallel)
    : parallel_(parallel) {
  gpuErrchk(cudaStreamCreate(&stream_));
};

template <typename RealType> PotentialExecutor<RealType>::~PotentialExecutor() {
  gpuErrchk(cudaStreamDestroy(stream_));
}

template <typename RealType>
int PotentialExecutor<RealType>::get_total_num_params(
    const std::vector<int> &param_sizes) const {
  int P = 0;
  for (int size : param_sizes) {
    P += size;
  }
  return P;
}

template <typename RealType>
void PotentialExecutor<RealType>::execute_potentials(
    const int N, const std::vector<int> &param_sizes,
    const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
    const RealType *h_x, const RealType *h_params, const RealType *h_box,
    unsigned long long *h_du_dx, unsigned long long *h_du_dp, __int128 *h_u) {

  const int num_pots = pots.size();
  const int P = get_total_num_params(param_sizes);
  DeviceBuffer<RealType> d_params(P);
  DeviceBuffer<RealType> d_x(N * D);
  DeviceBuffer<RealType> d_box(D * D);
  d_params.copy_from(h_params);
  d_x.copy_from(h_x);
  d_box.copy_from(h_box);

  DeviceBuffer<unsigned long long> d_du_dx_buffer(0);
  if (h_du_dx != nullptr) {
    d_du_dx_buffer.realloc(num_pots * N * D);
    gpuErrchk(cudaMemsetAsync(d_du_dx_buffer.data, 0, d_du_dx_buffer.size(),
                              stream_));
  }

  DeviceBuffer<__int128> d_u_buffer(0);
  if (h_u != nullptr) {
    d_u_buffer.realloc(num_pots);
    gpuErrchk(cudaMemsetAsync(d_u_buffer.data, 0, d_u_buffer.size(), stream_));
  }

  DeviceBuffer<unsigned long long> d_du_dp_buffer(0);
  if (h_du_dp != nullptr) {
    d_du_dp_buffer.realloc(P);
    gpuErrchk(cudaMemsetAsync(d_du_dp_buffer.data, 0, d_du_dp_buffer.size(),
                              stream_));
  }

  this->execute_potentials_device(
      N, param_sizes, pots, d_x.data, d_params.data, d_box.data,
      h_du_dx == nullptr ? nullptr : d_du_dx_buffer.data,
      h_du_dp == nullptr ? nullptr : d_du_dp_buffer.data,
      h_u == nullptr ? nullptr : d_u_buffer.data, stream_);

  if (h_du_dx != nullptr) {
    gpuErrchk(cudaMemcpyAsync(h_du_dx, d_du_dx_buffer.data,
                              d_du_dx_buffer.size(), cudaMemcpyDeviceToHost,
                              stream_));
  }

  if (h_du_dp != nullptr) {
    gpuErrchk(cudaMemcpyAsync(h_du_dp, d_du_dp_buffer.data,
                              d_du_dp_buffer.size(), cudaMemcpyDeviceToHost,
                              stream_));
  }

  if (h_u != nullptr) {
    gpuErrchk(cudaMemcpyAsync(h_u, d_u_buffer.data, d_u_buffer.size(),
                              cudaMemcpyDeviceToHost, stream_));
  }

  gpuErrchk(cudaStreamSynchronize(stream_));
};

template <typename RealType>
void PotentialExecutor<RealType>::execute_potentials_device(
    const int N, const std::vector<int> &param_sizes,
    const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
    const RealType *d_x, const RealType *d_params, const RealType *d_box,
    unsigned long long *d_du_dx, unsigned long long *d_du_dp, __int128 *d_u,
    cudaStream_t stream) {

  const int num_pots = pots.size();

  if (parallel_) {
    manager_.record_master_event(stream);
    for (int i = 0; i < num_pots; i++) {
      // Always sync the new streams with the incoming stream to ensure that the
      // state of the incoming buffers are valid
      manager_.wait_on_master(i, stream);
    }
  }
  cudaStream_t pot_stream = stream;

  int offset = 0;
  for (int i = 0; i < num_pots; i++) {
    if (parallel_) {
      pot_stream = manager_.get_stream(i);
    }
    const int pot_param_count = param_sizes[i];
    pots[i]->execute_device(
        1, N, pot_param_count, d_x,
        pot_param_count > 0 ? d_params + offset : nullptr, d_box,
        d_du_dx == nullptr ? nullptr : d_du_dx + (i * N * D),
        d_du_dp == nullptr ? nullptr : d_du_dp + offset,
        d_u == nullptr ? nullptr : d_u + i, pot_stream);

    offset += pot_param_count;
  }
  if (parallel_) {
    for (int i = 0; i < num_pots; i++) {
      manager_.record_and_wait_on_child(i, stream);
    }
  }
};

template <typename RealType>
void PotentialExecutor<RealType>::execute_batch_potentials_sparse(
    const int N, const std::vector<int> &param_sizes, const int batch_size,
    const int coord_batches, const int param_batches,
    const unsigned int *coords_batch_idxs,
    const unsigned int *params_batch_idxs,
    const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
    const RealType *h_x, const RealType *h_params, const RealType *h_box,
    unsigned long long *h_du_dx, unsigned long long *h_du_dp, __int128 *h_u) {

  const int num_pots = pots.size();
  const int P = get_total_num_params(param_sizes);
  DeviceBuffer<RealType> d_params(param_batches * P);
  DeviceBuffer<RealType> d_x(coord_batches * N * D);
  DeviceBuffer<RealType> d_box(coord_batches * D * D);
  d_params.copy_from(h_params);
  d_x.copy_from(h_x);
  d_box.copy_from(h_box);

  DeviceBuffer<unsigned long long> d_du_dx_buffer(0);
  if (h_du_dx != nullptr) {
    d_du_dx_buffer.realloc(batch_size * num_pots * N * D);
    gpuErrchk(cudaMemsetAsync(d_du_dx_buffer.data, 0, d_du_dx_buffer.size(),
                              stream_));
  }

  DeviceBuffer<__int128> d_u_buffer(0);
  if (h_u != nullptr) {
    d_u_buffer.realloc(batch_size * num_pots);
    gpuErrchk(cudaMemsetAsync(d_u_buffer.data, 0, d_u_buffer.size(), stream_));
  }

  DeviceBuffer<unsigned long long> d_du_dp_buffer(0);
  if (h_du_dp != nullptr) {
    d_du_dp_buffer.realloc(batch_size * P);
    gpuErrchk(cudaMemsetAsync(d_du_dp_buffer.data, 0, d_du_dp_buffer.size(),
                              stream_));
  }

  if (parallel_) {
    manager_.record_master_event(stream_);
  }
  // The final shape should be {num_pots, coord_batches, param_batches, ...}
  // which is why this doesn't call execute_potentials_device
  // TBD: Benchmark swapping inner and outer loop
  int param_offset = 0;
  int du_dp_offset = 0;
  for (int i = 0; i < num_pots; i++) {
    cudaStream_t pot_stream = stream_;
    if (parallel_) {
      // Always sync the new streams with the incoming stream to ensure that the
      // state of the incoming buffers are valid
      manager_.wait_on_master(i, stream_);
      pot_stream = manager_.get_stream(i);
    }
    const int pot_param_count = param_sizes[i];
    for (int j = 0; j < batch_size; j++) {
      int ic = coords_batch_idxs[j];
      int ip = params_batch_idxs[j];
      unsigned int offset_factor = (i * batch_size) + j;

      // Note that the parameters are stored {pots, param_batches, ....}
      // which is why we have to handle the offset for parameters differently
      pots[i]->execute_device(
          1, N, pot_param_count, d_x.data + (ic * N * D),
          pot_param_count > 0
              ? d_params.data + (ip * pot_param_count + param_offset)
              : nullptr,
          d_box.data + (ic * D * D),
          h_du_dx == nullptr ? nullptr
                             : d_du_dx_buffer.data + (offset_factor * N * D),
          h_du_dp == nullptr ? nullptr : d_du_dp_buffer.data + du_dp_offset,
          h_u == nullptr ? nullptr : d_u_buffer.data + offset_factor,
          pot_stream);

      du_dp_offset += pot_param_count;
    }
    param_offset += pot_param_count * param_batches;
    if (parallel_) {
      manager_.record_and_wait_on_child(i, stream_);
    }
  }

  if (h_du_dx != nullptr) {
    gpuErrchk(cudaMemcpyAsync(h_du_dx, d_du_dx_buffer.data,
                              d_du_dx_buffer.size(), cudaMemcpyDeviceToHost,
                              stream_));
  }

  if (h_du_dp != nullptr) {
    gpuErrchk(cudaMemcpyAsync(h_du_dp, d_du_dp_buffer.data,
                              d_du_dp_buffer.size(), cudaMemcpyDeviceToHost,
                              stream_));
  }

  if (h_u != nullptr) {
    gpuErrchk(cudaMemcpyAsync(h_u, d_u_buffer.data, d_u_buffer.size(),
                              cudaMemcpyDeviceToHost, stream_));
  }

  gpuErrchk(cudaStreamSynchronize(stream_));
}

template <typename RealType>
void PotentialExecutor<RealType>::du_dp_fixed_to_float(
    const int N, const std::vector<int> &param_sizes,
    const std::vector<std::shared_ptr<Potential<RealType>>> &pots,
    const unsigned long long *h_du_dp, RealType *h_du_dp_float) {
  int offset = 0;

  for (int i = 0; i < pots.size(); i++) {
    int bp_size = param_sizes[i];
    pots[i]->du_dp_fixed_to_float(N, bp_size, h_du_dp + offset,
                                  h_du_dp_float + offset);
    offset += bp_size;
  }
}

template class PotentialExecutor<double>;
template class PotentialExecutor<float>;

} // namespace tmd
