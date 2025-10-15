// Copyright 2019-2025, Relay Therapeutics
// Modifications Copyright 2025, Forrest York
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

#include <cuda_runtime.h>

namespace tmd {

// *Not* guaranteed to be thread-safe.
template <typename RealType> class Potential {

public:
  virtual ~Potential() {};

  static const int D;

  void execute_batch_host(const int coord_batch_size, const int N,
                          const int param_batch_size, const int P,
                          const RealType *h_x, const RealType *h_p,
                          const RealType *h_box, unsigned long long *h_du_dx,
                          unsigned long long *h_du_dp, __int128 *h_u);

  void execute_batch_sparse_host(
      const int coords_size, const int N, const int params_size, const int P,
      const int batch_size, const unsigned int *coords_batch_idxs,
      const unsigned int *params_batch_idxs, const RealType *h_x,
      const RealType *h_p, const RealType *h_box, unsigned long long *h_du_dx,
      unsigned long long *h_du_dp, __int128 *h_u);

  void execute_host(const int batches, const int N, const int P,
                    const RealType *h_x, const RealType *h_p,
                    const RealType *h_box, unsigned long long *h_du_dx,
                    unsigned long long *h_du_dp, __int128 *h_u);

  void execute_host_du_dx(const int N, const int P, const RealType *h_x,
                          const RealType *h_p, const RealType *h_box,
                          unsigned long long *h_du_dx);

  void execute_batch_device(const int coord_batch_size, const int N,
                            const int param_batch_size, const int P,
                            const RealType *d_x, const RealType *d_p,
                            const RealType *d_box, unsigned long long *d_du_dx,
                            unsigned long long *d_du_dp, __int128 *d_u,
                            cudaStream_t stream);

  void execute_batch_sparse_device(
      const int N, const int P, const int batch_size,
      const unsigned int *coords_batch_idxs,
      const unsigned int *params_batch_idxs, const RealType *d_x,
      const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
      unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream);

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx,
                              unsigned long long *d_du_dp, __int128 *h_u,
                              cudaStream_t stream) = 0;

  virtual void du_dp_fixed_to_float(const int N, const int P,
                                    const unsigned long long *du_dp,
                                    RealType *du_dp_float);

  virtual int batch_size() const;
};

} // namespace tmd
