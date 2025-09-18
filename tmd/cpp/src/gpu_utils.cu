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

namespace tmd {

int round_up_even(int count) { return count + (count % 2); }

curandStatus_t templateCurandNormal(curandGenerator_t generator,
                                    float *outputPtr, size_t n, float mean,
                                    float stddev) {
  return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

curandStatus_t templateCurandNormal(curandGenerator_t generator,
                                    double *outputPtr, size_t n, double mean,
                                    double stddev) {
  return curandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}

curandStatus_t templateCurandUniform(curandGenerator_t generator,
                                     float *outputPtr, size_t n) {
  return curandGenerateUniform(generator, outputPtr, n);
}

curandStatus_t templateCurandUniform(curandGenerator_t generator,
                                     double *outputPtr, size_t n) {
  return curandGenerateUniformDouble(generator, outputPtr, n);
}

void __global__ k_initialize_curand_states(const int count, const int seed,
                                           curandState_t *__restrict__ states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  while (idx < count) {
    // Set the sequence to idx to ensure values aren't statistically correlated
    // values
    curand_init(seed, idx, 0, &states[idx]);
    idx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
