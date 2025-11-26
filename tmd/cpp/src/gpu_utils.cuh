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

#pragma once

#include "curand.h"
#include "curand_kernel.h"
#include "exceptions.hpp"
#include "fixed_point.hpp"
#include "kernels/kernel_utils.cuh"
#include "math_utils.cuh"
#include <iostream>
#include <vector>

namespace tmd {

// round_up_even is important to generating random numbers with cuRand if
// generating Normal noise as the normal generators only generate sets that are
// divisible by the dimension (typically 2) and will return error
// CURAND_STATUS_LENGTH_NOT_MULTIPLE.
// https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437
int round_up_even(int count);

curandStatus_t templateCurandNormal(curandGenerator_t generator,
                                    float *outputPtr, size_t n, float mean,
                                    float stddev);

curandStatus_t templateCurandNormal(curandGenerator_t generator,
                                    double *outputPtr, size_t n, double mean,
                                    double stddev);

template <typename T>
T __device__ __forceinline__ template_curand_uniform(curandState_t *state);

template <>
float __device__ __forceinline__
template_curand_uniform<float>(curandState_t *state) {
  return curand_uniform(state);
}

template <>
double __device__ __forceinline__
template_curand_uniform<double>(curandState_t *state) {
  return curand_uniform_double(state);
}

template <typename T>
T __device__ __forceinline__ template_curand_normal(curandState_t *state);

template <>
float __device__ __forceinline__
template_curand_normal<float>(curandState_t *state) {
  return curand_normal(state);
}

template <>
double __device__ __forceinline__
template_curand_normal<double>(curandState_t *state) {
  return curand_normal_double(state);
}

void __device__ __forceinline__ template_curand_normal2(float &a, float &b,
                                                        curandState_t *state) {
  float2 val = curand_normal2(state);
  a = val.x;
  b = val.y;
}

void __device__ __forceinline__ template_curand_normal2(double &a, double &b,
                                                        curandState_t *state) {
  double2 val = curand_normal2_double(state);
  a = val.x;
  b = val.y;
}

curandStatus_t templateCurandUniform(curandGenerator_t generator,
                                     float *outputPtr, size_t n);

curandStatus_t templateCurandUniform(curandGenerator_t generator,
                                     double *outputPtr, size_t n);

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      // If the GPU is invalid or missing for some reason, raise an exception so
      // we can handle that Error codes can be found here:
      // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
      switch (code) {
      case cudaErrorInvalidDevice:
      case cudaErrorInsufficientDriver:
      case cudaErrorNoDevice:
      case cudaErrorStartupFailure:
      case cudaErrorInvalidPtx:
      case cudaErrorUnsupportedPtxVersion:
      case cudaErrorDevicesUnavailable:
      case cudaErrorUnknown:
        throw InvalidHardware(code);
      default:
        break;
      }
      exit(code);
    }
  }
}

#define curandErrchk(ans)                                                      \
  {                                                                            \
    curandAssert((ans), __FILE__, __LINE__);                                   \
  }
inline void curandAssert(curandStatus_t code, const char *file, int line,
                         bool abort = true) {
  if (code != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "curand failure, code: %d %s %d\n", code, file, line);
    if (abort)
      exit(code);
  }
}

/* cudaSafeMalloc is equivalent to gpuErrchk(cudaMalloc(...)) except that it
 * prints a warning message if the allocation is greater than a GiB.
 */
#define cudaSafeMalloc(ptr, size)                                              \
  ({                                                                           \
    const int cudaSafeMalloc__line = __LINE__;                                 \
    if (size > (1 << 30)) {                                                    \
      fprintf(stderr, "cudaSafeMalloc: allocation larger than 1GiB %s %d\n",   \
              __FILE__, cudaSafeMalloc__line);                                 \
    }                                                                          \
    gpuAssert(cudaMalloc(ptr, size), __FILE__, cudaSafeMalloc__line, true);    \
  })

// safe is for use of gpuErrchk
template <typename T>
T *gpuErrchkCudaMallocAndCopy(const T *host_array, const int count) {
  T *device_array;
  cudaSafeMalloc(&device_array, count * sizeof(*host_array));
  gpuErrchk(cudaMemcpy(device_array, host_array, count * sizeof(*host_array),
                       cudaMemcpyHostToDevice));
  return device_array;
}

// k_initialize_curand_states initializes an array of curandState_t objects such
// that each object uses the seed provided + the index in the array. Offsets and
// sequences always set to 0
void __global__ k_initialize_curand_states(const int count, const int seed,
                                           curandState_t *states);

template <typename T>
void __global__ k_initialize_array(const size_t count, T *__restrict__ array,
                                   const T val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < count) {
    array[idx] = val;

    idx += gridDim.x * blockDim.x;
  }
}

template <typename T>
void initializeArray(const int count, const T *array, T val) {
  // Nothing to allocate
  if (count == 0) {
    return;
  }
  const int tpb = DEFAULT_THREADS_PER_BLOCK;
  const int B =
      ceil_divide(count, tpb); // total number of blocks we need to process
  k_initialize_array<<<B, tpb, 0>>>(count, array, val);
  gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
std::vector<T> device_array_to_vector(const size_t length,
                                      const T *device_array) {
  std::vector<T> output(length);
  gpuErrchk(cudaMemcpy(&output[0], device_array, length * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return output;
}

float __device__ __forceinline__ rmul_rn(float a, float b) {
  return __fmul_rn(a, b);
}

double __device__ __forceinline__ rmul_rn(double a, double b) {
  return __dmul_rn(a, b);
}

float __device__ __forceinline__ radd_rn(float a, float b) {
  return __fadd_rn(a, b);
}

double __device__ __forceinline__ radd_rn(double a, double b) {
  return __dadd_rn(a, b);
}

float __device__ __forceinline__ rsub_rn(float a, float b) {
  return __fsub_rn(a, b);
}

double __device__ __forceinline__ rsub_rn(double a, double b) {
  return __dsub_rn(a, b);
}

float __device__ __forceinline__ rcp_rn(float x) { return __frcp_rn(x); }

double __device__ __forceinline__ rcp_rn(double x) { return __drcp_rn(x); }

} // namespace tmd
