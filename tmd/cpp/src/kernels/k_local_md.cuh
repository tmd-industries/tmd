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

// Kernels specific to Local MD implementation.

namespace tmd {

template <typename RealType>
void __global__ k_construct_bonded_params(
    const int num_idxs,               // Number of idxs
    const int num_atoms,              // Max value any idx can be
    const unsigned int reference_idx, // Atom index to create bonds to
    const RealType k, const RealType r_min, const RealType r_max,
    const unsigned int *__restrict__ idxs, // [num_idxs]
    int *__restrict__ bonds,               // [num_idxs * 2]
    RealType *__restrict__ params          // [num_idxs * 3]
) {
  const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_idxs) {
    return;
  }
  const unsigned int atom_idx = idxs[idx];
  if (atom_idx >= num_atoms) {
    return;
  }
  bonds[idx * 2 + 0] = reference_idx;
  bonds[idx * 2 + 1] = atom_idx;

  params[idx * 3 + 0] = k;
  params[idx * 3 + 1] = r_min;
  params[idx * 3 + 2] = r_max;
}

template <typename T>
void __global__ k_update_index(T *__restrict__ d_array, std::size_t idx,
                               T val) {
  d_array[idx] = val;
}

void __global__ k_idxs_intersection(const int N,
                                    const unsigned int *__restrict__ d_a,
                                    const unsigned int *__restrict__ d_b,
                                    unsigned int *__restrict__ d_dest) {
  const auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  // Set to the value if a and b match, else set to N;
  d_dest[idx] = d_a[idx] == d_b[idx] ? d_a[idx] : N;
}

void __global__ k_flag_free(const int max_index, const int K,
                            const unsigned int *__restrict__ indices,
                            char *__restrict__ flags) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (idx < K) {

    unsigned int flag_idx = indices[idx];
    if (flag_idx < max_index) {
      flags[flag_idx] = 1;
    }

    idx += gridDim.x * blockDim.x;
  }
}

// Iterates through a set of indices that contains chunks of idxs_dim size and
// outputs common_flags[i] = 1 for cases where at least one index in the chunk
// is contained within the comp_atoms. All other indices will set
// common_flags[i] == 0
void __global__ k_flag_indices_to_keep(
    const int num_idxs,           // Total number of idxs
    const int idxs_dims,          // Number of dimensions to the idxs
    const int num_atoms,          // Size
    const int *__restrict__ idxs, // [num_idxs, idxs_dims]
    const unsigned int
        *__restrict__ comp_atoms, // [num_atoms] comp_atoms[i] == i atom is
                                  // being considered, else comp_atoms[i] ==
                                  // num_atoms indicating to ignore the value
    char *__restrict__ overlapping_flags // [num_idxs]
) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  while (tidx < num_idxs) {

    bool is_free = false;
    for (int d = 0; d < idxs_dims; d++) {
      int atom_idx = idxs[tidx * idxs_dims + d];
      is_free |= comp_atoms[atom_idx] < num_atoms;
    }
    overlapping_flags[tidx] = is_free;

    tidx += gridDim.x * blockDim.x;
  }
}

// Forward permute will permute to match the permutation buffer, if false it
// will reverse the permutation. Useful for performing a permutation then
// undoing it.
template <typename T, bool FORWARD_PERMUTE>
void __global__
k_permute_chunks(const int N,          // Number of chunks
                 const int chunk_size, // Length of each chunk
                 const unsigned int *__restrict__ permutation, // [N]
                 const T *__restrict__ src_buffer, // [N, CHUNK_SIZE]
                 T *__restrict__ dest_buffer       // [N, CHUNK_SIZE]
) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  while (tidx < N) {

    const unsigned int src_idx = FORWARD_PERMUTE ? permutation[tidx] : tidx;
    const unsigned int dest_idx = FORWARD_PERMUTE ? tidx : permutation[tidx];
    for (int i = 0; i < chunk_size; i++) {
      dest_buffer[dest_idx * chunk_size + i] =
          src_buffer[src_idx * chunk_size + i];
    }

    tidx += gridDim.x * blockDim.x;
  }
}

} // namespace tmd
