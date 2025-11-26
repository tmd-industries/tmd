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

#include "kernel_utils.cuh"

namespace tmd {

#define FULL_MASK 0xffffffff

static const int TILE_SIZE = WARP_SIZE;

template <typename T>
void __global__
k_reset_system_idxs(const size_t num_systems, const size_t N,
                    T *__restrict__ col_system_idxs, // [num_systems, N]
                    T *__restrict__ row_system_idxs  // [num_systems, N]
) {
  auto system_idx = blockIdx.y;
  while (system_idx < num_systems) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
      col_system_idxs[N * system_idx + idx] = system_idx;
      row_system_idxs[N * system_idx + idx] = system_idx;
      idx += gridDim.x * blockDim.x;
    }
    system_idx += gridDim.y * blockDim.y;
  }
}

template <typename RealType>
void __global__ k_find_block_bounds(
    const int num_systems, // Number of systems
    const int N,           // Number of atoms/rows per system
    const unsigned int
        *__restrict__ system_row_indice_counts, // [num_systems] Number of
                                                // indices per system
    const unsigned int *__restrict__ row_idxs,  // [num_systems * N]
    const RealType *__restrict__ coords,        // [num_systems * N * 3]
    const RealType *__restrict__ box,           // [num_systems * 3 * 3]
    RealType *__restrict__ block_bounds_ctr,    // [num_systems * max_tiles * 3]
    RealType *__restrict__ block_bounds_ext,    // [num_systems * max_tiles * 3]
    unsigned int *__restrict__ ixn_counts       // [num_systems]
) {

  // Algorithm taken from
  // https://github.com/openmm/openmm/blob/master/platforms/cuda/src/kernels/findInteractingBlocks.cu#L7
  // Computes smaller bounding boxes than simpler form by accounting for
  // periodic box conditions

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }
  const int tile_offset = system_idx * 3 * ceil_divide(N, WARP_SIZE);

  const int num_row_idxs = system_row_indice_counts[system_idx];
  const int num_tiles = ceil_divide(num_row_idxs, WARP_SIZE);

  // each warp processes one tile
  const int tile_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  if (tile_idx >= num_tiles) {
    return;
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    ixn_counts[system_idx] = 0;
  }

  RealType pos_x;
  RealType pos_y;
  RealType pos_z;

  RealType min_pos_x;
  RealType min_pos_y;
  RealType min_pos_z;

  RealType max_pos_x;
  RealType max_pos_y;
  RealType max_pos_z;

  RealType imaged_pos;

  const RealType box_x = box[system_idx * 9 + 0 * 3 + 0];
  const RealType box_y = box[system_idx * 9 + 1 * 3 + 1];
  const RealType box_z = box[system_idx * 9 + 2 * 3 + 2];

  const RealType inv_bx = rcp_rn(box_x);
  const RealType inv_by = rcp_rn(box_y);
  const RealType inv_bz = rcp_rn(box_z);

  int row_idx = tile_idx * TILE_SIZE + (threadIdx.x % WARP_SIZE);
  // Reset the ixn counts

  if (row_idx < num_row_idxs) {
    int atom_idx = row_idxs[N * system_idx + row_idx];

    pos_x = coords[system_idx * N * 3 + atom_idx * 3 + 0];
    pos_y = coords[system_idx * N * 3 + atom_idx * 3 + 1];
    pos_z = coords[system_idx * N * 3 + atom_idx * 3 + 2];

    min_pos_x = pos_x;
    min_pos_y = pos_y;
    min_pos_z = pos_z;

    max_pos_x = min_pos_x;
    max_pos_y = min_pos_y;
    max_pos_z = min_pos_z;
  }

  // Only the first thread in each warp computes the min/max of the bounding box
  const bool compute_bounds = threadIdx.x % WARP_SIZE == 0;

  // Build up center over time, and recenter before computing
  // min and max, to reduce overall size of box thanks to accounting
  // for periodic boundary conditions
  const int src_lane = threadIdx.x + 1 % WARP_SIZE;
  for (int i = 0; i < WARP_SIZE; i++) {
    row_idx = __shfl_sync(0xffffffff, row_idx, src_lane);
    pos_x = __shfl_sync(0xffffffff, pos_x, src_lane);
    pos_y = __shfl_sync(0xffffffff, pos_y, src_lane);
    pos_z = __shfl_sync(0xffffffff, pos_z, src_lane);
    // Only evaluate for the first thread and when the row idx is valid
    if (compute_bounds && row_idx < num_row_idxs) {
      imaged_pos =
          pos_x - box_x * nearbyint((pos_x - static_cast<RealType>(0.5) *
                                                 (max_pos_x + min_pos_x)) *
                                    inv_bx);
      min_pos_x = min(min_pos_x, imaged_pos);
      max_pos_x = max(max_pos_x, imaged_pos);

      imaged_pos =
          pos_y - box_y * nearbyint((pos_y - static_cast<RealType>(0.5) *
                                                 (max_pos_y + min_pos_y)) *
                                    inv_by);
      min_pos_y = min(min_pos_y, imaged_pos);
      max_pos_y = max(max_pos_y, imaged_pos);

      imaged_pos =
          pos_z - box_z * nearbyint((pos_z - static_cast<RealType>(0.5) *
                                                 (max_pos_z + min_pos_z)) *
                                    inv_bz);
      min_pos_z = min(min_pos_z, imaged_pos);
      max_pos_z = max(max_pos_z, imaged_pos);
    }
  }
  if (compute_bounds) {
    block_bounds_ctr[tile_offset + tile_idx * 3 + 0] =
        static_cast<RealType>(0.5) * (max_pos_x + min_pos_x);
    block_bounds_ctr[tile_offset + tile_idx * 3 + 1] =
        static_cast<RealType>(0.5) * (max_pos_y + min_pos_y);
    block_bounds_ctr[tile_offset + tile_idx * 3 + 2] =
        static_cast<RealType>(0.5) * (max_pos_z + min_pos_z);

    block_bounds_ext[tile_offset + tile_idx * 3 + 0] =
        static_cast<RealType>(0.5) * (max_pos_x - min_pos_x);
    block_bounds_ext[tile_offset + tile_idx * 3 + 1] =
        static_cast<RealType>(0.5) * (max_pos_y - min_pos_y);
    block_bounds_ext[tile_offset + tile_idx * 3 + 2] =
        static_cast<RealType>(0.5) * (max_pos_z - min_pos_z);
  }
}

void __global__ k_compact_trim_atoms(
    const int num_systems, const int N, const int atom_buffer_size_per_system,
    const int Y,
    const unsigned int
        *system_row_counts, // [num_systems] Number of rows idxs for each system
    const unsigned int *__restrict__ trim_atoms,
    unsigned int *__restrict__ interactionCount,
    int *__restrict__ interactingTiles,
    unsigned int *__restrict__ interactingAtoms) {

  // we can probably get away with using only 32 if we do some fancier remainder
  // tricks, but this isn't a huge save
  __shared__ int ixn_j_buffer[2 * WARP_SIZE];

  const int system_idx = blockIdx.y;
  if (system_idx >= num_systems) {
    return;
  }

  const int max_tiles = ceil_divide(N, WARP_SIZE);
  const int tile_offset = system_idx * max_tiles;
  const int output_tile_offset = tile_offset * max_tiles;
  // Offset into the interacting atoms
  const int interacting_atom_offset = system_idx * atom_buffer_size_per_system;
  const int max_row_block =
      ceil_divide(system_row_counts[system_idx], WARP_SIZE);

  ixn_j_buffer[threadIdx.x] = N;
  ixn_j_buffer[WARP_SIZE + threadIdx.x] = N;

  const int indexInWarp = threadIdx.x % WARP_SIZE;
  const int warpMask = (1 << indexInWarp) - 1;
  const int row_block_idx = blockIdx.x;
  if (row_block_idx >= max_row_block) {
    return;
  }

  __shared__ volatile int sync_start[1];
  int neighborsInBuffer = 0;

  for (int trim_block_idx = 0; trim_block_idx < Y; trim_block_idx++) {

    int atom_j_idx =
        trim_atoms[tile_offset * Y * WARP_SIZE + row_block_idx * Y * WARP_SIZE +
                   trim_block_idx * WARP_SIZE + threadIdx.x];

    bool interacts = atom_j_idx < N;

    int includeAtomFlags = __ballot_sync(FULL_MASK, interacts);

    if (interacts) {
      // only interacting atoms partake in this
      int index = neighborsInBuffer +
                  __popc(includeAtomFlags &
                         warpMask); // where to store this in shared memory
      ixn_j_buffer[index] = atom_j_idx;
    }
    neighborsInBuffer += __popc(includeAtomFlags);

    if (neighborsInBuffer > WARP_SIZE) {
      int tilesToStore = 1;
      if (indexInWarp == 0) {
        *sync_start = atomicAdd(interactionCount + system_idx, tilesToStore);
      }
      __syncwarp();
      interactingTiles[output_tile_offset + *sync_start] =
          row_block_idx; // IS THIS CORRECT? CONTESTED
      interactingAtoms[interacting_atom_offset + *sync_start * WARP_SIZE +
                       threadIdx.x] = ixn_j_buffer[threadIdx.x];

      ixn_j_buffer[threadIdx.x] = ixn_j_buffer[WARP_SIZE + threadIdx.x];
      ixn_j_buffer[WARP_SIZE + threadIdx.x] = N; // reset old values
      neighborsInBuffer -= WARP_SIZE;
    }
  }

  if (neighborsInBuffer > 0) {
    int tilesToStore = 1;
    if (indexInWarp == 0) {
      *sync_start = atomicAdd(interactionCount + system_idx, tilesToStore);
    }
    __syncwarp();
    interactingTiles[output_tile_offset + *sync_start] = row_block_idx;
    interactingAtoms[interacting_atom_offset + *sync_start * WARP_SIZE +
                     threadIdx.x] = ixn_j_buffer[threadIdx.x];
  }
}

/*

This is a simplified algorithm of find_interacting_blocks.cu from OpenMM.
This is launched with a threadblock size of 32, (i.e. one warp).

Each block proceeds as follows:

1. Loads its own row block (TILE_SIZE = 32 atoms).
2. Compare the row block against TILE_SIZE = 32 other column blocks via bounding
box tests.
3. Determine which blocks potentially interact using warp-level programming.
4. Loop over each interacting block j, and see which row block atoms may
interact with j's bbox.
5. For atoms that interact, do a fine-grained comparison of each row block again
against each col block atom.
6. Store the column atoms that interact into shared memory
7. Store the leftover trim into its own buffer.

*/

// k_find_block_ixns determines the the column atoms that interact with the
// atoms in a row tile. In the case that UPPER_TRIAG is true, expect row_idxs
// and column_idxs to be identical and be the values of np.arange(0, N).
template <typename RealType, bool UPPER_TRIAG>
void __global__ k_find_blocks_with_ixns(
    const int num_systems,                    // Number of systems
    const int N,                              // Total number of atoms
    const int atom_buffer_size_per_system,    // Number of interactions in each
                                              // atom buffer for each system
    const unsigned int *system_column_counts, // [num_systems] Number of columns
                                              // idxs for each system
    const unsigned int
        *system_row_counts, // [num_systems] Number of rows idxs for each system
    const unsigned int *__restrict__ column_idxs, // [num_systems, NC]
    const unsigned int *__restrict__ row_idxs,    // [num_systems, NR]
    const RealType
        *__restrict__ column_bb_ctr, // [num_systems * N * 3] block centers
    const RealType
        *__restrict__ column_bb_ext, // [num_systems * N * 3] block extents
    const RealType
        *__restrict__ row_bb_ctr, // [num_systems * N * 3] block centers
    const RealType
        *__restrict__ row_bb_ext,        // [num_systems * N * 3] block extents
    const RealType *__restrict__ coords, // [num_systems * N * 3]
    const RealType *__restrict__ box,    // [num_systems * 3 * 3]
    unsigned int *__restrict__ interactionCount, // number of tiles that have
                                                 // interactions
    int *__restrict__ interactingTiles, // the row block idx of the tile that is
                                        // interacting
    unsigned int
        *__restrict__ interactingAtoms, // [num_systems * NR * WARP_SIZE] atom
                                        // indices interacting with each row
                                        // block
    unsigned int *__restrict__ trim_atoms, // the left-over trims that will
                                           // later be compacted
    const RealType base_cutoff, const RealType padding) {

  static_assert(TILE_SIZE == WARP_SIZE,
                "TILE_SIZE != WARP_SIZE is not currently supported");

  const int system_idx = blockIdx.z; // Z dimension is the systems
  if (system_idx >= num_systems) {
    return;
  }

  const int indexInWarp = threadIdx.x % WARP_SIZE;
  const int warpMask = (1 << indexInWarp) - 1;

  // we can probably get away with using only WARP_SIZE if we do some fancier
  // remainder tricks, but this isn't a huge save
  __shared__ int ixn_j_buffer[2 * WARP_SIZE];

  // initialize
  ixn_j_buffer[threadIdx.x] = N;
  ixn_j_buffer[WARP_SIZE + threadIdx.x] = N;

  __shared__ volatile int sync_start[1];

  const int NC = system_column_counts[system_idx];
  const int NR = system_row_counts[system_idx];
  const int max_tiles = ceil_divide(N, WARP_SIZE);
  const int tile_offset = system_idx * max_tiles;
  const int output_tile_offset = tile_offset * max_tiles;
  // Offset into the interacting atoms
  const int interacting_atom_offset = system_idx * atom_buffer_size_per_system;

  unsigned int row_i_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int atom_i_idx =
      row_i_idx < NR ? row_idxs[system_idx * N + row_i_idx] : N;

  const int row_block_idx = blockIdx.x;

  // Retrieve the center coords of row's box and outer limits of row box.
  RealType row_bb_ctr_x = row_bb_ctr[tile_offset * 3 + row_block_idx * 3 + 0];
  RealType row_bb_ctr_y = row_bb_ctr[tile_offset * 3 + row_block_idx * 3 + 1];
  RealType row_bb_ctr_z = row_bb_ctr[tile_offset * 3 + row_block_idx * 3 + 2];

  RealType row_bb_ext_x = row_bb_ext[tile_offset * 3 + row_block_idx * 3 + 0];
  RealType row_bb_ext_y = row_bb_ext[tile_offset * 3 + row_block_idx * 3 + 1];
  RealType row_bb_ext_z = row_bb_ext[tile_offset * 3 + row_block_idx * 3 + 2];

  int neighborsInBuffer = 0;

  RealType pos_i_x =
      atom_i_idx < N ? coords[system_idx * N * 3 + atom_i_idx * 3 + 0] : 0;
  RealType pos_i_y =
      atom_i_idx < N ? coords[system_idx * N * 3 + atom_i_idx * 3 + 1] : 0;
  RealType pos_i_z =
      atom_i_idx < N ? coords[system_idx * N * 3 + atom_i_idx * 3 + 2] : 0;

  const int NUM_COL_BLOCKS = ceil_divide(NC, TILE_SIZE);

  RealType bx = box[system_idx * 9 + 0 * 3 + 0];
  RealType by = box[system_idx * 9 + 1 * 3 + 1];
  RealType bz = box[system_idx * 9 + 2 * 3 + 2];

  RealType inv_bx = rcp_rn(bx);
  RealType inv_by = rcp_rn(by);
  RealType inv_bz = rcp_rn(bz);

  RealType non_periodic_dist_i = 0;
  RealType non_periodic_dist_j = 0;

  const RealType padded_cutoff = base_cutoff + padding;
  const RealType half_padded_cutoff =
      base_cutoff + padding * static_cast<RealType>(0.5);

  // Determine if the row block can be translated into a periodic box
  // to optimize distance calculations
  // https://github.com/proteneer/timemachine/issues/320
  const bool single_periodic_box =
      (static_cast<RealType>(0.5) * bx - row_bb_ext_x >= padded_cutoff &&
       static_cast<RealType>(0.5) * by - row_bb_ext_y >= padded_cutoff &&
       static_cast<RealType>(0.5) * bz - row_bb_ext_z >= padded_cutoff);

  if (single_periodic_box) {
    pos_i_x -= bx * nearbyint((pos_i_x - row_bb_ctr_x) * inv_bx);
    pos_i_y -= by * nearbyint((pos_i_y - row_bb_ctr_y) * inv_by);
    pos_i_z -= bz * nearbyint((pos_i_z - row_bb_ctr_z) * inv_bz);

    non_periodic_dist_i =
        static_cast<RealType>(0.5) *
        (pos_i_x * pos_i_x + pos_i_y * pos_i_y + pos_i_z * pos_i_z);
  }

  const RealType padded_cutoff_squared = padded_cutoff * padded_cutoff;

  int col_block_base = blockIdx.y * TILE_SIZE;

  int col_block_idx = col_block_base + indexInWarp;

  bool include_col_block = (col_block_idx < NUM_COL_BLOCKS) &&
                           (!UPPER_TRIAG || col_block_idx >= row_block_idx);

  if (include_col_block) {

    // Compute center of column box and extent coords.
    RealType col_bb_ctr_x =
        column_bb_ctr[tile_offset * 3 + col_block_idx * 3 + 0];
    RealType col_bb_ctr_y =
        column_bb_ctr[tile_offset * 3 + col_block_idx * 3 + 1];
    RealType col_bb_ctr_z =
        column_bb_ctr[tile_offset * 3 + col_block_idx * 3 + 2];

    RealType col_bb_ext_x =
        column_bb_ext[tile_offset * 3 + col_block_idx * 3 + 0];
    RealType col_bb_ext_y =
        column_bb_ext[tile_offset * 3 + col_block_idx * 3 + 1];
    RealType col_bb_ext_z =
        column_bb_ext[tile_offset * 3 + col_block_idx * 3 + 2];

    // Find delta between boxes
    RealType box_box_dx = row_bb_ctr_x - col_bb_ctr_x;
    RealType box_box_dy = row_bb_ctr_y - col_bb_ctr_y;
    RealType box_box_dz = row_bb_ctr_z - col_bb_ctr_z;

    // Recenter delta box
    box_box_dx -= bx * nearbyint(box_box_dx * inv_bx);
    box_box_dy -= by * nearbyint(box_box_dy * inv_by);
    box_box_dz -= bz * nearbyint(box_box_dz * inv_bz);

    // If boxes overlap, treat distance as 0
    box_box_dx = max(static_cast<RealType>(0.0),
                     fabs(box_box_dx) - row_bb_ext_x - col_bb_ext_x);
    box_box_dy = max(static_cast<RealType>(0.0),
                     fabs(box_box_dy) - row_bb_ext_y - col_bb_ext_y);
    box_box_dz = max(static_cast<RealType>(0.0),
                     fabs(box_box_dz) - row_bb_ext_z - col_bb_ext_z);

    // Check if the deltas between boxes are within cutoff
    include_col_block &= (box_box_dx * box_box_dx + box_box_dy * box_box_dy +
                          box_box_dz * box_box_dz) < (padded_cutoff_squared);
  }

  // __ballot returns bit flags to indicate which thread in the warp identified
  // a column block within the cutoff.
  unsigned includeBlockFlags = __ballot_sync(FULL_MASK, include_col_block);

  // Loop over the col blocks we identified as potentially containing neighbors.
  while (includeBlockFlags != 0) {

    // (ytz): CUDA ffs returns an inclusive [0,32] such that:
    // ffs(0) == 0
    // ffs(2^0=1) == 1
    // ffs(2^1=2) == 2
    // ffs(2^2=4) == 3
    // ffs(2^3=8) == 4
    // ffs(2^31) == 32

    int offset = __ffs(includeBlockFlags) - 1;
    includeBlockFlags &= includeBlockFlags - 1;

    int col_block = col_block_base + offset;
    int col_j_idx = col_block * WARP_SIZE +
                    threadIdx.x; // each thread loads a different atom
    int atom_j_idx =
        col_j_idx < NC ? column_idxs[system_idx * N + col_j_idx] : N;

    // Compute overlap between column bounding box and row atom
    RealType col_bb_ctr_x = column_bb_ctr[tile_offset * 3 + col_block * 3 + 0];
    RealType col_bb_ctr_y = column_bb_ctr[tile_offset * 3 + col_block * 3 + 1];
    RealType col_bb_ctr_z = column_bb_ctr[tile_offset * 3 + col_block * 3 + 2];

    RealType col_bb_ext_x = column_bb_ext[tile_offset * 3 + col_block * 3 + 0];
    RealType col_bb_ext_y = column_bb_ext[tile_offset * 3 + col_block * 3 + 1];
    RealType col_bb_ext_z = column_bb_ext[tile_offset * 3 + col_block * 3 + 2];

    // Don't use pos_i_* here, as might have been shifted to center of row box
    RealType atom_box_dx =
        (atom_i_idx < N ? coords[system_idx * N * 3 + atom_i_idx * 3 + 0] : 0) -
        col_bb_ctr_x;
    RealType atom_box_dy =
        (atom_i_idx < N ? coords[system_idx * N * 3 + atom_i_idx * 3 + 1] : 0) -
        col_bb_ctr_y;
    RealType atom_box_dz =
        (atom_i_idx < N ? coords[system_idx * N * 3 + atom_i_idx * 3 + 2] : 0) -
        col_bb_ctr_z;

    atom_box_dx -= bx * nearbyint(atom_box_dx * inv_bx);
    atom_box_dy -= by * nearbyint(atom_box_dy * inv_by);
    atom_box_dz -= bz * nearbyint(atom_box_dz * inv_bz);

    atom_box_dx =
        max(static_cast<RealType>(0.0), fabs(atom_box_dx) - col_bb_ext_x);
    atom_box_dy =
        max(static_cast<RealType>(0.0), fabs(atom_box_dy) - col_bb_ext_y);
    atom_box_dz =
        max(static_cast<RealType>(0.0), fabs(atom_box_dz) - col_bb_ext_z);

    bool check_column_atoms =
        atom_i_idx < N && atom_box_dx * atom_box_dx +
                                  atom_box_dy * atom_box_dy +
                                  atom_box_dz * atom_box_dz <
                              padded_cutoff_squared;
    // Find rows where the row atom and column boxes are within cutoff
    unsigned atomFlags = __ballot_sync(FULL_MASK, check_column_atoms);
    bool interacts = false;

    //       threadIdx
    //      0 1 2 3 4 5
    //   0  0 0 0 0 0 0
    // a 1  0 1 0 1 1 0  row_atom
    // t 0  0 0 0 0 0 0
    // o 0  0 0 0 0 0 0
    // m 0  0 0 0 0 0 0
    // f 1  1 0 0 0 1 1  row_atom
    // l 0  0 0 0 0 0 0
    // a 1  0 1 0 0 1 0  row_atom
    // g 1  1 1 0 0 0 1  row_atom
    // s 0  0 0 0 0 0 0
    //   0  0 0 0 0 0 0

    RealType pos_j_x =
        atom_j_idx < N ? coords[system_idx * N * 3 + atom_j_idx * 3 + 0] : 0;
    RealType pos_j_y =
        atom_j_idx < N ? coords[system_idx * N * 3 + atom_j_idx * 3 + 1] : 0;
    RealType pos_j_z =
        atom_j_idx < N ? coords[system_idx * N * 3 + atom_j_idx * 3 + 2] : 0;

    // Note: this optimization makes certain assumptions specific to local MD
    // when certain particles are frozen and likely does not do what we want in
    // a more general setting, if the padding semantic in the neighborlist is
    // used outside of a nonbonded interaction group. In particular, the
    // heuristic assumes that:

    // - atom_i is always a free atom.
    // - If: atom_j >= NR, then it is a frozen atom, so that a free atom_i can
    // move at most padding increment
    //   for the neighborlist to be valid.
    // - Else: atom_j < NR, then it is a free atom, then either atom_i or atom_j
    // can move at most padding/2
    //   increment for the neighborlist to be valid.

    // In general, this logic assumes that: (atom_i in row_idxs < atom_j in
    // col_idx) AND (row_idxs prefixes col_idxs)
    //
    // in the upstream calls. Consider the case of a general disjoint row_idxs /
    // col_idxs:
    //
    //   row_idxs = 1,7,3 (NR=3)
    //   col_idxs = 0,4,9,2,5 (NR=5)
    //
    //  the markers (x) denote "buggy" padding (where UPPER_TRIAG=true)
    //
    //     0 4 9 2 5
    //   1 o x o x o
    //   7 . . o x o
    //   3 . . . x o
    //
    // In practice, the nonbonded interaction group that calls the nblist
    // upstream always permutes the indices such that we're never in the general
    // case. UPPER_TRIAG helps
    bool j_is_frozen = UPPER_TRIAG ? atom_j_idx >= NR : false;

    const RealType j_padded_cutoff =
        j_is_frozen ? half_padded_cutoff : padded_cutoff;
    const RealType j_padded_cutoff_squared = j_padded_cutoff * j_padded_cutoff;

    if (single_periodic_box) {
      // Recenter using **row** box center
      pos_j_x -= bx * nearbyint((pos_j_x - row_bb_ctr_x) * inv_bx);
      pos_j_y -= by * nearbyint((pos_j_y - row_bb_ctr_y) * inv_by);
      pos_j_z -= bz * nearbyint((pos_j_z - row_bb_ctr_z) * inv_bz);

      non_periodic_dist_j =
          static_cast<RealType>(0.5) *
          (pos_j_x * pos_j_x + pos_j_y * pos_j_y + pos_j_z * pos_j_z);
    }

    unsigned includeAtomFlags = 0;
    while (atomFlags) {
      const int row_atom = __ffs(atomFlags) - 1;
      atomFlags &= atomFlags - 1;
      RealType row_i_x = __shfl_sync(FULL_MASK, pos_i_x, row_atom);
      RealType row_i_y = __shfl_sync(FULL_MASK, pos_i_y, row_atom);
      RealType row_i_z = __shfl_sync(FULL_MASK, pos_i_z, row_atom);

      if (!single_periodic_box) {
        RealType atom_atom_dx = row_i_x - pos_j_x;
        RealType atom_atom_dy = row_i_y - pos_j_y;
        RealType atom_atom_dz = row_i_z - pos_j_z;

        atom_atom_dx -= bx * nearbyint(atom_atom_dx * inv_bx);
        atom_atom_dy -= by * nearbyint(atom_atom_dy * inv_by);
        atom_atom_dz -= bz * nearbyint(atom_atom_dz * inv_bz);

        interacts |=
            (atom_atom_dx * atom_atom_dx + atom_atom_dy * atom_atom_dy +
             atom_atom_dz * atom_atom_dz) < j_padded_cutoff_squared;
      } else {
        // All threads in warp need single_periodic_box to be true for this not
        // to hang
        RealType corrected_i =
            __shfl_sync(FULL_MASK, non_periodic_dist_i, row_atom);

        // Below is half the magnitude of the distance equation, expanded.
        RealType half_dist = corrected_i + non_periodic_dist_j -
                             row_i_x * pos_j_x - row_i_y * pos_j_y -
                             row_i_z * pos_j_z;
        interacts |=
            half_dist < (static_cast<RealType>(0.5) * j_padded_cutoff_squared);
      }
      includeAtomFlags = __ballot_sync(FULL_MASK, interacts);
      // If all threads in the warp have found interactions, can terminate early
      if (includeAtomFlags == FULL_MASK) {
        break;
      }
    }

    // Add any interacting atoms to the buffer.
    if (interacts) {
      int index = neighborsInBuffer +
                  __popc(includeAtomFlags &
                         warpMask); // where to store this in shared memory
      // Indices can be at most 64
      ixn_j_buffer[index] = atom_j_idx;
    }
    neighborsInBuffer += __popc(includeAtomFlags);

    if (neighborsInBuffer > WARP_SIZE) {
      int tilesToStore = 1;
      if (indexInWarp == 0) {
        *sync_start = atomicAdd(interactionCount + system_idx, tilesToStore);
      }
      __syncwarp();
      interactingTiles[output_tile_offset + *sync_start] = row_block_idx;

      interactingAtoms[interacting_atom_offset + *sync_start * WARP_SIZE +
                       threadIdx.x] = ixn_j_buffer[threadIdx.x];

      ixn_j_buffer[threadIdx.x] = ixn_j_buffer[WARP_SIZE + threadIdx.x];
      ixn_j_buffer[WARP_SIZE + threadIdx.x] = N; // reset old values
      neighborsInBuffer -= WARP_SIZE;
    }
  }

  // store trim
  const int Y = gridDim.y;
  trim_atoms[tile_offset * Y * WARP_SIZE + blockIdx.x * Y * WARP_SIZE +
             blockIdx.y * WARP_SIZE + threadIdx.x] = ixn_j_buffer[threadIdx.x];
}

} // namespace tmd
