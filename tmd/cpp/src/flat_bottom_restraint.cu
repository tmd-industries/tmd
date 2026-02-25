#include "flat_bottom_restraint.hpp"
#include "gpu_utils.cuh"
#include "k_flat_bottom_restraint.cuh"
#include "kernel_utils.cuh"
#include "math_utils.cuh"
#include <cub/cub.cuh>
#include <vector>

namespace tmd {

template <typename RealType>
FlatBottomRestraint<RealType>::FlatBottomRestraint(
    const int num_systems, const int num_atoms,
    const std::vector<int> &restrained_atoms,      // [B]
    const std::vector<RealType> &restraint_coords, // [B, 3]
    const std::vector<int> &system_idxs)
    : num_systems_(num_systems), num_atoms_(num_atoms),
      max_idxs_(restrained_atoms.size()), cur_num_idxs_(max_idxs_),
      nrg_accum_(num_systems_, cur_num_idxs_),
      kernel_ptrs_({// enumerate over every possible kernel combination
                    // U: Compute U
                    // X: Compute DU_DX
                    // P: Compute DU_DP           U  X  P
                    &k_flat_bottom_restraint<RealType, 0, 0, 0>,
                    &k_flat_bottom_restraint<RealType, 0, 0, 1>,
                    &k_flat_bottom_restraint<RealType, 0, 1, 0>,
                    &k_flat_bottom_restraint<RealType, 0, 1, 1>,
                    &k_flat_bottom_restraint<RealType, 1, 0, 0>,
                    &k_flat_bottom_restraint<RealType, 1, 0, 1>,
                    &k_flat_bottom_restraint<RealType, 1, 1, 0>,
                    &k_flat_bottom_restraint<RealType, 1, 1, 1>}) {

  // validate restrained_atoms: even length, all idxs non-negative, and no
  // self-edges
  if (restrained_atoms.size() % IDXS_DIM != 0) {
    throw std::runtime_error("restrained_atoms.size() must be exactly " +
                             std::to_string(IDXS_DIM) + "*k!");
  }
  if (restraint_coords.size() % 3 != 0) {
    throw std::runtime_error("restraint_coords.size() must be exactly " +
                             std::to_string(3) + "*k!");
  }
  for (int b = 0; b < cur_num_idxs_; b++) {
    auto atom = restrained_atoms[b * IDXS_DIM + 0];

    if (atom < 0) {
      throw std::runtime_error("idxs must be non-negative");
    } else if (atom >= num_atoms) {
      throw std::runtime_error(
          "idxs must be less than the number of atoms in the system");
    }
  }
  // Copy coordinates to device
  cudaSafeMalloc(&d_restraint_coords_,
                 cur_num_idxs_ * 3 * sizeof(*d_restraint_coords_));
  gpuErrchk(cudaMemcpy(d_restraint_coords_, &restraint_coords[0],
                       cur_num_idxs_ * 3 * sizeof(*d_restraint_coords_),
                       cudaMemcpyHostToDevice));

  // copy idxs to device
  cudaSafeMalloc(&d_restrained_atoms_,
                 cur_num_idxs_ * sizeof(*d_restrained_atoms_));
  gpuErrchk(cudaMemcpy(d_restrained_atoms_, &restrained_atoms[0],
                       cur_num_idxs_ * sizeof(*d_restrained_atoms_),
                       cudaMemcpyHostToDevice));

  cudaSafeMalloc(&d_u_buffer_, cur_num_idxs_ * sizeof(*d_u_buffer_));
  cudaSafeMalloc(&d_system_idxs_, cur_num_idxs_ * sizeof(*d_system_idxs_));

  gpuErrchk(cudaMemcpy(d_system_idxs_, &system_idxs[0],
                       cur_num_idxs_ * sizeof(*d_system_idxs_),
                       cudaMemcpyHostToDevice));
};

template <typename RealType>
FlatBottomRestraint<RealType>::~FlatBottomRestraint() {
  gpuErrchk(cudaFree(d_restrained_atoms_));
  gpuErrchk(cudaFree(d_restraint_coords_));
  gpuErrchk(cudaFree(d_u_buffer_));
  gpuErrchk(cudaFree(d_system_idxs_));
};

template <typename RealType>
void FlatBottomRestraint<RealType>::execute_device(
    const int batches, const int N, const int P, const RealType *d_x,
    const RealType *d_p, const RealType *d_box, unsigned long long *d_du_dx,
    unsigned long long *d_du_dp, __int128 *d_u, cudaStream_t stream) {

  const int num_params_per_atom = 3;
  const int expected_P = num_params_per_atom * cur_num_idxs_;

  if (N != num_atoms_) {
    throw std::runtime_error("N != num_atoms_");
  }

  if (P != expected_P) {
    throw std::runtime_error(
        "FlatBottomRestraint::execute_device(): expected P == " +
        std::to_string(expected_P) + ", got P=" + std::to_string(P));
  }

  if (cur_num_idxs_ > 0) {
    const int tpb = DEFAULT_THREADS_PER_BLOCK;
    const int blocks = ceil_divide(cur_num_idxs_, tpb);

    int kernel_idx = 0;
    kernel_idx |= d_du_dp ? 1 << 0 : 0;
    kernel_idx |= d_du_dx ? 1 << 1 : 0;
    kernel_idx |= d_u ? 1 << 2 : 0;

    kernel_ptrs_[kernel_idx]<<<blocks, tpb, 0, stream>>>(
        num_atoms_, cur_num_idxs_, d_x, d_box, d_p, d_restrained_atoms_,
        d_system_idxs_, d_restraint_coords_, d_du_dx, d_du_dp,
        d_u == nullptr ? nullptr : d_u_buffer_);
    gpuErrchk(cudaPeekAtLastError());

    if (d_u) {
      nrg_accum_.sum_device(cur_num_idxs_, d_u_buffer_, d_system_idxs_, d_u,
                            stream);
    }
  }
};

template <typename RealType>
void FlatBottomRestraint<RealType>::set_restrained_atoms(
    const int restrained_atoms, const int *d_atoms,
    const RealType *d_restraint_pos, const cudaStream_t stream) {
  if (max_idxs_ < restrained_atoms) {
    throw std::runtime_error(
        "set_bonds_device(): Max number of bonds " + std::to_string(max_idxs_) +
        " is less than new idxs " + std::to_string(restrained_atoms));
  }
  gpuErrchk(cudaMemcpyAsync(d_restrained_atoms_, d_atoms,
                            restrained_atoms * sizeof(*d_atoms),
                            cudaMemcpyDeviceToDevice, stream));
  gpuErrchk(cudaMemcpyAsync(d_restraint_coords_, d_restraint_pos,
                            restrained_atoms * 3 * sizeof(*d_restraint_pos),
                            cudaMemcpyDeviceToDevice, stream));
  cur_num_idxs_ = restrained_atoms;
}

template <typename RealType>
void FlatBottomRestraint<RealType>::set_system_idxs_device(
    const int num_idxs, const int *d_new_system_idxs, cudaStream_t stream) {
  if (cur_num_idxs_ != num_idxs) {
    throw std::runtime_error(
        "FlatBottomRestraint::set_system_idxs_device(): num idxs must match");
  }
  gpuErrchk(cudaMemcpyAsync(d_system_idxs_, d_new_system_idxs,
                            num_idxs * sizeof(*d_system_idxs_),
                            cudaMemcpyDeviceToDevice, stream));
}

template <typename RealType>
int FlatBottomRestraint<RealType>::num_systems() const {
  return num_systems_;
}

template class FlatBottomRestraint<double>;
template class FlatBottomRestraint<float>;

} // namespace tmd
