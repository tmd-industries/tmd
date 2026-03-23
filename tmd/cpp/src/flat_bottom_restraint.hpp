/**
 * A similar implementation to FlatBottomBond, but rather than forming bonds (ie
 *restraining two atoms together), FlatBottomRestraint allows specifying a
 *location. The location does not have to represent a real atom, and can be any
 *arbitrary point.
 *
 * This class honors PBCs, so location will be re-imaged into the nearest PBC
 *when considering the restraint.
 **/
#pragma once

#include "energy_accum.hpp"
#include "potential.hpp"
#include <array>
#include <vector>

namespace tmd {

template <typename RealType>
class FlatBottomRestraint : public Potential<RealType> {

  typedef void (*k_flat_bond_fn)(
      const int N, // Number of atoms
      const int R, // number of atom indices
      const RealType *__restrict__ coords, const RealType *__restrict__ box,
      const RealType *__restrict__ params,           // [B, 3]
      const int *__restrict__ atom_idxs,             // [B]
      const int *__restrict__ system_idxs,           // [B]
      const RealType *__restrict__ restraint_coords, // [B, 3]
      unsigned long long *__restrict__ du_dx,
      unsigned long long *__restrict__ du_dp, __int128 *__restrict__ u);

private:
  const int num_systems_;
  const int num_atoms_;

  const int max_idxs_;
  int cur_num_idxs_;

  int *d_restrained_atoms_;
  int *d_system_idxs_; // Which system each bond is associated with
  RealType *d_restraint_coords_;
  __int128 *d_u_buffer_;

  EnergyAccumulator nrg_accum_;

  std::array<k_flat_bond_fn, 8> kernel_ptrs_;

public:
  static const int IDXS_DIM = 1;

  FlatBottomRestraint(const int num_systems, const int num_atoms,
                      const std::vector<int> &restrained_atoms,      // [B, 2]
                      const std::vector<RealType> &restraint_coords, // [B, 3]
                      const std::vector<int> &system_idxs);

  ~FlatBottomRestraint();

  virtual void execute_device(const int batches, const int N, const int P,
                              const RealType *d_x, const RealType *d_p,
                              const RealType *d_box,
                              unsigned long long *d_du_dx,
                              unsigned long long *d_du_dp, __int128 *d_u,
                              cudaStream_t stream) override;

  int num_atoms() const { return cur_num_idxs_; }

  void set_restrained_atoms(const int restrained_atoms, const int *d_atoms,
                            const RealType *d_restraint_pos,
                            const cudaStream_t stream);

  void set_system_idxs_device(const int num_idxs, const int *d_new_system_idxs,
                              cudaStream_t stream);

  virtual int num_systems() const override;
};

} // namespace tmd
