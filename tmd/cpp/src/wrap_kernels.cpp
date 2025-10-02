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

#include <cmath>
#include <numeric>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>

#include "all_atom_energies.hpp"
#include "barostat.hpp"
#include "bd_exchange_move.hpp"
#include "bound_potential.hpp"
#include "centroid_restraint.hpp"
#include "chiral_atom_restraint.hpp"
#include "chiral_bond_restraint.hpp"
#include "context.hpp"
#include "exceptions.hpp"
#include "exchange.hpp"
#include "fanout_summed_potential.hpp"
#include "fixed_point.hpp"
#include "flat_bottom_bond.hpp"
#include "harmonic_angle.hpp"
#include "harmonic_bond.hpp"
#include "langevin_integrator.hpp"
#include "local_md_potentials.hpp"
#include "local_md_utils.hpp"
#include "log_flat_bottom_bond.hpp"
#include "mover.hpp"
#include "neighborlist.hpp"
#include "nonbonded_common.hpp"
#include "nonbonded_interaction_group.hpp"
#include "nonbonded_mol_energy.hpp"
#include "nonbonded_pair_list.hpp"
#include "nonbonded_precomputed.hpp"
#include "periodic_torsion.hpp"
#include "potential.hpp"
#include "potential_executor.hpp"
#include "rmsd_align.hpp"
#include "rotations.hpp"
#include "segmented_sumexp.hpp"
#include "segmented_weighted_random_sampler.hpp"
#include "set_utils.hpp"
#include "summed_potential.hpp"
#include "tibd_exchange_move.hpp"
#include "translations.hpp"
#include "verlet_integrator.hpp"

#include <iostream>

namespace py = pybind11;
using namespace tmd;

void verify_bond_idxs(const py::array_t<int, py::array::c_style> &bond_idxs,
                      const int idxs_per_bond) {
  size_t bond_dims = bond_idxs.ndim();
  if (bond_dims != 2) {
    throw std::runtime_error("idxs dimensions must be 2");
  }
  if (bond_idxs.shape(bond_dims - 1) != idxs_per_bond) {
    throw std::runtime_error("idxs must be of length " +
                             std::to_string(idxs_per_bond));
  }
}

void verify_coords(const py::array_t<double, py::array::c_style> &coords) {
  size_t coord_dimensions = coords.ndim();
  if (coord_dimensions != 2) {
    throw std::runtime_error("coords dimensions must be 2");
  }
  if (coords.shape(coord_dimensions - 1) != 3) {
    throw std::runtime_error("coords must have a shape that is 3 dimensional");
  }
}

// A utility to make sure that the coords and box shapes are correct
void verify_coords_and_box(
    const py::array_t<double, py::array::c_style> &coords,
    const py::array_t<double, py::array::c_style> &box) {
  verify_coords(coords);
  if (box.ndim() != 2 || box.shape(0) != 3 || box.shape(1) != 3) {
    throw std::runtime_error("box must be 3x3");
  }
  auto box_data = box.data();
  for (int i = 0; i < box.size(); i++) {
    if (i == 0 || i == 4 || i == 8) {
      if (box_data[i] <= 0.0) {
        throw std::runtime_error(
            "box must have positive values along diagonal");
      }
    } else if (box_data[i] != 0.0) {
      throw std::runtime_error("box must be ortholinear");
    }
  }
}

// convert_energy_to_fp handles the combining of energies, summing them up
// deterministically and returning nan if there are overflows. The energies are
// collected in int128
double convert_energy_to_fp(__int128 fixed_u) {
  double res = std::numeric_limits<double>::quiet_NaN();
  if (!fixed_point_overflow(fixed_u)) {
    res = FIXED_ENERGY_TO_FLOAT<double>(fixed_u);
  }
  return res;
}

template <typename T>
std::vector<T>
py_array_to_vector(const py::array_t<T, py::array::c_style> &arr) {
  std::vector<T> v(arr.data(), arr.data() + arr.size());
  return v;
}

template <typename T1, typename T2>
std::vector<T2>
py_array_to_vector_with_cast(const py::array_t<T1, py::array::c_style> &arr) {
  std::vector<T2> v(arr.size());
  for (int i = 0; i < arr.size(); i++) {
    v[i] = static_cast<T2>(arr.data()[i]);
  }
  return v;
}

template <typename T1, typename T2>
std::vector<T2> py_vector_to_vector_with_cast(const std::vector<T1> &arr) {
  std::vector<T2> v(arr.size());
  for (unsigned long i = 0; i < arr.size(); i++) {
    v[i] = static_cast<T2>(arr[i]);
  }
  return v;
}

template <typename RealType>
void declare_neighborlist(py::module &m, const char *typestr) {

  using Class = Neighborlist<RealType>;
  std::string pyclass_name = std::string("Neighborlist_") + typestr;
  py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(),
                    py::dynamic_attr())
      .def(py::init([](int N, bool compute_upper_triangular) {
             return new Neighborlist<RealType>(N, compute_upper_triangular);
           }),
           py::arg("N"), py::arg("compute_upper_triangular"))
      .def(
          "compute_block_bounds",
          [](Neighborlist<RealType> &nblist,
             const py::array_t<double, py::array::c_style> &coords,
             const py::array_t<double, py::array::c_style> &box,
             const int block_size) -> py::tuple {
            if (block_size != 32) {
              // The neighborlist kernel implementation assumes that block size
              // is fixed to the CUDA warpSize
              throw std::runtime_error("Block size must be 32.");
            }
            verify_coords_and_box(coords, box);
            int N = coords.shape()[0];
            int D = coords.shape()[1];
            int B = (N + block_size - 1) / block_size;

            py::array_t<RealType, py::array::c_style> py_bb_ctrs({B, D});
            py::array_t<RealType, py::array::c_style> py_bb_exts({B, D});

            std::vector<RealType> real_coords =
                py_array_to_vector_with_cast<double, RealType>(coords);
            std::vector<RealType> real_box =
                py_array_to_vector_with_cast<double, RealType>(box);

            nblist.compute_block_bounds_host(
                N, real_coords.data(), real_box.data(),
                py_bb_ctrs.mutable_data(), py_bb_exts.mutable_data());

            // returns real type
            return py::make_tuple(py_bb_ctrs, py_bb_exts);
          },
          py::arg("coords"), py::arg("box"), py::arg("block_size"))
      .def(
          "get_nblist",
          [](Neighborlist<RealType> &nblist,
             const py::array_t<double, py::array::c_style> &coords,
             const py::array_t<double, py::array::c_style> &box,
             const double cutoff,
             const double padding) -> std::vector<std::vector<int>> {
            int N = coords.shape()[0];
            verify_coords_and_box(coords, box);

            std::vector<RealType> real_coords =
                py_array_to_vector_with_cast<double, RealType>(coords);
            std::vector<RealType> real_box =
                py_array_to_vector_with_cast<double, RealType>(box);

            std::vector<std::vector<int>> ixn_list = nblist.get_nblist_host(
                N, real_coords.data(), real_box.data(), cutoff, padding);

            return ixn_list;
          },
          py::arg("coords"), py::arg("box"), py::arg("cutoff"),
          py::arg("padding"))
      .def(
          "set_row_idxs",
          [](Neighborlist<RealType> &nblist,
             const py::array_t<unsigned int, py::array::c_style> &idxs_i) {
            std::vector<unsigned int> idxs = py_array_to_vector(idxs_i);
            nblist.set_row_idxs(idxs);
          },
          py::arg("idxs"))
      .def(
          "set_row_idxs_and_col_idxs",
          [](Neighborlist<RealType> &nblist,
             const py::array_t<unsigned int, py::array::c_style> &row_idxs_i,
             const py::array_t<unsigned int, py::array::c_style> &col_idxs_i) {
            std::vector<unsigned int> row_idxs = py_array_to_vector(row_idxs_i);
            std::vector<unsigned int> col_idxs = py_array_to_vector(col_idxs_i);
            nblist.set_row_idxs_and_col_idxs(row_idxs, col_idxs);
          },
          py::arg("row_idxs"), py::arg("col_idxs"))
      .def("set_compute_upper_triangular",
           &Neighborlist<RealType>::set_compute_upper_triangular)
      .def("reset_row_idxs", &Neighborlist<RealType>::reset_row_idxs)
      .def("get_tile_ixn_count", &Neighborlist<RealType>::num_tile_ixns)
      .def("get_max_ixn_count", &Neighborlist<RealType>::max_ixn_count)
      .def("resize", &Neighborlist<RealType>::resize, py::arg("size"))
      .def("get_num_row_idxs", &Neighborlist<RealType>::get_num_row_idxs);
}

template <typename RealType>
void declare_hilbert_sort(py::module &m, const char *typestr) {

  using Class = HilbertSort<RealType>;
  std::string pyclass_name = std::string("HilbertSort_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int N) { return new HilbertSort<RealType>(N); }),
           py::arg("size"))
      .def(
          "sort",
          [](HilbertSort<RealType> &sorter,
             const py::array_t<double, py::array::c_style> &coords,
             const py::array_t<double, py::array::c_style> &box)
              -> const py::array_t<uint32_t, py::array::c_style> {
            const int N = coords.shape()[0];
            verify_coords_and_box(coords, box);

            std::vector<RealType> real_coords =
                py_array_to_vector_with_cast<double, RealType>(coords);
            std::vector<RealType> real_box =
                py_array_to_vector_with_cast<double, RealType>(box);

            std::vector<unsigned int> sort_perm =
                sorter.sort_host(N, real_coords.data(), real_box.data());
            py::array_t<uint32_t, py::array::c_style> output_perm(
                sort_perm.size(), sort_perm.data());
            return output_perm;
          },
          py::arg("coords"), py::arg("box"));
}

template <typename RealType>
void declare_segmented_weighted_random_sampler(py::module &m,
                                               const char *typestr) {

  using Class = SegmentedWeightedRandomSampler<RealType>;
  std::string pyclass_name =
      std::string("SegmentedWeightedRandomSampler_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int N, const int segments, const int seed) {
             return new Class(N, segments, seed);
           }),
           py::arg("max_vals_per_segment"), py::arg("segments"),
           py::arg("seed"))
      .def(
          "sample",
          [](Class &sampler, const std::vector<std::vector<double>> &weights)
              -> std::vector<int> {
            std::vector<std::vector<RealType>> real_batches(weights.size());
            for (unsigned long i = 0; i < weights.size(); i++) {
              real_batches[i] =
                  py_vector_to_vector_with_cast<double, RealType>(weights[i]);
            }

            std::vector<int> samples = sampler.sample_host(real_batches);
            return samples;
          },
          py::arg("weights"),
          R"pbdoc(
        Randomly select a value from batches of weights.

        Parameters
        ----------

        weights: vector of vectors containing doubles
            Weights to sample from. Do not need to be normalized.

        Returns
        -------
        Array of sample indices
            Shape (num_batches, )
        )pbdoc");
}

template <typename RealType>
void declare_nonbonded_mol_energy(py::module &m, const char *typestr) {

  using Class = NonbondedMolEnergyPotential<RealType>;
  std::string pyclass_name =
      std::string("NonbondedMolEnergyPotential_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int N,
                       const std::vector<std::vector<int>> &target_mols,
                       const RealType beta, const RealType cutoff) {
             return new Class(N, target_mols, beta, cutoff);
           }),
           py::arg("N"), py::arg("target_mols"), py::arg("beta"),
           py::arg("cutoff"))
      .def(
          "execute",
          [](Class &potential,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &params,
             const py::array_t<RealType, py::array::c_style> &box)
              -> py::array_t<RealType, py::array::c_style> {
            verify_coords_and_box(coords, box);
            int N = coords.shape()[0];
            if (N != params.shape()[0]) {
              throw std::runtime_error("params N != coords N");
            }

            int P = params.size();

            std::vector<__int128> fixed_energies = potential.mol_energies_host(
                N, P, coords.data(), params.data(), box.data());

            py::array_t<RealType, py::array::c_style> py_u(
                fixed_energies.size());

            for (unsigned int i = 0; i < fixed_energies.size(); i++) {
              py_u.mutable_data()[i] = convert_energy_to_fp(fixed_energies[i]);
            }
            return py_u;
          },
          py::arg("coords"), py::arg("params"), py::arg("box"),
          R"pbdoc(
        Compute the energies of the target molecules

        Parameters
        ----------
        coords: NDArray
            A set of coordinates.

        params: NDArray
            A set of nonbonded parameters for all atoms.

        box: NDArray
            A box.

        Returns
        -------
        Array of energies
            Shape (num_mols, )
        )pbdoc");
}

template <typename RealType>
void declare_context(py::module &m, const char *typestr) {

  using Class = Context<RealType>;
  std::string pyclass_name = std::string("Context_") + typestr;
  py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(),
                    py::dynamic_attr())
      .def(py::init(
               [](const py::array_t<RealType, py::array::c_style> &x0,
                  const py::array_t<RealType, py::array::c_style> &v0,
                  const py::array_t<RealType, py::array::c_style> &box0,
                  std::shared_ptr<Integrator<RealType>> intg,
                  std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
                  std::optional<std::vector<std::shared_ptr<Mover<RealType>>>>
                      movers) {
                 int N = x0.shape()[0];
                 int D = x0.shape()[1];
                 verify_coords_and_box(x0, box0);
                 if (N != v0.shape()[0]) {
                   throw std::runtime_error("v0 N != x0 N");
                 }

                 if (D != v0.shape()[1]) {
                   throw std::runtime_error("v0 D != x0 D");
                 }

                 std::vector<std::shared_ptr<Mover<RealType>>> v_movers(0);
                 if (movers.has_value()) {
                   v_movers = movers.value();
                 }
                 for (auto mover : v_movers) {
                   if (mover == nullptr) {
                     throw std::runtime_error("got nullptr instead of mover");
                   }
                 }

                 return new Context<RealType>(N, x0.data(), v0.data(),
                                              box0.data(), intg, bps, v_movers);
               }),
           py::arg("x0"), py::arg("v0"), py::arg("box"), py::arg("integrator"),
           py::arg("bps"), py::arg("movers") = py::none())
      .def("step", &Context<RealType>::step,
           R"pbdoc(
        Take a single step.

        Note: Must call `initialize` before stepping and `finalize` after stepping to ensure the correct velocities and positions to be returned by `get_x_t()` and `get_v_t()`.
        )pbdoc")
      .def("finalize", &Context<RealType>::finalize)
      .def("initialize", &Context<RealType>::initialize)
      .def(
          "multiple_steps",
          [](Context<RealType> &ctxt, const int n_steps,
             int store_x_interval) -> py::tuple {
            if (store_x_interval < 0) {
              throw std::runtime_error(
                  "store_x_interval must be greater than or equal to zero");
            }
            // (ytz): I hate C++
            int N = ctxt.num_atoms();
            int D = 3;

            // If the store_x_interval is 0, collect the last frame by setting
            // x_interval to n_steps
            const int x_interval =
                (store_x_interval == 0) ? n_steps : store_x_interval;
            // n_samples determines the number of frames that will be collected,
            // computed to be able to allocate the buffers up front to avoid
            // allocating memory twice
            const int n_samples = n_steps / x_interval;
            py::array_t<RealType, py::array::c_style> out_x_buffer(
                {n_samples, N, D});
            py::array_t<RealType, py::array::c_style> box_buffer(
                {n_samples, D, D});
            auto res = py::make_tuple(out_x_buffer, box_buffer);
            ctxt.multiple_steps(n_steps, n_samples, out_x_buffer.mutable_data(),
                                box_buffer.mutable_data());
            return res;
          },
          py::arg("n_steps"), py::arg("store_x_interval") = 0,
          R"pbdoc(
        Take multiple steps.

        Frames are stored after having taken the number of steps specified by store_x_interval. E.g. if
        store_x_interval is 5, then on the 5th step the frame will be stored.

        Parameters
        ----------
        n_steps: int
            Number of steps

        store_x_interval: int
            How often we store the frames, store after every store_x_interval iterations. Setting to zero collects frames
            at the last step. Setting store_x_interval > n_steps will return no frames and skip runtime validation of box
            size.

        Returns
        -------
        2-tuple of coordinates, boxes
            F = floor(n_steps/store_x_interval).
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

        Raises
        ------
            RuntimeError:
                Box dimensions are invalid when a frame is collected

    )pbdoc")
      .def(
          "multiple_steps_local",
          [](Context<RealType> &ctxt, const int n_steps,
             const py::array_t<int, py::array::c_style> &local_idxs,
             const int store_x_interval, const RealType radius,
             const RealType k, const int seed) -> py::tuple {
            if (n_steps <= 0) {
              throw std::runtime_error("local steps must be at least one");
            }
            if (store_x_interval < 0) {
              throw std::runtime_error(
                  "store_x_interval must be greater than or equal to zero");
            }
            verify_local_md_parameters<RealType>(radius, k);

            const int N = ctxt.num_atoms();
            const int D = 3;

            std::vector<int> vec_local_idxs = py_array_to_vector(local_idxs);
            verify_atom_idxs(N, vec_local_idxs);

            // If the store_x_interval is 0, collect the last frame by setting
            // x_interval to n_steps
            const int x_interval =
                (store_x_interval == 0) ? n_steps : store_x_interval;
            // n_samples determines the number of frames that will be collected,
            // computed to be able to allocate the buffers up front to avoid
            // allocating memory twice
            const int n_samples = n_steps / x_interval;
            py::array_t<RealType, py::array::c_style> out_x_buffer(
                {n_samples, N, D});
            py::array_t<RealType, py::array::c_style> box_buffer(
                {n_samples, D, D});
            auto res = py::make_tuple(out_x_buffer, box_buffer);

            ctxt.multiple_steps_local(
                n_steps, vec_local_idxs, n_samples, radius, k, seed,
                out_x_buffer.mutable_data(), box_buffer.mutable_data());
            return res;
          },
          py::arg("n_steps"), py::arg("local_idxs"),
          py::arg("store_x_interval") = 0, py::arg("radius") = 1.2,
          py::arg("k") = 10000.0, py::arg("seed") = 2022,
          R"pbdoc(
        Take multiple steps using particles selected based on the log probability using a random particle from the local_idxs,
        the random particle is frozen for all steps.

        Running a barostat and local MD at the same time are not currently supported. If a barostat is
        assigned to the context, the barostat won't run.

        Note: Running this multiple times with small number of steps (< 100) may result in a vacuum around the local idxs due to
        discretization error caused by switching on the restraint after a particle has moved beyond the radius.

        F = iterations / store_x_interval

        The first call to `multiple_steps_local` takes longer than subsequent calls, if setup_local_md has not been called previously,
        initializes potentials needed for local MD. The default local MD parameters are to freeze the reference and to
        use the temperature of the integrator, which must be a LangevinIntegrator.

        Parameters
        ----------
        n_steps: int
            Number of steps to run.

        local_idxs: np.array of int32
            The idxs that defines the atoms to use as the region(s) to run local MD. A random idx will be
            selected to be frozen and used as the center of the shell of particles to be simulated. The selected
            idx is constant across all steps.

        store_x_interval: int
            How often we store the frames, store after every store_x_interval iterations. Setting to zero collects frames
            at the last step. Setting store_x_interval > n_steps will return no frames and skip runtime validation of box
            size.

        radius: float
            The radius in nanometers from the selected idx to simulate for local MD.

        k: float
            The flat bottom restraint K value to use for selection and restraint of atoms within the inner shell.

        seed: int
            The seed that is used to randomly select a particle to freeze and for the probabilistic selection of
            free particles. It is recommended to provide a new seed each time this function is called.

        Returns
        -------
        2-tuple of coordinates, boxes
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

        Raises
        ------
            RuntimeError:
                Box dimensions are invalid when a frame is collected

        Note: All boxes returned will be identical as local MD only runs under constant volume.
    )pbdoc")
      .def(
          "multiple_steps_local_selection",
          [](Context<RealType> &ctxt, const int n_steps,
             const int reference_idx,
             const py::array_t<int, py::array::c_style> &selection_idxs,
             const int store_x_interval, const RealType radius,
             const RealType k) -> py::tuple {
            if (n_steps <= 0) {
              throw std::runtime_error("local steps must be at least one");
            }
            if (store_x_interval < 0) {
              throw std::runtime_error(
                  "store_x_interval must be greater than or equal to zero");
            }
            verify_local_md_parameters<RealType>(radius, k);

            const int N = ctxt.num_atoms();
            const int D = 3;

            if (reference_idx < 0 || reference_idx >= N) {
              throw std::runtime_error(
                  "reference idx must be at least 0 and less than " +
                  std::to_string(N));
            }
            std::vector<int> vec_selection_idxs =
                py_array_to_vector(selection_idxs);
            verify_atom_idxs(N, vec_selection_idxs);
            std::set<int> selection_set(vec_selection_idxs.begin(),
                                        vec_selection_idxs.end());
            if (selection_set.find(reference_idx) != selection_set.end()) {
              throw std::runtime_error(
                  "reference idx must not be in selection idxs");
            }
            // If the store_x_interval is 0, collect the last frame by setting
            // x_interval to n_steps
            const int x_interval =
                (store_x_interval == 0) ? n_steps : store_x_interval;
            // n_samples determines the number of frames that will be collected,
            // computed to be able to allocate the buffers up front to avoid
            // allocating memory twice
            const int n_samples = n_steps / x_interval;
            py::array_t<RealType, py::array::c_style> out_x_buffer(
                {n_samples, N, D});
            py::array_t<RealType, py::array::c_style> box_buffer(
                {n_samples, D, D});
            auto res = py::make_tuple(out_x_buffer, box_buffer);

            ctxt.multiple_steps_local_selection(
                n_steps, reference_idx, vec_selection_idxs, n_samples, radius,
                k, out_x_buffer.mutable_data(), box_buffer.mutable_data());
            return res;
          },
          py::arg("n_steps"), py::arg("reference_idx"),
          py::arg("selection_idxs"), py::arg("store_x_interval") = 0,
          py::arg("radius") = 1.2, py::arg("k") = 10000.0,
          R"pbdoc(
        Take multiple steps using a selection of free particles restrained to a reference particle. Useful for avoiding the bias
        introduced by switching on and off the restraint on different particles as is done with multiple_steps_local.

        Running a barostat and local MD at the same time are not currently supported. If a barostat is
        assigned to the context, the barostat won't run.

        Note: Running this multiple times with small number of steps (< 100) may result in a vacuum around the local idxs due to
        discretization error caused by switching on the restraint after a particle has moved beyond the radius.

        F = iterations / store_x_interval

        The first call to `multiple_steps_local_selection` takes longer than subsequent calls, if setup_local_md has not been called previously,
        initializes potentials needed for local MD. The default local MD parameters are to freeze the reference and to
        use the temperature of the integrator, which must be a LangevinIntegrator.

        Parameters
        ----------
        n_steps: int
            Number of steps to run.

        reference_idx: int
            Idx of particle to use as reference.

        selection_idxs: np.array of int32
            The idxs of particles that should be free during local MD. Will be restrained to the particle specified by reference_idx particle using a
            flat bottom restraint which is defined by the radius and k values. Can be up to N - 1 particles, IE all particles except the reference_idx.

        store_x_interval: int
            How often we store the frames, store after every store_x_interval iterations. Setting to zero collects frames
            at the last step. Setting store_x_interval > n_steps will return no frames and skip runtime validation of box
            size.

        radius: float
            The radius in nanometers from the reference idx to allow particles to be unrestrained in, afterwards apply a restraint to the reference particle.

        k: float
            The flat bottom restraint K value to use for restraint of atoms to the reference particle.

        Returns
        -------
        2-tuple of coordinates, boxes
            Coordinates have shape (F, N, 3)
            Boxes have shape (F, 3, 3)

        Raises
        ------
            RuntimeError:
                Box dimensions are invalid when a frame is collected

        Note: All boxes returned will be identical as local MD only runs under constant volume.
    )pbdoc")
      .def("setup_local_md", &Context<RealType>::setup_local_md,
           py::arg("temperature"), py::arg("freeze_reference"),
           py::arg("nblist_padding") =
               LocalMDPotentials<RealType>::DEFAULT_NBLIST_PADDING,
           R"pbdoc(
        Configures the potential for local MD. This is automatically done when calling local MD methods,
        but can be done explicitly and with different parameters. This function is idempotent, provided the same input parameters.

        Parameters
        ----------
        temperature: float
            Temperature in kelvin

        freeze_reference: bool
            Whether or not to freeze reference, otherwise applies restraint between frozen
            particles and the reference.

        nblist_padding: float
            Size of the neighborlist padding to set while running local md. Defaults to 0.3, which may be too large for large numbers of free particles.

        Raises
        ------
            RuntimeError:
                Called with different parameters than previously initialized with.
    )pbdoc")
      .def("get_local_md_potentials",
           &Context<RealType>::get_local_md_potentials,
           R"pbdoc(
        Return the bound potentials that are used when running Local MD.

        Returns
        -------
        list of BoundPotential objects

        Raises
        ------
            RuntimeError:
                Local MD is not configured.
    )pbdoc")
      .def(
          "_truncate_potentials_local_selection",
          [](Context<RealType> &ctxt, const int reference_idx,
             const py::array_t<int, py::array::c_style> &selection_idxs,
             const RealType radius, const RealType k)
              -> std::vector<std::shared_ptr<BoundPotential<RealType>>> {
            verify_local_md_parameters<RealType>(radius, k);

            const int N = ctxt.num_atoms();

            if (reference_idx < 0 || reference_idx >= N) {
              throw std::runtime_error(
                  "reference idx must be at least 0 and less than " +
                  std::to_string(N));
            }
            std::vector<int> vec_selection_idxs =
                py_array_to_vector(selection_idxs);
            verify_atom_idxs(N, vec_selection_idxs);
            std::set<int> selection_set(vec_selection_idxs.begin(),
                                        vec_selection_idxs.end());
            if (selection_set.find(reference_idx) != selection_set.end()) {
              throw std::runtime_error(
                  "reference idx must not be in selection idxs");
            }
            return ctxt.truncate_potentials_local_selection(
                reference_idx, vec_selection_idxs, radius, k);
          },
          py::arg("reference_idx"), py::arg("selection_idxs"),
          py::arg("radius") = 1.2, py::arg("k") = 10000.0,
          R"pbdoc(
        Strictly a testing function that will truncate the potentials associated with a context based on a selection of a system. The potential and the
        context should not be used after calling this function.

        Parameters
        ----------
        reference_idx: int
            Idx of particle to use as reference.

        selection_idxs: np.array of int32
            The idxs of particles that should be free during local MD. Will be restrained to the particle specified by reference_idx particle using a
            flat bottom restraint which is defined by the radius and k values. Can be up to N - 1 particles, IE all particles except the reference_idx.

        radius: float
            The radius in nanometers from the reference idx to allow particles to be unrestrained in, afterwards apply a restraint to the reference particle.

        k: float
            The flat bottom restraint K value to use for restraint of atoms to the reference particle.

        Returns
        -------
        list of BoundPotential objects
    )pbdoc")
      .def(
          "set_x_t",
          [](Context<RealType> &ctxt,
             const py::array_t<RealType, py::array::c_style> &new_x_t) {
            if (new_x_t.shape()[0] != ctxt.num_atoms()) {
              throw std::runtime_error(
                  "number of new coords disagree with current coords");
            }
            ctxt.set_x_t(new_x_t.data());
          },
          py::arg("coords"))
      .def(
          "set_v_t",
          [](Context<RealType> &ctxt,
             const py::array_t<RealType, py::array::c_style> &new_v_t) {
            if (new_v_t.shape()[0] != ctxt.num_atoms()) {
              throw std::runtime_error(
                  "number of new velocities disagree with current coords");
            }
            ctxt.set_v_t(new_v_t.data());
          },
          py::arg("velocities"))
      .def(
          "set_box",
          [](Context<RealType> &ctxt,
             const py::array_t<RealType, py::array::c_style> &new_box_t) {
            if (new_box_t.size() != 9 || new_box_t.shape()[0] != 3) {
              throw std::runtime_error("box must be 3x3");
            }
            ctxt.set_box(new_box_t.data());
          },
          py::arg("box"))
      .def("get_x_t",
           [](Context<RealType> &ctxt)
               -> py::array_t<RealType, py::array::c_style> {
             unsigned int N = ctxt.num_atoms();
             unsigned int D = 3;
             py::array_t<RealType, py::array::c_style> buffer({N, D});
             ctxt.get_x_t(buffer.mutable_data());
             return buffer;
           })
      .def("get_v_t",
           [](Context<RealType> &ctxt)
               -> py::array_t<RealType, py::array::c_style> {
             unsigned int N = ctxt.num_atoms();
             unsigned int D = 3;
             py::array_t<RealType, py::array::c_style> buffer({N, D});
             ctxt.get_v_t(buffer.mutable_data());
             return buffer;
           })
      .def("get_box",
           [](Context<RealType> &ctxt)
               -> py::array_t<RealType, py::array::c_style> {
             unsigned int D = 3;
             py::array_t<RealType, py::array::c_style> buffer({D, D});
             ctxt.get_box(buffer.mutable_data());
             return buffer;
           })
      .def("get_integrator", &Context<RealType>::get_integrator)
      .def("get_potentials", &Context<RealType>::get_potentials)
      .def("get_barostat", &Context<RealType>::get_barostat)
      .def("get_movers", &Context<RealType>::get_movers);
}

template <typename RealType>
void declare_integrator(py::module &m, const char *typestr) {
  using Class = Integrator<RealType>;
  std::string pyclass_name = std::string("Integrator_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr());
}

template <typename RealType>
void declare_langevin_integrator(py::module &m, const char *typestr) {

  using Class = LangevinIntegrator<RealType>;
  std::string pyclass_name = std::string("LangevinIntegrator_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Integrator<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const py::array_t<RealType, py::array::c_style> &masses,
                       RealType temperature, RealType dt, RealType friction,
                       int seed) {
             return new Class(masses.size(), masses.data(), temperature, dt,
                              friction, seed);
           }),
           py::arg("masses"), py::arg("temperature"), py::arg("dt"),
           py::arg("friction"), py::arg("seed"));
}

template <typename RealType>
void declare_velocity_verlet_integrator(py::module &m, const char *typestr) {

  using Class = VelocityVerletIntegrator<RealType>;
  std::string pyclass_name = std::string("VelocityVerletIntegrator_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Integrator<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](RealType dt,
                       const py::array_t<RealType, py::array::c_style> &cbs) {
             return new VelocityVerletIntegrator<RealType>(cbs.size(), dt,
                                                           cbs.data());
           }),
           py::arg("dt"), py::arg("cbs"));
}

template <typename RealType>
void declare_potential(py::module &m, const char *typestr) {

  using Class = Potential<RealType>;
  std::string pyclass_name = std::string("Potential_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def("batch_size", &Class::batch_size)
      .def(
          "execute_batch",
          [](Potential<RealType> &pot,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &params,
             const py::array_t<RealType, py::array::c_style> &boxes,
             const bool compute_du_dx, const bool compute_du_dp,
             const bool compute_u) -> py::tuple {
            if (coords.ndim() != 3 || boxes.ndim() != 3) {
              throw std::runtime_error(
                  "coords and boxes must have 3 dimensions");
            }
            if (coords.shape()[0] != boxes.shape()[0]) {
              throw std::runtime_error(
                  "number of batches of coords and boxes don't match");
            }
            if (params.ndim() < 2) {
              throw std::runtime_error(
                  "parameters must have at least 2 dimensions");
            }

            const long unsigned int coord_batches = coords.shape()[0];
            const long unsigned int N = coords.shape()[1];
            const long unsigned int D = coords.shape()[2];

            const long unsigned int param_batches = params.shape()[0];
            const long unsigned int P = params.size() / param_batches;

            const long unsigned int total_executions =
                coord_batches * param_batches;

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_batch_host`) Only
            // initialize memory when needed, as buffers can be quite large
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(total_executions * N * D, 9999);
            }
            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(total_executions * P, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(total_executions, 9999);
            }

            pot.execute_batch_host(coord_batches, N, param_batches, P,
                                   coords.data(), params.data(), boxes.data(),
                                   compute_du_dx ? du_dx.data() : nullptr,
                                   compute_du_dp ? du_dp.data() : nullptr,
                                   compute_u ? u.data() : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());
            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {coord_batches, param_batches, N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }

            if (compute_du_dp) {
              std::vector<ssize_t> pshape(params.shape(),
                                          params.shape() + params.ndim());
              // Remove the first dimension of the parameters shape to be
              // consistent in ordering of return values
              pshape.erase(pshape.begin());
              // Append the new dimensions for the du_dps
              std::vector<unsigned long int> shape(
                  {coord_batches, param_batches});
              pshape.insert(pshape.begin(), shape.begin(), shape.end());

              py::array_t<RealType, py::array::c_style> py_du_dp(pshape);
              for (unsigned int i = 0; i < total_executions; i++) {
                pot.du_dp_fixed_to_float(N, P, &du_dp[0] + (i * P),
                                         py_du_dp.mutable_data() + (i * P));
              }
              result[1] = py_du_dp;
            }

            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u(
                  {coord_batches, param_batches});

              for (unsigned int i = 0; i < py_u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              result[2] = py_u;
            }

            return result;
          },
          py::arg("coords"), py::arg("params"), py::arg("boxes"),
          py::arg("compute_du_dx"), py::arg("compute_du_dp"),
          py::arg("compute_u"),
          R"pbdoc(
        Execute the potential over a batch of coords and parameters. The total number of executions of the potential is
        num_coord_batches * num_param_batches.

        Note: This function allocates memory for all of the inputs on the GPU. This may lead to OOMs.

        Parameters
        ----------
        coords: NDArray
            A three dimensional array containing a batch of coordinates.

        params: NDArray
            A multi dimensional array containing a batch of parameters. First dimension
            determines the batch size, the rest of the array is passed to the potential as the
            parameters.

        boxes: NDArray
            A three dimensional array containing a batch of boxes.

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx.

        compute_du_dp: bool
            Indicates to compute du_dp, else returns None for du_dp.

        compute_u: bool
            Indicates to compute u, else returns None for u.


        Returns
        -------
        3-tuple of du_dx, du_dp, u
            coord_batch_size = coords.shape[0]
            param_batch_size = params.shape[0]
            du_dx has shape (coords_batch_size, param_batch_size, N, 3)
            du_dp has shape (coords_batch_size, param_batch_size, P)
            u has shape (coords_batch_size, param_batch_size)

    )pbdoc")
      .def(
          "execute_batch_sparse",
          [](Potential<RealType> &pot,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &params,
             const py::array_t<RealType, py::array::c_style> &boxes,
             const py::array_t<unsigned int, py::array::c_style>
                 &coords_batch_idxs,
             const py::array_t<unsigned int, py::array::c_style>
                 &params_batch_idxs,
             const bool compute_du_dx, const bool compute_du_dp,
             const bool compute_u) -> py::tuple {
            if (coords.ndim() != 3 || boxes.ndim() != 3) {
              throw std::runtime_error(
                  "coords and boxes must have 3 dimensions");
            }
            if (coords.shape()[0] != boxes.shape()[0]) {
              throw std::runtime_error(
                  "number of coord arrays and boxes don't match");
            }
            if (params.ndim() < 2) {
              throw std::runtime_error(
                  "parameters must have at least 2 dimensions");
            }
            if (coords_batch_idxs.ndim() != 1 ||
                params_batch_idxs.ndim() != 1) {
              throw std::runtime_error(
                  "coords_batch_idxs and params_batch_idxs must be "
                  "one-dimensional arrays");
            }
            if (coords_batch_idxs.size() != params_batch_idxs.size()) {
              throw std::runtime_error(
                  "coords_batch_idxs and params_batch_idxs must have the same "
                  "length");
            }

            const long unsigned int batch_size = coords_batch_idxs.size();
            const unsigned int *coords_batch_idxs_data =
                coords_batch_idxs.data();
            const unsigned int *params_batch_idxs_data =
                params_batch_idxs.data();

            for (long unsigned int i = 0; i < batch_size; i++) {
              if (coords_batch_idxs_data[i] >= coords.shape()[0]) {
                throw std::runtime_error("coords_batch_idxs contains an index "
                                         "that is out of bounds");
              }
              if (params_batch_idxs_data[i] >= params.shape()[0]) {
                throw std::runtime_error("params_batch_idxs contains an index "
                                         "that is out of bounds");
              }
            }

            const long unsigned int coords_size = coords.shape()[0];
            const long unsigned int N = coords.shape()[1];
            const long unsigned int D = coords.shape()[2];

            const long unsigned int params_size = params.shape()[0];
            const long unsigned int P = params.size() / params_size;

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_batch_host`) Only
            // initialize memory when needed, as buffers can be quite large
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(batch_size * N * D, 9999);
            }
            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(batch_size * P, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(batch_size, 9999);
            }

            pot.execute_batch_sparse_host(
                coords_size, N, params_size, P, batch_size,
                coords_batch_idxs_data, params_batch_idxs_data, coords.data(),
                params.data(), boxes.data(),
                compute_du_dx ? du_dx.data() : nullptr,
                compute_du_dp ? du_dp.data() : nullptr,
                compute_u ? u.data() : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());
            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {batch_size, N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }

            if (compute_du_dp) {
              std::vector<ssize_t> pshape(params.shape(),
                                          params.shape() + params.ndim());
              pshape[0] = batch_size;

              py::array_t<RealType, py::array::c_style> py_du_dp(pshape);
              for (unsigned int i = 0; i < batch_size; i++) {
                pot.du_dp_fixed_to_float(N, P, &du_dp[0] + (i * P),
                                         py_du_dp.mutable_data() + (i * P));
              }
              result[1] = py_du_dp;
            }

            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u(batch_size);

              for (unsigned int i = 0; i < py_u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              result[2] = py_u;
            }

            return result;
          },
          py::arg("coords"), py::arg("params"), py::arg("boxes"),
          py::arg("coords_batch_idxs"), py::arg("params_batch_idxs"),
          py::arg("compute_du_dx"), py::arg("compute_du_dp"),
          py::arg("compute_u"),
          R"pbdoc(
        Execute the potential over a batch of coordinates and parameters. Similar to execute_batch, except that instead
        of evaluating the potential on the dense matrix of pairs of coordinates and parameters, this accepts arrays
        specifying the indices of the coordinates and parameters to use for each evaluation, allowing evaluation of
        arbitrary elements of the matrix. The total number of evaluations is len(coords_batch_idxs)
        [= len(params_batch_idxs)].

        Notes
        -----
        * This function allocates memory for all of the inputs on the GPU. This may lead to OOMs.
        * When using with stateful potentials, care should be taken in the ordering of the evaluations (as specified by
          coords_batch_idxs and params_batch_idxs) to maintain efficiency. For example, batch evaluation of a nonbonded
          all-pairs potential may be most efficient in the order [(coords_1, params_1), (coords_1, params_2), ... ,
          (coords_2, params_1), ..., (coords_n, params_n)], i.e. with an "outer loop" over the coordinates and "inner
          loop" over parameters, to avoid unnecessary rebuilds of the neighborlist.

        Parameters
        ----------
        coords: NDArray
            (coords_size, n_atoms, 3) array containing multiple coordinate arrays

        params: NDArray
            (params_size, P) array containing multiple parameter arrays

        boxes: NDArray
            (coords_size, 3, 3) array containing a batch of boxes

        coords_batch_idxs: NDArray
            (batch_size,) indices of the coordinates to use for each evaluation

        params_batch_idxs: NDArray
            (batch_size,) indices of the parameters to use for each evaluation

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx

        compute_du_dp: bool
            Indicates to compute du_dp, else returns None for du_dp

        compute_u: bool
            Indicates to compute u, else returns None for u


        Returns
        -------
        3-tuple of du_dx, du_dp, u
            batch_size = coords_batch_idxs.shape[0]
            du_dx has shape (batch_size, N, 3)
            du_dp has shape (batch_size, P)
            u has shape (batch_size,)

    )pbdoc")
      .def(
          "execute",
          [](Potential<RealType> &pot,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &params,
             const py::array_t<RealType, py::array::c_style> &box,
             bool compute_du_dx, bool compute_du_dp,
             bool compute_u) -> py::tuple {
            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.size();
            verify_coords_and_box(coords, box);

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_host`)
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(N * D, 9999);
            }
            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(P, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(1, 9999);
            }

            pot.execute_host(1, N, P, coords.data(), params.data(), box.data(),
                             compute_du_dx ? &du_dx[0] : nullptr,
                             compute_du_dp ? &du_dp[0] : nullptr,
                             compute_u ? &u[0] : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());

            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx({N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }
            if (compute_du_dp) {
              std::vector<ssize_t> pshape(params.shape(),
                                          params.shape() + params.ndim());

              py::array_t<RealType, py::array::c_style> py_du_dp(pshape);
              pot.du_dp_fixed_to_float(N, P, &du_dp[0],
                                       py_du_dp.mutable_data());
              result[1] = py_du_dp;
            }
            if (compute_u) {
              RealType u_sum = convert_energy_to_fp(u[0]);
              // returning a raw float32 causes python to interpret it as a
              // 'float' type (which is 64bit)
              result[2] = u_sum;
            }

            return result;
          },
          py::arg("coords"), py::arg("params"), py::arg("box"),
          py::arg("compute_du_dx") = true, py::arg("compute_du_dp") = true,
          py::arg("compute_u") = true)
      .def(
          "execute_dim",
          [](Potential<RealType> &pot,
             const py::array_t<RealType, py::array::c_style> &coords,
             const std::vector<py::array_t<RealType, py::array::c_style>>
                 &params,
             const py::array_t<RealType, py::array::c_style> &boxes,
             const bool compute_du_dx, const bool compute_du_dp,
             const bool compute_u) -> py::tuple {
            const long batches = coords.shape(0);
            if (static_cast<long>(params.size()) != batches) {
              throw std::runtime_error(
                  "Parameters must have same number of batches as coords");
            } else if (boxes.shape(0) != batches) {
              throw std::runtime_error(
                  "Boxes must have same number of batches as coords");
            }

            const long unsigned int N = coords.shape(1);
            const long unsigned int D = coords.shape(2);
            // verify_coords_and_box(coords, box);

            long unsigned int P = 0;
            for (auto batch : params) {
              P += batch.size();
            }
            py::array_t<RealType, py::array::c_style> param_data(P);
            int offset = 0;
            for (auto batch : params) {
              std::memcpy(param_data.mutable_data() + offset, batch.data(),
                          batch.size() * sizeof(RealType));
              offset += batch.size();
            }

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_host`)
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(batches * N * D, 9999);
            }
            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(P, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(batches, 9999);
            }

            pot.execute_host(batches, N, P, coords.data(), param_data.data(),
                             boxes.data(), compute_du_dx ? &du_dx[0] : nullptr,
                             compute_du_dp ? &du_dp[0] : nullptr,
                             compute_u ? &u[0] : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());

            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {static_cast<long unsigned int>(batches), N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }
            if (compute_du_dp) {
              std::vector<py::array_t<RealType, py::array::c_style>> py_du_dp;
              int offset = 0;
              for (auto batch : params) {
                // There has to be a cleaner way to handle this
                if (batch.ndim() == 2) {
                  py::array_t<RealType, py::array::c_style> batch_du_dp(
                      {batch.shape(0), batch.shape(1)});
                  pot.du_dp_fixed_to_float(N, batch.size(), &du_dp[0] + offset,
                                           batch_du_dp.mutable_data());
                  py_du_dp.push_back(batch_du_dp);
                } else {
                  py::array_t<RealType, py::array::c_style> batch_du_dp(
                      {batch.shape(0)});
                  pot.du_dp_fixed_to_float(N, batch.size(), &du_dp[0] + offset,
                                           batch_du_dp.mutable_data());
                  py_du_dp.push_back(batch_du_dp);
                }
                offset += batch.size();
              }
              result[1] = py_du_dp;
            }
            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u({batches});
              for (unsigned int i = 0; i < u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              // returning a raw float32 causes python to interpret it as a
              // 'float' type (which is 64bit)
              result[2] = py_u;
            }

            return result;
          },
          py::arg("coords"), py::arg("params"), py::arg("box"),
          py::arg("compute_du_dx") = true, py::arg("compute_du_dp") = true,
          py::arg("compute_u") = true)
      .def(
          "execute_du_dx",
          [](Potential<RealType> &pot,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &params,
             const py::array_t<RealType, py::array::c_style> &box)
              -> py::array_t<RealType, py::array::c_style> {
            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            const long unsigned int P = params.size();
            verify_coords_and_box(coords, box);

            std::vector<unsigned long long> du_dx(N * D);

            pot.execute_host_du_dx(N, P, coords.data(), params.data(),
                                   box.data(), &du_dx[0]);

            py::array_t<RealType, py::array::c_style> py_du_dx({N, D});
            for (unsigned int i = 0; i < du_dx.size(); i++) {
              py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
            }

            return py_du_dx;
          },
          py::arg("coords"), py::arg("params"), py::arg("box"));
}

template <typename RealType>
void declare_bound_potential(py::module &m, const char *typestr) {

  using Class = BoundPotential<RealType>;
  std::string pyclass_name = std::string("BoundPotential_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(
          py::init([](std::shared_ptr<Potential<RealType>> potential,
                      const py::array_t<RealType, py::array::c_style> &params) {
            int params_dim = 1; // Has to be at least one
            if (params.ndim() == 2) {
              params_dim = params.shape()[1];
            }
            return new BoundPotential<RealType>(
                potential, py_array_to_vector(params), params_dim);
          }),
          py::arg("potential"), py::arg("params"))
      .def("get_potential",
           [](const BoundPotential<RealType> &bp) { return bp.potential; })
      .def("get_flat_params", &BoundPotential<RealType>::get_params)
      .def(
          "set_params",
          [](BoundPotential<RealType> &bp,
             const py::array_t<RealType, py::array::c_style> &params) {
            bp.set_params(py_array_to_vector(params));
          },
          py::arg("params"))
      .def("size", [](const BoundPotential<RealType> &bp) { return bp.size; })
      .def(
          "execute",
          [](BoundPotential<RealType> &bp,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &box,
             bool compute_du_dx, bool compute_u) -> py::tuple {
            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            verify_coords_and_box(coords, box);

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_host`)
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(N * D, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(1, 9999);
            }

            bp.execute_host(1, N, coords.data(), box.data(),
                            compute_du_dx ? &du_dx[0] : nullptr,
                            compute_u ? &u[0] : nullptr);

            auto result = py::make_tuple(py::none(), py::none());

            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx({N, D});
              if (compute_du_dx) {
                for (unsigned int i = 0; i < du_dx.size(); i++) {
                  py_du_dx.mutable_data()[i] =
                      FIXED_TO_FLOAT<RealType>(du_dx[i]);
                }
              }
              result[0] = py_du_dx;
            }
            if (compute_u) {
              RealType u_sum = convert_energy_to_fp(u[0]);
              result[1] = u_sum;
            }

            return result;
          },
          py::arg("coords"), py::arg("box"), py::arg("compute_du_dx") = true,
          py::arg("compute_u") = true)
      .def(
          "execute_batch",
          [](BoundPotential<RealType> &bp,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &boxes,
             const bool compute_du_dx, const bool compute_u) -> py::tuple {
            if (coords.ndim() != 3 && boxes.ndim() != 3) {
              throw std::runtime_error(
                  "coords and boxes must have 3 dimensions");
            }
            if (coords.shape()[0] != boxes.shape()[0]) {
              throw std::runtime_error(
                  "number of batches of coords and boxes don't match");
            }

            const long unsigned int coord_batches = coords.shape()[0];
            const long unsigned int N = coords.shape()[1];
            const long unsigned int D = coords.shape()[2];

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_batch_host`) Only
            // initialize memory when needed, as buffers can be quite large
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(coord_batches * N * D, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(coord_batches, 9999);
            }

            bp.execute_batch_host(coord_batches, N, coords.data(), boxes.data(),
                                  compute_du_dx ? du_dx.data() : nullptr,
                                  compute_u ? u.data() : nullptr);

            auto result = py::make_tuple(py::none(), py::none());
            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {coord_batches, N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }

            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u(coord_batches);

              for (unsigned int i = 0; i < py_u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              result[1] = py_u;
            }

            return result;
          },
          py::arg("coords"), py::arg("boxes"), py::arg("compute_du_dx"),
          py::arg("compute_u"),
          R"pbdoc(
        Execute the potential over a batch of coordinates and boxes.

        Note: This function allocates memory for all of the inputs on the GPU. This may lead to OOMs.

        Parameters
        ----------
        coords: NDArray
            A three dimensional array containing a batch of coordinates.

        boxes: NDArray
            A three dimensional array containing a batch of boxes.

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx.

        compute_u: bool
            Indicates to compute u, else returns None for u.


        Returns
        -------
        2-tuple of du_dx, u
            coord_batch_size = coords.shape[0]
            du_dx has shape (coords_batch_size, N, 3)
            u has shape (coords_batch_size)

    )pbdoc")
      .def(
          "execute_fixed",
          [](BoundPotential<RealType> &bp,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &box)
              -> const py::array_t<uint64_t, py::array::c_style> {
            const long unsigned int N = coords.shape()[0];
            verify_coords_and_box(coords, box);
            std::vector<__int128> u(1, 9999);

            bp.execute_host(1, N, coords.data(), box.data(), nullptr, &u[0]);

            py::array_t<uint64_t, py::array::c_style> py_u(1);
            if (fixed_point_overflow(u[0])) {
              // Force it to a specific value, else conversion borks
              py_u.mutable_data()[0] = LLONG_MAX;
            } else {
              py_u.mutable_data()[0] = u[0];
            }
            return py_u;
          },
          py::arg("coords"), py::arg("box"));
}

template <typename RealType>
void declare_potential_executor(py::module &m, const char *typestr) {

  using Class = tmd::PotentialExecutor<RealType>;
  std::string pyclass_name = std::string("PotentialExecutor_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const bool parallel) {
             return new tmd::PotentialExecutor<RealType>(parallel);
           }),
           py::arg("parallel") = true)
      .def(
          "execute",
          [](tmd::PotentialExecutor<RealType> &runner,
             std::vector<std::shared_ptr<tmd::Potential<RealType>>> &pots,
             const py::array_t<RealType, py::array::c_style> &coords,
             const std::vector<py::array_t<RealType, py::array::c_style>>
                 &params,
             const py::array_t<RealType, py::array::c_style> &box,
             const bool compute_du_dx, const bool compute_du_dp,
             const bool compute_u) -> py::tuple {
            const long unsigned int N = coords.shape()[0];
            const long unsigned int D = coords.shape()[1];
            verify_coords_and_box(coords, box);

            if (!compute_du_dx && !compute_u && !compute_du_dp) {
              throw std::runtime_error(
                  "must compute either du_dx, du_dp or energy");
            }

            if (params.size() != pots.size()) {
              throw std::runtime_error("number of potentials and the number of "
                                       "parameter sets must match");
            }
            const long unsigned int num_pots = pots.size();

            int offset = 0;
            std::vector<RealType> combined_params(0);

            std::vector<int> vec_param_sizes;

            for (unsigned int i = 0; i < num_pots; i++) {
              const int size = params[i].size();
              vec_param_sizes.push_back(size);
              combined_params.resize(combined_params.size() + size);
              std::memcpy(combined_params.data() + offset, params[i].data(),
                          size * sizeof(RealType));
              offset += size;
            }
            const int P = runner.get_total_num_params(vec_param_sizes);

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_potentials`)
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(num_pots * N * D, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(num_pots, 9999);
            }

            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(P, 9999);
            }

            runner.execute_potentials(N, vec_param_sizes, pots, coords.data(),
                                      combined_params.data(), box.data(),
                                      compute_du_dx ? &du_dx[0] : nullptr,
                                      compute_du_dp ? &du_dp[0] : nullptr,
                                      compute_u ? &u[0] : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());
            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {num_pots, N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }

            if (compute_du_dp) {
              std::vector<py::array_t<RealType, py::array::c_style>>
                  output_du_dp;
              offset = 0;
              for (unsigned int i = 0; i < num_pots; i++) {
                std::vector<ssize_t> pshape(
                    params[i].shape(), params[i].shape() + params[i].ndim());
                py::array_t<RealType, py::array::c_style> py_du_dp(pshape);
                pots[i]->du_dp_fixed_to_float(N, params[i].size(),
                                              &du_dp[0] + offset,
                                              py_du_dp.mutable_data());
                offset += params[i].size();
                output_du_dp.push_back(py_du_dp);
              }
              result[1] = output_du_dp;
            }

            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u(num_pots);

              for (unsigned int i = 0; i < py_u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              result[2] = py_u;
            }

            return result;
          },
          py::arg("pots"), py::arg("coords"), py::arg("params"), py::arg("box"),
          py::arg("compute_du_dx") = true, py::arg("compute_du_dp") = true,
          py::arg("compute_u") = true,
          R"pbdoc("Execute potentials over the a set of coordinates and box.

        Parameters
        ----------
        pots: list[Potential]
            list of potentials to execute over

        coords: NDArray
            (n_atoms, 3) array containing the coordinates

        params: list[NDArray]
            (n_pots, P) list containing multiple parameter arrays

        boxes: NDArray
            (3, 3) array containing the boxes

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx

        compute_du_dp: bool
            Indicates to compute du_dp, else returns None for du_dp

        compute_u: bool
            Indicates to compute u, else returns None for u


        Returns
        -------
        3-tuple of du_dx, du_dp, u
            du_dx has shape (n_pots, N, 3)
            du_dp has shape (n_pots,  P)
            u has shape (n_pots)

    )pbdoc")
      .def(
          "execute_batch",
          [](tmd::PotentialExecutor<RealType> &runner,
             std::vector<std::shared_ptr<tmd::Potential<RealType>>> &pots,
             const py::array_t<RealType, py::array::c_style> &coords,
             const std::vector<py::array_t<RealType, py::array::c_style>>
                 &params,
             const py::array_t<RealType, py::array::c_style> &boxes,
             const bool compute_du_dx, const bool compute_du_dp,
             const bool compute_u) -> py::tuple {
            if (!compute_du_dx && !compute_u && !compute_du_dp) {
              throw std::runtime_error(
                  "must compute either du_dx, du_dp or energy");
            }
            if (coords.ndim() != 3 || boxes.ndim() != 3) {
              throw std::runtime_error(
                  "coords and boxes must have 3 dimensions");
            }
            if (coords.shape()[0] != boxes.shape()[0]) {
              throw std::runtime_error(
                  "number of batches of coords and boxes don't match");
            }

            const long unsigned int coord_batches = coords.shape()[0];
            const long unsigned int N = coords.shape()[1];
            const long unsigned int D = coords.shape()[2];

            if (params.size() != pots.size()) {
              throw std::runtime_error("number of potentials and the number of "
                                       "parameter sets must match");
            }

            const long unsigned int param_batches = params[0].shape()[0];
            for (auto param_batch : params) {
              if (param_batch.shape()[0] !=
                  static_cast<long int>(param_batches)) {
                throw std::runtime_error("number of parameter batches must "
                                         "match for each potential");
              }
              if (param_batch.ndim() < 2) {
                throw std::runtime_error(
                    "params must have at least 2 dimensions");
              }
            }
            const long unsigned int total_combinations =
                coord_batches * param_batches;
            const long unsigned int num_pots = pots.size();

            std::vector<RealType> combined_params(0);

            std::vector<int> vec_batch_param_sizes;

            // Probably a better way to do this
            int offset = 0;
            for (unsigned int i = 0; i < num_pots; i++) {
              const int size = params[i].size();
              // Only count the size of parameters per batch
              vec_batch_param_sizes.push_back(size / params[i].shape()[0]);

              combined_params.resize(combined_params.size() + size);

              std::memcpy(combined_params.data() + offset, params[i].data(),
                          size * sizeof(RealType));
              offset += size;
            }
            const int P = runner.get_total_num_params(vec_batch_param_sizes);

            // Setup the indices to use. Reduces duplicate code with
            // execute_batch_sparse
            std::vector<unsigned int> coords_batch_idxs;
            std::vector<unsigned int> param_batch_idxs;
            for (unsigned int i = 0; i < coord_batches; i++) {
              for (unsigned int j = 0; j < param_batches; j++) {
                coords_batch_idxs.push_back(i);
                param_batch_idxs.push_back(j);
              }
            }
            if (coords_batch_idxs.size() != param_batch_idxs.size() ||
                coords_batch_idxs.size() != total_combinations) {
              throw std::runtime_error("Something went wrong, report a bug");
            }

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_potentials`)
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(total_combinations * num_pots * N * D, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(total_combinations * num_pots, 9999);
            }

            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(total_combinations * P, 9999);
            }

            runner.execute_batch_potentials_sparse(
                N, vec_batch_param_sizes, total_combinations, coord_batches,
                param_batches, coords_batch_idxs.data(),
                param_batch_idxs.data(), pots, coords.data(),
                combined_params.data(), boxes.data(),
                compute_du_dx ? &du_dx[0] : nullptr,
                compute_du_dp ? &du_dp[0] : nullptr,
                compute_u ? &u[0] : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());
            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {num_pots, coord_batches, param_batches, N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }

            if (compute_du_dp) {
              std::vector<py::array_t<RealType, py::array::c_style>>
                  output_du_dp;
              int offset = 0;
              for (unsigned int i = 0; i < num_pots; i++) {
                std::vector<ssize_t> pshape(
                    params[i].shape(), params[i].shape() + params[i].ndim());
                // Add the coords batch dimension
                pshape.insert(pshape.begin(),
                              static_cast<ssize_t>(coord_batches));
                py::array_t<RealType, py::array::c_style> py_du_dp(pshape);
                pots[i]->du_dp_fixed_to_float(
                    N * total_combinations, py_du_dp.size(), &du_dp[0] + offset,
                    py_du_dp.mutable_data());
                offset += py_du_dp.size();
                output_du_dp.push_back(py_du_dp);
              }
              result[1] = output_du_dp;
            }

            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u(
                  {num_pots, coord_batches, param_batches});

              for (unsigned int i = 0; i < py_u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              result[2] = py_u;
            }

            return result;
          },
          py::arg("pots"), py::arg("coords"), py::arg("params"), py::arg("box"),
          py::arg("compute_du_dx") = true, py::arg("compute_du_dp") = true,
          py::arg("compute_u") = true,
          R"pbdoc("Execute potentials over a batch of coordinates and parameters. The total number of evaluations is len(pots) * len(coords) * len(params)

        Notes
        -----
        * This function allocates memory for all of the inputs on the GPU. This may lead to OOMs.
        * When using with stateful potentials, care should be taken in the ordering of the evaluations (as specified by
          coords_batch_idxs and params_batch_idxs) to maintain efficiency. For example, batch evaluation of a nonbonded
          all-pairs potential may be most efficient in the order [(coords_1, params_1), (coords_1, params_2), ... ,
          (coords_2, params_1), ..., (coords_n, params_n)], i.e. with an "outer loop" over the coordinates and "inner
          loop" over parameters, to avoid unnecessary rebuilds of the neighborlist.

        Parameters
        ----------
        pots: list[Potential]
            list of potentials to execute over

        coords: NDArray
            (coord_batches, n_atoms, 3) array containing multiple coordinate arrays

        params: list[NDArray]
            (n_pots, param_batches, P) list containing multiple parameter arrays

        boxes: NDArray
            (coord_batches, 3, 3) array containing a batch of boxes

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx

        compute_du_dp: bool
            Indicates to compute du_dp, else returns None for du_dp

        compute_u: bool
            Indicates to compute u, else returns None for u


        Returns
        -------
        3-tuple of du_dx, du_dp, u
            coord_batches = coords.shape[0]
            param_batches = params[0].shape[0]
            du_dx has shape (n_pots, coord_batches, param_batches, N, 3)
            du_dp has shape (n_pots, coord_batches, param_batches, P)
            u has shape (n_pots, coord_batches, param_batches)

    )pbdoc")
      .def(
          "execute_batch_sparse",
          [](tmd::PotentialExecutor<RealType> &runner,
             std::vector<std::shared_ptr<tmd::Potential<RealType>>> &pots,
             const py::array_t<RealType, py::array::c_style> &coords,
             const std::vector<py::array_t<RealType, py::array::c_style>>
                 &params,
             const py::array_t<RealType, py::array::c_style> &boxes,
             const py::array_t<unsigned int, py::array::c_style>
                 &coords_batch_idxs,
             const py::array_t<unsigned int, py::array::c_style>
                 &params_batch_idxs,
             const bool compute_du_dx, const bool compute_du_dp,
             const bool compute_u) -> py::tuple {
            if (!compute_du_dx && !compute_u && !compute_du_dp) {
              throw std::runtime_error(
                  "must compute either du_dx, du_dp or energy");
            }
            if (coords.ndim() != 3 || boxes.ndim() != 3) {
              throw std::runtime_error(
                  "coords and boxes must have 3 dimensions");
            }
            if (coords.shape()[0] != boxes.shape()[0]) {
              throw std::runtime_error(
                  "number of batches of coords and boxes don't match");
            }

            const long unsigned int coord_batches = coords.shape()[0];
            const long unsigned int N = coords.shape()[1];
            const long unsigned int D = coords.shape()[2];

            if (params.size() != pots.size()) {
              throw std::runtime_error("number of potentials and the number of "
                                       "parameter sets must match");
            }

            const long unsigned int param_batches = params[0].shape()[0];
            for (auto param_batch : params) {
              if (param_batch.shape()[0] !=
                  static_cast<long int>(param_batches)) {
                throw std::runtime_error("number of parameter batches must "
                                         "match for each potential");
              }
              if (param_batch.ndim() < 2) {
                throw std::runtime_error(
                    "params must have at least 2 dimensions");
              }
            }

            if (coords_batch_idxs.ndim() != 1 ||
                params_batch_idxs.ndim() != 1) {
              throw std::runtime_error(
                  "coords_batch_idxs and params_batch_idxs must be "
                  "one-dimensional arrays");
            }
            if (coords_batch_idxs.size() != params_batch_idxs.size()) {
              throw std::runtime_error(
                  "coords_batch_idxs and params_batch_idxs must have the same "
                  "length");
            }

            const int batch_size = coords_batch_idxs.size();
            const unsigned int *coords_batch_idxs_data =
                coords_batch_idxs.data();
            const unsigned int *params_batch_idxs_data =
                params_batch_idxs.data();

            for (int i = 0; i < batch_size; i++) {
              if (coords_batch_idxs_data[i] >= coord_batches) {
                throw std::runtime_error("coords_batch_idxs contains an index "
                                         "that is out of bounds");
              }
              if (params_batch_idxs_data[i] >= param_batches) {
                throw std::runtime_error("params_batch_idxs contains an index "
                                         "that is out of bounds");
              }
            }

            const long unsigned int num_pots = pots.size();

            std::vector<RealType> combined_params(0);

            std::vector<int> vec_batch_param_sizes;

            // Probably a better way to do this
            int offset = 0;
            for (unsigned int i = 0; i < num_pots; i++) {
              const int size = params[i].size();
              // Only count the size of parameters per batch
              vec_batch_param_sizes.push_back(size / params[i].shape()[0]);

              combined_params.resize(combined_params.size() + size);

              std::memcpy(combined_params.data() + offset, params[i].data(),
                          size * sizeof(RealType));
              offset += size;
            }
            const int P = runner.get_total_num_params(vec_batch_param_sizes);

            // initialize with fixed garbage values for debugging convenience
            // (these should be overwritten by `execute_potentials`)
            std::vector<unsigned long long> du_dx;
            if (compute_du_dx) {
              du_dx.assign(batch_size * num_pots * N * D, 9999);
            }
            std::vector<__int128> u;
            if (compute_u) {
              u.assign(batch_size * num_pots, 9999);
            }

            std::vector<unsigned long long> du_dp;
            if (compute_du_dp) {
              du_dp.assign(batch_size * P, 9999);
            }

            runner.execute_batch_potentials_sparse(
                N, vec_batch_param_sizes, batch_size, coord_batches,
                param_batches, coords_batch_idxs_data, params_batch_idxs_data,
                pots, coords.data(), combined_params.data(), boxes.data(),
                compute_du_dx ? &du_dx[0] : nullptr,
                compute_du_dp ? &du_dp[0] : nullptr,
                compute_u ? &u[0] : nullptr);

            auto result = py::make_tuple(py::none(), py::none(), py::none());
            if (compute_du_dx) {
              py::array_t<RealType, py::array::c_style> py_du_dx(
                  {num_pots, static_cast<long unsigned int>(batch_size), N, D});
              for (unsigned int i = 0; i < du_dx.size(); i++) {
                py_du_dx.mutable_data()[i] = FIXED_TO_FLOAT<RealType>(du_dx[i]);
              }
              result[0] = py_du_dx;
            }

            if (compute_du_dp) {
              std::vector<py::array_t<RealType, py::array::c_style>>
                  output_du_dp;
              int offset = 0;
              for (unsigned int i = 0; i < num_pots; i++) {
                std::vector<ssize_t> pshape(
                    params[i].shape(), params[i].shape() + params[i].ndim());
                // set the batch dimension
                pshape[0] = batch_size;
                py::array_t<RealType, py::array::c_style> py_du_dp(pshape);
                pots[i]->du_dp_fixed_to_float(N * batch_size, py_du_dp.size(),
                                              &du_dp[0] + offset,
                                              py_du_dp.mutable_data());
                offset += py_du_dp.size();
                output_du_dp.push_back(py_du_dp);
              }
              result[1] = output_du_dp;
            }

            if (compute_u) {
              py::array_t<RealType, py::array::c_style> py_u(
                  {num_pots, static_cast<long unsigned int>(batch_size)});

              for (unsigned int i = 0; i < py_u.size(); i++) {
                py_u.mutable_data()[i] = convert_energy_to_fp(u[i]);
              }
              result[2] = py_u;
            }

            return result;
          },
          py::arg("pots"), py::arg("coords"), py::arg("params"), py::arg("box"),
          py::arg("coords_batch_idxs"), py::arg("params_batch_idxs"),
          py::arg("compute_du_dx") = true, py::arg("compute_du_dp") = true,
          py::arg("compute_u") = true,
          R"pbdoc("Execute potentials over a batch of coordinates and parameters. Similar to execute_batch, except that instead
        of evaluating the potential on the dense matrix of pairs of coordinates and parameters, this accepts arrays
        specifying the indices of the coordinates and parameters to use for each evaluation, allowing evaluation of
        arbitrary elements of the matrix. The total number of evaluations is len(coords_batch_idxs)
        [= len(params_batch_idxs)].

        Notes
        -----
        * This function allocates memory for all of the inputs on the GPU. This may lead to OOMs.
        * When using with stateful potentials, care should be taken in the ordering of the evaluations (as specified by
          coords_batch_idxs and params_batch_idxs) to maintain efficiency. For example, batch evaluation of a nonbonded
          all-pairs potential may be most efficient in the order [(coords_1, params_1), (coords_1, params_2), ... ,
          (coords_2, params_1), ..., (coords_n, params_n)], i.e. with an "outer loop" over the coordinates and "inner
          loop" over parameters, to avoid unnecessary rebuilds of the neighborlist.

        Parameters
        ----------
        pots: list[Potential]
            list of potentials to execute over

        coords: NDArray
            (coords_size, n_atoms, 3) array containing multiple coordinate arrays

        params: list[NDArray]
            (n_pots, params_size, P) array containing multiple parameter arrays

        boxes: NDArray
            (coords_size, 3, 3) array containing a batch of boxes

        coords_batch_idxs: NDArray
            (batch_size,) indices of the coordinates to use for each evaluation

        params_batch_idxs: NDArray
            (batch_size,) indices of the parameters to use for each evaluation

        compute_du_dx: bool
            Indicates to compute du_dx, else returns None for du_dx

        compute_du_dp: bool
            Indicates to compute du_dp, else returns None for du_dp

        compute_u: bool
            Indicates to compute u, else returns None for u


        Returns
        -------
        3-tuple of du_dx, du_dp, u
            batch_size = coords_batch_idxs.shape[0]
            du_dx has shape (n_pots, batch_size, N, 3)
            du_dp has shape (n_pots, batch_size, P)
            u has shape (n_pots, batch_size,)

    )pbdoc");
}

// TBD: Is this needed?
template <typename T>
std::vector<T> flatten_vector_of_arrays(
    const std::vector<py::array_t<T, py::array::c_style>> &input) {
  int offset = 0;
  std::vector<T> output;

  for (int i = 0; i < input.size(); i++) {
    const unsigned long arr_size = input[i].size();
    output.resize(output.size() + arr_size);
    std::memcpy(output.data() + offset, input[i].data(), arr_size * sizeof(T));
    offset += arr_size;
  }
}

template <typename RealType>
void declare_harmonic_bond(py::module &m, const char *typestr) {

  using Class = HarmonicBond<RealType>;
  std::string pyclass_name = std::string("HarmonicBond_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int num_atoms,
                       const py::array_t<int, py::array::c_style> &bond_idxs) {
             verify_bond_idxs(bond_idxs, 2);
             // Create a vector with all zeros
             std::vector<int> bond_system_idxs(bond_idxs.shape(0), 0);
             const int num_batches = 1;
             return new HarmonicBond<RealType>(num_batches, num_atoms,
                                               py_array_to_vector(bond_idxs),
                                               bond_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("bond_idxs"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &bond_idxs) {
             const int num_batches = bond_idxs.size();
             std::vector<int> combined_bond_vec;
             std::vector<int> bond_system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(bond_idxs[i], 2);
               const unsigned long bond_arr_size = bond_idxs[i].size();
               combined_bond_vec.resize(combined_bond_vec.size() +
                                        bond_arr_size);
               std::memcpy(combined_bond_vec.data() + offset,
                           bond_idxs[i].data(), bond_arr_size * sizeof(int));
               offset += bond_arr_size;

               bond_system_idxs.resize(bond_system_idxs.size() +
                                       bond_idxs[i].shape(0));
               std::fill(bond_system_idxs.end() - bond_idxs[i].shape(0),
                         bond_system_idxs.end(), i);
             }
             return new HarmonicBond<RealType>(
                 num_batches, num_atoms, combined_bond_vec, bond_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("bond_idxs"))
      .def("get_idxs", [](Class &pot) -> py::array_t<int, py::array::c_style> {
        std::vector<int> output_idxs = pot.get_idxs_host();
        py::array_t<int, py::array::c_style> out_idx_buffer(
            {pot.get_num_idxs(), pot.IDXS_DIM}, output_idxs.data());
        return out_idx_buffer;
      });
}

template <typename RealType>
void declare_flat_bottom_bond(py::module &m, const char *typestr) {

  using Class = FlatBottomBond<RealType>;
  std::string pyclass_name = std::string("FlatBottomBond_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int num_atoms,
                       const py::array_t<int, py::array::c_style> &bond_idxs) {
             verify_bond_idxs(bond_idxs, 2);
             // Create a vector with all zeros
             std::vector<int> bond_system_idxs(bond_idxs.shape(0), 0);
             const int num_batches = 1;
             return new FlatBottomBond<RealType>(num_batches, num_atoms,
                                                 py_array_to_vector(bond_idxs),
                                                 bond_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("bond_idxs"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &bond_idxs) {
             const int num_batches = bond_idxs.size();
             std::vector<int> combined_bond_vec;
             std::vector<int> bond_system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(bond_idxs[i], 2);
               const unsigned long bond_arr_size = bond_idxs[i].size();
               combined_bond_vec.resize(combined_bond_vec.size() +
                                        bond_arr_size);
               std::memcpy(combined_bond_vec.data() + offset,
                           bond_idxs[i].data(), bond_arr_size * sizeof(int));
               offset += bond_arr_size;

               bond_system_idxs.resize(bond_system_idxs.size() +
                                       bond_idxs[i].shape(0));
               std::fill(bond_system_idxs.end() - bond_idxs[i].shape(0),
                         bond_system_idxs.end(), i);
             }
             return new FlatBottomBond<RealType>(
                 num_batches, num_atoms, combined_bond_vec, bond_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("bond_idxs"))
      .def("get_num_bonds", &Class::num_bonds);
}

template <typename RealType>
void declare_log_flat_bottom_bond(py::module &m, const char *typestr) {

  using Class = LogFlatBottomBond<RealType>;
  std::string pyclass_name = std::string("LogFlatBottomBond_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int num_atoms,
                       const py::array_t<int, py::array::c_style> &bond_idxs,
                       const double beta) {
             verify_bond_idxs(bond_idxs, 2);
             std::vector<int> bond_system_idxs(bond_idxs.shape(0), 0);
             const int num_batches = 1;
             return new LogFlatBottomBond<RealType>(
                 num_batches, num_atoms, py_array_to_vector(bond_idxs),
                 bond_system_idxs, beta);
           }),
           py::arg("num_atoms"), py::arg("bond_idxs"), py::arg("beta"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &bond_idxs,
                       const double beta) {
             const int num_batches = bond_idxs.size();
             std::vector<int> combined_bond_vec;
             std::vector<int> bond_system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(bond_idxs[i], 2);
               const unsigned long bond_arr_size = bond_idxs[i].size();
               combined_bond_vec.resize(combined_bond_vec.size() +
                                        bond_arr_size);
               std::memcpy(combined_bond_vec.data() + offset,
                           bond_idxs[i].data(), bond_arr_size * sizeof(int));
               offset += bond_arr_size;

               bond_system_idxs.resize(bond_system_idxs.size() +
                                       bond_idxs[i].shape(0));
               std::fill(bond_system_idxs.end() - bond_idxs[i].shape(0),
                         bond_system_idxs.end(), i);
             }
             return new LogFlatBottomBond<RealType>(num_batches, num_atoms,
                                                    combined_bond_vec,
                                                    bond_system_idxs, beta);
           }),
           py::arg("num_atoms"), py::arg("bond_idxs"), py::arg("beta"))
      .def("get_num_bonds", &Class::num_bonds);
}

template <typename RealType>
void declare_nonbonded_precomputed(py::module &m, const char *typestr) {

  using Class = NonbondedPairListPrecomputed<RealType>;
  std::string pyclass_name =
      std::string("NonbondedPairListPrecomputed_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int num_atoms,
                       const py::array_t<int, py::array::c_style> &pair_idxs,
                       const double beta, const double cutoff) {
             verify_bond_idxs(pair_idxs, 2);
             std::vector<int> system_idxs(pair_idxs.shape(0), 0);
             const int num_batches = 1;
             return new NonbondedPairListPrecomputed<RealType>(
                 num_batches, num_atoms, py_array_to_vector(pair_idxs),
                 system_idxs, beta, cutoff);
           }),
           py::arg("num_atoms"), py::arg("pair_idxs"), py::arg("beta"),
           py::arg("cutoff"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &pair_idxs,
                       const double beta, const double cutoff) {
             const int num_batches = pair_idxs.size();
             std::vector<int> combined_pair_idxs;
             std::vector<int> system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(pair_idxs[i], 2);
               const unsigned long bond_arr_size = pair_idxs[i].size();
               combined_pair_idxs.resize(combined_pair_idxs.size() +
                                         bond_arr_size);
               std::memcpy(combined_pair_idxs.data() + offset,
                           pair_idxs[i].data(), bond_arr_size * sizeof(int));
               offset += bond_arr_size;

               system_idxs.resize(system_idxs.size() + pair_idxs[i].shape(0));
               std::fill(system_idxs.end() - pair_idxs[i].shape(0),
                         system_idxs.end(), i);
             }
             return new NonbondedPairListPrecomputed<RealType>(
                 num_batches, num_atoms, combined_pair_idxs, system_idxs, beta,
                 cutoff);
           }),
           py::arg("num_atoms"), py::arg("pair_idxs"), py::arg("beta"),
           py::arg("cutoff"));
}

template <typename RealType>
void declare_chiral_atom_restraint(py::module &m, const char *typestr) {

  using Class = ChiralAtomRestraint<RealType>;
  std::string pyclass_name = std::string("ChiralAtomRestraint_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(
          py::init([](const int num_atoms,
                      const py::array_t<int, py::array::c_style> &idxs) {
            verify_bond_idxs(idxs, 4);
            std::vector<int> system_idxs(idxs.shape(0), 0);
            return new ChiralAtomRestraint<RealType>(
                1, num_atoms, py_array_to_vector(idxs), system_idxs);
          }),
          py::arg("num_atoms"), py::arg("idxs"),
          R"pbdoc(Please refer to tmd.potentials.chiral_restraints for documentation on arguments)pbdoc")
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &restraint_idxs) {
             const int num_batches = restraint_idxs.size();
             std::vector<int> combined_restraint_vec;
             std::vector<int> restraint_system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(restraint_idxs[i], 4);
               const unsigned long restraint_arr_size =
                   restraint_idxs[i].size();
               combined_restraint_vec.resize(combined_restraint_vec.size() +
                                             restraint_arr_size);
               std::memcpy(combined_restraint_vec.data() + offset,
                           restraint_idxs[i].data(),
                           restraint_arr_size * sizeof(int));
               offset += restraint_arr_size;

               restraint_system_idxs.resize(restraint_system_idxs.size() +
                                            restraint_idxs[i].shape(0));
               std::fill(restraint_system_idxs.end() -
                             restraint_idxs[i].shape(0),
                         restraint_system_idxs.end(), i);
             }
             return new ChiralAtomRestraint<RealType>(num_batches, num_atoms,
                                                      combined_restraint_vec,
                                                      restraint_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("idxs"));
}

template <typename RealType>
void declare_chiral_bond_restraint(py::module &m, const char *typestr) {

  using Class = ChiralBondRestraint<RealType>;
  std::string pyclass_name = std::string("ChiralBondRestraint_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(
          py::init([](const int num_atoms,
                      const py::array_t<int, py::array::c_style> &idxs,
                      const py::array_t<int, py::array::c_style> &signs) {
            verify_bond_idxs(idxs, 4);
            std::vector<int> system_idxs(idxs.shape(0), 0);
            return new ChiralBondRestraint<RealType>(
                1, num_atoms, py_array_to_vector(idxs),
                py_array_to_vector(signs), system_idxs);
          }),
          py::arg("num_atoms"), py::arg("idxs"), py::arg("signs"),
          R"pbdoc(Please refer to tmd.potentials.chiral_restraints for documentation on arguments)pbdoc")
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &restraint_idxs,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &signs) {
             const int num_batches = restraint_idxs.size();
             std::vector<int> combined_restraint_vec;
             std::vector<int> combined_signs_vec;
             std::vector<int> restraint_system_idxs;
             int offset = 0;
             int sign_offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(restraint_idxs[i], 4);
               const unsigned long restraint_arr_size =
                   restraint_idxs[i].size();
               const unsigned long sign_arr_size = signs[i].size();
               combined_restraint_vec.resize(combined_restraint_vec.size() +
                                             restraint_arr_size);
               std::memcpy(combined_restraint_vec.data() + offset,
                           restraint_idxs[i].data(),
                           restraint_arr_size * sizeof(int));

               combined_signs_vec.resize(combined_signs_vec.size() +
                                         sign_arr_size);
               std::memcpy(combined_signs_vec.data() + sign_offset,
                           signs[i].data(), sign_arr_size * sizeof(int));
               offset += restraint_arr_size;
               sign_offset += sign_arr_size;

               restraint_system_idxs.resize(restraint_system_idxs.size() +
                                            restraint_idxs[i].shape(0));
               std::fill(restraint_system_idxs.end() -
                             restraint_idxs[i].shape(0),
                         restraint_system_idxs.end(), i);
             }
             return new ChiralBondRestraint<RealType>(
                 num_batches, num_atoms, combined_restraint_vec,
                 combined_signs_vec, restraint_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("idxs"), py::arg("signs"));
}

template <typename RealType>
void declare_harmonic_angle(py::module &m, const char *typestr) {

  using Class = HarmonicAngle<RealType>;
  std::string pyclass_name = std::string("HarmonicAngle_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int num_atoms,
                       const py::array_t<int, py::array::c_style> &angle_idxs) {
             verify_bond_idxs(angle_idxs, 3);
             std::vector<int> system_idxs(angle_idxs.shape(0), 0);
             std::vector<int> vec_angle_idxs = py_array_to_vector(angle_idxs);
             return new HarmonicAngle<RealType>(1, num_atoms, vec_angle_idxs,
                                                system_idxs);
           }),
           py::arg("num_atoms"), py::arg("angle_idxs"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &angle_idxs) {
             const int num_batches = angle_idxs.size();
             std::vector<int> combined_angle_vec;
             std::vector<int> angle_system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(angle_idxs[i], 3);
               const unsigned long bond_arr_size = angle_idxs[i].size();
               combined_angle_vec.resize(combined_angle_vec.size() +
                                         bond_arr_size);
               std::memcpy(combined_angle_vec.data() + offset,
                           angle_idxs[i].data(), bond_arr_size * sizeof(int));
               offset += bond_arr_size;

               angle_system_idxs.resize(angle_system_idxs.size() +
                                        angle_idxs[i].shape(0));
               std::fill(angle_system_idxs.end() - angle_idxs[i].shape(0),
                         angle_system_idxs.end(), i);
             }
             return new HarmonicAngle<RealType>(
                 num_batches, num_atoms, combined_angle_vec, angle_system_idxs);
           }),
           py::arg("num_atoms"), py::arg("angle_idxs"))
      .def("get_idxs", [](Class &pot) -> py::array_t<int, py::array::c_style> {
        std::vector<int> output_idxs = pot.get_idxs_host();
        py::array_t<int, py::array::c_style> out_idx_buffer(
            {pot.get_num_idxs(), pot.IDXS_DIM}, output_idxs.data());
        return out_idx_buffer;
      });
}

template <typename RealType>
void declare_centroid_restraint(py::module &m, const char *typestr) {

  using Class = CentroidRestraint<RealType>;
  std::string pyclass_name = std::string("CentroidRestraint_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const py::array_t<int, py::array::c_style> &group_a_idxs,
                       const py::array_t<int, py::array::c_style> &group_b_idxs,
                       const double kb, const double b0) {
             std::vector<int> vec_group_a_idxs =
                 py_array_to_vector(group_a_idxs);
             std::vector<int> vec_group_b_idxs =
                 py_array_to_vector(group_b_idxs);

             return new CentroidRestraint<RealType>(vec_group_a_idxs,
                                                    vec_group_b_idxs, kb, b0);
           }),
           py::arg("group_a_idxs"), py::arg("group_b_idxs"), py::arg("kb"),
           py::arg("b0"));
}

template <typename RealType>
void declare_periodic_torsion(py::module &m, const char *typestr) {

  using Class = PeriodicTorsion<RealType>;
  std::string pyclass_name = std::string("PeriodicTorsion_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init(
               [](const int num_atoms,
                  const py::array_t<int, py::array::c_style> &torsion_idxs) {
                 verify_bond_idxs(torsion_idxs, 4);
                 // Create a vector with all zeros
                 std::vector<int> system_idxs(torsion_idxs.shape(0), 0);
                 std::vector<int> vec_torsion_idxs =
                     py_array_to_vector(torsion_idxs);

                 const int num_batches = 1;
                 return new PeriodicTorsion<RealType>(
                     num_batches, num_atoms, vec_torsion_idxs, system_idxs);
               }),
           py::arg("num_atoms"), py::arg("torsion_idxs"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &torsion_idxs) {
             const int num_batches = torsion_idxs.size();
             std::vector<int> combined_torsion_vec;
             std::vector<int> system_idxs;
             int offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(torsion_idxs[i], 4);
               const unsigned long bond_arr_size = torsion_idxs[i].size();
               combined_torsion_vec.resize(combined_torsion_vec.size() +
                                           bond_arr_size);
               std::memcpy(combined_torsion_vec.data() + offset,
                           torsion_idxs[i].data(), bond_arr_size * sizeof(int));
               offset += bond_arr_size;

               system_idxs.resize(system_idxs.size() +
                                  torsion_idxs[i].shape(0));
               std::fill(system_idxs.end() - torsion_idxs[i].shape(0),
                         system_idxs.end(), i);
             }
             return new PeriodicTorsion<RealType>(
                 num_batches, num_atoms, combined_torsion_vec, system_idxs);
           }),
           py::arg("num_atoms"), py::arg("torsion_idxs"))
      .def("get_idxs", [](Class &pot) -> py::array_t<int, py::array::c_style> {
        std::vector<int> output_idxs = pot.get_idxs_host();
        py::array_t<int, py::array::c_style> out_idx_buffer(
            {pot.get_num_idxs(), pot.IDXS_DIM}, output_idxs.data());
        return out_idx_buffer;
      });
}

template <typename RealType>
void declare_nonbonded_interaction_group(py::module &m, const char *typestr) {
  using Class = NonbondedInteractionGroup<RealType>;
  std::string pyclass_name =
      std::string("NonbondedInteractionGroup_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(
          py::init(
              [](const int N,
                 const py::array_t<int, py::array::c_style> &row_atom_idxs_i,
                 const double beta, const double cutoff,
                 std::optional<py::array_t<int, py::array::c_style>>
                     &col_atom_idxs_i,
                 const bool disable_hilbert_sort, const double nblist_padding) {
                std::vector<int> row_atom_idxs =
                    py_array_to_vector(row_atom_idxs_i);

                std::vector<int> col_atom_idxs;
                if (col_atom_idxs_i) {
                  col_atom_idxs.resize(col_atom_idxs_i->size());
                  std::memcpy(col_atom_idxs.data(), col_atom_idxs_i->data(),
                              col_atom_idxs_i->size() * sizeof(int));
                } else {
                  std::set<int> unique_row_atom_idxs =
                      unique_idxs(row_atom_idxs);
                  col_atom_idxs =
                      get_indices_difference(N, unique_row_atom_idxs);
                }

                return new NonbondedInteractionGroup<RealType>(
                    N, row_atom_idxs, col_atom_idxs, beta, cutoff,
                    disable_hilbert_sort, nblist_padding);
              }),
          py::arg("num_atoms"), py::arg("row_atom_idxs_i"), py::arg("beta"),
          py::arg("cutoff"), py::arg("col_atom_idxs_i") = py::none(),
          py::arg("disable_hilbert_sort") = false,
          py::arg("nblist_padding") = 0.1,
          R"pbdoc(
                    Set up the NonbondedInteractionGroup.

                    Parameters
                    ----------
                    num_atoms: int
                        Number of atoms.

                    row_atom_idxs: NDArray
                        First group of atoms in the interaction.

                    beta: float

                    cutoff: float
                        Ignore all interactions beyond this distance in nm.

                    col_atom_idxs: Optional[NDArray]
                        Second group of atoms in the interaction. If not specified,
                        use all of the atoms not in the `row_atom_idxs`.

                    disable_hilbert_sort: bool
                        Set to True to disable the Hilbert sort.

                    nblist_padding: float
                        Margin for the neighborlist.

            )pbdoc")
      .def("set_atom_idxs", &Class::set_atom_idxs, py::arg("row_atom_idxs"),
           py::arg("col_atom_idxs"),
           R"pbdoc(
                    Set up the atom idxs for the NonbondedInteractionGroup.
                    The interaction is defined between two groups of atom idxs,
                    `row_atom_idxs` and `col_atom_idxs`. These should be a disjoint
                    list of idxs.

                    Parameters
                    ----------
                    row_atom_idxs: NDArray
                        First group of atoms in the interaction.

                    col_atom_idxs: NDArray
                        Second group of atoms in the interaction.

            )pbdoc")
      .def("get_row_idxs", &Class::get_row_idxs)
      .def("get_col_idxs", &Class::get_col_idxs)
      .def("get_nblist_padding", &Class::get_nblist_padding)
      .def("set_nblist_padding", &Class::set_nblist_padding)
      .def("get_compute_col_grads", &Class::get_compute_col_grads)
      .def("set_compute_col_grads", &Class::set_compute_col_grads);
}

template <typename RealType, bool Negated>
void declare_nonbonded_pair_list(py::module &m, const char *typestr) {
  using Class = NonbondedPairList<RealType, Negated>;
  std::string pyclass_name;
  // If the pair list is negated, it is intended to be used for exclusions
  if (Negated) {
    pyclass_name = std::string("NonbondedExclusions_") + typestr;
  } else {
    pyclass_name = std::string("NonbondedPairList_") + typestr;
  }
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(
          py::init([](const int num_atoms,
                      const py::array_t<int, py::array::c_style> &pair_idxs_i,
                      const py::array_t<RealType, py::array::c_style> &scales_i,
                      const RealType beta, const RealType cutoff) {
            verify_bond_idxs(pair_idxs_i, 2);
            std::vector<int> pair_idxs = py_array_to_vector(pair_idxs_i);

            std::vector<RealType> scales = py_array_to_vector(scales_i);
            std::vector<int> system_idxs(pair_idxs_i.shape(0), 0);
            const int num_batches = 1;
            return new NonbondedPairList<RealType, Negated>(
                num_batches, num_atoms, pair_idxs, scales, system_idxs, beta,
                cutoff);
          }),
          py::arg("num_atoms"), py::arg("pair_idxs_i"), py::arg("scales_i"),
          py::arg("beta"), py::arg("cutoff"))
      .def(py::init([](const int num_atoms,
                       const std::vector<py::array_t<int, py::array::c_style>>
                           &pair_idxs,
                       const std::vector<
                           py::array_t<RealType, py::array::c_style>> &scales,
                       const RealType beta, const RealType cutoff) {
             const int num_batches = pair_idxs.size();
             std::vector<int> combined_pair_idxs;
             std::vector<RealType> combined_scales;
             std::vector<int> system_idxs;
             int offset = 0;
             int scale_offset = 0;
             for (int i = 0; i < num_batches; i++) {
               verify_bond_idxs(pair_idxs[i], 2);
               const unsigned long pair_idxs_arr_size = pair_idxs[i].size();
               const unsigned long sign_arr_size = scales[i].size();
               combined_pair_idxs.resize(combined_pair_idxs.size() +
                                         pair_idxs_arr_size);
               std::memcpy(combined_pair_idxs.data() + offset,
                           pair_idxs[i].data(),
                           pair_idxs_arr_size * sizeof(int));

               combined_scales.resize(combined_scales.size() + sign_arr_size);
               std::memcpy(combined_scales.data() + scale_offset,
                           scales[i].data(), sign_arr_size * sizeof(RealType));
               offset += pair_idxs_arr_size;
               scale_offset += sign_arr_size;

               system_idxs.resize(system_idxs.size() + pair_idxs[i].shape(0));
               std::fill(system_idxs.end() - pair_idxs[i].shape(0),
                         system_idxs.end(), i);
             }
             return new NonbondedPairList<RealType, Negated>(
                 num_batches, num_atoms, combined_pair_idxs, combined_scales,
                 system_idxs, beta, cutoff);
           }),
           py::arg("num_atoms"), py::arg("pair_idxs"), py::arg("scales"),
           py::arg("beta"), py::arg("cutoff"))
      .def("get_idxs",
           [](Class &pot) -> py::array_t<int, py::array::c_style> {
             std::vector<int> output_idxs = pot.get_idxs_host();
             py::array_t<int, py::array::c_style> out_idx_buffer(
                 {pot.get_num_idxs(), pot.IDXS_DIM}, output_idxs.data());
             return out_idx_buffer;
           })
      .def("get_scales",
           [](Class &pot) -> py::array_t<RealType, py::array::c_style> {
             std::vector<RealType> output_scales = pot.get_scales_host();
             py::array_t<RealType, py::array::c_style> out_scale_buffer(
                 {pot.get_num_idxs(), pot.IDXS_DIM}, output_scales.data());
             return out_scale_buffer;
           });
  ;
}

template <typename RealType>
void declare_mover(py::module &m, const char *typestr) {

  using Class = Mover<RealType>;
  std::string pyclass_name = std::string("Mover_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def("set_interval", &Class::set_interval, py::arg("interval"))
      .def("get_interval", &Class::get_interval)
      .def("set_step", &Class::set_step, py::arg("step"))
      .def(
          "move",
          [](Class &mover,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &box)
              -> py::tuple {
            verify_coords_and_box(coords, box);
            const int N = coords.shape()[0];
            const int D = box.shape()[0];

            std::array<std::vector<RealType>, 2> result =
                mover.move_host(N, coords.data(), box.data());

            py::array_t<RealType, py::array::c_style> out_x_buffer(
                {N, D}, result[0].data());

            py::array_t<RealType, py::array::c_style> box_buffer(
                {D, D}, result[1].data());
            return py::make_tuple(out_x_buffer, box_buffer);
          },
          py::arg("coords"), py::arg("box"));
}

template <typename RealType>
void declare_barostat(py::module &m, const char *typestr) {

  using Class = MonteCarloBarostat<RealType>;
  std::string pyclass_name = std::string("MonteCarloBarostat_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Mover<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init(
               [](const int N, const RealType pressure,
                  const RealType temperature,
                  std::vector<std::vector<int>> &group_idxs, const int interval,
                  std::vector<std::shared_ptr<BoundPotential<RealType>>> &bps,
                  const int seed, const bool adaptive_scaling_enabled,
                  const RealType initial_volume_scale_factor) {
                 return new Class(N, pressure, temperature, group_idxs,
                                  interval, bps, seed, adaptive_scaling_enabled,
                                  initial_volume_scale_factor);
               }),
           py::arg("N"), py::arg("pressure"), py::arg("temperature"),
           py::arg("group_idxs"), py::arg("interval"), py::arg("bps"),
           py::arg("seed"), py::arg("adaptive_scaling_enabled"),
           py::arg("initial_volume_scale_factor"))
      .def("set_volume_scale_factor", &Class::set_volume_scale_factor,
           py::arg("volume_scale_factor"))
      .def("get_volume_scale_factor", &Class::get_volume_scale_factor)
      .def("set_adaptive_scaling", &Class::set_adaptive_scaling,
           py::arg("adaptive_scaling_enabled"))
      .def("get_adaptive_scaling", &Class::get_adaptive_scaling)
      .def("set_pressure", &Class::set_pressure, py::arg("pressure"));
}

template <typename RealType>
void declare_summed_potential(py::module &m, const char *typestr) {

  using Class = SummedPotential<RealType>;
  std::string pyclass_name = std::string("SummedPotential_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init(
               [](std::vector<std::shared_ptr<Potential<RealType>>> &potentials,
                  std::vector<int> &params_sizes, bool parallel) {
                 return new SummedPotential<RealType>(potentials, params_sizes,
                                                      parallel);
               }),

           py::arg("potentials"), py::arg("params_sizes"),
           py::arg("parallel") = true)
      .def("get_potentials", &SummedPotential<RealType>::get_potentials);
}

template <typename RealType>
void declare_fanout_summed_potential(py::module &m, const char *typestr) {

  using Class = FanoutSummedPotential<RealType>;
  std::string pyclass_name = std::string("FanoutSummedPotential_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Potential<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init(
               [](std::vector<std::shared_ptr<Potential<RealType>>> &potentials,
                  bool parallel) {
                 return new FanoutSummedPotential<RealType>(potentials,
                                                            parallel);
               }),
           py::arg("potentials"), py::arg("parallel") = true)
      .def("get_potentials", &FanoutSummedPotential<RealType>::get_potentials);
}

template <typename RealType>
void declare_segmented_sum_exp(py::module &m, const char *typestr) {

  using Class = SegmentedSumExp<RealType>;
  std::string pyclass_name = std::string("SegmentedSumExp_") + typestr;
  py::class_<Class, std::shared_ptr<Class>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int max_vals_per_segment, const int segments) {
             return new Class(max_vals_per_segment, segments);
           }),
           py::arg("max_vals_per_segment"), py::arg("num_segments"))
      .def(
          "logsumexp",
          [](Class &summer, const std::vector<std::vector<double>> &vals)
              -> std::vector<RealType> {
            std::vector<std::vector<RealType>> real_batches(vals.size());
            for (unsigned long i = 0; i < vals.size(); i++) {
              real_batches[i] =
                  py_vector_to_vector_with_cast<double, RealType>(vals[i]);
            }
            std::vector<RealType> results = summer.logsumexp_host(real_batches);

            return results;
          },
          py::arg("values"),
          R"pbdoc(
        Compute the logsumexp of a batch of vectors

        Parameters
        ----------

        vals: vector of vectors containing doubles
            A vector of vectors to compute the logsumexp

        Returns
        -------
        Array of sample indices
            Shape (vals.size(), )
        )pbdoc");
}

template <typename RealType>
void declare_biased_deletion_exchange_move(py::module &m, const char *typestr) {

  using Class = BDExchangeMove<RealType>;
  std::string pyclass_name = std::string("BDExchangeMove_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, Mover<RealType>>(
      m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
      .def(py::init([](const int N,
                       const std::vector<std::vector<int>> &target_mols,
                       const py::array_t<RealType, py::array::c_style> &params,
                       const RealType temperature, const RealType nb_beta,
                       const RealType cutoff, const int seed,
                       const int num_proposals_per_move, const int interval,
                       const int batch_size) {
             size_t params_dim = params.ndim();
             if (num_proposals_per_move <= 0) {
               throw std::runtime_error(
                   "proposals per move must be greater than 0");
             }
             if (params_dim != 2) {
               throw std::runtime_error("parameters dimensions must be 2");
             }
             if (params.shape(0) != N) {
               throw std::runtime_error("Number of parameters must match N");
             }
             if (target_mols.size() == 0) {
               throw std::runtime_error("must provide at least one molecule");
             }
             if (interval <= 0) {
               throw std::runtime_error("must provide interval greater than 0");
             }
             if (batch_size <= 0) {
               throw std::runtime_error(
                   "must provide batch size greater than 0");
             }
             if (batch_size > num_proposals_per_move) {
               throw std::runtime_error("number of proposals per move must be "
                                        "greater than batch size");
             }
             std::vector<RealType> v_params = py_array_to_vector(params);
             return new Class(N, target_mols, v_params, temperature, nb_beta,
                              cutoff, seed, num_proposals_per_move, interval,
                              batch_size);
           }),
           py::arg("N"), py::arg("target_mols"), py::arg("params"),
           py::arg("temperature"), py::arg("nb_beta"), py::arg("cutoff"),
           py::arg("seed"), py::arg("num_proposals_per_move"),
           py::arg("interval"), py::arg("batch_size") = 1)
      .def(
          "move",
          [](Class &mover,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &box)
              -> py::tuple {
            verify_coords_and_box(coords, box);
            const int N = coords.shape()[0];
            const int D = coords.shape()[1];

            std::array<std::vector<RealType>, 2> result =
                mover.move_host(N, coords.data(), box.data());

            py::array_t<RealType, py::array::c_style> out_x_buffer(
                {N, D}, result[0].data());

            py::array_t<RealType, py::array::c_style> box_buffer(
                {D, D}, result[1].data());

            return py::make_tuple(out_x_buffer, box_buffer);
          },
          py::arg("coords"), py::arg("box"))
      .def(
          "compute_incremental_log_weights",
          [](Class &mover,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &box,
             const py::array_t<int, py::array::c_style> &mol_idxs,
             const py::array_t<RealType, py::array::c_style> &quaternions,
             const py::array_t<RealType, py::array::c_style> &translations)
              -> std::vector<std::vector<RealType>> {
            verify_coords_and_box(coords, box);
            const int N = coords.shape()[0];

            if (mol_idxs.size() != static_cast<ssize_t>(mover.batch_size())) {
              throw std::runtime_error(
                  "number of mol idxs must match batch size");
            }

            if (quaternions.shape()[0] !=
                static_cast<ssize_t>(mover.batch_size())) {
              throw std::runtime_error(
                  "number of quaternions must match batch size");
            }
            if (quaternions.shape()[1] != 4) {
              throw std::runtime_error("each quaternion must be of length 4");
            }

            if (translations.shape()[0] !=
                static_cast<ssize_t>(mover.batch_size())) {
              throw std::runtime_error(
                  "number of translations must match batch size");
            }
            if (translations.shape()[1] != 3) {
              throw std::runtime_error("each translation must be of length 3");
            }

            std::vector<RealType> h_quats =
                py_array_to_vector_with_cast<RealType, RealType>(quaternions);
            std::vector<RealType> h_translations =
                py_array_to_vector_with_cast<RealType, RealType>(translations);

            std::vector<std::vector<RealType>> weights =
                mover.compute_incremental_log_weights_host(
                    N, coords.data(), box.data(), mol_idxs.data(), &h_quats[0],
                    &h_translations[0]);
            return weights;
          },
          py::arg("coords"), py::arg("box"), py::arg("mol_idxs"),
          py::arg("quaternions"), py::arg("translation"))
      .def(
          "compute_initial_log_weights",
          [](Class &mover,
             const py::array_t<RealType, py::array::c_style> &coords,
             const py::array_t<RealType, py::array::c_style> &box)
              -> std::vector<RealType> {
            verify_coords_and_box(coords, box);
            const int N = coords.shape()[0];

            std::vector<RealType> weights =
                mover.compute_initial_log_weights_host(N, coords.data(),
                                                       box.data());
            return weights;
          },
          py::arg("coords"), py::arg("box"))
      .def("get_params",
           [](Class &mover) -> py::array_t<RealType, py::array::c_style> {
             std::vector<RealType> flat_params = mover.get_params();
             const int D = PARAMS_PER_ATOM;
             const int N = flat_params.size() / D;
             py::array_t<RealType, py::array::c_style> out_params(
                 {N, D}, flat_params.data());
             return out_params;
           })
      .def(
          "set_params",
          [](Class &mover,
             const py::array_t<RealType, py::array::c_style> &params) {
            mover.set_params(py_array_to_vector(params));
          },
          py::arg("params"))
      .def("last_log_probability", &Class::log_probability_host,
           R"pbdoc(
        Returns the last log probability.

        Only meaningful/valid when batch_size == 1 and num_proposals_per_move == 1 else
        the value is simply the first value in the buffer which in the case of a batch size greater than
        1 is the first proposal in the batch and in the case of num_proposals_per_move greater than 1
        the probability of the last move, which may or may not have been accepted.
        )pbdoc")
      .def("last_raw_log_probability", &Class::raw_log_probability_host)
      .def("n_accepted", &Class::n_accepted)
      .def("n_proposed", &Class::n_proposed)
      .def("acceptance_fraction", &Class::acceptance_fraction)
      .def("get_before_log_weights", &Class::get_before_log_weights)
      .def("get_after_log_weights", &Class::get_after_log_weights)
      .def("batch_size", &Class::batch_size);
}

template <typename RealType>
void declare_targeted_insertion_biased_deletion_exchange_move(
    py::module &m, const char *typestr) {

  using Class = TIBDExchangeMove<RealType>;
  std::string pyclass_name = std::string("TIBDExchangeMove_") + typestr;
  py::class_<Class, std::shared_ptr<Class>, BDExchangeMove<RealType>,
             Mover<RealType>>(m, pyclass_name.c_str(), py::buffer_protocol(),
                              py::dynamic_attr())
      .def(py::init([](const int N, const std::vector<int> &ligand_idxs,
                       const std::vector<std::vector<int>> &target_mols,
                       const py::array_t<RealType, py::array::c_style> &params,
                       const RealType temperature, const RealType nb_beta,
                       const RealType cutoff, const RealType radius,
                       const int seed, const int num_proposals_per_move,
                       const int interval, const int batch_size) {
             size_t params_dim = params.ndim();
             if (num_proposals_per_move <= 0) {
               throw std::runtime_error(
                   "proposals per move must be greater than 0");
             }
             if (params_dim != 2) {
               throw std::runtime_error("parameters dimensions must be 2");
             }
             if (params.shape(0) != N) {
               throw std::runtime_error("Number of parameters must match N");
             }
             if (ligand_idxs.size() == 0) {
               throw std::runtime_error(
                   "must provide at least one atom for the ligand indices");
             }
             if (target_mols.size() == 0) {
               throw std::runtime_error("must provide at least one molecule");
             }
             if (interval <= 0) {
               throw std::runtime_error("must provide interval greater than 0");
             }
             if (batch_size <= 0) {
               throw std::runtime_error(
                   "must provide batch size greater than 0");
             }
             if (batch_size > num_proposals_per_move) {
               throw std::runtime_error("number of proposals per move must be "
                                        "greater than batch size");
             }
             std::vector<RealType> v_params = py_array_to_vector(params);
             return new Class(N, ligand_idxs, target_mols, v_params,
                              temperature, nb_beta, cutoff, radius, seed,
                              num_proposals_per_move, interval, batch_size);
           }),
           py::arg("N"), py::arg("ligand_idxs"), py::arg("target_mols"),
           py::arg("params"), py::arg("temperature"), py::arg("nb_beta"),
           py::arg("cutoff"), py::arg("radius"), py::arg("seed"),
           py::arg("num_proposals_per_move"), py::arg("interval"),
           py::arg("batch_size") = 1);
}

const py::array_t<double, py::array::c_style>
py_rmsd_align(const py::array_t<double, py::array::c_style> &x1,
              const py::array_t<double, py::array::c_style> &x2) {

  int N1 = x1.shape()[0];
  int N2 = x2.shape()[0];

  int D1 = x1.shape()[1];
  int D2 = x2.shape()[1];

  if (N1 != N2) {
    throw std::runtime_error("N1 != N2");
  }

  if (D1 != 3) {
    throw std::runtime_error("D1 != 3");
  }

  if (D2 != 3) {
    throw std::runtime_error("D2 != 3");
  }

  py::array_t<double, py::array::c_style> py_x2_aligned({N1, D1});

  rmsd_align_cpu(N1, x1.data(), x2.data(), py_x2_aligned.mutable_data());

  return py_x2_aligned;
}

template <typename RealType>
py::array_t<RealType, py::array::c_style> py_atom_by_atom_energies(
    const py::array_t<int, py::array::c_style> &target_atoms,
    const py::array_t<RealType, py::array::c_style> &coords,
    const py::array_t<RealType, py::array::c_style> &params,
    const py::array_t<RealType, py::array::c_style> &box,
    const RealType nb_beta, const RealType cutoff) {

  const int N = coords.shape()[0];
  verify_coords_and_box(coords, box);

  std::vector<int> v_target_atoms = py_array_to_vector(target_atoms);
  std::vector<RealType> v_coords = py_array_to_vector(coords);
  std::vector<RealType> v_params = py_array_to_vector(params);
  std::vector<RealType> v_box = py_array_to_vector(box);

  std::vector<RealType> output_energies =
      compute_atom_by_atom_energies<RealType>(
          N, v_target_atoms, v_coords, v_params, v_box,
          static_cast<RealType>(nb_beta), static_cast<RealType>(cutoff));

  py::array_t<RealType, py::array::c_style> py_energy(
      {static_cast<int>(target_atoms.size()), N});
  for (unsigned int i = 0; i < output_energies.size(); i++) {
    py_energy.mutable_data()[i] = output_energies[i];
  }
  return py_energy;
}

template <typename RealType>
py::tuple
py_inner_outer_mols(const py::array_t<int, py::array::c_style> &center_atoms,
                    const py::array_t<RealType, py::array::c_style> &coords,
                    const py::array_t<RealType, py::array::c_style> &box,
                    const std::vector<std::vector<int>> &group_idxs,
                    const RealType radius) {

  verify_coords_and_box(coords, box);

  std::vector<int> v_center_atoms = py_array_to_vector(center_atoms);
  std::vector<RealType> v_coords = py_array_to_vector(coords);
  std::vector<RealType> v_box = py_array_to_vector(box);

  std::array<std::vector<int>, 2> inner_and_outer =
      get_inner_and_outer_mols<RealType>(v_center_atoms, v_coords, v_box,
                                         group_idxs, radius);

  return py::make_tuple(inner_and_outer[0], inner_and_outer[1]);
}

template <typename RealType>
py::array_t<RealType, py::array::c_style>
py_rotate_coords(const py::array_t<RealType, py::array::c_style> &coords,
                 const py::array_t<RealType, py::array::c_style> &quaternions) {
  verify_coords(coords);

  size_t quaternions_ndims = quaternions.ndim();
  if (quaternions_ndims != 2) {
    throw std::runtime_error("quaternions dimensions must be 2");
  }
  if (quaternions.shape(quaternions_ndims - 1) != 4) {
    throw std::runtime_error(
        "quaternions must have a shape that is 4 dimensional");
  }

  std::vector<RealType> v_quaternions =
      py_array_to_vector_with_cast<RealType, RealType>(quaternions);

  const int N = coords.shape(0);
  const int num_rotations = quaternions.shape(0);
  py::array_t<RealType, py::array::c_style> py_rotated_coords(
      {N, num_rotations, 3});
  rotate_coordinates_host<RealType>(N, num_rotations, coords.data(),
                                    &v_quaternions[0],
                                    py_rotated_coords.mutable_data());
  return py_rotated_coords;
}

template <typename RealType>
py::array_t<RealType, py::array::c_style> py_rotate_and_translate_mol(
    const py::array_t<RealType, py::array::c_style> &coords,
    const py::array_t<RealType, py::array::c_style> &box,
    const py::array_t<RealType, py::array::c_style> &quaternions,
    const py::array_t<RealType, py::array::c_style> &translations) {
  verify_coords_and_box(coords, box);

  if (quaternions.ndim() != 2) {
    throw std::runtime_error("quaternions dimensions must be 2");
  }
  if (quaternions.shape(1) != 4) {
    throw std::runtime_error("quaternions must be of length 4");
  }

  if (translations.ndim() != 2) {
    throw std::runtime_error("translations dimensions must be 2");
  }
  if (translations.shape(1) != 3) {
    throw std::runtime_error("translations must be of size 3");
  }

  if (quaternions.shape(0) != translations.shape(0)) {
    throw std::runtime_error(
        "Number of quaternions and translations must match");
  }

  std::vector<RealType> v_quaternions =
      py_array_to_vector_with_cast<RealType, RealType>(quaternions);
  std::vector<RealType> v_translations =
      py_array_to_vector_with_cast<RealType, RealType>(translations);

  const int batch_size = quaternions.shape(0);

  const int N = coords.shape(0);
  py::array_t<RealType, py::array::c_style> py_rotated_coords(
      {batch_size, N, 3});
  rotate_coordinates_and_translate_mol_host<RealType>(
      N, batch_size, coords.data(), box.data(), &v_quaternions[0],
      &v_translations[0], py_rotated_coords.mutable_data());
  return py_rotated_coords;
}

template <typename RealType>
py::array_t<RealType, py::array::c_style>
py_translations_inside_and_outside_sphere_host(
    const int num_translations,
    const py::array_t<RealType, py::array::c_style> &box,
    const py::array_t<RealType, py::array::c_style> &center,
    const RealType radius, const int seed) {

  if (center.size() != 3) {
    throw std::runtime_error("Center must be of length 3");
  }

  std::vector<RealType> v_center =
      py_array_to_vector_with_cast<RealType, RealType>(center);
  std::vector<RealType> v_box = py_array_to_vector(box);

  std::vector<RealType> translations =
      translations_inside_and_outside_sphere_host<RealType>(
          num_translations, v_box, v_center, static_cast<RealType>(radius),
          seed);
  py::array_t<RealType, py::array::c_style> py_translations(
      {num_translations, 2, 3});
  for (unsigned int i = 0; i < translations.size(); i++) {
    py_translations.mutable_data()[i] = translations[i];
  }
  return py_translations;
}

void py_cuda_device_reset() { cudaDeviceReset(); }

PYBIND11_MODULE(custom_ops, m) {
  py::register_exception<InvalidHardware>(m, "InvalidHardware");

  m.def("rmsd_align", &py_rmsd_align, "RMSD align two molecules", py::arg("x1"),
        py::arg("x2"));

  declare_mover<double>(m, "f64");
  declare_mover<float>(m, "f32");
  declare_barostat<double>(m, "f64");
  declare_barostat<float>(m, "f32");

  declare_integrator<double>(m, "f64");
  declare_integrator<float>(m, "f32");
  declare_langevin_integrator<double>(m, "f64");
  declare_langevin_integrator<float>(m, "f32");
  declare_velocity_verlet_integrator<double>(m, "f64");
  declare_velocity_verlet_integrator<float>(m, "f32");

  declare_potential<double>(m, "f64");
  declare_potential<float>(m, "f32");
  declare_bound_potential<double>(m, "f64");
  declare_bound_potential<float>(m, "f32");
  declare_summed_potential<double>(m, "f64");
  declare_summed_potential<float>(m, "f32");
  declare_fanout_summed_potential<double>(m, "f64");
  declare_fanout_summed_potential<float>(m, "f32");

  declare_potential_executor<double>(m, "f64");
  declare_potential_executor<float>(m, "f32");

  declare_neighborlist<double>(m, "f64");
  declare_neighborlist<float>(m, "f32");

  declare_hilbert_sort<double>(m, "f64");
  declare_hilbert_sort<float>(m, "f32");

  declare_centroid_restraint<double>(m, "f64");
  declare_centroid_restraint<float>(m, "f32");

  declare_harmonic_bond<double>(m, "f64");
  declare_harmonic_bond<float>(m, "f32");

  declare_flat_bottom_bond<double>(m, "f64");
  declare_flat_bottom_bond<float>(m, "f32");

  declare_log_flat_bottom_bond<double>(m, "f64");
  declare_log_flat_bottom_bond<float>(m, "f32");

  declare_chiral_atom_restraint<double>(m, "f64");
  declare_chiral_atom_restraint<float>(m, "f32");

  declare_chiral_bond_restraint<double>(m, "f64");
  declare_chiral_bond_restraint<float>(m, "f32");

  declare_harmonic_angle<double>(m, "f64");
  declare_harmonic_angle<float>(m, "f32");

  declare_periodic_torsion<double>(m, "f64");
  declare_periodic_torsion<float>(m, "f32");

  declare_nonbonded_interaction_group<double>(m, "f64");
  declare_nonbonded_interaction_group<float>(m, "f32");

  declare_nonbonded_precomputed<double>(m, "f64");
  declare_nonbonded_precomputed<float>(m, "f32");

  declare_nonbonded_pair_list<double, false>(m, "f64");
  declare_nonbonded_pair_list<float, false>(m, "f32");

  declare_nonbonded_pair_list<double, true>(m, "f64");
  declare_nonbonded_pair_list<float, true>(m, "f32");

  declare_segmented_weighted_random_sampler<double>(m, "f64");
  declare_segmented_weighted_random_sampler<float>(m, "f32");

  declare_segmented_sum_exp<double>(m, "f64");
  declare_segmented_sum_exp<float>(m, "f32");

  declare_nonbonded_mol_energy<double>(m, "f64");
  declare_nonbonded_mol_energy<float>(m, "f32");

  declare_biased_deletion_exchange_move<double>(m, "f64");
  declare_biased_deletion_exchange_move<float>(m, "f32");

  declare_targeted_insertion_biased_deletion_exchange_move<double>(m, "f64");
  declare_targeted_insertion_biased_deletion_exchange_move<float>(m, "f32");

  declare_context<double>(m, "f64");
  declare_context<float>(m, "f32");

  // TESTING DEFINITIONS
  m.def("cuda_device_reset", &py_cuda_device_reset,
        "Destroy all allocations and reset all state on the current device in "
        "the current process.");
  m.def("rotate_coords_f32", &py_rotate_coords<float>,
        "Function for testing rotation of coordinates in CUDA",
        py::arg("coords"), py::arg("quaternions"));
  m.def("rotate_coords_f64", &py_rotate_coords<double>,
        "Function for testing rotation of coordinates in CUDA",
        py::arg("coords"), py::arg("quaternions"));
  m.def(
      "rotate_and_translate_mol_f32", &py_rotate_and_translate_mol<float>,
      "Function for testing kernel for rotating and translating a mol in CUDA",
      py::arg("coords"), py::arg("box"), py::arg("quaternion"),
      py::arg("translation"));
  m.def(
      "rotate_and_translate_mol_f64", &py_rotate_and_translate_mol<double>,
      "Function for testing kernel for rotating and translating a mol in CUDA",
      py::arg("coords"), py::arg("box"), py::arg("quaternion"),
      py::arg("translation"));
  m.def("atom_by_atom_energies_f32", &py_atom_by_atom_energies<float>,
        "Function for testing atom by atom energies", py::arg("target_atoms"),
        py::arg("coords"), py::arg("params"), py::arg("box"),
        py::arg("nb_beta"), py::arg("nb_cutoff"));
  m.def("atom_by_atom_energies_f64", &py_atom_by_atom_energies<double>,
        "Function for testing atom by atom energies", py::arg("target_atoms"),
        py::arg("coords"), py::arg("params"), py::arg("box"),
        py::arg("nb_beta"), py::arg("nb_cutoff"));
  m.def("inner_and_outer_mols_f32", &py_inner_outer_mols<float>,
        "Function to test computation of inner and outer mols",
        py::arg("center_atoms"), py::arg("coords"), py::arg("box"),
        py::arg("group_idxs"), py::arg("radius"));
  m.def("inner_and_outer_mols_f64", &py_inner_outer_mols<double>,
        "Function to test computation of inner and outer mols",
        py::arg("center_atoms"), py::arg("coords"), py::arg("box"),
        py::arg("group_idxs"), py::arg("radius"));
  m.def("translations_inside_and_outside_sphere_host_f32",
        &py_translations_inside_and_outside_sphere_host<float>,
        "Function to test translations within sphere",
        py::arg("num_translations"), py::arg("box"), py::arg("center"),
        py::arg("radius"), py::arg("seed"));
  m.def("translations_inside_and_outside_sphere_host_f64",
        &py_translations_inside_and_outside_sphere_host<double>,
        "Function to test translations within sphere",
        py::arg("num_translations"), py::arg("box"), py::arg("center"),
        py::arg("radius"), py::arg("seed"));

  m.attr("FIXED_EXPONENT") = py::int_(FIXED_EXPONENT);
}
